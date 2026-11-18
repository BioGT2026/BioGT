"""
Graph Attention Network (GAT) Model for Disease-Biomarker Association Prediction
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv
from dgl.nn.pytorch import GATConv
from Model.common import DEBUG_LOGGING


class HeteroGATLayer(nn.Module):
    """Single layer of heterogeneous GAT"""
    def __init__(self, in_feats, out_feats, ntypes, etypes, num_heads=8):
        super().__init__()
        self.conv = HeteroGraphConv(
            {
                etype: GATConv(in_feats, out_feats // num_heads, num_heads, bias=True)
                for etype in etypes
            },
            aggregate='sum'
        )
        self.ntypes = ntypes
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads

    def forward(self, g, h_dict):
        """
        Args:
            g: DGL heterogeneous graph
            h_dict: Dictionary of node features {ntype: tensor}
        
        Returns:
            Dictionary of updated node features
        """
        # Debug: Check input dimensions
        if DEBUG_LOGGING:
            logging.debug(f"[HeteroGATLayer] Input dimensions:")
            for ntype, feat in h_dict.items():
                logging.debug(f"  {ntype}: {feat.shape}")
        
        for ntype, feat in h_dict.items():
            if feat.shape[-1] != self.in_feats:
                logging.error(f"[HeteroGATLayer] ERROR: Input feature dimension mismatch for '{ntype}'!")
                logging.error(f"  Expected: {self.in_feats}, Got: {feat.shape[-1]}")
                raise RuntimeError(
                    f"HeteroGATLayer input dimension mismatch for '{ntype}': "
                    f"expected {self.in_feats}, got {feat.shape[-1]}"
                )
        
        h_out_dict = self.conv(g, h_dict)
        
        # Reshape GAT outputs from [num_nodes, num_heads, out_feats_per_head] to [num_nodes, num_heads * out_feats_per_head]
        for ntype in h_out_dict:
            feat = h_out_dict[ntype]
            if len(feat.shape) == 3:  # [num_nodes, num_heads, out_feats_per_head]
                num_nodes, num_heads, out_feats_per_head = feat.shape
                h_out_dict[ntype] = feat.reshape(num_nodes, num_heads * out_feats_per_head)
                if DEBUG_LOGGING:
                    logging.debug(f"[HeteroGATLayer] Reshaped {ntype} output from {(num_nodes, num_heads, out_feats_per_head)} to {h_out_dict[ntype].shape}")
        
        # Apply ReLU activation
        for ntype in h_out_dict:
            h_out_dict[ntype] = F.relu(h_out_dict[ntype])
        
        # IMPORTANT: HeteroGraphConv only returns node types that received messages.
        # We need to preserve node types that didn't receive messages but exist in the input.
        # This ensures all node types from the input are present in the output.
        missing_ntypes = set(h_dict.keys()) - set(h_out_dict.keys())
        if missing_ntypes:
            if DEBUG_LOGGING:
                logging.debug(f"HeteroGATLayer: Node types without incoming messages: {missing_ntypes}. Preserving their input features.")
            for ntype in missing_ntypes:
                # Preserve the input features for node types that didn't receive messages
                h_out_dict[ntype] = h_dict[ntype]
        
        return h_out_dict


class GATModel(nn.Module):
    """
    Graph Attention Network Model for heterogeneous graphs
    Designed for link prediction in disease-biomarker association networks
    """
    def __init__(self, node_feature_embeddings, gat_hidden, gat_out,
                 num_gat_layers, ntypes, etypes, target_node_types,
                 num_nodes_dict, node_info=None, dropout=0.1,
                 num_heads=8, freeze_pretrained=True):
        super().__init__()
        
        if DEBUG_LOGGING:
            logging.info("[GATModel.__init__] Starting initialization")
            logging.info(f"[GATModel.__init__] Parameters: gat_hidden={gat_hidden}, gat_out={gat_out}, num_gat_layers={num_gat_layers}, num_heads={num_heads}")
            logging.info(f"[GATModel.__init__] Input node_feature_embeddings keys: {list(node_feature_embeddings.keys()) if node_feature_embeddings else None}")
        
        self.disable_hgcn = False
        self.disable_transformer = True
        self.disable_moe = False
        self.fusion_mode = 'sequential'
        self.enable_explainability = False
        self.node_info = node_info
        
        # Initialize node embeddings (consistent with DiseaseAssociationModel)
        self.node_embeddings = nn.ModuleDict()
        for ntype in ntypes:
            if node_feature_embeddings and ntype in node_feature_embeddings and node_feature_embeddings[ntype] is not None:
                # Use pre-trained embeddings
                embedding = node_feature_embeddings[ntype]
                if DEBUG_LOGGING:
                    logging.info(f"[GATModel.__init__] Node type '{ntype}': embedding type={type(embedding)}, shape={embedding.shape if hasattr(embedding, 'shape') else 'N/A'}")
                if isinstance(embedding, nn.Embedding):
                    self.node_embeddings[ntype] = embedding
                    embedding.weight.requires_grad = not freeze_pretrained
                elif isinstance(embedding, torch.Tensor):
                    # Convert tensor to Embedding
                    num_embeddings, embedding_dim = embedding.shape
                    if DEBUG_LOGGING:
                        logging.info(f"[GATModel.__init__] Converting tensor to Embedding for '{ntype}': {num_embeddings} nodes × {embedding_dim} dim")
                    self.node_embeddings[ntype] = nn.Embedding(
                        num_embeddings,
                        embedding_dim,
                        _weight=embedding,
                    )
                    self.node_embeddings[ntype].weight.requires_grad = not freeze_pretrained
                else:
                    raise TypeError(
                        f"Unsupported embedding type for ntype '{ntype}': {type(embedding)}"
                    )
            else:
                # Initialize new learnable embedding for types without pretrained features
                if ntype in num_nodes_dict:
                    num_nodes = num_nodes_dict[ntype]
                    embedding_dim = gat_hidden  # Use GAT first layer input dim
                    self.node_embeddings[ntype] = nn.Embedding(num_nodes, embedding_dim)
                else:
                    # Fallback: use default values if ntype not in num_nodes_dict
                    print(f"Warning: Node type '{ntype}' not found in num_nodes_dict. Using default 1000 nodes.")
                    self.node_embeddings[ntype] = nn.Embedding(1000, gat_hidden)
        
        # Determine initial feature dimension
        self.initial_feature_dim = -1
        self.node_type_dims = {}  # Track dimension for each node type
        
        if isinstance(self.node_embeddings, nn.ModuleDict):
            for ntype, emb in self.node_embeddings.items():
                if hasattr(emb, 'embedding_dim'):
                    dim = emb.embedding_dim
                    self.node_type_dims[ntype] = dim
                    if self.initial_feature_dim == -1:
                        self.initial_feature_dim = dim
                    if DEBUG_LOGGING:
                        logging.info(f"[GATModel.__init__] {ntype} embedding dimension: {dim}")
        
        if self.initial_feature_dim == -1:
            self.initial_feature_dim = 128
            if DEBUG_LOGGING:
                logging.warning(f"[GATModel.__init__] Could not determine initial_feature_dim, using default: {self.initial_feature_dim}")
        
        if DEBUG_LOGGING:
            logging.info(f"[GATModel.__init__] Detected node_type_dims: {self.node_type_dims}")
            logging.info(f"[GATModel.__init__] Using initial_feature_dim: {self.initial_feature_dim} for all ntypes")
            logging.info(f"[GATModel.__init__] GAT hidden dimension: {gat_hidden}")
        
        self.gat_hidden = gat_hidden
        self.gat_out = gat_out
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.ntypes = ntypes
        self.etypes = etypes
        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }
        self.dropout_rate = dropout
        
        # Feature projection layer to handle arbitrary input feature dimensions
        # This projects input features from initial_feature_dim to gat_hidden
        self.feature_projection = nn.ModuleDict()
        if self.initial_feature_dim != gat_hidden:
            if DEBUG_LOGGING:
                logging.info(f"[GATModel.__init__] Creating projection layers: {self.initial_feature_dim} → {gat_hidden}")
            for ntype in ntypes:
                self.feature_projection[ntype] = nn.Linear(self.initial_feature_dim, gat_hidden)
                if DEBUG_LOGGING:
                    logging.info(f"[GATModel.__init__]   Created projection for '{ntype}': {self.initial_feature_dim} → {gat_hidden}")
        else:
            if DEBUG_LOGGING:
                logging.info(f"[GATModel.__init__] No projection needed: initial_feature_dim ({self.initial_feature_dim}) == gat_hidden ({gat_hidden})")
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        if num_gat_layers > 0:
            # First layer - input is either original features or projected features (to gat_hidden dim)
            first_layer_input_dim = gat_hidden if self.initial_feature_dim != gat_hidden else self.initial_feature_dim
            self.gat_layers.append(
                HeteroGATLayer(first_layer_input_dim, gat_hidden, ntypes, etypes, num_heads)
            )
            
            # Hidden layers
            for _ in range(num_gat_layers - 2):
                self.gat_layers.append(
                    HeteroGATLayer(gat_hidden, gat_hidden, ntypes, etypes, num_heads)
                )
            
            # Output layer
            if num_gat_layers > 1:
                self.gat_layers.append(
                    HeteroGATLayer(gat_hidden, gat_out, ntypes, etypes, num_heads)
                )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Predictor for link prediction
        predictor_input_dim = gat_out * 2
        self.predictor_input_dim = predictor_input_dim
        self.moe_predictors = nn.ModuleDict()
        
        for etype in self.target_etypes:
            src_type, _, dst_type = etype
            self.moe_predictors[str(etype)] = nn.Sequential(
                nn.Linear(predictor_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

    def set_target_etypes(self, target_etypes):
        """
        Dynamically sets the edge types the model should generate predictions for.
        Ensures predictors exist for all target etypes.
        """
        if not target_etypes:
            print("Warning: set_target_etypes called with empty list. Keeping existing target_etypes.")
            return
        self.target_etypes = set(target_etypes)
        for etype in self.target_etypes:
            key = str(etype)
            if key not in self.moe_predictors:
                self.moe_predictors[key] = nn.Sequential(
                    nn.Linear(self.predictor_input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(64, 1)
                )
        print(f"Model target etypes updated. Now predicting for {len(self.target_etypes)} types.")
    
    def get_embeddings(self, g, initial_features, edge_weights=None, return_branch_outputs=False):
        """
        Get node embeddings from GAT
        
        Args:
            g: DGL graph
            initial_features: Dictionary of initial node features
            edge_weights: Optional edge weights
            return_branch_outputs: Whether to return intermediate outputs
        
        Returns:
            Dictionary of node embeddings
        """
        h = {}
        
        # Log initial features for debugging
        if DEBUG_LOGGING:
            logging.info("[GAT get_embeddings] Initial features check:")
            for ntype in self.ntypes:
                if ntype in initial_features and initial_features[ntype] is not None:
                    feat = initial_features[ntype]
                    logging.info(f"[GAT get_embeddings]   '{ntype}' from initial_features: shape={feat.shape}, dtype={feat.dtype}")
                elif ntype in self.node_embeddings:
                    logging.info(f"[GAT get_embeddings]   '{ntype}' will use node_embeddings: embedding_dim={self.node_embeddings[ntype].embedding_dim}")
        
        # Initialize embeddings for all node types from initial_features or node_embeddings
        for ntype in self.ntypes:
            if ntype in initial_features and initial_features[ntype] is not None:
                # Use provided initial features
                h[ntype] = initial_features[ntype]
            elif ntype in self.node_embeddings:
                # Generate embeddings from node_embeddings module
                num_nodes = g.num_nodes(ntype)
                if num_nodes > 0:
                    indices = torch.arange(num_nodes, device=g.device)
                    h[ntype] = self.node_embeddings[ntype](indices)
        
        # Project features to gat_hidden dimension if necessary
        # Check actual feature dimension dynamically at runtime
        for ntype in h:
            actual_feat_dim = h[ntype].shape[-1]
            if DEBUG_LOGGING:
                logging.info(f"[GAT get_embeddings] Processing '{ntype}': actual_feat_dim={actual_feat_dim}, target={self.gat_hidden}")
            
            if actual_feat_dim != self.gat_hidden:
                # Need projection
                if ntype in self.feature_projection and self.feature_projection[ntype] is not None:
                    h[ntype] = self.feature_projection[ntype](h[ntype])
                    if DEBUG_LOGGING:
                        logging.info(f"[GAT get_embeddings]   Applied existing projection for '{ntype}': {actual_feat_dim} → {h[ntype].shape[-1]}")
                else:
                    # Create projection layer on-the-fly if missing
                    if DEBUG_LOGGING:
                        logging.warning(f"[GAT get_embeddings]   Creating on-the-fly projection for '{ntype}': {actual_feat_dim} → {self.gat_hidden}")
                    self.feature_projection[ntype] = nn.Linear(actual_feat_dim, self.gat_hidden).to(h[ntype].device)
                    h[ntype] = self.feature_projection[ntype](h[ntype])
                    if DEBUG_LOGGING:
                        logging.info(f"[GAT get_embeddings]   On-the-fly projection created and applied for '{ntype}': {actual_feat_dim} → {h[ntype].shape[-1]}")
            else:
                if DEBUG_LOGGING:
                    logging.info(f"[GAT get_embeddings]   No projection needed for '{ntype}': dimension already {actual_feat_dim}")
        
        # CRITICAL: Verify all features have the correct dimension before passing to GAT layers
        if DEBUG_LOGGING:
            logging.info("[GAT get_embeddings] Final feature dimensions before GAT layers:")
            for ntype in h:
                logging.info(f"[GAT get_embeddings]   {ntype}: shape={h[ntype].shape}")
        
        for ntype in h:
            if h[ntype].shape[-1] != self.gat_hidden:
                logging.error(f"[GAT get_embeddings] ERROR: '{ntype}' has dimension {h[ntype].shape[-1]}, expected {self.gat_hidden}")
                raise RuntimeError(
                    f"Feature dimension mismatch for node type '{ntype}': "
                    f"got {h[ntype].shape[-1]}, expected {self.gat_hidden}"
                )
        
        # Apply GAT layers
        if DEBUG_LOGGING:
            logging.info(f"[GAT get_embeddings] Applying {len(self.gat_layers)} GAT layers...")
        for layer_idx, gat_layer in enumerate(self.gat_layers):
            if DEBUG_LOGGING:
                logging.info(f"[GAT get_embeddings] Before layer {layer_idx}: h keys={list(h.keys())}")
                for ntype in h:
                    logging.info(f"[GAT get_embeddings]   {ntype}: shape={h[ntype].shape}")
            
            h = gat_layer(g, h)
            
            if DEBUG_LOGGING:
                logging.info(f"[GAT get_embeddings] After layer {layer_idx} (before dropout):")
                for ntype in h:
                    logging.info(f"[GAT get_embeddings]   {ntype}: shape={h[ntype].shape}")
            
            h = {ntype: self.dropout(feat) for ntype, feat in h.items()}
            
            if DEBUG_LOGGING:
                logging.info(f"[GAT get_embeddings] After layer {layer_idx} (after dropout):")
                for ntype in h:
                    logging.info(f"[GAT get_embeddings]   {ntype}: shape={h[ntype].shape}")
        
        if return_branch_outputs:
            return h, {'gnn': h}
        
        return h
    
    def forward(self, g, input_features, pos_graph, neg_graph, edge_weights=None, target_etypes_override=None):
        """
        Forward pass for link prediction
        
        Args:
            g: Message passing graph
            input_features: Initial node features
            pos_graph: Positive graph for link prediction
            neg_graph: Negative graph for link prediction
            edge_weights: Optional edge weights
            target_etypes_override: Override target edge types
        
        Returns:
            pos_scores, neg_scores: Dictionaries of scores for positive and negative edges
        """
        # Log input features
        if DEBUG_LOGGING:
            logging.info("[GAT forward] Input features received:")
            for ntype, feat in input_features.items():
                if isinstance(feat, torch.Tensor):
                    logging.info(f"[GAT forward]   {ntype}: shape={feat.shape}, dtype={feat.dtype}, device={feat.device}")
                else:
                    logging.info(f"[GAT forward]   {ntype}: type={type(feat)} (NOT a tensor!)")
        
        # Ensure features are float tensors
        for ntype in list(input_features.keys()):
            feat = input_features[ntype]
            if isinstance(feat, torch.Tensor):
                input_features[ntype] = feat.float()
            else:
                logging.error(f"[GAT forward] ERROR: Feature for '{ntype}' is not a tensor: {type(feat)}")
                raise TypeError(f"Feature for node type '{ntype}' must be a torch.Tensor, got {type(feat)}")
        
        # Get embeddings from GAT
        h_gnn = self.get_embeddings(g, input_features, edge_weights)
        if DEBUG_LOGGING:
            logging.info("[GAT forward] After get_embeddings:")
            for ntype, feat in h_gnn.items():
                logging.info(f"[GAT forward]   {ntype}: shape={feat.shape}")
        h_fused = h_gnn
        
        # Get target edge types
        target_etypes = target_etypes_override if target_etypes_override else self.target_etypes
        
        # Compute link prediction scores
        if DEBUG_LOGGING:
            logging.info(f"GAT forward: target_etypes = {target_etypes}")
            logging.info(f"GAT forward: pos_graph.canonical_etypes = {pos_graph.canonical_etypes}")
            logging.info(f"GAT forward: h_fused keys = {list(h_fused.keys())}")
            logging.info(f"GAT forward: pos_graph has {sum(pos_graph.num_edges(etype) for etype in pos_graph.canonical_etypes)} total edges")
        
        pos_scores = {}
        neg_scores = {}
        
        for etype in target_etypes:
            src_type, _, dst_type = etype
            
            if src_type not in h_fused or dst_type not in h_fused:
                continue
            
            # Get edges
            if etype in pos_graph.canonical_etypes:
                src_nodes, dst_nodes = pos_graph.edges(etype=etype)
                if len(src_nodes) > 0:
                    src_feats = h_fused[src_type][src_nodes]
                    dst_feats = h_fused[dst_type][dst_nodes]
                    edge_feats = torch.cat([src_feats, dst_feats], dim=1)
                    
                    if str(etype) in self.moe_predictors:
                        pos_scores[etype] = self.moe_predictors[str(etype)](edge_feats).view(-1)
            
            if etype in neg_graph.canonical_etypes:
                src_nodes, dst_nodes = neg_graph.edges(etype=etype)
                if len(src_nodes) > 0:
                    src_feats = h_fused[src_type][src_nodes]
                    dst_feats = h_fused[dst_type][dst_nodes]
                    edge_feats = torch.cat([src_feats, dst_feats], dim=1)
                    
                    if str(etype) in self.moe_predictors:
                        neg_scores[etype] = self.moe_predictors[str(etype)](edge_feats).view(-1)
        
        # DEBUG
        if DEBUG_LOGGING:
            logging.info(f"GAT forward: computed pos_scores keys = {list(pos_scores.keys())}")
            logging.info(f"GAT forward: computed neg_scores keys = {list(neg_scores.keys())}")
            for etype in pos_scores:
                logging.info(f"  pos_scores[{etype}]: shape={pos_scores[etype].shape}, numel={pos_scores[etype].numel()}")
        
        return pos_scores, neg_scores
    
    def predict_links(self, g, h_initial_for_g, source_ntype, source_dgl_ids, target_nodes_dict_with_ids):
        """
        Predict links from source nodes to target nodes
        
        Args:
            g: Graph
            h_initial_for_g: Initial node features
            source_ntype: Source node type
            source_dgl_ids: Source node DGL IDs
            target_nodes_dict_with_ids: Dictionary of target nodes by type
        
        Returns:
            Predictions dictionary
        """
        self.eval()
        predictions = {}
        
        with torch.no_grad():
            h = self.get_embeddings(g, h_initial_for_g)
            
            source_feats = h[source_ntype][source_dgl_ids]
            
            for target_ntype, target_info in target_nodes_dict_with_ids.items():
                target_dgl_ids = target_info['dgl_ids']
                target_feats = h[target_ntype][target_dgl_ids]
                
                # Compute scores for all pairs
                scores = torch.mm(source_feats, target_feats.t())  # [num_source, num_target]
                
                # Get top predictions
                for i, src_id in enumerate(source_dgl_ids):
                    predictions[i] = {
                        'scores': scores[i].cpu().numpy(),
                        'target_ids': target_info['original_ids']
                    }
        
        return predictions
