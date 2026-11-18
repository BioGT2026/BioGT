"""
Graph Convolutional Network (GCN) Model for Disease-Biomarker Association Prediction
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, GraphConv
from Model.common import DEBUG_LOGGING


class HeteroGCNLayer(nn.Module):
    """Single layer of heterogeneous GCN"""
    def __init__(self, in_feats, out_feats, ntypes, etypes):
        super().__init__()
        self.conv = HeteroGraphConv(
            {
                etype: GraphConv(in_feats, out_feats, bias=True)
                for etype in etypes
            },
            aggregate='sum'
        )
        self.ntypes = ntypes
        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(self, g, h_dict):
        """
        Args:
            g: DGL heterogeneous graph
            h_dict: Dictionary of node features {ntype: tensor}
        
        Returns:
            Dictionary of updated node features
        """
        h_out_dict = self.conv(g, h_dict)
        
        # Apply ReLU activation
        for ntype in h_out_dict:
            h_out_dict[ntype] = F.relu(h_out_dict[ntype])
        
        # IMPORTANT: HeteroGraphConv only returns node types that received messages.
        # We need to preserve node types that didn't receive messages but exist in the input.
        # This ensures all node types from the input are present in the output.
        missing_ntypes = set(h_dict.keys()) - set(h_out_dict.keys())
        if missing_ntypes:
            if DEBUG_LOGGING:
                logging.debug(f"HeteroGCNLayer: Node types without incoming messages: {missing_ntypes}. Preserving their input features.")
            for ntype in missing_ntypes:
                # Preserve the input features for node types that didn't receive messages
                h_out_dict[ntype] = h_dict[ntype]
        
        return h_out_dict


class GCNModel(nn.Module):
    """
    Graph Convolutional Network Model for heterogeneous graphs
    Designed for link prediction in disease-biomarker association networks
    """
    def __init__(self, node_feature_embeddings, gcn_hidden, gcn_out,
                 num_gcn_layers, ntypes, etypes, target_node_types,
                 num_nodes_dict, node_info=None, dropout=0.1, 
                 freeze_pretrained=True):
        super().__init__()
        
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
                if isinstance(embedding, nn.Embedding):
                    self.node_embeddings[ntype] = embedding
                    embedding.weight.requires_grad = not freeze_pretrained
                elif isinstance(embedding, torch.Tensor):
                    # Convert tensor to Embedding
                    num_embeddings, embedding_dim = embedding.shape
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
                    embedding_dim = gcn_hidden  # Use GCN first layer input dim
                    self.node_embeddings[ntype] = nn.Embedding(num_nodes, embedding_dim)
                else:
                    # Fallback: use default values if ntype not in num_nodes_dict
                    print(f"Warning: Node type '{ntype}' not found in num_nodes_dict. Using default 1000 nodes.")
                    self.node_embeddings[ntype] = nn.Embedding(1000, gcn_hidden)
        
        # Determine initial feature dimension
        self.initial_feature_dim = -1
        if isinstance(self.node_embeddings, nn.ModuleDict):
            for emb in self.node_embeddings.values():
                if hasattr(emb, 'embedding_dim'):
                    self.initial_feature_dim = emb.embedding_dim
                    break
        
        if self.initial_feature_dim == -1:
            self.initial_feature_dim = 128
        
        self.gcn_hidden = gcn_hidden
        self.gcn_out = gcn_out
        self.num_gcn_layers = num_gcn_layers
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
        # This projects input features from initial_feature_dim to gcn_hidden
        self.feature_projection = nn.ModuleDict()
        if self.initial_feature_dim != gcn_hidden:
            for ntype in ntypes:
                self.feature_projection[ntype] = nn.Linear(self.initial_feature_dim, gcn_hidden)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        
        if num_gcn_layers > 0:
            # First layer - input is either original features or projected features (to gcn_hidden dim)
            first_layer_input_dim = gcn_hidden if self.initial_feature_dim != gcn_hidden else self.initial_feature_dim
            self.gcn_layers.append(
                HeteroGCNLayer(first_layer_input_dim, gcn_hidden, ntypes, etypes)
            )
            
            # Hidden layers
            for _ in range(num_gcn_layers - 2):
                self.gcn_layers.append(
                    HeteroGCNLayer(gcn_hidden, gcn_hidden, ntypes, etypes)
                )
            
            # Output layer
            if num_gcn_layers > 1:
                self.gcn_layers.append(
                    HeteroGCNLayer(gcn_hidden, gcn_out, ntypes, etypes)
                )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Predictor for link prediction
        predictor_input_dim = gcn_out * 2
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
        Get node embeddings from GCN
        
        Args:
            g: DGL graph
            initial_features: Dictionary of initial node features
            edge_weights: Optional edge weights
            return_branch_outputs: Whether to return intermediate outputs
        
        Returns:
            Dictionary of node embeddings
        """
        h = {}
        
        if DEBUG_LOGGING:
            logging.info(f"GCN get_embeddings: self.ntypes = {self.ntypes}")
            logging.info(f"GCN get_embeddings: initial_features keys = {list(initial_features.keys())}")
            logging.info(f"GCN get_embeddings: self.node_embeddings keys = {list(self.node_embeddings.keys())}")
        
        # Initialize embeddings for all node types from initial_features or node_embeddings
        for ntype in self.ntypes:
            if ntype in initial_features and initial_features[ntype] is not None:
                # Use provided initial features
                h[ntype] = initial_features[ntype]
                if DEBUG_LOGGING:
                    logging.info(f"GCN get_embeddings: Using initial_features for {ntype}, shape={h[ntype].shape}")
            elif ntype in self.node_embeddings:
                # Generate embeddings from node_embeddings module
                num_nodes = g.num_nodes(ntype)
                if num_nodes > 0:
                    indices = torch.arange(num_nodes, device=g.device)
                    h[ntype] = self.node_embeddings[ntype](indices)
                    if DEBUG_LOGGING:
                        logging.info(f"GCN get_embeddings: Generated embeddings for {ntype}, shape={h[ntype].shape}")
                else:
                    logging.warning(f"GCN get_embeddings: {ntype} has 0 nodes in graph")
            else:
                logging.warning(f"GCN get_embeddings: {ntype} not in initial_features or node_embeddings!")
        
        if DEBUG_LOGGING:
            logging.info(f"GCN get_embeddings: h keys before GCN layers = {list(h.keys())}")
        
        # Project features to gcn_hidden dimension if necessary
        # Check actual feature dimension dynamically at runtime
        for ntype in h:
            actual_feat_dim = h[ntype].shape[-1]
            if actual_feat_dim != self.gcn_hidden:
                # Need projection
                if ntype in self.feature_projection:
                    h[ntype] = self.feature_projection[ntype](h[ntype])
                else:
                    # Create projection layer on-the-fly if missing
                    self.feature_projection[ntype] = nn.Linear(actual_feat_dim, self.gcn_hidden).to(h[ntype].device)
                    h[ntype] = self.feature_projection[ntype](h[ntype])
        
        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(g, h)
            h = {ntype: self.dropout(feat) for ntype, feat in h.items()}
        
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
        # Ensure features are float
        for ntype, feat in input_features.items():
            input_features[ntype] = feat.float()
        
        # DEBUG - Log inputs before get_embeddings
        if DEBUG_LOGGING:
            logging.info(f"GCN forward: About to call get_embeddings")
            logging.info(f"GCN forward: input_features keys = {list(input_features.keys())}")
            logging.info(f"GCN forward: g.ntypes = {g.ntypes}")
        
        # Get embeddings from GCN
        h_gnn = self.get_embeddings(g, input_features, edge_weights)
        h_fused = h_gnn
        
        if DEBUG_LOGGING:
            logging.info(f"GCN forward: After get_embeddings, h_fused keys = {list(h_fused.keys())}")
        
        # Get target edge types
        target_etypes = target_etypes_override if target_etypes_override else self.target_etypes
        
        # DEBUG - use INFO level so it's always visible
        if DEBUG_LOGGING:
            logging.info(f"GCN forward: target_etypes = {target_etypes}")
            logging.info(f"GCN forward: pos_graph.canonical_etypes = {pos_graph.canonical_etypes if hasattr(pos_graph, 'canonical_etypes') else 'N/A'}")
            logging.info(f"GCN forward: pos_graph has {sum(pos_graph.num_edges(etype) for etype in pos_graph.canonical_etypes)} total edges")
        
        # Compute link prediction scores
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
                    else:
                        if DEBUG_LOGGING:
                            logging.info(f"GCN forward: predictor {str(etype)} not in moe_predictors! Keys: {list(self.moe_predictors.keys())}")
            
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
            logging.info(f"GCN forward: computed pos_scores keys = {list(pos_scores.keys())}")
            logging.info(f"GCN forward: computed neg_scores keys = {list(neg_scores.keys())}")
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
