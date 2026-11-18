"""
Transformer Model for Disease-Biomarker Association Prediction
Simple Transformer without Graph Convolution layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import logging
from Model.common import DEBUG_LOGGING

class TransformerEncoderBlock(nn.Module):
    """Standard Transformer Encoder Block"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src.float()
        
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed Forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder for sequence of node embeddings"""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        """
        Args:
            src: (seq_len, d_model) or (batch_size, seq_len, d_model)
        
        Returns:
            Transformed features with same shape as input
        """
        for layer in self.transformer_layers:
            src = layer(src)
            src = self.dropout(src)
        
        return src


class SimpleTransformerModel(nn.Module):
    """
    Simple Transformer Model for heterogeneous graphs
    Uses node embeddings as sequences and applies Transformer encoding
    Designed for link prediction in disease-biomarker association networks
    """
    def __init__(self, node_feature_embeddings, transformer_hidden, transformer_out,
                 num_transformer_layers, ntypes, etypes, target_node_types,
                 num_nodes_dict, node_info=None, dropout=0.1, 
                 transformer_heads=8, transformer_ff=512, freeze_pretrained=True):
        super().__init__()
        
        self.disable_hgcn = True
        self.disable_transformer = False
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
                    embedding_dim = transformer_hidden  # Use Transformer first layer input dim
                    self.node_embeddings[ntype] = nn.Embedding(num_nodes, embedding_dim)
                else:
                    # Fallback: use default values if ntype not in num_nodes_dict
                    print(f"Warning: Node type '{ntype}' not found in num_nodes_dict. Using default 1000 nodes.")
                    self.node_embeddings[ntype] = nn.Embedding(1000, transformer_hidden)
        
        # Determine initial feature dimension
        self.initial_feature_dim = -1
        if isinstance(self.node_embeddings, nn.ModuleDict):
            for emb in self.node_embeddings.values():
                if hasattr(emb, 'embedding_dim'):
                    self.initial_feature_dim = emb.embedding_dim
                    break
        
        if self.initial_feature_dim == -1:
            self.initial_feature_dim = 128
        
        self.transformer_hidden = transformer_hidden
        self.transformer_out = transformer_out
        self.num_transformer_layers = num_transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ff = transformer_ff
        self.ntypes = ntypes
        self.etypes = etypes
        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }
        self.dropout_rate = dropout
        
        # Transformer encoder stacks for each node type
        self.transformer_stacks = nn.ModuleDict()
        
        for ntype in ntypes:
            self.transformer_stacks[ntype] = TransformerSequenceEncoder(
                d_model=transformer_out,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff,
                num_layers=num_transformer_layers,
                dropout=dropout
            )
        
        # Projection layer to map initial features to transformer hidden dim
        self.feature_projection = nn.ModuleDict()
        for ntype in ntypes:
            self.feature_projection[ntype] = nn.Linear(self.initial_feature_dim, transformer_out)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Predictor for link prediction
        predictor_input_dim = transformer_out * 2
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
        if DEBUG_LOGGING:
            print(f"Model target etypes updated. Now predicting for {len(self.target_etypes)} types.")
    
    def get_embeddings(self, g, initial_features, edge_weights=None, return_branch_outputs=False):
        """
        Get node embeddings from Transformer
        
        Args:
            g: DGL graph
            initial_features: Dictionary of initial node features
            edge_weights: Optional edge weights
            return_branch_outputs: Whether to return intermediate outputs
        
        Returns:
            Dictionary of node embeddings
        """
        h = {}
        
        # Initialize and project features for all node types
        for ntype in self.ntypes:
            if ntype in initial_features and initial_features[ntype] is not None:
                # Use provided initial features
                feat = initial_features[ntype]
            elif ntype in self.node_embeddings:
                # Generate embeddings from node_embeddings module
                num_nodes = g.num_nodes(ntype)
                if num_nodes > 0:
                    indices = torch.arange(num_nodes, device=g.device)
                    feat = self.node_embeddings[ntype](indices)
                else:
                    continue
            else:
                continue
            
            # Project to transformer hidden dimension
            feat_proj = self.feature_projection[ntype](feat)
            
            # Apply Transformer encoding
            feat_transformed = self.transformer_stacks[ntype](feat_proj.unsqueeze(1)).squeeze(1)
            
            h[ntype] = self.dropout(feat_transformed)
        
        if return_branch_outputs:
            return h, {'transformer': h}
        
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
        
        # Get embeddings from Transformer
        h_transformer = self.get_embeddings(g, input_features, edge_weights)
        h_fused = h_transformer
        
        # Get target edge types
        target_etypes = target_etypes_override if target_etypes_override else self.target_etypes
        
        # Compute link prediction scores
        if DEBUG_LOGGING:
            logging.info(f"Transformer forward: target_etypes = {target_etypes}")
            logging.info(f"Transformer forward: pos_graph.canonical_etypes = {pos_graph.canonical_etypes}")
            logging.info(f"Transformer forward: h_fused keys = {list(h_fused.keys())}")
            logging.info(f"Transformer forward: pos_graph has {sum(pos_graph.num_edges(etype) for etype in pos_graph.canonical_etypes)} total edges")
        
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
            logging.info(f"Transformer forward: computed pos_scores keys = {list(pos_scores.keys())}")
            logging.info(f"Transformer forward: computed neg_scores keys = {list(neg_scores.keys())}")
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
