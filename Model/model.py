import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from dgl.nn import HeteroGraphConv, GraphConv
import logging
import heapq
from collections import defaultdict, deque

class MultiheadLatentAttention(nn.Module):
    """
    Implements Multi-head Latent Attention.
    Uses a set of learnable latent vectors to process the input sequence via cross-attention.
    """
    def __init__(self, d_model, nhead, num_latents, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents
        self.nhead = nhead
        self.d_model = d_model

        # Learnable latent vectors, initialized randomly
        # Shape: (1, num_latents, d_model) - batch dim added for broadcasting
        self.latent_vectors = nn.Parameter(torch.randn(1, num_latents, d_model))

        # Attention layer for input -> latents (Query: latents, Key/Value: input)
        self.cross_attn_in_to_lat = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Optional: Self-attention among latents (Query/Key/Value: latents)
        self.self_attn_lat = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Attention layer for latents -> output (Query: input, Key/Value: latents)
        self.cross_attn_lat_to_out = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Layer norms and dropouts (can be adjusted based on specific architecture needs)
        self.norm_lat1 = nn.LayerNorm(d_model)
        self.norm_lat2 = nn.LayerNorm(d_model) # If using latent self-attention
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout_lat = nn.Dropout(dropout)
        self.dropout_lat2 = nn.Dropout(dropout) # If using latent self-attention
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src (Tensor): Input sequence. Shape: (batch_size, seq_len, d_model)
            src_key_padding_mask (Tensor, optional): Mask for padding tokens in src. Shape: (batch_size, seq_len)

        Returns:
            Tensor: Output sequence after latent attention. Shape: (batch_size, seq_len, d_model)
        """
        batch_size = src.shape[0]
        # Expand latents to match batch size
        latents = self.latent_vectors.expand(batch_size, -1, -1) # Shape: (batch_size, num_latents, d_model)

        # 1. Input -> Latents Cross-Attention
        # Latents query the input sequence
        lat_attn_output = self.cross_attn_in_to_lat(
            query=latents,
            key=src,
            value=src,
            key_padding_mask=src_key_padding_mask # Apply padding mask to keys/values from src
        )[0] # Get attention output
        latents = self.norm_lat1(latents + self.dropout_lat(lat_attn_output))

        # Optional: 2. Latent Self-Attention (uncomment if needed)
        # Process information within the latent space
        lat_attn_output2 = self.self_attn_lat(latents, latents, latents)[0]
        latents = self.norm_lat2(latents + self.dropout_lat2(lat_attn_output2))

        # 3. Latents -> Output Cross-Attention
        # Original input sequence queries the processed latents
        out_attn_output = self.cross_attn_lat_to_out(
            query=src, # Query comes from the original input sequence positions
            key=latents,
            value=latents
        )[0] # Get attention output

        # The output here replaces the self-attention output in the standard block
        # We don't add it back to src here, that happens in the TransformerEncoderBlock
        return out_attn_output # Shape: (batch_size, seq_len, d_model)


class HGCNWithWeight(nn.Module):
    """ Heterogeneous Graph Neural Network """
    def __init__(self, in_feats_dict, hidden_size, out_size, ntypes, etypes, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.ntypes = ntypes
        self.canonical_etypes = etypes # Full graph canonical etypes
        self.num_layers = num_layers

        if num_layers == 0:
            pass # No layers
        elif num_layers == 1: # Input -> Output
            output_layer_mods = {}
            # Correctly determine input dimension for each relation based on its source node types
            for rel_str in set(e[1] for e in etypes):
                current_rel_input_dim = None
                for s_canon, r_canon, d_canon in etypes:
                    if r_canon == rel_str and s_canon in in_feats_dict:
                        current_rel_input_dim = in_feats_dict[s_canon]
                        break # Found a valid source type for this relation
                
                if current_rel_input_dim is None: # Fallback if no source type for this rel_str had features
                    ref_dim = next(iter(in_feats_dict.values()), out_size if out_size > 0 else hidden_size) # A sensible fallback
                    print(f"Warning (HGCNWithWeight init layer 1): No source type for relation '{rel_str}' in in_feats_dict. Using ref_dim {ref_dim} for GCN input.")
                    current_rel_input_dim = ref_dim

                output_layer_mods[rel_str] = GraphConv(current_rel_input_dim, out_size, allow_zero_in_degree=True)

            if output_layer_mods:
                self.layers.append(HeteroGraphConv(output_layer_mods, aggregate='sum'))
            else:
                print(f"Warning (HGCNWithWeight init): num_layers=1 but output layer is empty. Check in_feats_dict: {list(in_feats_dict.keys()) if in_feats_dict else 'None'}, etypes: {etypes}")
        else: # num_layers >= 2
            # Input layer (input_sizes -> hidden_size)
            input_layer_mods_actual = {}
            for rel_str in set(e[1] for e in etypes):
                current_rel_input_dim = None
                for s_canon, r_canon, d_canon in etypes:
                    if r_canon == rel_str and s_canon in in_feats_dict:
                        current_rel_input_dim = in_feats_dict[s_canon]
                        break # Found a valid source type for this relation

                if current_rel_input_dim is None:
                    ref_dim = next(iter(in_feats_dict.values()), hidden_size)
                    print(f"Warning (HGCNWithWeight init input layer): No source type for relation '{rel_str}' in in_feats_dict. Using ref_dim {ref_dim} for GCN input.")
                    current_rel_input_dim = ref_dim
                input_layer_mods_actual[rel_str] = GraphConv(current_rel_input_dim, hidden_size, allow_zero_in_degree=True)

            if input_layer_mods_actual:
                self.layers.append(HeteroGraphConv(input_layer_mods_actual, aggregate='sum'))
            else: 
                 print(f"Warning (HGCNWithWeight init): Input GCN layer for num_layers>=2 is empty. in_feats: {list(in_feats_dict.keys()) if in_feats_dict else 'None'}, etypes: {etypes}")

            # Hidden layers (hidden_size -> hidden_size)
            for _ in range(num_layers - 2):
                hidden_layer_mods_actual = {r_etype_str: GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True) for r_etype_str in set(e[1] for e in etypes)}
                if hidden_layer_mods_actual: self.layers.append(HeteroGraphConv(hidden_layer_mods_actual, aggregate='sum'))
                else: print("Warning (HGCNWithWeight init): A hidden GCN layer for num_layers>=2 is empty.")
            
            # Output layer (hidden_size -> out_size)
            if self.layers: 
                output_layer_mods_actual = {r_etype_str: GraphConv(hidden_size, out_size, allow_zero_in_degree=True) for r_etype_str in set(e[1] for e in etypes)}
                if output_layer_mods_actual: self.layers.append(HeteroGraphConv(output_layer_mods_actual, aggregate='sum'))
                else: print("Warning (HGCNWithWeight init): Output GCN layer for num_layers>=2 is empty.")
        
        if not self.layers and self.num_layers > 0:
            print(f"Critical Warning (HGCNWithWeight init): Configured for {self.num_layers} layers, but self.layers is empty. Forward pass will return input features.")


    def forward(self, blocks_or_graph, h_initial_dict, edge_weights=None):
        h = h_initial_dict

        if not self.layers: 
            print("Info (HGCNWithWeight.forward): No GCN layers to apply. Returning features as is.")
            return h

        if isinstance(blocks_or_graph, list):
            # MFGs (blocks) path
            if len(blocks_or_graph) != len(self.layers):
                print(f"Error (HGCNWithWeight.forward): Number of blocks ({len(blocks_or_graph)}) "
                      f"must match number of GCN layers ({len(self.layers)}). ")
                # This is a critical mismatch. Returning initial features to prevent further errors.
                return h_initial_dict


            current_h_features = h 

            for i in range(len(self.layers)): # Iterate GCN layers
                block_mfg = blocks_or_graph[i].to(next(self.layers[i].parameters()).device) # Move block to layer's device
                layer_module = self.layers[i] 
                
                h_input_for_block = {}
                mod_kwargs_block = {} # Initialize mod_kwargs_block here
                block_device = block_mfg.device
                for srctype in block_mfg.srctypes:
                    # Ensure features are on the same device as the block_mfg and layer_module
                    block_device = block_mfg.device
                    if srctype in current_h_features:
                        feat_on_device = current_h_features[srctype].to(block_device)
                        if feat_on_device.shape[0] == block_mfg.num_src_nodes(srctype):
                            h_input_for_block[srctype] = feat_on_device
                        else:
                            expected_dim_for_srctype = None
                            for rel_str_in_layer, conv_in_layer in layer_module.mods.items():
                                for s_canon, r_canon, d_canon in self.canonical_etypes: 
                                    if s_canon == srctype and r_canon == rel_str_in_layer:
                                        expected_dim_for_srctype = conv_in_layer._in_feats # type: ignore
                                        break
                                if expected_dim_for_srctype: break
                            
                            if expected_dim_for_srctype is not None:
                                print(f"Warning (HGCN L{i} Block Input): Correcting feature for srctype '{srctype}' (block nodes: {block_mfg.num_src_nodes(srctype)}, feat nodes: {feat_on_device.shape[0]}) to zero-init with dim {expected_dim_for_srctype}.")
                                h_input_for_block[srctype] = torch.zeros(
                                    (block_mfg.num_src_nodes(srctype), expected_dim_for_srctype),
                                    device=block_device
                                )
                            else:
                                print(f"Error (HGCN L{i} Block Input): Dim unknown for zero-init of srctype '{srctype}' with shape mismatch.")
                                # Potentially return or raise error to avoid DGL error
                                return current_h_features # Or a more specific error handling

                    elif block_mfg.num_src_nodes(srctype) > 0 : 
                        expected_dim_for_srctype = None
                        for rel_str_in_layer, conv_in_layer in layer_module.mods.items():
                            for s_canon, r_canon, d_canon in self.canonical_etypes:
                                if s_canon == srctype and r_canon == rel_str_in_layer:
                                    expected_dim_for_srctype = conv_in_layer._in_feats # type: ignore
                                    break
                            if expected_dim_for_srctype: break
                        if expected_dim_for_srctype is not None:
                             print(f"Info (HGCN L{i} Block Input): Zero-initializing missing srctype '{srctype}' "
                                   f"(nodes: {block_mfg.num_src_nodes(srctype)}, dim: {expected_dim_for_srctype}) for block {i}.")
                             h_input_for_block[srctype] = torch.zeros(
                                 (block_mfg.num_src_nodes(srctype), expected_dim_for_srctype),
                                 device=block_device
                             )
                        else:
                             print(f"Warning (HGCN L{i} Block Input): srctype '{srctype}' missing from features and dim unknown for block {i}. HeteroGraphConv might error.")
                
                mod_kwargs_block = {}
                for etype_b in block_mfg.canonical_etypes:
                    src_type_b, rel_name_b, dst_type_b = etype_b
                    if rel_name_b in layer_module.mods:
                        if block_mfg.num_edges(etype_b) > 0 and 'weight' in block_mfg.edges[etype_b].data:
                            edge_weights_b = block_mfg.edges[etype_b].data['weight'].to(block_device)
                            if edge_weights_b.dtype != torch.float32:
                                edge_weights_b = edge_weights_b.float()
                            mod_kwargs_block[rel_name_b] = {'edge_weight': edge_weights_b}
                
                if not h_input_for_block and any(block_mfg.num_edges(et_b_nw) > 0 for et_b_nw in block_mfg.canonical_etypes if et_b_nw[1] in layer_module.mods):
                    print(f"Warning: HGCN layer {i} for block {i} has no prepared input features, but block has relevant edges. Layer output will be empty.")

                elif not h_input_for_block:
                     h_output_from_layer = {dsttype: torch.empty((block_mfg.num_dst_nodes(dsttype), layer_module.mods[next(iter(layer_module.mods))]._out_feats), device=block_device) for dsttype in block_mfg.dsttypes if block_mfg.num_dst_nodes(dsttype) > 0}
                else:
                    try:
                        h_output_from_layer = layer_module(block_mfg, h_input_for_block, mod_kwargs=mod_kwargs_block)
                    except dgl.DGLError as e_dgl_block:
                        print(f"Error during HGCN layer {i} on block {i} (device: {block_mfg.device}): {e_dgl_block}")
                        print(f"  Input feats keys: {list(h_input_for_block.keys())}")
                        for k,v in h_input_for_block.items(): print(f"    {k}: {v.shape}, {v.device}")
                        print(f"  Layer mods: {list(layer_module.mods.keys())}")
                        raise
                    except Exception as e_gen_block:
                        print(f"Generic Error during HGCN layer {i} on block {i}: {e_gen_block}")
                        raise
                
                current_h_features = {
                    ntype_dst: F.relu(feat_dst)
                    for ntype_dst, feat_dst in h_output_from_layer.items()
                }
            return current_h_features
        
        else: # Full graph training
            g = blocks_or_graph
            for i, layer in enumerate(self.layers):
                h_input_for_layer = h
                
                try:
                    layer_device = next(layer.parameters()).device
                except StopIteration:
                    example_tensor = next(iter(h_input_for_layer.values()), None)
                    layer_device = example_tensor.device if example_tensor is not None else torch.device('cpu')

                layer_edge_weights = None
                if edge_weights is not None and isinstance(edge_weights, dict):
                    # Get the edge weights for this layer's graph
                    layer_edge_weights = {k: v.to(layer_device) for k, v in edge_weights.items() if k in g.canonical_etypes}

                # Ensure features exist for every source node type required by this layer.
                if isinstance(layer, HeteroGraphConv):
                    for rel_name, conv_module in layer.mods.items():
                        expected_srctype = None
                        for s_canon, r_canon, _ in self.canonical_etypes:
                            if r_canon == rel_name:
                                expected_srctype = s_canon
                                break

                        if expected_srctype is None:
                            continue

                        if expected_srctype in h_input_for_layer:
                            h_input_for_layer[expected_srctype] = h_input_for_layer[expected_srctype].to(layer_device)
                            continue

                        num_src_nodes = g.num_nodes(expected_srctype)

                        expected_dim = getattr(conv_module, '_in_src_feats', None)
                        if isinstance(expected_dim, tuple):
                            expected_dim = expected_dim[0]
                        if expected_dim is None:
                            expected_dim = getattr(conv_module, '_in_feats', None)
                        if isinstance(expected_dim, tuple):
                            expected_dim = expected_dim[0]
                        if expected_dim is None:
                            expected_dim = getattr(conv_module, 'in_feats', None)
                        if isinstance(expected_dim, tuple):
                            expected_dim = expected_dim[0]
                        if expected_dim is None:
                            expected_dim = getattr(conv_module, 'in_feats', None)
                            if isinstance(expected_dim, tuple):
                                expected_dim = expected_dim[0]
                        if expected_dim is None:
                            sample_feat = next(iter(h_input_for_layer.values()), None)
                            if sample_feat is not None and sample_feat.dim() == 2:
                                expected_dim = sample_feat.shape[1]
                        if expected_dim is None:
                            expected_dim = getattr(conv_module, '_out_feats', None)
                        if isinstance(expected_dim, tuple):
                            expected_dim = expected_dim[0]
                        if expected_dim is None:
                            expected_dim = getattr(conv_module, 'out_feats', None)
                        if isinstance(expected_dim, tuple):
                            expected_dim = expected_dim[0]
                        if expected_dim is None:
                            expected_dim = 1

                        expected_dim = int(expected_dim)
                        h_input_for_layer[expected_srctype] = torch.zeros(
                            (num_src_nodes, expected_dim),
                            device=layer_device,
                            dtype=torch.float32,
                        )

                h = layer(g, h_input_for_layer, mod_kwargs={'edge_weight': layer_edge_weights})

                # After a layer, h contains features only for destination node types.
                # We must ensure all node types are present for the next layer's input.
                # Also ensure the final output contains all node types.
                h_next_input = {}
                for ntype in g.ntypes:
                    if ntype in h:
                        h_next_input[ntype] = h[ntype]
                    elif ntype in h_input_for_layer:
                        # If a node type was a source but not a destination, carry its features over.
                        h_next_input[ntype] = h_input_for_layer[ntype]
                h = h_next_input
            return h


class TransformerEncoderBlock(nn.Module):
    """ Transformer Encoder Block using Multi-head Latent Attention """
    def __init__(self, d_model, nhead, dim_feedforward, num_latents, dropout=0.1, disable_mla=False):
        super().__init__()
        self.disable_mla = disable_mla
        if self.disable_mla:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        else:
            self.latent_attn = MultiheadLatentAttention(d_model, nhead, num_latents, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # Dropout after attention
        self.dropout2 = nn.Dropout(dropout) # Dropout after feedforward

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Ensure the source tensor is float type before attention operation.
        src = src.float()
        
        if self.disable_mla:
            # When called from nn.Sequential, only `src` is passed.
            # The `self_attn` layer returns a tuple (attn_output, attn_output_weights)
            # if need_weights is True (default). We only need the output.
            # Explicitly setting need_weights=False ensures it only returns the tensor.
            src2 = self.self_attn(src, src, src,
                                  key_padding_mask=src_key_padding_mask,
                                  need_weights=False)[0]
        else:
            src2 = self.latent_attn(src, src_key_padding_mask=src_key_padding_mask)

        # Add & Norm (Residual connection after attention)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed Forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        # Add & Norm (Residual connection after feedforward)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class HeteroTransformerEncoder(nn.Module):
    """
    Applies multiple layers of TransformerEncoderBlock to each node type
    in a heterogeneous graph's feature dictionary.
    """
    def __init__(self, ntypes_with_features, d_model, nhead, dim_feedforward, num_layers, num_latents, dropout=0.1, disable_mla=False):
        """
        Args:
            ntypes_with_features (list or set): Node types that will have input features.
            d_model (int): Input feature dimension (expected output dim of HGCN).
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            num_layers (int): Number of TransformerEncoderBlock layers.
            num_latents (int): Number of latent vectors for MultiheadLatentAttention.
            dropout (float): Dropout rate.
            disable_mla (bool): If True, use standard self-attention instead of latent attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.transformer_stacks = nn.ModuleDict()

        for ntype in ntypes_with_features:
            if num_layers > 0:
                layers = [TransformerEncoderBlock(d_model, nhead, dim_feedforward, num_latents, dropout, disable_mla=disable_mla)
                          for _ in range(num_layers)]
                self.transformer_stacks[ntype] = nn.Sequential(*layers)
            # else: If num_layers is 0, the ModuleDict will be empty for this type

    def forward(self, h_dict):
        """
        Applies the Transformer stacks to the input feature dictionary.

        Args:
            h_dict (dict[str, Tensor]): Dictionary of node features (output from HGCN).
                                        Expected shape per type: (num_nodes, d_model)

        Returns:
            dict[str, Tensor]: Dictionary of node features after Transformer encoding.
                               Shape per type: (num_nodes, d_model)
        """
        h_transformed = {}
        for ntype, h in h_dict.items():
            if ntype in self.transformer_stacks and h.shape[0] > 0 and self.num_layers > 0:
                # Transformer expects batch_first=True -> (batch, seq, feature)
                # Here, batch=1, seq=num_nodes, feature=d_model
                # The error "softmax not implemented for 'Long'" indicates that `h` is a LongTensor.
                # It must be converted to a float type before being passed to the attention mechanism.
                h_seq = h.float().unsqueeze(0)
                transformed_seq = self.transformer_stacks[ntype](h_seq)
                h_transformed[ntype] = transformed_seq.squeeze(0)
            else:
                # Pass through features if no transformer stack for this type,
                # no nodes, or num_layers is 0.
                h_transformed[ntype] = h
        return h_transformed


class PairwiseInteractionFeatures(nn.Module):
    """Expand [context, target] pairs with multiplicative and distance features."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.in_features = self.embedding_dim * 2
        self.out_features = self.embedding_dim * 4

    def forward(self, pair_features):
        context, target = torch.split(
            pair_features,
            self.embedding_dim,
            dim=-1,
        )
        return torch.cat(
            [
                context,
                target,
                context * target,
                torch.abs(context - target),
            ],
            dim=-1,
        )


class StructuralPriorFusion(nn.Module):
    """Fuse a train-graph structural prior with the neural pair score."""

    def __init__(self, neural_weight=0.0, eps=1e-8):
        super().__init__()
        self.neural_weight = float(neural_weight)
        self.eps = float(eps)

    def _standardize(self, scores, candidate_ids=None):
        reference = scores
        if candidate_ids is not None:
            reference = scores[candidate_ids]
        mean = reference.mean()
        std = reference.std(unbiased=False).clamp_min(self.eps)
        return (scores - mean) / std

    def forward(
        self,
        neural_scores,
        prior_scores,
        candidate_ids=None,
        neural_weight=None,
    ):
        neural_z = self._standardize(neural_scores, candidate_ids)
        prior_z = self._standardize(prior_scores, candidate_ids)
        weight = self.neural_weight if neural_weight is None else float(neural_weight)
        return prior_z + weight * neural_z


class DiseaseAssociationModel(nn.Module):
    def __init__(self, node_feature_embeddings, hgcn_hidden, hgcn_out,
                 num_transformer_layers,
                 transformer_heads, transformer_ff,
                 num_latents,
                 ntypes, etypes, target_node_types, 
                 num_nodes_dict, # Added to get node counts for embedding init
                 node_info=None,
                 dropout=0.1,
                 num_gnn_layers=2,
                 freeze_pretrained=True, # Added to control freezing
                 disable_hgcn=False, disable_transformer=False, disable_moe=False, disable_mla=False, disable_feature_gating=False,
                 fusion_mode='sequential', enable_explainability=False, explainability_top_k=5, explainability_max_hops=3,
                 explainability_hop_penalty=0.1, explainability_virtual_edge_penalty=0.35,
                 explainability_virtual_sim_threshold=0.3, explainability_attribution_method='input_x_gradient',
                 scorer_mode='concat_mlp',
                 structural_prior_mode='none',
                 structural_prior_neural_weight=0.0):
        super().__init__()
        self.disable_hgcn = disable_hgcn
        self.disable_transformer = disable_transformer
        self.disable_moe = disable_moe
        self.disable_mla = disable_mla
        self.disable_feature_gating = disable_feature_gating
        self.fusion_mode = fusion_mode
        self.scorer_mode = scorer_mode
        self.structural_prior_mode = str(structural_prior_mode or 'none').lower()
        self.structural_prior_fusion = StructuralPriorFusion(
            neural_weight=structural_prior_neural_weight,
        )
        self.enable_explainability = enable_explainability
        self.explainability_top_k = max(1, int(explainability_top_k)) if explainability_top_k is not None else 1
        self.explainability_max_hops = max(1, int(explainability_max_hops)) if explainability_max_hops is not None else 1
        self.explainability_hop_penalty = max(0.0, float(explainability_hop_penalty))
        self.explainability_virtual_edge_penalty = max(0.0, float(explainability_virtual_edge_penalty))
        self.explainability_virtual_sim_threshold = min(1.0, max(-1.0, float(explainability_virtual_sim_threshold)))
        self.explainability_attribution_method = str(explainability_attribution_method or 'input_x_gradient').lower()
        if self.explainability_attribution_method not in {'input_x_gradient', 'gradient_norm'}:
            self.explainability_attribution_method = 'input_x_gradient'
        self._last_explainability = None
        self._last_fused_embeddings = None
        self._explainability_adj_cache = None
        self._explainability_adj_cache_key = None

        # Optional node_info mapping (from DGLGraphCreator.get_node_info())
        # Used for explainability formatting (mapping dgl id -> original id -> name)
        self.node_info = node_info

        # This model now assumes it will be moved to a single device in the training script.
        # All `device_map` and `.to(device)` calls have been removed.

        # Initialize embeddings for all node types, creating new ones if not pretrained.
        self.node_embeddings = nn.ModuleDict()

        for ntype in ntypes:
            if node_feature_embeddings and ntype in node_feature_embeddings and node_feature_embeddings[ntype] is not None:
                # Use pre-trained embeddings
                pretrained_weight = node_feature_embeddings[ntype]
                num_embeddings, embedding_dim = pretrained_weight.shape
                self.node_embeddings[ntype] = nn.Embedding(
                    num_embeddings,
                    embedding_dim,
                    _weight=pretrained_weight,
                    sparse=True
                )
                self.node_embeddings[ntype].weight.requires_grad = not freeze_pretrained
            else:
                # Initialize new learnable embedding for types without pretrained features
                if ntype in num_nodes_dict:
                    num_nodes = num_nodes_dict[ntype]
                    embedding_dim = hgcn_out  # Use HGCN output dim for new embeddings
                    self.node_embeddings[ntype] = nn.Embedding(num_nodes, embedding_dim, sparse=True)
                else:
                    # This should not happen if num_nodes_dict is correctly populated
                    print(f"Warning: Cannot initialize embedding for ntype '{ntype}' because its count is unknown.")

        self.embedding_sizes = {} 
        if isinstance(self.node_embeddings, nn.ModuleDict):
            for ntype, emb in self.node_embeddings.items():
                if isinstance(emb, nn.Embedding):
                    self.embedding_sizes[ntype] = {"num_nodes": emb.weight.shape[0], "dim": emb.weight.shape[1]}
        
        initial_feature_dim = -1
        # Try to infer initial_feature_dim from nn.Embedding layers
        if isinstance(self.node_embeddings, nn.ModuleDict):
            first_emb_layer = next((emb for emb in self.node_embeddings.values() if isinstance(emb, nn.Embedding)), None)
            if first_emb_layer: initial_feature_dim = first_emb_layer.embedding_dim
        
        # If not found, try to infer from a dict of tensors (less common for this model structure's init)
        if initial_feature_dim == -1 and isinstance(self.node_embeddings, dict) and self.node_embeddings:
            first_feat_tensor = next(iter(self.node_embeddings.values()), None)
            if first_feat_tensor is not None and isinstance(first_feat_tensor, torch.Tensor): 
                initial_feature_dim = first_feat_tensor.shape[1]

        # If still not found, it's an issue unless features are always passed with known dims
        if initial_feature_dim == -1:
             print("Warning: Could not reliably determine initial_feature_dim from node_feature_embeddings. "
                   "This might be okay if features are passed externally with known dimensions. "
                   "HGCN input dimension will rely on hgcn_in_feats_actual.")
        
        self.initial_feature_dim = initial_feature_dim 
        self.hgcn_out_dim = hgcn_out 
        self.num_gnn_layers = num_gnn_layers

        hgcn_in_feats_actual = {}
        # Populate hgcn_in_feats_actual based on what node_embeddings provides
        # This assumes that if a type is in node_embeddings, its feature dim is self.initial_feature_dim
        # Or, if node_embeddings is a dict of tensors, it uses their specific dims.
        if isinstance(self.node_embeddings, nn.ModuleDict):
            for ntype_key, emb_module in self.node_embeddings.items():
                if ntype_key in ntypes: 
                    if isinstance(emb_module, nn.Embedding):
                         hgcn_in_feats_actual[ntype_key] = emb_module.embedding_dim
                    elif hasattr(emb_module, 'out_features'): # e.g. a Linear layer
                         hgcn_in_feats_actual[ntype_key] = emb_module.out_features
                    elif self.initial_feature_dim != -1 : # Fallback to inferred general dim
                         hgcn_in_feats_actual[ntype_key] = self.initial_feature_dim

        elif isinstance(self.node_embeddings, dict): 
             for ntype_key, feat_tensor_val in self.node_embeddings.items():
                 if ntype_key in ntypes and isinstance(feat_tensor_val, torch.Tensor):
                     hgcn_in_feats_actual[ntype_key] = feat_tensor_val.shape[1]
        
        if not hgcn_in_feats_actual and self.initial_feature_dim != -1:
            print(f"Warning: hgcn_in_feats_actual is empty. Falling back to uniform initial_feature_dim ({self.initial_feature_dim}) for all ntypes if they appear in etypes.")
            # This fallback might be too broad. HGCN init needs specific dims for source types of its relations.
            # The HGCN init itself has fallback if a src type for a relation is not in its in_feats_dict.
            # So, passing an empty or partially filled hgcn_in_feats_actual is okay if HGCN init handles it.
            pass # Let HGCN init handle missing types in hgcn_in_feats_actual

        if self.disable_hgcn:
            print("HGCN component is DISABLED for ablation.")
            self.hgnn = None
        else:
            self.hgnn = HGCNWithWeight(
                hgcn_in_feats_actual,
                hgcn_hidden, 
                self.hgcn_out_dim,
                ntypes, etypes,
                num_layers=self.num_gnn_layers
            )
        
        transformer_input_ntypes_actual = ntypes 
        
        _num_transformer_layers = num_transformer_layers
        if self.disable_transformer:
            print("Transformer component is DISABLED for ablation.")
            _num_transformer_layers = 0

        # Adjust Transformer d_model based on fusion mode
        transformer_d_model = hgcn_out
        if self.fusion_mode in ['concat', 'weighted', 'routing'] and not self.disable_hgcn:
            # In parallel modes, Transformer runs on initial features.
            # We'll project initial features to hgcn_out dim before Transformer.
            transformer_d_model = hgcn_out
            self.transformer_input_projection = nn.ModuleDict()
            if initial_feature_dim != -1 and initial_feature_dim != hgcn_out:
                print(f"Fusion mode '{self.fusion_mode}' requires projecting features for Transformer from {initial_feature_dim} to {hgcn_out}.")
                for ntype in ntypes:
                    if ntype in hgcn_in_feats_actual and hgcn_in_feats_actual[ntype] > 0:
                         self.transformer_input_projection[ntype] = nn.Linear(hgcn_in_feats_actual[ntype], hgcn_out)
        elif self.fusion_mode in ['sequential', 'hierarchical']:
            # In sequential/hierarchical modes, Transformer runs on HGCN output (or a fusion of it).
            transformer_d_model = self.hgcn_out_dim
        
        self.transformer_encoder_parallel = HeteroTransformerEncoder( 
            ntypes_with_features=transformer_input_ntypes_actual, 
            d_model=transformer_d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            num_layers=_num_transformer_layers,
            num_latents=num_latents,
            dropout=dropout,
            disable_mla=self.disable_mla
        )
        
        # Add feature gating mechanism (for sequential and hierarchical modes)
        self.feature_gates = None
        if self.fusion_mode in ['sequential', 'hierarchical'] and not self.disable_feature_gating and not self.disable_hgcn and not self.disable_transformer:
            print(f"Feature Gating is ENABLED for {self.fusion_mode} fusion.")
            self.feature_gates = nn.ModuleDict()
            for ntype in transformer_input_ntypes_actual:
                gate_input_dim = self.hgcn_out_dim * 2 if self.fusion_mode == 'hierarchical' else self.hgcn_out_dim
                self.feature_gates[ntype] = nn.Sequential(
                    nn.Linear(gate_input_dim, self.hgcn_out_dim),
                    nn.Sigmoid()
                )
        
        # Add learnable weights for 'weighted' fusion mode
        self.fusion_weights = None
        if self.fusion_mode == 'weighted' and not self.disable_hgcn and not self.disable_transformer:
            print("Weighted fusion is ENABLED.")
            self.fusion_weights = nn.ParameterDict()
            for ntype in ntypes:
                # Initialize with 0.5, so both have equal contribution initially
                self.fusion_weights[ntype] = nn.Parameter(torch.tensor([0.5]))

        # Add dynamic routing networks for 'routing' fusion mode
        self.routing_networks = None
        if self.fusion_mode == 'routing' and not self.disable_hgcn and not self.disable_transformer:
            print("Dynamic Routing fusion is ENABLED.")
            self.routing_networks = nn.ModuleDict()
            for ntype in ntypes:
                self.routing_networks[ntype] = nn.Sequential(
                    nn.Linear(self.hgcn_out_dim * 2, self.hgcn_out_dim),
                    nn.ReLU(),
                    nn.Linear(self.hgcn_out_dim, 2), # Outputs two weights (for h_g and h_t)
                    nn.Softmax(dim=-1)
                )

        # Determine the fused dimension based on active components and fusion mode
        fused_dim = 0
        if self.fusion_mode in ['sequential', 'hierarchical']:
            if not self.disable_hgcn:
                fused_dim = self.hgcn_out_dim
            else: # HGCN disabled, Transformer (if enabled) processes initial features
                fused_dim = self.initial_feature_dim if self.initial_feature_dim != -1 else self.hgcn_out_dim
                if not self.disable_transformer:
                    self.transformer_encoder_parallel.d_model = fused_dim
        elif self.fusion_mode == 'concat':
            if not self.disable_hgcn: fused_dim += self.hgcn_out_dim
            if not self.disable_transformer: fused_dim += transformer_d_model
            if fused_dim == 0: fused_dim = self.initial_feature_dim if self.initial_feature_dim != -1 else self.hgcn_out_dim
        elif self.fusion_mode in ['weighted', 'routing']:
            # Both branches must have the same dimension
            fused_dim = self.hgcn_out_dim
            if self.disable_hgcn or self.disable_transformer: # If one is disabled, it's just the other's output
                fused_dim = self.hgcn_out_dim if not self.disable_hgcn else (transformer_d_model if not self.disable_transformer else self.initial_feature_dim)

        if fused_dim == 0:
            print("CRITICAL WARNING: All feature extractors are disabled or fusion resulted in 0 dim. Using a fallback dimension.")
            fused_dim = self.initial_feature_dim if self.initial_feature_dim != -1 else self.hgcn_out_dim

        predictor_input_dim_concat = fused_dim * 2
        def make_predictor():
            if self.scorer_mode == 'interaction_mlp':
                return nn.Sequential(
                    PairwiseInteractionFeatures(fused_dim),
                    nn.Linear(fused_dim * 4, fused_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fused_dim * 2, 1),
                )
            return nn.Sequential(
                nn.Linear(predictor_input_dim_concat, predictor_input_dim_concat // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(predictor_input_dim_concat // 2, 1)
            )

        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }

        if self.disable_moe:
            print("Mixture-of-Experts (MoE) is DISABLED for ablation. Using a single shared predictor.")
            self.expert_predictors = None
            self.shared_predictor = make_predictor()
        else:
            print("Mixture-of-Experts (MoE) is ENABLED.")
            self.expert_predictors = nn.ModuleDict()
            self.shared_predictor = None
            for target_type in self.target_node_types: 
                self.expert_predictors[target_type] = make_predictor()

    def fuse_structural_scores(
        self,
        neural_scores,
        prior_scores,
        candidate_ids=None,
        neural_weight=None,
    ):
        if self.structural_prior_mode == 'none':
            return neural_scores
        return self.structural_prior_fusion(
            neural_scores,
            prior_scores,
            candidate_ids=candidate_ids,
            neural_weight=neural_weight,
        )

    def set_target_etypes(self, target_etypes):
        """
        Dynamically sets the edge types the model should generate predictions for.
        This is useful for focusing evaluation on specific edge types.
        """
        self.target_etypes = target_etypes
        print(f"Model target etypes updated. Now predicting for {len(target_etypes)} types.")
        print(f"New target etypes for prediction: {target_etypes}")

    def get_embeddings(self, g, initial_features, edge_weights=None, return_branch_outputs=False):
        # Four Fusion Strategies: sequential, concat, weighted, hierarchical, routing
        
        # 1. Get HGCN embeddings (or pass through if disabled)
        h_g = {}
        if not self.disable_hgcn and self.hgnn is not None:
            h_g = self.hgnn(g, initial_features, edge_weights=edge_weights)
        else:
            h_g = initial_features

        # 2. Get Transformer embeddings (or pass through if disabled)
        h_t = {}
        transformer_input = {}
        if not self.disable_transformer and self.transformer_encoder_parallel.num_layers > 0:
            # Determine input for transformer based on fusion mode
            if self.fusion_mode == 'sequential':
                transformer_input = h_g # Input is the output of HGCN
            elif self.fusion_mode == 'hierarchical':
                # Fuse HGCN output with initial features before transformer
                projected_initial = {}
                if hasattr(self, 'transformer_input_projection'):
                    for ntype, feat in initial_features.items():
                        if ntype in self.transformer_input_projection:
                            projected_initial[ntype] = self.transformer_input_projection[ntype](feat)
                        else:
                            projected_initial[ntype] = feat
                else:
                    projected_initial = initial_features
                
                hier_fused_input = {}
                for ntype in h_g.keys():
                    if ntype in projected_initial:
                        if not self.disable_feature_gating and self.feature_gates and ntype in self.feature_gates:
                            gate_input = torch.cat([h_g[ntype], projected_initial[ntype]], dim=1)
                            gate = self.feature_gates[ntype](gate_input)
                            hier_fused_input[ntype] = (1 - gate) * h_g[ntype] + gate * projected_initial[ntype]
                        else:
                            hier_fused_input[ntype] = h_g[ntype] + projected_initial[ntype]
                    else:
                        hier_fused_input[ntype] = h_g[ntype]
                transformer_input = hier_fused_input
            else: # For 'concat', 'weighted', and 'routing', input is the initial features
                transformer_input = initial_features
                # Apply projection if necessary for parallel modes
                if hasattr(self, 'transformer_input_projection'):
                    projected_input = {}
                    for ntype, feat in transformer_input.items():
                        if ntype in self.transformer_input_projection:
                            projected_input[ntype] = self.transformer_input_projection[ntype](feat)
                        else:
                            projected_input[ntype] = feat # Pass through if no projection needed/defined
                    transformer_input = projected_input

            h_t = self.transformer_encoder_parallel(transformer_input)
        else:
            # If transformer is disabled, h_t remains empty.
            # For sequential mode, this means we just return h_g.
            # For parallel modes, this means we only use the h_g branch.
            pass

        # 3. Fuse the features based on the selected mode
        h_fused = {}
        if self.fusion_mode == 'sequential':
            # In sequential mode, h_g was the input to the transformer.
            # We fuse the transformer output (h_t) with its input (h_g) via residual connection.
            for ntype in h_g.keys():
                if ntype in h_t:
                    if not self.disable_feature_gating and self.feature_gates and ntype in self.feature_gates:
                        gate = self.feature_gates[ntype](h_t[ntype])
                        h_fused[ntype] = h_g[ntype] + gate * h_t[ntype] # Gated residual
                    else:
                        h_fused[ntype] = h_g[ntype] + h_t[ntype] # Simple residual
                else:
                    h_fused[ntype] = h_g[ntype] # No transformer output for this type

        elif self.fusion_mode == 'hierarchical':
            # In hierarchical mode, transformer_input was the fusion of h_g and initial_features.
            # The final output is the output of the transformer.
            # A final residual connection can also be added.
            # transformer_input is the dict `hier_fused_input` computed above
            for ntype in transformer_input.keys():
                if ntype in h_t:
                    # Residual connection between transformer's input and output
                    h_fused[ntype] = transformer_input[ntype] + h_t[ntype]
                else:
                    h_fused[ntype] = transformer_input[ntype]

        elif self.fusion_mode == 'concat':
            all_ntypes = set(h_g.keys()) | set(h_t.keys())
            for ntype in all_ntypes:
                if ntype in h_g and ntype in h_t:
                    h_fused[ntype] = torch.cat([h_g[ntype], h_t[ntype]], dim=1)
                elif ntype in h_g:
                    # If transformer is disabled, we need to pad with zeros
                    padding = torch.zeros_like(h_g[ntype])
                    h_fused[ntype] = torch.cat([h_g[ntype], padding], dim=1)
                elif ntype in h_t:
                    # If HGCN is disabled, we need to pad with zeros
                    padding = torch.zeros_like(h_t[ntype])
                    h_fused[ntype] = torch.cat([padding, h_t[ntype]], dim=1)

        elif self.fusion_mode == 'weighted':
            all_ntypes = set(h_g.keys()) | set(h_t.keys())
            for ntype in all_ntypes:
                # If one of the models is disabled, no fusion is needed. Just pass the output.
                if self.disable_hgcn:
                    if ntype in h_t: h_fused[ntype] = h_t[ntype]
                    continue
                if self.disable_transformer:
                    if ntype in h_g: h_fused[ntype] = h_g[ntype]
                    continue

                if ntype in h_g and ntype in h_t:
                    # Both HGCN and Transformer were active
                    weight = torch.sigmoid(self.fusion_weights[ntype]) # Sigmoid to keep it between 0 and 1
                    h_fused[ntype] = (1 - weight) * h_g[ntype] + weight * h_t[ntype]
                elif ntype in h_g:
                    h_fused[ntype] = h_g[ntype] # Only HGCN was active
                elif ntype in h_t:
                    h_fused[ntype] = h_t[ntype] # Only Transformer was active
            
        elif self.fusion_mode == 'routing':
            all_ntypes = set(h_g.keys()) | set(h_t.keys())
            for ntype in all_ntypes:
                if ntype in h_g and ntype in h_t and ntype in self.routing_networks:
                    # Concatenate features to be fed into the routing network
                    routing_input = torch.cat([h_g[ntype], h_t[ntype]], dim=1)
                    # Get dynamic weights [w_g, w_t] for each node
                    weights = self.routing_networks[ntype](routing_input) # Shape: (num_nodes, 2)
                    # Apply weights
                    h_fused[ntype] = weights[:, 0].unsqueeze(1) * h_g[ntype] + weights[:, 1].unsqueeze(1) * h_t[ntype]
                elif ntype in h_g:
                    h_fused[ntype] = h_g[ntype] # Only HGCN was active
                elif ntype in h_t:
                    h_fused[ntype] = h_t[ntype] # Only Transformer was active
        else: # Fallback, should not be reached
            h_fused = h_g

        if return_branch_outputs:
            fusion_weights = {}
            if self.fusion_weights is not None:
                for ntype, weight_param in self.fusion_weights.items():
                    try:
                        fusion_weights[ntype] = torch.sigmoid(weight_param).item()
                    except Exception:
                        fusion_weights[ntype] = float('nan')
            branch_outputs = {
                'hgcn': h_g,
                'transformer': h_t,
                'fused': h_fused,
                'fusion_weights': fusion_weights
            }
            return h_fused, branch_outputs

        return h_fused


    def forward(self, g, input_features, pos_graph, neg_graph, edge_weights=None, target_etypes_override=None):
        # Ensure all input features are float type before processing
        for ntype, feat in input_features.items():
            if feat.dtype != torch.float32:
                input_features[ntype] = feat.float()

        try:
            device = next(self.parameters()).device
        except StopIteration:
            print("Warning: Model has no parameters. Using CPU as fallback device.")
            device = torch.device("cpu")

        if isinstance(g, list): 
            if not g :
                print("Error: `blocks` list is empty in model.forward(). Cannot proceed.")
                return {}, {}
            if len(g) != self.num_gnn_layers:
                 print(f"Error: Number of blocks ({len(g)}) != num_gnn_layers ({self.num_gnn_layers}). This will cause issues in HGCN processing.")
                 # This is critical. HGCN expects this match.
                 return {}, {}


            h_fused_dst_nodes = self.get_embeddings(g, input_features) # for blocks[-1].dstdata

            pos_scores = {}
            neg_scores = {}
            
            # Helper to map global NIDs from pos/neg graph to local indices in h_fused_dst_nodes
            # h_fused_dst_nodes is {ntype: tensor_for_dst_nodes_of_blocks[-1]}
            # We need blocks[-1].dstdata[dgl.NID] for mapping.
            
            dst_node_global_to_local_idx_map = {} # {ntype: {global_nid_val: local_idx_in_tensor}}
            if hasattr(g[-1], 'dstdata') and dgl.NID in g[-1].dstdata:
                for ntype_mfg_dst in g[-1].dsttypes:
                    if ntype_mfg_dst in g[-1].dstdata[dgl.NID]:
                        nids_tensor_global = g[-1].dstdata[dgl.NID][ntype_mfg_dst]
                        dst_node_global_to_local_idx_map[ntype_mfg_dst] = {
                            nid_val.item(): idx for idx, nid_val in enumerate(nids_tensor_global)
                        }
            else:
                print("Error: blocks[-1].dstdata[dgl.NID] not found. Cannot map features for scoring.")
                return {}, {}


            def get_edge_scores_mfg(graph_to_score, fused_embeddings_dict):
                scores_dict = {}
                if graph_to_score is None: return scores_dict

                for etype in graph_to_score.canonical_etypes:
                    src_ntype, rel_name, dst_ntype = etype
                    if dst_ntype not in self.target_node_types or graph_to_score.num_edges(etype) == 0:
                        continue
                    
                    # Edges in graph_to_score (pos_graph/neg_graph from EdgePredictionSampler)
                    # Their nodes are global NIDs.
                    # We need to use these global NIDs to look up indices in dst_node_global_to_local_idx_map
                    # to fetch from fused_embeddings_dict.
                    
                    # graph_to_score.edges() returns (u, v) which are local to graph_to_score.
                    # We need their global NIDs. These should be stored in graph_to_score.ndata[dgl.NID]
                    if dgl.NID not in graph_to_score.nodes[src_ntype].data or \
                       dgl.NID not in graph_to_score.nodes[dst_ntype].data:
                        print(f"Warning: dgl.NID missing in ndata of pos/neg graph for etype {etype}. Cannot score.")
                        continue

                    u_local, v_local = graph_to_score.edges(etype=etype)
                    u_global_nids = graph_to_score.nodes[src_ntype].data[dgl.NID][u_local]
                    v_global_nids = graph_to_score.nodes[dst_ntype].data[dgl.NID][v_local]

                    src_indices_in_fused = [dst_node_global_to_local_idx_map.get(src_ntype, {}).get(gid.item()) for gid in u_global_nids]
                    dst_indices_in_fused = [dst_node_global_to_local_idx_map.get(dst_ntype, {}).get(gid.item()) for gid in v_global_nids]

                    # Filter out None if any NID was not found in the map (should not happen if graphs are correct)
                    valid_mask = [(s_idx is not None and d_idx is not None) for s_idx, d_idx in zip(src_indices_in_fused, dst_indices_in_fused)]
                    if not any(valid_mask): continue

                    src_indices_final = torch.tensor([s for i,s in enumerate(src_indices_in_fused) if valid_mask[i]], dtype=torch.long)
                    dst_indices_final = torch.tensor([d for i,d in enumerate(dst_indices_in_fused) if valid_mask[i]], dtype=torch.long)
                    
                    if src_indices_final.numel() == 0 : continue


                    if src_ntype in fused_embeddings_dict and dst_ntype in fused_embeddings_dict:
                        src_emb_all = fused_embeddings_dict[src_ntype]
                        dst_emb_all = fused_embeddings_dict[dst_ntype]

                        src_emb_batch = src_emb_all[src_indices_final.to(src_emb_all.device)]
                        dst_emb_batch = dst_emb_all[dst_indices_final.to(dst_emb_all.device)]
                        
                        pairs = torch.cat([src_emb_batch, dst_emb_batch], dim=1)
                        expert_predictor = self.expert_predictors[dst_ntype]
                        actual_predictor_in_dim = expert_predictor[0].in_features
                        
                        if pairs.shape[1] == actual_predictor_in_dim:
                            score = expert_predictor(pairs).view(-1)
                            # Store scores corresponding to the valid_mask subset of original edges
                            # This requires careful handling if not all edges were valid.
                            # For now, assume all edges in batch are validly mapped. If valid_mask filters, score length changes.
                            # If we only score valid_mask edges, then it's fine.
                            scores_dict[etype] = score
                        else:
                            print(f"Warning (Score MFG): Predictor input dim mismatch for {etype}. Expected {actual_predictor_in_dim}, got {pairs.shape[1]}. Skipping.")
                return scores_dict

            pos_scores = get_edge_scores_mfg(pos_graph, h_fused_dst_nodes)
            neg_scores = get_edge_scores_mfg(neg_graph, h_fused_dst_nodes) # Re-use for neg_graph
            return pos_scores, neg_scores

        elif isinstance(g, dgl.DGLGraph): 
            collect_explainability = (
                self.enable_explainability
                and not self.training
                and self.fusion_mode == 'weighted'
            )

            if collect_explainability:
                h_fused_all_nodes, branch_outputs = self.get_embeddings(
                    g,
                    input_features,
                    edge_weights=edge_weights,
                    return_branch_outputs=True
                )
                self._capture_explainability(branch_outputs, g)
            else:
                self._last_explainability = None
                h_fused_all_nodes = self.get_embeddings(g, input_features, edge_weights=edge_weights)

            pos_scores = {}
            neg_scores = {}

            # Determine which edge types to use for scoring
            active_target_etypes = target_etypes_override if target_etypes_override is not None else self.target_etypes

            # Determine the fallback predictor if MoE is enabled but a specific expert is missing
            fallback_predictor = None
            if not self.disable_moe:
                if self.expert_predictors:
                    # Use the first available expert as a fallback
                    fallback_predictor = next(iter(self.expert_predictors.values()))
                else:
                    # This case should ideally not happen if MoE is on and there are target types, but as a safeguard:
                    logging.warning("MoE is enabled, but no expert predictors were created. Link prediction will fail.")

            if pos_graph is not None: # pos_graph is typically `g` itself or a subgraph
                for etype in pos_graph.canonical_etypes:
                    if etype not in active_target_etypes:
                        continue

                    src_ntype, _, dst_ntype = etype

                    # Prefer destination as the target side, but fall back to source when needed.
                    target_ntype = None
                    if dst_ntype in self.target_node_types:
                        target_ntype = dst_ntype
                    elif src_ntype in self.target_node_types:
                        target_ntype = src_ntype

                    predictor = None
                    if not self.disable_moe:
                        if target_ntype and target_ntype in self.expert_predictors:
                            predictor = self.expert_predictors[target_ntype]
                        else:
                            # Fallback for edge types whose endpoint is not backed by a dedicated expert
                            predictor = fallback_predictor
                    else:
                        predictor = self.shared_predictor

                    if predictor is None:
                        logging.warning(
                            "No predictor found for edge type %s (target endpoint: %s). Skipping.",
                            etype,
                            target_ntype,
                        )
                        continue

                    src_nodes, dst_nodes = pos_graph.edges(etype=etype)
                    if src_nodes.numel() == 0:
                        continue
                    
                    h_src = h_fused_all_nodes[src_ntype][src_nodes]
                    h_dst = h_fused_all_nodes[dst_ntype][dst_nodes]
                    
                    # Keep the expert input order as context/source node first, target node second.
                    if target_ntype == src_ntype and dst_ntype not in self.target_node_types:
                        edge_features = torch.cat([h_dst, h_src], dim=1)
                    else:
                        edge_features = torch.cat([h_src, h_dst], dim=1)
                    pos_scores[etype] = predictor(edge_features).view(-1)

            if neg_graph is not None:
                 for etype in neg_graph.canonical_etypes:
                    if etype not in active_target_etypes:
                        continue

                    src_ntype, _, dst_ntype = etype

                    target_ntype = None
                    if dst_ntype in self.target_node_types:
                        target_ntype = dst_ntype
                    elif src_ntype in self.target_node_types:
                        target_ntype = src_ntype

                    predictor = None
                    if not self.disable_moe:
                        if target_ntype and target_ntype in self.expert_predictors:
                            predictor = self.expert_predictors[target_ntype]
                        else:
                            predictor = fallback_predictor
                    else:
                        predictor = self.shared_predictor

                    if predictor is None:
                        logging.warning(
                            "No predictor found for negative edge type %s (target endpoint: %s). Skipping.",
                            etype,
                            target_ntype,
                        )
                        continue

                    src_nodes, dst_nodes = neg_graph.edges(etype=etype)
                    if src_nodes.numel() == 0:
                        continue

                    h_src = h_fused_all_nodes[src_ntype][src_nodes]
                    h_dst = h_fused_all_nodes[dst_ntype][dst_nodes]
                    
                    if target_ntype == src_ntype and dst_ntype not in self.target_node_types:
                        edge_features = torch.cat([h_dst, h_src], dim=1)
                    else:
                        edge_features = torch.cat([h_src, h_dst], dim=1)
                    neg_scores[etype] = predictor(edge_features).view(-1)
            
            # Apply numerical stability measures to prevent gradient explosion
            # especially important for sparse edge types like RNA
            for etype in pos_scores:
                if pos_scores[etype].numel() > 0:
                    # Ensure scores stay within reasonable range to prevent numerical issues
                    pos_scores[etype] = torch.clamp(pos_scores[etype], min=-10.0, max=10.0)
            
            for etype in neg_scores:
                if neg_scores[etype].numel() > 0:
                    neg_scores[etype] = torch.clamp(neg_scores[etype], min=-10.0, max=10.0)
                    
            return pos_scores, neg_scores
        else:
            logging.error(f"ERROR: model.py forward(): Unexpected type for 'g' argument: {type(g)}.")
            return {}, {}

    def _capture_explainability(self, branch_outputs, graph):
        try:
            explainability_payload = self._compute_explainability(branch_outputs, graph)
            self._last_explainability = explainability_payload
        except Exception as exc:
            logging.warning(f"Explainability generation failed: {exc}")
            self._last_explainability = None
        # Store fused embeddings on CPU for embedding-bridge path discovery
        try:
            fused = branch_outputs.get('fused', {}) if branch_outputs else {}
            self._last_fused_embeddings = {
                ntype: t.detach().cpu() for ntype, t in fused.items()
                if t is not None and t.numel() > 0
            }
        except Exception:
            self._last_fused_embeddings = None

    def _compute_explainability(self, branch_outputs, graph):
        if not branch_outputs or not isinstance(branch_outputs, dict):
            return None

        hgcn_outputs = branch_outputs.get('hgcn', {}) or {}
        transformer_outputs = branch_outputs.get('transformer', {}) or {}
        fused_outputs = branch_outputs.get('fused', {}) or {}
        fusion_weights = branch_outputs.get('fusion_weights', {}) or {}
        top_k = max(1, int(self.explainability_top_k))

        explain = {
            'fusion_mode': self.fusion_mode,
            'fusion_weights': {k: float(v) for k, v in fusion_weights.items()},
            'per_type': {},
            'edge_alignment': {}
        }

        if isinstance(graph, dgl.DGLGraph):
            self._ensure_explainability_adj_cache(graph)

        for ntype, hgcn_tensor in hgcn_outputs.items():
            if hgcn_tensor is None or hgcn_tensor.numel() == 0:
                continue

            per_type_entry = {}
            hgcn_cpu = hgcn_tensor.detach().float().cpu()
            per_type_entry['hgcn_topk_norm'] = self._topk_norms(hgcn_cpu, top_k)

            transformer_tensor = transformer_outputs.get(ntype)
            if transformer_tensor is not None and transformer_tensor.numel() > 0 and transformer_tensor.shape == hgcn_tensor.shape:
                transformer_cpu = transformer_tensor.detach().float().cpu()
                per_type_entry['transformer_topk_norm'] = self._topk_norms(transformer_cpu, top_k)

                alignment = F.cosine_similarity(hgcn_cpu, transformer_cpu, dim=1)
                per_type_entry['alignment_topk'] = self._topk_values(alignment, top_k)
                per_type_entry['alignment_summary'] = {
                    'mean': float(alignment.mean().item()),
                    'std': float(alignment.std(unbiased=False).item()) if alignment.numel() > 1 else 0.0,
                    'min': float(alignment.min().item()),
                    'max': float(alignment.max().item())
                }
            else:
                per_type_entry['transformer_topk_norm'] = []

            fused_tensor = fused_outputs.get(ntype)
            if fused_tensor is not None and fused_tensor.numel() > 0:
                fused_cpu = fused_tensor.detach().float().cpu()
                per_type_entry['fused_topk_norm'] = self._topk_norms(fused_cpu, top_k)
            else:
                per_type_entry['fused_topk_norm'] = []

            # Determine dominant branch per node type (preference order: fusion_weights -> norm comparison)
            try:
                dom = 'mixed'
                if ntype in fusion_weights:
                    w = float(fusion_weights.get(ntype, float('nan')))
                    if not math.isnan(w):
                        if w >= 0.66:
                            dom = 'transformer'
                        elif w <= 0.33:
                            dom = 'hgcn'
                        else:
                            dom = 'mixed'
                else:
                    # fallback: use mean of topk norms if available
                    try:
                        hg_norms = [item.get('norm', 0.0) for item in per_type_entry.get('hgcn_topk_norm', [])]
                        tf_norms = [item.get('norm', 0.0) for item in per_type_entry.get('transformer_topk_norm', [])]
                        mean_h = float(sum(hg_norms) / len(hg_norms)) if hg_norms else 0.0
                        mean_t = float(sum(tf_norms) / len(tf_norms)) if tf_norms else 0.0
                        denom = mean_h + mean_t + 1e-9
                        ratio_t = mean_t / denom
                        if ratio_t >= 0.66:
                            dom = 'transformer'
                        elif ratio_t <= 0.33:
                            dom = 'hgcn'
                        else:
                            dom = 'mixed'
                    except Exception:
                        dom = 'mixed'
                per_type_entry['dominant_branch'] = dom
            except Exception:
                per_type_entry['dominant_branch'] = 'mixed'

            explain['per_type'][ntype] = per_type_entry

        if isinstance(graph, dgl.DGLGraph):
            for etype in graph.canonical_etypes:
                if etype not in self.target_etypes:
                    continue
                src_ntype, _, dst_ntype = etype
                if src_ntype not in fused_outputs or dst_ntype not in fused_outputs:
                    continue

                src_tensor = fused_outputs[src_ntype]
                dst_tensor = fused_outputs[dst_ntype]
                if src_tensor is None or dst_tensor is None or src_tensor.numel() == 0 or dst_tensor.numel() == 0:
                    continue

                src_nodes, dst_nodes = graph.edges(etype=etype)
                if src_nodes.numel() == 0:
                    continue

                src_embed = src_tensor[src_nodes].detach().float().cpu()
                dst_embed = dst_tensor[dst_nodes].detach().float().cpu()
                similarities = F.cosine_similarity(src_embed, dst_embed, dim=1)
                src_nodes_cpu = src_nodes.detach().cpu()
                dst_nodes_cpu = dst_nodes.detach().cpu()
                explain['edge_alignment'][etype] = self._topk_edges(
                    graph,
                    etype,
                    similarities,
                    src_nodes_cpu,
                    dst_nodes_cpu,
                    top_k,
                    hgcn_outputs=hgcn_outputs,
                    transformer_outputs=transformer_outputs,
                    fused_outputs=fused_outputs,
                    fusion_weights=fusion_weights
                )

        return explain

    @staticmethod
    def _topk_norms(tensor, top_k):
        if tensor.ndim < 2:
            return []
        norms = torch.norm(tensor, dim=1)
        if norms.numel() == 0:
            return []
        k = min(top_k, norms.shape[0])
        values, indices = torch.topk(norms, k)
        return [
            {
                'node_idx': int(indices[i].item()),
                'norm': float(values[i].item())
            }
            for i in range(k)
        ]

    @staticmethod
    def _topk_values(values_tensor, top_k):
        if values_tensor.numel() == 0:
            return []
        k = min(top_k, values_tensor.shape[0])
        values, indices = torch.topk(values_tensor, k)
        return [
            {
                'node_idx': int(indices[i].item()),
                'value': float(values[i].item())
            }
            for i in range(k)
        ]

    def _topk_edges(self, graph, etype, similarities, src_nodes, dst_nodes, top_k,
                    hgcn_outputs=None, transformer_outputs=None, fused_outputs=None, fusion_weights=None):
        if similarities.numel() == 0:
            return []
        k = min(top_k, similarities.shape[0])
        values, indices = torch.topk(similarities, k)
        src_ntype, _, dst_ntype = etype
        entries = []
        for i in range(k):
            edge_idx = int(indices[i].item())
            src_idx = int(src_nodes[edge_idx].item())
            dst_idx = int(dst_nodes[edge_idx].item())
            entry = {
                'edge_idx': edge_idx,
                'src_idx': src_idx,
                'dst_idx': dst_idx,
                'similarity': float(values[i].item())
            }
            # compute branch-specific similarities if available
            try:
                if hgcn_outputs and src_ntype in hgcn_outputs and dst_ntype in hgcn_outputs:
                    h_src = hgcn_outputs[src_ntype][src_idx].detach().float().cpu().unsqueeze(0)
                    h_dst = hgcn_outputs[dst_ntype][dst_idx].detach().float().cpu().unsqueeze(0)
                    entry['hgcn_sim'] = float(F.cosine_similarity(h_src, h_dst, dim=1).item())
                else:
                    entry['hgcn_sim'] = None
            except Exception:
                entry['hgcn_sim'] = None

            try:
                if transformer_outputs and src_ntype in transformer_outputs and dst_ntype in transformer_outputs:
                    t_src = transformer_outputs[src_ntype][src_idx].detach().float().cpu().unsqueeze(0)
                    t_dst = transformer_outputs[dst_ntype][dst_idx].detach().float().cpu().unsqueeze(0)
                    entry['transformer_sim'] = float(F.cosine_similarity(t_src, t_dst, dim=1).item())
                else:
                    entry['transformer_sim'] = None
            except Exception:
                entry['transformer_sim'] = None

            # fusion contribution: if fusion_weights provided, include per-ntype weight (transformer share)
            try:
                if fusion_weights and isinstance(fusion_weights, dict):
                    # fusion_weights are stored as transformer weight in our convention
                    entry['fusion_weight_src'] = float(fusion_weights.get(src_ntype, float('nan')))
                    entry['fusion_weight_dst'] = float(fusion_weights.get(dst_ntype, float('nan')))
                else:
                    entry['fusion_weight_src'] = None
                    entry['fusion_weight_dst'] = None
            except Exception:
                entry['fusion_weight_src'] = None
                entry['fusion_weight_dst'] = None

            # Determine dominant branch for this edge (hgcn | transformer | mixed)
            try:
                dom = 'mixed'
                fws = entry.get('fusion_weight_src')
                fwd = entry.get('fusion_weight_dst')
                if fws is not None and fwd is not None and (not (math.isnan(fws) or math.isnan(fwd))):
                    avg_w = float((fws + fwd) / 2.0)
                    if avg_w >= 0.66:
                        dom = 'transformer'
                    elif avg_w <= 0.33:
                        dom = 'hgcn'
                    else:
                        dom = 'mixed'
                else:
                    # Fallback: compare branch similarities if available
                    hsim = entry.get('hgcn_sim')
                    tsim = entry.get('transformer_sim')
                    if hsim is not None and tsim is not None:
                        try:
                            if tsim > hsim * 1.2:
                                dom = 'transformer'
                            elif hsim > tsim * 1.2:
                                dom = 'hgcn'
                            else:
                                dom = 'mixed'
                        except Exception:
                            dom = 'mixed'
                entry['dominant_branch'] = dom
            except Exception:
                entry['dominant_branch'] = 'mixed'

            path = self._find_weighted_shortest_path(
                graph,
                src_ntype,
                src_idx,
                dst_ntype,
                dst_idx
            )
            if path:
                entry['path'] = self._format_path(graph, path)
            entries.append(entry)
        return entries

    def _ensure_explainability_adj_cache(self, graph):
        if not isinstance(graph, dgl.DGLGraph):
            return
        graph_identity = id(graph)
        if self._explainability_adj_cache_key == graph_identity and self._explainability_adj_cache is not None:
            return

        adjacency = defaultdict(list)
        for etype in graph.canonical_etypes:
            src_ntype, _, dst_ntype = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            if src_nodes.numel() == 0:
                continue
            src_list = src_nodes.detach().cpu().tolist()
            dst_list = dst_nodes.detach().cpu().tolist()
            for s_node, d_node in zip(src_list, dst_list):
                src_key = (src_ntype, int(s_node))
                dst_key = (dst_ntype, int(d_node))
                adjacency[src_key].append(dst_key)
                adjacency[dst_key].append(src_key)

        self._explainability_adj_cache = adjacency
        self._explainability_adj_cache_key = graph_identity

    def _get_embedding_similarity(self, ntype_a, idx_a, ntype_b, idx_b):
        fused = getattr(self, '_last_fused_embeddings', None)
        if fused is None:
            return 0.0
        emb_a = fused.get(ntype_a)
        emb_b = fused.get(ntype_b)
        if emb_a is None or emb_b is None:
            return 0.0
        if int(idx_a) >= emb_a.shape[0] or int(idx_b) >= emb_b.shape[0]:
            return 0.0
        try:
            va = emb_a[int(idx_a)].float().unsqueeze(0)
            vb = emb_b[int(idx_b)].float().unsqueeze(0)
            sim = F.cosine_similarity(va, vb, dim=1).item()
            return max(-1.0, min(1.0, float(sim)))
        except Exception:
            return 0.0

    def _get_embedding_edge_cost(self, ntype_a, idx_a, ntype_b, idx_b, edge_kind='physical'):
        """Return embedding-based edge cost with explicit path penalties.

        Physical edge cost: 1 - cosine_similarity + hop_penalty
        Virtual edge cost: 1 - cosine_similarity + virtual_edge_penalty
        """
        sim = self._get_embedding_similarity(ntype_a, idx_a, ntype_b, idx_b)
        base_cost = max(0.0, 1.0 - sim)
        if edge_kind == 'virtual':
            return base_cost + self.explainability_virtual_edge_penalty
        return base_cost + self.explainability_hop_penalty

    def _get_attribution_score(self, feature_tensor, gradient_tensor):
        if self.explainability_attribution_method == 'gradient_norm':
            return gradient_tensor.norm(dim=1)
        return (feature_tensor * gradient_tensor).abs().sum(dim=1)

    def _find_weighted_shortest_path(self, graph, src_ntype, src_idx,
                                     dst_ntype, dst_idx, max_hops=None):
        """Dijkstra on the physical heterogeneous graph with embedding-cosine
        edge costs.  Returns a list of ((ntype, idx), edge_cost) tuples
        representing the cheapest path, or None if unreachable within max_hops."""
        hop_limit = int(max_hops) if max_hops is not None else int(self.explainability_max_hops)
        if hop_limit <= 0:
            return None
        if self._explainability_adj_cache is None:
            self._ensure_explainability_adj_cache(graph)
        adjacency = self._explainability_adj_cache or {}

        start = (src_ntype, int(src_idx))
        goal = (dst_ntype, int(dst_idx))
        if start == goal:
            return [(start, 0.0)]

        # Priority queue: (cumulative_cost, hop_count, node, path_with_costs)
        # path_with_costs: list of ((ntype, idx), edge_cost_to_arrive)
        init_entry = (0.0, 0, start, [(start, 0.0)])
        heap = [init_entry]
        best_cost = {start: 0.0}

        while heap:
            cum_cost, hops, current, path = heapq.heappop(heap)
            if current == goal:
                return path
            if hops >= hop_limit:
                continue
            if cum_cost > best_cost.get(current, float('inf')):
                continue  # stale entry

            for neighbor in adjacency.get(current, []):
                edge_cost = self._get_embedding_edge_cost(
                    current[0], current[1], neighbor[0], neighbor[1], edge_kind='physical'
                )
                new_cost = cum_cost + edge_cost
                if new_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = new_cost
                    new_path = path + [(neighbor, edge_cost)]
                    heapq.heappush(heap, (new_cost, hops + 1, neighbor, new_path))
        return None  # unreachable

    def _format_path(self, graph, path_nodes):
        """Format a path for display.

        Accepts two formats:
        - Legacy: list of (ntype, idx) tuples
        - Weighted: list of ((ntype, idx), edge_cost) tuples from Dijkstra
        """
        if not path_nodes:
            return None

        # Detect format
        first = path_nodes[0]
        is_weighted = isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (list, tuple))

        formatted_segments = []
        raw_sims = []
        prev_node = None
        for entry in path_nodes:
            if is_weighted:
                (ntype, nid), edge_cost = entry
            else:
                ntype, nid = entry
                edge_cost = None

            seg = f"{ntype}:{int(nid)}"
            node_name = self._resolve_node_name(graph, ntype, int(nid))
            if node_name:
                seg = f"{seg} ({node_name})"
            if prev_node is not None:
                raw_sim = self._get_embedding_similarity(prev_node[0], prev_node[1], ntype, int(nid))
                raw_sims.append(raw_sim)
                seg += f" [sim:{raw_sim:.3f}]"
            formatted_segments.append(seg)
            prev_node = (ntype, int(nid))

        path_str = " -> ".join(formatted_segments)
        if is_weighted and raw_sims:
            avg_sim = sum(raw_sims) / len(raw_sims)
            path_str += f" (avg_sim:{avg_sim:.3f})"
        return path_str

    def get_last_explainability(self):
        return self._last_explainability

    def get_explainability_path_details(self, graph, src_ntype, src_idx, dst_ntype, dst_idx,
                                        max_hops=None, h_initial=None):
        """Return structured explainability path details.

        physical_shortest_path always refers to Tier 1 physical-graph Dijkstra.
        final_displayed_path is the path shown to the user: physical if available,
        otherwise the Tier 2 gradient-guided fallback.
        """
        if not self.enable_explainability:
            return None

        hop_limit = int(max_hops) if max_hops is not None else int(self.explainability_max_hops)
        details = {
            'physical_shortest_path': None,
            'final_displayed_path': None,
            'final_path_kind': 'none',
            'physical_path_status': f"N/A (no physical path within {hop_limit} hops)",
        }
        try:
            self._ensure_explainability_adj_cache(graph)

            weighted_path = self._find_weighted_shortest_path(
                graph, src_ntype, src_idx, dst_ntype, dst_idx,
                max_hops=max_hops
            )
            if weighted_path:
                physical_path = self._format_path(graph, weighted_path)
                details['physical_shortest_path'] = physical_path
                details['physical_path_status'] = physical_path
                details['final_displayed_path'] = physical_path
                details['final_path_kind'] = 'physical'
                return details

            if h_initial is not None:
                guided_path = self._find_gradient_guided_path(
                    graph, h_initial,
                    src_ntype, int(src_idx), dst_ntype, int(dst_idx),
                    max_hops=max_hops
                )
                if guided_path:
                    details['final_displayed_path'] = guided_path
                    details['final_path_kind'] = 'gradient_guided'
                    return details

            return details
        except Exception as exc:
            logging.debug(f"Explainability path lookup failed: {exc}")
            return details

    def get_explainability_path(self, graph, src_ntype, src_idx, dst_ntype, dst_idx,
                               max_hops=None, h_initial=None):
        """Backward-compatible wrapper returning only the final displayed path."""
        details = self.get_explainability_path_details(
            graph,
            src_ntype,
            src_idx,
            dst_ntype,
            dst_idx,
            max_hops=max_hops,
            h_initial=h_initial,
        )
        if not details:
            return None
        return details.get('final_displayed_path')

    # ------------------------------------------------------------------
    # Gradient-guided path (fallback when physical Dijkstra fails)
    # ------------------------------------------------------------------

    def _compute_node_attributions(self, g, h_initial, src_ntype, src_idx,
                                   dst_ntype, dst_idx):
        """Return dict { (ntype, idx): attribution_score } for nodes with
        nonzero influence on the (src→dst) prediction score."""
        was_training = self.training
        self.eval()

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        h_grad = {}
        for ntype, feat in h_initial.items():
            t = feat.to(device).detach().clone().float()
            t.requires_grad_(True)
            h_grad[ntype] = t

        g_dev = g.to(device) if g.device != device else g
        h_fused = self.get_embeddings(g_dev, h_grad)

        if src_ntype not in h_fused or dst_ntype not in h_fused:
            if was_training:
                self.train()
            return {}
        if int(src_idx) >= h_fused[src_ntype].shape[0] or int(dst_idx) >= h_fused[dst_ntype].shape[0]:
            if was_training:
                self.train()
            return {}

        src_embed = h_fused[src_ntype][int(src_idx)]
        dst_embed = h_fused[dst_ntype][int(dst_idx)]
        pairs = torch.cat([src_embed, dst_embed], dim=0).unsqueeze(0)

        if self.disable_moe:
            predictor = self.shared_predictor
        else:
            predictor = self.expert_predictors.get(dst_ntype)
            if predictor is None and self.expert_predictors:
                predictor = next(iter(self.expert_predictors.values()))
        if predictor is None:
            if was_training:
                self.train()
            return {}

        score = predictor(pairs).squeeze()
        score.backward()

        attr_map = {}
        for ntype, feat in h_grad.items():
            if feat.grad is None:
                continue
            attribution_scores = self._get_attribution_score(feat.detach(), feat.grad.detach()).cpu()
            for idx in range(attribution_scores.shape[0]):
                val = attribution_scores[idx].item()
                if val > 1e-8:
                    attr_map[(ntype, idx)] = val

        if was_training:
            self.train()
        return attr_map

    def _find_gradient_guided_path(self, graph, h_initial,
                                   src_ntype, src_idx, dst_ntype, dst_idx,
                                   max_hops=None, top_k_nodes=50,
                                   sim_threshold=None):
        """Build a *path* through high-attribution nodes.

        Steps:
        1. Compute gradient attribution for all nodes.
        2. Select the top-K attributed nodes (plus src and dst).
        3. Build a small augmented adjacency among these nodes:
           - Physical edges from the original graph.
           - Virtual embedding edges for pairs with cosine sim > threshold.
        4. Run Dijkstra (embedding-weighted) on this subgraph.
        5. Return a formatted path with per-hop annotation.
        """
        hop_limit = int(max_hops) if max_hops is not None else int(self.explainability_max_hops)
        # Allow a relaxed hop limit for the augmented graph since the subgraph
        # is small and paths through virtual edges may be longer.
        relaxed_hop_limit = max(hop_limit * 2, 6)
        sim_threshold = self.explainability_virtual_sim_threshold if sim_threshold is None else float(sim_threshold)

        fused = getattr(self, '_last_fused_embeddings', None)
        if fused is None:
            return None

        # Step 1: attribution
        attr_map = self._compute_node_attributions(
            graph, h_initial, src_ntype, src_idx, dst_ntype, dst_idx
        )
        if not attr_map:
            return None

        # Step 2: select subgraph nodes
        start = (src_ntype, int(src_idx))
        goal = (dst_ntype, int(dst_idx))
        sorted_nodes = sorted(attr_map.items(), key=lambda x: x[1], reverse=True)
        subgraph_nodes = {start, goal}
        for node_key, _ in sorted_nodes[:top_k_nodes]:
            subgraph_nodes.add(node_key)

        # Step 3: build augmented adjacency among subgraph nodes
        # 3a: physical edges
        physical_adj = self._explainability_adj_cache or {}
        sub_adj = defaultdict(list)  # node -> list of (neighbor, edge_cost, edge_type)
        for node in subgraph_nodes:
            for neighbor in physical_adj.get(node, []):
                if neighbor in subgraph_nodes:
                    cost = self._get_embedding_edge_cost(
                        node[0], node[1], neighbor[0], neighbor[1], edge_kind='physical'
                    )
                    sub_adj[node].append((neighbor, cost, 'physical'))

        # 3b: virtual embedding edges
        node_list = list(subgraph_nodes)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                na, nb = node_list[i], node_list[j]
                # Skip if already physically connected
                if any(nbr == nb for nbr, _, _ in sub_adj.get(na, [])):
                    continue
                emb_a = fused.get(na[0])
                emb_b = fused.get(nb[0])
                if emb_a is None or emb_b is None:
                    continue
                if na[1] >= emb_a.shape[0] or nb[1] >= emb_b.shape[0]:
                    continue
                va = emb_a[na[1]].float().unsqueeze(0)
                vb = emb_b[nb[1]].float().unsqueeze(0)
                sim = F.cosine_similarity(va, vb, dim=1).item()
                if sim >= sim_threshold:
                    cost = self._get_embedding_edge_cost(na[0], na[1], nb[0], nb[1], edge_kind='virtual')
                    sub_adj[na].append((nb, cost, 'virtual'))
                    sub_adj[nb].append((na, cost, 'virtual'))

        # Step 4: Dijkstra on augmented subgraph
        # Entries: (cumulative_cost, hops, node, path)
        # path: list of ((ntype, idx), edge_cost, edge_type_str)
        heap = [(0.0, 0, start, [(start, 0.0, 'start')])]
        best_cost = {start: 0.0}

        result_path = None
        while heap:
            cum_cost, hops, current, path = heapq.heappop(heap)
            if current == goal:
                result_path = path
                break
            if hops >= relaxed_hop_limit:
                continue
            if cum_cost > best_cost.get(current, float('inf')):
                continue
            for neighbor, edge_cost, edge_type in sub_adj.get(current, []):
                new_cost = cum_cost + edge_cost
                if new_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = new_cost
                    new_path = path + [(neighbor, edge_cost, edge_type)]
                    heapq.heappush(heap, (new_cost, hops + 1, neighbor, new_path))

        if result_path is None:
            return None

        # Step 5: format with per-hop annotations
        formatted_segments = []
        raw_sims = []
        prev_node = None
        for (ntype, nid), edge_cost, edge_type in result_path:
            seg = f"{ntype}:{int(nid)}"
            node_name = self._resolve_node_name(graph, ntype, int(nid))
            if node_name:
                seg += f" ({node_name})"
            # Attribution strength indicator
            attr_val = attr_map.get((ntype, int(nid)))
            if attr_val is not None and attr_val > 1e-6:
                seg += f" [attr:{attr_val:.4f}]"
            # Edge annotation
            if prev_node is not None:
                raw_sim = self._get_embedding_similarity(prev_node[0], prev_node[1], ntype, int(nid))
                raw_sims.append(raw_sim)
                if edge_type == 'virtual':
                    seg += f" [virtual: inferred-functional-similarity, sim:{raw_sim:.3f}]"
                elif edge_type == 'physical':
                    seg += f" [sim:{raw_sim:.3f}]"

            formatted_segments.append(seg)
            prev_node = (ntype, int(nid))

        path_str = " -> ".join(formatted_segments)
        avg_sim = (sum(raw_sims) / len(raw_sims)) if raw_sims else 0.0
        has_virtual = any(et == 'virtual' for _, _, et in result_path)
        tag = "[gradient-guided path]" if has_virtual else "[weighted shortest path]"
        if has_virtual:
            path_str += " [virtual edges denote inferred functional similarity, not observed graph edges]"
        return f"{tag} {path_str} (avg_sim:{avg_sim:.3f})"

    def _resolve_node_name(self, graph, ntype, node_idx):
        """Best-effort resolution of a DGL node index to a human-readable name."""
        node_idx = int(node_idx)
        try:
            if isinstance(graph, dgl.DGLGraph) and ntype in graph.ntypes:
                ndata = graph.nodes[ntype].data
                if 'name' in ndata:
                    val = ndata['name'][node_idx]
                    if isinstance(val, bytes):
                        return val.decode('utf-8', errors='ignore')
                    return str(val)
        except Exception:
            pass
        try:
            if self.node_info:
                dgl_map = self.node_info.get('dglid_to_orig', {})
                id_map = self.node_info.get('id_to_node_data', {})
                orig_id = dgl_map.get(ntype, {}).get(node_idx)
                if orig_id is not None:
                    data = id_map.get(orig_id, {})
                    name = data.get('name') or data.get('Name')
                    if name is not None:
                        return str(name)
        except Exception:
            pass
        return None

    def predict_links(self, g, h_initial_for_g, source_ntype, source_dgl_ids, target_nodes_dict_with_ids):
        try:
            device = next(self.parameters()).device
        except StopIteration:
            print("Warning: Model has no parameters. Using CPU as fallback device.")
            device = torch.device("cpu")
        
        collect_explainability = (
            self.enable_explainability
            and self.fusion_mode == 'weighted'
        )

        if collect_explainability:
            h_fused_all_nodes_in_g, branch_outputs = self.get_embeddings(
                g,
                h_initial_for_g,
                return_branch_outputs=True
            )
            self._capture_explainability(branch_outputs, g)
        else:
            self._last_explainability = None
            h_fused_all_nodes_in_g = self.get_embeddings(g, h_initial_for_g)

        # Store fused embeddings for bridge-path / attribution in non-weighted modes
        if self.enable_explainability and not collect_explainability:
            try:
                self._last_fused_embeddings = {
                    ntype: t.detach().cpu() for ntype, t in h_fused_all_nodes_in_g.items()
                    if t is not None and t.numel() > 0
                }
            except Exception:
                self._last_fused_embeddings = None

        if source_ntype not in h_fused_all_nodes_in_g:
            print(f"Warning (Predict): No embeddings for source type '{source_ntype}'. Returning empty.")
            return {ntype: torch.empty(0, device=device) for ntype in target_nodes_dict_with_ids}

        source_node_embeddings_all_type = h_fused_all_nodes_in_g[source_ntype]
        
        # Ensure source_dgl_ids are valid indices
        s_ids_tensor = torch.as_tensor(source_dgl_ids, device=source_node_embeddings_all_type.device)
        s_ids_mask = (s_ids_tensor >= 0) & (s_ids_tensor < source_node_embeddings_all_type.shape[0])
        valid_source_ids_tensor = s_ids_tensor[s_ids_mask]

        if valid_source_ids_tensor.numel() == 0:
            print(f"Error (Predict): No valid source DGL IDs found for type '{source_ntype}' after filtering.")
            return {ntype: torch.empty(0, device=device) for ntype in target_nodes_dict_with_ids}
        
        selected_source_embeddings = source_node_embeddings_all_type[valid_source_ids_tensor]

        predictions = {}
        
        for target_ntype, target_dgl_ids_list in target_nodes_dict_with_ids.items():
            if target_ntype not in self.target_node_types:
                predictions[target_ntype] = torch.empty((selected_source_embeddings.shape[0], 0), device=device)
                continue
            if target_ntype not in h_fused_all_nodes_in_g:
                predictions[target_ntype] = torch.empty((selected_source_embeddings.shape[0], 0), device=device)
                continue
            if not target_dgl_ids_list:
                predictions[target_ntype] = torch.empty((selected_source_embeddings.shape[0], 0), device=device)
                continue

            target_node_embeddings_all_type = h_fused_all_nodes_in_g[target_ntype]
            t_ids_tensor = torch.as_tensor(target_dgl_ids_list, device=target_node_embeddings_all_type.device)
            t_ids_mask = (t_ids_tensor >= 0) & (t_ids_tensor < target_node_embeddings_all_type.shape[0])
            valid_target_ids_tensor = t_ids_tensor[t_ids_mask]
            valid_positions = torch.nonzero(t_ids_mask, as_tuple=False).flatten()

            num_requested_targets = len(target_dgl_ids_list)

            if valid_target_ids_tensor.numel() == 0:
                fill_scores = torch.full(
                    (selected_source_embeddings.shape[0], num_requested_targets),
                    float('-inf'),
                    device=device,
                    dtype=selected_source_embeddings.dtype,
                )
                predictions[target_ntype] = fill_scores
                continue

            selected_target_embeddings = target_node_embeddings_all_type[valid_target_ids_tensor]

            num_inputs = selected_source_embeddings.shape[0]
            num_outputs = selected_target_embeddings.shape[0]
            
            if self.disable_moe:
                expert_predictor = self.shared_predictor
            else:
                expert_predictor = self.expert_predictors[target_ntype]

            actual_predictor_in_dim_pl = expert_predictor[0].in_features

            mem_estimate_gb = (num_inputs * num_outputs * actual_predictor_in_dim_pl * 4) / (1024**3)

            valid_scores = torch.empty((num_inputs, 0), device=device)

            if mem_estimate_gb > 2.0 and num_inputs > 0 and num_outputs > 0: # Added checks for num_inputs/outputs
                max_targets_per_batch = min(1000, max(100, int(2.0 * (1024**3) / (num_inputs * actual_predictor_in_dim_pl * 4))))
                if max_targets_per_batch == 0:
                    max_targets_per_batch = 1 # Ensure at least 1 if calculation is too small
                all_scores_list = []
                for i in range(0, num_outputs, max_targets_per_batch):
                    batch_target_embeddings = selected_target_embeddings[i : i + max_targets_per_batch]
                    current_batch_size = batch_target_embeddings.shape[0]
                    if current_batch_size == 0:
                        continue # Should not happen if num_outputs > 0

                    input_emb_expanded = selected_source_embeddings.unsqueeze(1).expand(-1, current_batch_size, -1)
                    output_emb_expanded = batch_target_embeddings.unsqueeze(0).expand(num_inputs, -1, -1)

                    pairs = torch.cat([input_emb_expanded, output_emb_expanded], dim=2)

                    if pairs.shape[-1] == actual_predictor_in_dim_pl:
                        batch_scores = expert_predictor(pairs.reshape(-1, actual_predictor_in_dim_pl)).reshape(num_inputs, current_batch_size)
                        all_scores_list.append(batch_scores)
                    else:
                        print(f"Error (Predict Batch): Dim mismatch for predictor '{target_ntype}'. Exp {actual_predictor_in_dim_pl}, Got {pairs.shape[-1]}. Appending empty.")
                        all_scores_list.append(torch.empty((num_inputs, current_batch_size), device=device, dtype=selected_source_embeddings.dtype))
                        break
                    del pairs, batch_scores
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if all_scores_list:
                    valid_scores = torch.cat(all_scores_list, dim=1)
            elif num_inputs > 0 and num_outputs > 0: # Small prediction task
                input_emb_expanded = selected_source_embeddings.unsqueeze(1).expand(-1, num_outputs, -1)
                output_emb_expanded = selected_target_embeddings.unsqueeze(0).expand(num_inputs, -1, -1)
                pairs = torch.cat([input_emb_expanded, output_emb_expanded], dim=2)

                if pairs.shape[-1] == actual_predictor_in_dim_pl:
                    valid_scores = expert_predictor(pairs.reshape(-1, actual_predictor_in_dim_pl)).reshape(num_inputs, num_outputs)
                else:
                    print(f"Error (Predict Small): Dim mismatch for predictor '{target_ntype}'. Exp {actual_predictor_in_dim_pl}, Got {pairs.shape[-1]}. Skipping.")
            # else: num_inputs or num_outputs is 0, leave valid_scores empty

            if valid_positions.numel() == num_requested_targets and valid_scores.shape[1] == num_requested_targets:
                predictions[target_ntype] = valid_scores
                continue

            fill_scores = torch.full(
                (num_inputs, num_requested_targets),
                float('-inf'),
                device=device,
                dtype=selected_source_embeddings.dtype,
            )

            if valid_scores.numel() > 0 and valid_positions.numel() == valid_scores.shape[1]:
                indices = valid_positions.to(device=device, dtype=torch.long)
                fill_scores.index_copy_(1, indices, valid_scores)

            predictions[target_ntype] = fill_scores
        return predictions
