"""Relation-aware heterogeneous GNN baseline."""

import math
import re

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv


def _etype_key(etype):
    return re.sub(r"[^0-9A-Za-z_]+", "_", "__".join(map(str, etype)))


class RGCNLayer(nn.Module):
    """Relation-specific graph convolution over a DGL heterograph."""

    def __init__(self, in_dim, out_dim, etypes, activation=True):
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.conv = HeteroGraphConv(
            {
                etype[1]: GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
                for etype in etypes
            },
            aggregate="sum",
        )

    def forward(self, g, h, edge_weights=None):
        mod_kwargs = {}
        if edge_weights:
            for etype in g.canonical_etypes:
                if etype in edge_weights:
                    mod_kwargs[etype[1]] = {"edge_weight": edge_weights[etype].float()}

        try:
            h_out = self.conv(g, h, mod_kwargs=mod_kwargs)
        except dgl.DGLError:
            h_out = self.conv(g, h)

        for ntype, feat in h.items():
            if ntype not in h_out:
                h_out[ntype] = feat

        if self.activation:
            h_out = {ntype: F.relu(feat) for ntype, feat in h_out.items()}
        return h_out


class RGCNModel(nn.Module):
    """R-GCN baseline with a DistMult decoder."""

    def __init__(
        self,
        node_feature_embeddings,
        rgcn_hidden,
        rgcn_out,
        num_rgcn_layers,
        ntypes,
        etypes,
        target_node_types,
        num_nodes_dict,
        node_info=None,
        dropout=0.1,
        freeze_pretrained=True,
    ):
        super().__init__()
        self.disable_hgcn = False
        self.disable_transformer = True
        self.disable_moe = True
        self.fusion_mode = "rgcn"
        self.enable_explainability = False
        self.node_info = node_info
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.rgcn_hidden = int(rgcn_hidden)
        self.rgcn_out = int(rgcn_out)
        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }
        self.dropout = nn.Dropout(dropout)

        self.node_embeddings = nn.ModuleDict()
        input_dims = {}
        for ntype in self.ntypes:
            raw = node_feature_embeddings.get(ntype) if node_feature_embeddings else None
            if isinstance(raw, nn.Embedding):
                emb = raw
                emb.weight.requires_grad = not freeze_pretrained
            elif isinstance(raw, torch.Tensor):
                emb = nn.Embedding(raw.shape[0], raw.shape[1], _weight=raw.float())
                emb.weight.requires_grad = not freeze_pretrained
            else:
                emb = nn.Embedding(int(num_nodes_dict.get(ntype, 1)), self.rgcn_hidden)
                nn.init.xavier_uniform_(emb.weight)
            self.node_embeddings[ntype] = emb
            input_dims[ntype] = emb.embedding_dim

        self.input_projection = nn.ModuleDict()
        for ntype, in_dim in input_dims.items():
            if in_dim != self.rgcn_hidden:
                self.input_projection[ntype] = nn.Linear(in_dim, self.rgcn_hidden)

        layer_count = max(1, int(num_rgcn_layers))
        self.layers = nn.ModuleList()
        if layer_count == 1:
            self.layers.append(RGCNLayer(self.rgcn_hidden, self.rgcn_out, self.etypes, activation=False))
        else:
            self.layers.append(RGCNLayer(self.rgcn_hidden, self.rgcn_hidden, self.etypes, activation=True))
            for _ in range(layer_count - 2):
                self.layers.append(RGCNLayer(self.rgcn_hidden, self.rgcn_hidden, self.etypes, activation=True))
            self.layers.append(RGCNLayer(self.rgcn_hidden, self.rgcn_out, self.etypes, activation=False))

        self.relation_embeddings = nn.ModuleDict()
        self.relation_bias = nn.ParameterDict()
        self._ensure_relations(self.etypes)

    def _ensure_relations(self, etypes):
        for etype in etypes:
            key = _etype_key(etype)
            if key not in self.relation_embeddings:
                rel_emb = nn.Embedding(1, self.rgcn_out)
                nn.init.xavier_uniform_(rel_emb.weight)
                self.relation_embeddings[key] = rel_emb
            if key not in self.relation_bias:
                self.relation_bias[key] = nn.Parameter(torch.zeros(1))

    def set_target_etypes(self, target_etypes):
        if not target_etypes:
            return
        self.target_etypes = set(target_etypes)
        self._ensure_relations(target_etypes)

    def get_embeddings(self, g, initial_features=None, edge_weights=None, return_branch_outputs=False):
        h = {}
        for ntype in g.ntypes:
            if initial_features and ntype in initial_features and initial_features[ntype] is not None:
                feat = initial_features[ntype].float()
            else:
                node_ids = torch.arange(g.num_nodes(ntype), device=g.device)
                feat = self.node_embeddings[ntype](node_ids)
            if ntype in self.input_projection:
                feat = self.input_projection[ntype](feat)
            h[ntype] = feat

        for layer in self.layers:
            h = layer(g, h, edge_weights=edge_weights)
            h = {ntype: self.dropout(feat) for ntype, feat in h.items()}

        if return_branch_outputs:
            return h, {"rgcn": h}
        return h

    def _score_edges(self, h, graph, etype):
        src_type, _, dst_type = etype
        if src_type not in h or dst_type not in h:
            return None
        src_nodes, dst_nodes = graph.edges(etype=etype)
        if src_nodes.numel() == 0:
            return None
        key = _etype_key(etype)
        src_feat = h[src_type][src_nodes]
        dst_feat = h[dst_type][dst_nodes]
        rel_feat = self.relation_embeddings[key].weight[0]
        score = (src_feat * rel_feat * dst_feat).sum(dim=-1) / math.sqrt(self.rgcn_out)
        return score + self.relation_bias[key]

    def forward(self, g, input_features, pos_graph, neg_graph, edge_weights=None, target_etypes_override=None):
        h = self.get_embeddings(g, input_features, edge_weights=edge_weights)
        target_etypes = target_etypes_override if target_etypes_override else self.target_etypes
        self._ensure_relations(target_etypes)

        pos_scores = {}
        neg_scores = {}
        for etype in target_etypes:
            if etype in pos_graph.canonical_etypes:
                score = self._score_edges(h, pos_graph, etype)
                if score is not None:
                    pos_scores[etype] = score.view(-1)
            if etype in neg_graph.canonical_etypes:
                score = self._score_edges(h, neg_graph, etype)
                if score is not None:
                    neg_scores[etype] = score.view(-1)
        return pos_scores, neg_scores

    def predict_links(self, g, h_initial_for_g, source_ntype, source_dgl_ids, target_nodes_dict_with_ids):
        self.eval()
        predictions = {}
        with torch.no_grad():
            h = self.get_embeddings(g, h_initial_for_g)
            source_feats = h[source_ntype][source_dgl_ids]
            for target_ntype, target_info in target_nodes_dict_with_ids.items():
                target_dgl_ids = target_info["dgl_ids"]
                target_feats = h[target_ntype][target_dgl_ids]
                scores = torch.mm(source_feats, target_feats.t())
                for i, _ in enumerate(source_dgl_ids):
                    predictions[i] = {
                        "scores": scores[i].detach().cpu().numpy(),
                        "target_ids": target_info["original_ids"],
                    }
        return predictions
