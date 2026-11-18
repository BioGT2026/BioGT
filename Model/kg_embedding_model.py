"""KG embedding baselines for disease-biomarker link prediction."""

import math
import re

import torch
import torch.nn as nn


def _etype_key(etype):
    return re.sub(r"[^0-9A-Za-z_]+", "_", "__".join(map(str, etype)))


class KGEmbeddingModel(nn.Module):
    """DistMult-style knowledge graph embedding baseline."""

    def __init__(
        self,
        node_feature_embeddings,
        embedding_dim,
        ntypes,
        etypes,
        target_node_types,
        num_nodes_dict,
        node_info=None,
        dropout=0.1,
        freeze_pretrained=False,
        score_function="distmult",
        rotate_margin=12.0,
    ):
        super().__init__()
        self.disable_hgcn = True
        self.disable_transformer = True
        self.disable_moe = True
        self.score_function = score_function
        self.fusion_mode = score_function
        self.enable_explainability = False
        self.node_info = node_info
        self.embedding_dim = int(embedding_dim)
        self.entity_dim = self.embedding_dim
        if self.score_function in {"complex", "rotate"} and self.entity_dim % 2 != 0:
            self.entity_dim += 1
        self.complex_rank = max(1, self.entity_dim // 2)
        self.relation_dim = self.complex_rank if self.score_function == "rotate" else self.entity_dim
        self.rotate_margin = float(rotate_margin)
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }
        self.dropout = nn.Dropout(dropout)

        self.node_embeddings = nn.ModuleDict()
        for ntype in self.ntypes:
            num_nodes = int(num_nodes_dict.get(ntype, 0))
            emb = nn.Embedding(max(num_nodes, 1), self.entity_dim)
            nn.init.xavier_uniform_(emb.weight)

            init_value = None
            if node_feature_embeddings and ntype in node_feature_embeddings:
                raw = node_feature_embeddings[ntype]
                if isinstance(raw, nn.Embedding):
                    init_value = raw.weight.detach()
                elif isinstance(raw, torch.Tensor):
                    init_value = raw.detach()

            if init_value is not None and init_value.numel() > 0 and init_value.shape[0] == num_nodes:
                if init_value.shape[1] == self.entity_dim:
                    emb.weight.data[:num_nodes].copy_(init_value.float())
                else:
                    projection = nn.Linear(init_value.shape[1], self.entity_dim)
                    nn.init.xavier_uniform_(projection.weight)
                    with torch.no_grad():
                        emb.weight.data[:num_nodes].copy_(projection(init_value.float()))

            emb.weight.requires_grad = not freeze_pretrained
            self.node_embeddings[ntype] = emb

        self.relation_embeddings = nn.ModuleDict()
        self.relation_bias = nn.ParameterDict()
        self._ensure_relations(self.etypes)

    def _ensure_relations(self, etypes):
        for etype in etypes:
            key = _etype_key(etype)
            if key not in self.relation_embeddings:
                rel_emb = nn.Embedding(1, self.relation_dim)
                if self.score_function == "rotate":
                    nn.init.uniform_(rel_emb.weight, -math.pi, math.pi)
                else:
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
            if ntype not in self.node_embeddings:
                continue
            node_ids = torch.arange(g.num_nodes(ntype), device=g.device)
            h[ntype] = self.dropout(self.node_embeddings[ntype](node_ids))
        if return_branch_outputs:
            return h, {"kg_embedding": h}
        return h

    def _score_edges(self, h, graph, etype):
        src_type, _, dst_type = etype
        if src_type not in h or dst_type not in h:
            return None
        src_nodes, dst_nodes = graph.edges(etype=etype)
        if src_nodes.numel() == 0:
            return None
        rel_key = _etype_key(etype)
        src_feat = h[src_type][src_nodes]
        dst_feat = h[dst_type][dst_nodes]
        rel_feat = self.relation_embeddings[rel_key].weight[0]

        if self.score_function == "complex":
            src_real, src_imag = torch.chunk(src_feat, 2, dim=-1)
            dst_real, dst_imag = torch.chunk(dst_feat, 2, dim=-1)
            rel_real, rel_imag = torch.chunk(rel_feat, 2, dim=-1)
            score = (
                src_real * rel_real * dst_real
                + src_imag * rel_real * dst_imag
                + src_real * rel_imag * dst_imag
                - src_imag * rel_imag * dst_real
            ).sum(dim=-1) / math.sqrt(self.complex_rank)
        elif self.score_function == "rotate":
            src_real, src_imag = torch.chunk(src_feat, 2, dim=-1)
            dst_real, dst_imag = torch.chunk(dst_feat, 2, dim=-1)
            rel_real = torch.cos(rel_feat)
            rel_imag = torch.sin(rel_feat)
            rot_real = src_real * rel_real - src_imag * rel_imag
            rot_imag = src_real * rel_imag + src_imag * rel_real
            distance = torch.stack(
                [rot_real - dst_real, rot_imag - dst_imag],
                dim=0,
            ).norm(dim=0).sum(dim=-1)
            score = self.rotate_margin - distance
        else:
            score = (src_feat * rel_feat * dst_feat).sum(dim=-1) / math.sqrt(self.entity_dim)
        return score + self.relation_bias[rel_key]

    def forward(self, g, input_features, pos_graph, neg_graph, edge_weights=None, target_etypes_override=None):
        h = self.get_embeddings(g, input_features, edge_weights)
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


class ComplExModel(KGEmbeddingModel):
    """ComplEx knowledge graph embedding baseline."""

    def __init__(self, *args, **kwargs):
        kwargs["score_function"] = "complex"
        super().__init__(*args, **kwargs)


class RotatEModel(KGEmbeddingModel):
    """RotatE knowledge graph embedding baseline."""

    def __init__(self, *args, **kwargs):
        kwargs["score_function"] = "rotate"
        super().__init__(*args, **kwargs)
