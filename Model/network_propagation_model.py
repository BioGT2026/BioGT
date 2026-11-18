"""Network propagation baseline for disease-biomarker link prediction."""

from collections import defaultdict

import torch
import torch.nn as nn


class NetworkPropagationModel(nn.Module):
    """Non-parametric random-walk-with-restart style baseline.

    Scores are computed from the message-passing graph only. Validation/test
    positive edges are supplied separately by the training script, so held-out
    target labels do not leak into the propagation graph.
    """

    def __init__(
        self,
        ntypes,
        etypes,
        target_node_types,
        node_info=None,
        restart_prob=0.5,
        max_steps=3,
        dropout=0.0,
    ):
        super().__init__()
        self.disable_hgcn = True
        self.disable_transformer = True
        self.disable_moe = True
        self.fusion_mode = "network_propagation"
        self.enable_explainability = False
        self.node_info = node_info
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }
        self.restart_prob = float(restart_prob)
        self.max_steps = int(max_steps)
        self.node_embeddings = nn.ModuleDict()

        # Kept as calibration parameters for API compatibility if users decide
        # to fine-tune this baseline, but train_test_single skips training by default.
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))
        self._adjacency_cache = {}
        self._propagation_cache = {}

    def set_target_etypes(self, target_etypes):
        if target_etypes:
            self.target_etypes = set(target_etypes)

    def get_embeddings(self, g, initial_features=None, edge_weights=None, return_branch_outputs=False):
        if return_branch_outputs:
            return {}, {"network_propagation": {}}
        return {}

    def _graph_cache_key(self, g):
        edge_counts = tuple((etype, int(g.num_edges(etype))) for etype in g.canonical_etypes)
        node_counts = tuple((ntype, int(g.num_nodes(ntype))) for ntype in g.ntypes)
        return id(g), node_counts, edge_counts

    def _build_adjacency(self, g):
        cache_key = self._graph_cache_key(g)
        if cache_key in self._adjacency_cache:
            return self._adjacency_cache[cache_key]

        adjacency = defaultdict(list)
        for etype in g.canonical_etypes:
            src_type, _, dst_type = etype
            src_nodes, dst_nodes = g.edges(etype=etype)
            if src_nodes.numel() == 0:
                continue
            for src_id, dst_id in zip(
                src_nodes.detach().cpu().tolist(),
                dst_nodes.detach().cpu().tolist(),
            ):
                src_key = (src_type, int(src_id))
                dst_key = (dst_type, int(dst_id))
                adjacency[src_key].append(dst_key)
                adjacency[dst_key].append(src_key)

        self._adjacency_cache[cache_key] = adjacency
        self._propagation_cache = {
            key: value
            for key, value in self._propagation_cache.items()
            if key[0] == cache_key
        }
        return adjacency

    def _propagate_from_source(self, g, source_key):
        graph_key = self._graph_cache_key(g)
        cache_key = (graph_key, source_key, self.restart_prob, self.max_steps)
        if cache_key in self._propagation_cache:
            return self._propagation_cache[cache_key]

        adjacency = self._build_adjacency(g)
        restart = min(max(self.restart_prob, 0.0), 1.0)
        continue_prob = 1.0 - restart
        scores = defaultdict(float)
        frontier = {source_key: 1.0}
        scores[source_key] = restart

        for _ in range(max(0, self.max_steps)):
            next_frontier = defaultdict(float)
            for node_key, mass in frontier.items():
                neighbors = adjacency.get(node_key, [])
                if not neighbors:
                    continue
                share = continue_prob * mass / float(len(neighbors))
                for neighbor in neighbors:
                    next_frontier[neighbor] += share

            if not next_frontier:
                break
            for node_key, mass in next_frontier.items():
                scores[node_key] += restart * mass
            frontier = dict(next_frontier)

        self._propagation_cache[cache_key] = scores
        return scores

    def _score_edges(self, g, edge_graph, etype):
        src_type, _, dst_type = etype
        src_nodes, dst_nodes = edge_graph.edges(etype=etype)
        if src_nodes.numel() == 0:
            return None

        src_list = src_nodes.detach().cpu().tolist()
        dst_list = dst_nodes.detach().cpu().tolist()
        values = []
        for src_id, dst_id in zip(src_list, dst_list):
            source_key = (src_type, int(src_id))
            target_key = (dst_type, int(dst_id))
            scores = self._propagate_from_source(g, source_key)
            values.append(float(scores.get(target_key, 0.0)))

        probs = torch.tensor(values, dtype=torch.float32, device=src_nodes.device)
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
        logits = torch.log(probs / (1.0 - probs))
        return logits * self.score_scale.to(logits.device) + self.score_bias.to(logits.device)

    def forward(self, g, input_features, pos_graph, neg_graph, edge_weights=None, target_etypes_override=None):
        target_etypes = target_etypes_override if target_etypes_override else self.target_etypes
        pos_scores = {}
        neg_scores = {}
        for etype in target_etypes:
            if etype in pos_graph.canonical_etypes:
                score = self._score_edges(g, pos_graph, etype)
                if score is not None:
                    pos_scores[etype] = score.view(-1)
            if etype in neg_graph.canonical_etypes:
                score = self._score_edges(g, neg_graph, etype)
                if score is not None:
                    neg_scores[etype] = score.view(-1)
        return pos_scores, neg_scores
