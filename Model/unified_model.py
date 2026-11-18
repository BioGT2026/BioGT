import logging
import math
from collections import OrderedDict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv


QUERY_HOP_KEY = "_query_hop"
QUERY_SEED_KEY = "_query_seed"
EVIDENCE_SCORE_KEY = "_evidence_score"


def parse_modality_fusion_modes(value, target_types=None, fallback="weighted"):
    if isinstance(value, dict):
        parsed = {str(key).strip(): str(mode).strip().lower() for key, mode in value.items()}
    else:
        parsed = {}
        for entry in str(value or "").split(","):
            if ":" not in entry:
                continue
            node_type, mode = entry.split(":", 1)
            node_type = node_type.strip()
            mode = mode.strip().lower()
            if node_type:
                parsed[node_type] = mode

    valid_modes = {"weighted", "concat", "hierarchical", "sequential"}
    result = {}
    for target_type in target_types or parsed.keys():
        mode = parsed.get(target_type, fallback)
        if mode not in valid_modes:
            logging.warning(
                "Unsupported fusion mode '%s' for modality '%s'; using '%s'.",
                mode,
                target_type,
                fallback,
            )
            mode = fallback
        result[target_type] = mode
    return result


class QueryConditionedLowRankRelationAdapter(nn.Module):
    """Zero-output low-rank relation delta conditioned on the query state."""

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.message_down = nn.Linear(hidden_dim, rank, bias=False)
        self.query_gate = nn.Linear(hidden_dim, rank, bias=False)
        self.output_up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.message_down.weight)
        nn.init.xavier_uniform_(self.query_gate.weight)
        nn.init.zeros_(self.output_up.weight)

    def forward(self, messages, condition):
        gate = torch.tanh(self.query_gate(condition)).unsqueeze(1)
        return self.output_up(self.message_down(messages) * gate)


class QueryConditionedIndicatorBoundaryAdapter(nn.Module):
    """Zero-output query delta for an indicator source boundary."""

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.condition_down = nn.Linear(hidden_dim, rank, bias=False)
        self.output_up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.condition_down.weight)
        nn.init.zeros_(self.output_up.weight)

    def forward(self, condition):
        frozen_condition = condition.detach()
        return self.output_up(torch.tanh(self.condition_down(frozen_condition)))


class QueryConditionedIndicatorMultiRuleResidual(nn.Module):
    """Query-gated low-rank readouts for independent indicator rules."""

    def __init__(self, hidden_dim, rule_count, readout_rank):
        super().__init__()
        self.rule_count = int(rule_count)
        self.rule_tokens = nn.Parameter(
            torch.empty(self.rule_count, hidden_dim)
        )
        nn.init.normal_(
            self.rule_tokens,
            mean=0.0,
            std=hidden_dim ** -0.5,
        )
        self.rule_gate = nn.Linear(hidden_dim, self.rule_count, bias=True)
        nn.init.xavier_uniform_(self.rule_gate.weight)
        nn.init.zeros_(self.rule_gate.bias)
        self.rule_readouts = nn.ModuleList()
        for _ in range(self.rule_count):
            readout = nn.Sequential(
                nn.Linear(3 * hidden_dim, readout_rank, bias=False),
                nn.GELU(),
                nn.Linear(readout_rank, 1, bias=False),
            )
            nn.init.xavier_uniform_(readout[0].weight)
            nn.init.zeros_(readout[-1].weight)
            self.rule_readouts.append(readout)

    def gate_weights(self, condition):
        return torch.softmax(self.rule_gate(condition.detach()), dim=-1)


class QueryConditionedRelationTransitionResidual(nn.Module):
    """Zero-init relation weights propagated over a train-only relation graph."""

    def __init__(self, hidden_dim, relation_count, rank, layers):
        super().__init__()
        self.relation_count = int(relation_count)
        self.layers = int(layers)
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.layers)
        ])
        self.state_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(self.layers)
        ])
        self.readout_down = nn.Linear(hidden_dim, rank, bias=False)
        self.readout_up = nn.Linear(rank, 1, bias=False)
        for layer in self.message_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.readout_down.weight)
        nn.init.zeros_(self.readout_up.weight)

    def forward(self, condition, adjacency, boundary_index):
        batch_size, hidden_dim = condition.shape
        states = condition.new_zeros(
            (batch_size, self.relation_count, hidden_dim)
        )
        states[:, int(boundary_index)] = condition.detach()
        transition = adjacency.transpose(0, 1).coalesce()
        for message_layer, state_norm in zip(
            self.message_layers,
            self.state_norms,
        ):
            flat_states = states.permute(1, 0, 2).reshape(
                self.relation_count,
                batch_size * hidden_dim,
            )
            propagated = torch.sparse.mm(
                transition,
                flat_states,
            ).reshape(
                self.relation_count,
                batch_size,
                hidden_dim,
            ).permute(1, 0, 2)
            states = F.gelu(state_norm(states + message_layer(propagated)))
        deltas = self.readout_up(
            torch.tanh(self.readout_down(states))
        ).squeeze(-1)
        return 1.0 + torch.tanh(deltas)


class QueryLocalRelationTransitionResidual(nn.Module):
    """Zero-init train-only relation propagation with query-local incidence."""

    def __init__(self, hidden_dim, relation_count, rank, layers):
        super().__init__()
        self.relation_count = int(relation_count)
        self.layers = int(layers)
        self.local_gate = nn.Linear(1, hidden_dim, bias=False)
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.layers)
        ])
        self.state_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(self.layers)
        ])
        self.readout_down = nn.Linear(hidden_dim, rank, bias=False)
        self.readout_up = nn.Linear(rank, 1, bias=False)
        nn.init.xavier_uniform_(self.local_gate.weight)
        for layer in self.message_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.readout_down.weight)
        nn.init.zeros_(self.readout_up.weight)

    def forward(self, condition, adjacency, boundary_index, local_incidence):
        if local_incidence.shape != (condition.shape[0], self.relation_count):
            raise ValueError(
                "Query-local relation incidence must align with batch and "
                "relation count"
            )
        stable_incidence = torch.log1p(local_incidence.clamp_min(0.0))
        local_mean = stable_incidence.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized_incidence = stable_incidence / local_mean
        local_weights = 2.0 * torch.sigmoid(
            self.local_gate(normalized_incidence.unsqueeze(-1))
        )

        batch_size, hidden_dim = condition.shape
        states = condition.new_zeros(
            (batch_size, self.relation_count, hidden_dim)
        )
        states[:, int(boundary_index)] = condition.detach()
        transition = adjacency.transpose(0, 1).coalesce()
        for message_layer, state_norm in zip(
            self.message_layers,
            self.state_norms,
        ):
            gated_states = states * local_weights
            flat_states = gated_states.permute(1, 0, 2).reshape(
                self.relation_count,
                batch_size * hidden_dim,
            )
            propagated = torch.sparse.mm(
                transition,
                flat_states,
            ).reshape(
                self.relation_count,
                batch_size,
                hidden_dim,
            ).permute(1, 0, 2)
            states = F.gelu(
                state_norm(states + message_layer(propagated) * local_weights)
            )
        deltas = self.readout_up(
            torch.tanh(self.readout_down(states))
        ).squeeze(-1)
        return 1.0 + torch.tanh(deltas), normalized_incidence, local_weights


class CandidateConditionedPathLayerAttentionResidual(nn.Module):
    """Candidate-specific attention over frozen RelBF path-layer states."""

    layer_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.query_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.candidate_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.attention_readout = nn.Linear(rank, 1, bias=False)
        self.evidence_down = nn.Linear(4 * hidden_dim, rank, bias=False)
        self.evidence_up = nn.Linear(rank, 1, bias=False)
        for layer in (
            self.token_projection,
            self.query_projection,
            self.candidate_projection,
            self.attention_readout,
            self.evidence_down,
        ):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.evidence_up.weight)

    def forward(self, layer_states, condition, candidate_state):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.layer_labels
        ):
            raise ValueError(
                "RelBF path-layer states must have shape [query, candidate, "
                "3, hidden]"
            )
        expected_prefix = (condition.shape[0], candidate_state.shape[0])
        if layer_states.shape[:2] != expected_prefix:
            raise ValueError(
                "RelBF path-layer states must align with query and candidate "
                "counts"
            )

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        attention_hidden = torch.tanh(
            self.token_projection(frozen_layers)
            + self.query_projection(frozen_condition)[:, None, None, :]
            + self.candidate_projection(frozen_candidates)[None, :, None, :]
        )
        attention = torch.softmax(
            self.attention_readout(attention_hidden).squeeze(-1),
            dim=-1,
        )
        evidence = (
            frozen_layers * attention.unsqueeze(-1)
        ).sum(dim=2)
        candidate_matrix = frozen_candidates.unsqueeze(0)
        query_matrix = frozen_condition.unsqueeze(1)
        readout_input = torch.cat([
            evidence,
            evidence * candidate_matrix,
            evidence * query_matrix,
            candidate_matrix * query_matrix,
        ], dim=-1)
        residual = self.evidence_up(
            F.gelu(self.evidence_down(readout_input))
        ).squeeze(-1)
        return residual, attention


class CandidateConditionedPathTokenTransformerResidual(nn.Module):
    """One-layer candidate-specific Transformer over BioGT path tokens."""

    token_labels = ("query", "candidate", "shallow", "middle", "deep")
    path_labels = token_labels[2:]
    attention_heads = 2

    def __init__(self, hidden_dim):
        super().__init__()
        if hidden_dim % self.attention_heads:
            raise ValueError(
                "BioGT path-token Transformer hidden dimension must be "
                f"divisible by {self.attention_heads}"
            )
        self.token_type_embeddings = nn.Parameter(
            torch.empty(len(self.token_labels), hidden_dim)
        )
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, self.attention_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.normal_(self.token_type_embeddings, std=0.02)
        nn.init.zeros_(self.score_readout.weight)

    def forward(self, layer_states, condition, candidate_state):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "RelBF path-token states must have shape [query, candidate, "
                "3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Path-token query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Path-token candidate states do not align")
        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        query_tokens = frozen_condition[:, None, :].expand(
            query_count, candidate_count, -1
        )
        candidate_tokens = frozen_candidates[None, :, :].expand(
            query_count, candidate_count, -1
        )
        tokens = torch.cat([
            query_tokens.unsqueeze(2),
            candidate_tokens.unsqueeze(2),
            frozen_layers,
        ], dim=2)
        flat_tokens = tokens.reshape(
            query_count * candidate_count,
            len(self.token_labels),
            tokens.shape[-1],
        )
        flat_tokens = flat_tokens + self.token_type_embeddings.unsqueeze(0)
        attended, attention = self.self_attention(
            flat_tokens,
            flat_tokens,
            flat_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = self.attention_norm(flat_tokens + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        residual = self.score_readout(hidden[:, 1, :]).reshape(
            query_count, candidate_count
        )
        path_attention = attention[:, :, 1, 2:].mean(dim=1)
        path_attention = path_attention / (
            path_attention.sum(dim=-1, keepdim=True) + 1e-12
        )
        return residual, path_attention.reshape(
            query_count, candidate_count, len(self.path_labels)
        )


class CandidateCompetitionBoundaryResidual(nn.Module):
    """Linear-in-candidates competition against fixed parent-score anchors."""

    anchor_count = 32
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim):
        super().__init__()
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.boundary_projection = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "Candidate competition path states must have shape "
                "[query, candidate, 3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Candidate competition query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Candidate competition candidate states do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Candidate competition parent scores do not align")
        if candidate_count == 0:
            raise ValueError("Candidate competition requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_parent_scores = parent_scores.detach()
        candidate_matrix = frozen_candidates[None, :, :].expand(
            query_count, candidate_count, -1
        )
        token_input = torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        )
        tokens = self.candidate_token(token_input)

        ranking_scores = frozen_parent_scores
        if candidate_valid is not None:
            if candidate_valid.shape != (candidate_count,):
                raise ValueError("Candidate competition validity mask does not align")
            ranking_scores = ranking_scores.masked_fill(
                ~candidate_valid.detach().bool().unsqueeze(0),
                torch.finfo(ranking_scores.dtype).min,
            )
        anchor_count = min(self.anchor_count, candidate_count)
        anchor_indices = torch.topk(
            ranking_scores, anchor_count, dim=1, sorted=True
        ).indices
        gather_index = anchor_indices.unsqueeze(-1).expand(
            -1, -1, tokens.shape[-1]
        )
        anchor_tokens = torch.gather(tokens, 1, gather_index)

        queries = self.query_projection(
            tokens + frozen_condition.unsqueeze(1)
        )
        keys = self.key_projection(anchor_tokens)
        attention = torch.softmax(
            torch.einsum("qch,qkh->qck", queries, keys)
            / math.sqrt(tokens.shape[-1]),
            dim=-1,
        )
        context = torch.einsum(
            "qck,qkh->qch",
            attention,
            self.value_projection(anchor_tokens),
        )
        boundary_hidden = self.boundary_projection(torch.cat([
            tokens,
            context,
            tokens - context,
            frozen_condition.unsqueeze(1).expand_as(tokens),
        ], dim=-1))
        residual = self.score_readout(boundary_hidden).squeeze(-1)
        return residual, anchor_indices, attention


class ParentTopBoundarySlotComparatorResidual(nn.Module):
    """Compare every candidate with the frozen parent rank-10/11 slots."""

    boundary_ranks = (10, 11)
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim):
        super().__init__()
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gap_projection = nn.Sequential(
            nn.Linear(len(self.boundary_ranks), hidden_dim),
            nn.Tanh(),
        )
        self.comparator = nn.Sequential(
            nn.Linear(7 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "Boundary-slot path states must have shape "
                "[query, candidate, 3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Boundary-slot query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Boundary-slot candidate states do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Boundary-slot parent scores do not align")
        if candidate_count == 0:
            raise ValueError("Boundary-slot comparator requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_parent_scores = parent_scores.detach()
        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        candidate_tokens = self.candidate_token(torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        ))

        ranking_scores = frozen_parent_scores
        valid_count = candidate_count
        if candidate_valid is not None:
            if candidate_valid.shape != (candidate_count,):
                raise ValueError("Boundary-slot validity mask does not align")
            frozen_valid = candidate_valid.detach().bool()
            valid_count = int(frozen_valid.sum().item())
            if valid_count == 0:
                raise ValueError("Boundary-slot comparator requires a valid candidate")
            ranking_scores = ranking_scores.masked_fill(
                ~frozen_valid.unsqueeze(0),
                torch.finfo(ranking_scores.dtype).min,
            )

        selected_count = min(max(self.boundary_ranks), candidate_count)
        sorted_indices = torch.topk(
            ranking_scores, selected_count, dim=1, sorted=True
        ).indices
        slot_positions = torch.tensor(
            [min(rank - 1, valid_count - 1) for rank in self.boundary_ranks],
            device=sorted_indices.device,
            dtype=torch.long,
        )
        boundary_indices = sorted_indices.index_select(1, slot_positions)
        token_index = boundary_indices.unsqueeze(-1).expand(
            -1, -1, candidate_tokens.shape[-1]
        )
        boundary_tokens = torch.gather(candidate_tokens, 1, token_index)
        score_slots = torch.gather(frozen_parent_scores, 1, boundary_indices)
        score_gaps = frozen_parent_scores.unsqueeze(-1) - score_slots.unsqueeze(1)

        rank10_token = boundary_tokens[:, 0, :].unsqueeze(1).expand_as(
            candidate_tokens
        )
        rank11_token = boundary_tokens[:, 1, :].unsqueeze(1).expand_as(
            candidate_tokens
        )
        query_token = frozen_condition.unsqueeze(1).expand_as(candidate_tokens)
        gap_token = self.gap_projection(score_gaps)
        hidden = self.comparator(torch.cat([
            candidate_tokens,
            rank10_token,
            rank11_token,
            candidate_tokens - rank10_token,
            candidate_tokens - rank11_token,
            query_token,
            gap_token,
        ], dim=-1))
        residual = self.score_readout(hidden).squeeze(-1)
        return residual, boundary_indices, score_slots, score_gaps


class BoundaryWindowRankSlotExpertResidual(nn.Module):
    """Refine only frozen-parent rank slots 5 through 20."""

    window_start_rank = 5
    window_end_rank = 20
    boundary_ranks = (10, 11)
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim):
        super().__init__()
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rank_position_embedding = nn.Embedding(
            self.window_end_rank, hidden_dim
        )
        self.gap_projection = nn.Sequential(
            nn.Linear(len(self.boundary_ranks), hidden_dim),
            nn.Tanh(),
        )
        self.expert = nn.Sequential(
            nn.Linear(8 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "Boundary-window path states must have shape "
                "[query, candidate, 3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Boundary-window query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Boundary-window candidate states do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Boundary-window parent scores do not align")
        if candidate_count == 0:
            raise ValueError("Boundary-window expert requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_parent_scores = parent_scores.detach()
        ranking_scores = frozen_parent_scores
        valid_count = candidate_count
        if candidate_valid is not None:
            if candidate_valid.shape != (candidate_count,):
                raise ValueError("Boundary-window validity mask does not align")
            frozen_valid = candidate_valid.detach().bool()
            valid_count = int(frozen_valid.sum().item())
            if valid_count == 0:
                raise ValueError("Boundary-window expert requires a valid candidate")
            ranking_scores = ranking_scores.masked_fill(
                ~frozen_valid.unsqueeze(0),
                torch.finfo(ranking_scores.dtype).min,
            )

        selected_count = min(self.window_end_rank, valid_count)
        sorted_indices = torch.topk(
            ranking_scores, selected_count, dim=1, sorted=True
        ).indices
        window_start = self.window_start_rank - 1
        window_indices = sorted_indices[:, window_start:selected_count]
        rank_positions = torch.arange(
            window_start + 1,
            max(window_start + 1, selected_count + 1),
            device=sorted_indices.device,
            dtype=torch.long,
        )

        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        candidate_tokens = self.candidate_token(torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        ))
        window_token_index = window_indices.unsqueeze(-1).expand(
            -1, -1, candidate_tokens.shape[-1]
        )
        window_tokens = torch.gather(
            candidate_tokens, 1, window_token_index
        )

        boundary_positions = torch.tensor(
            [min(rank, valid_count) - 1 for rank in self.boundary_ranks],
            device=sorted_indices.device,
            dtype=torch.long,
        )
        boundary_indices = sorted_indices.index_select(1, boundary_positions)
        boundary_token_index = boundary_indices.unsqueeze(-1).expand(
            -1, -1, candidate_tokens.shape[-1]
        )
        boundary_tokens = torch.gather(
            candidate_tokens, 1, boundary_token_index
        )
        boundary_scores = torch.gather(
            frozen_parent_scores, 1, boundary_indices
        )
        window_scores = torch.gather(
            frozen_parent_scores, 1, window_indices
        )
        score_gaps = window_scores.unsqueeze(-1) - boundary_scores.unsqueeze(1)

        rank10_token = boundary_tokens[:, 0, :].unsqueeze(1).expand_as(
            window_tokens
        )
        rank11_token = boundary_tokens[:, 1, :].unsqueeze(1).expand_as(
            window_tokens
        )
        query_token = frozen_condition.unsqueeze(1).expand_as(window_tokens)
        position_token = self.rank_position_embedding(
            rank_positions - 1
        ).unsqueeze(0).expand_as(window_tokens)
        gap_token = self.gap_projection(score_gaps)
        hidden = self.expert(torch.cat([
            window_tokens,
            position_token,
            rank10_token,
            rank11_token,
            window_tokens - rank10_token,
            window_tokens - rank11_token,
            query_token,
            gap_token,
        ], dim=-1))
        window_residual = self.score_readout(hidden).squeeze(-1)
        residual = parent_scores.new_zeros(query_count, candidate_count)
        residual = residual.scatter(1, window_indices, window_residual)
        window_mask = torch.zeros_like(residual, dtype=torch.bool)
        window_mask = window_mask.scatter(
            1, window_indices, torch.ones_like(window_indices, dtype=torch.bool)
        )
        return (
            residual,
            window_indices,
            rank_positions,
            boundary_indices,
            boundary_scores,
            score_gaps,
            window_mask,
        )


class BoundaryExchangeResidual(nn.Module):
    """Exchange evidence between frozen-parent top-10 boundary groups."""

    window_start_rank = 5
    window_end_rank = 20
    incumbent_ranks = tuple(range(5, 11))
    challenger_ranks = tuple(range(11, 21))
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim, candidate_dim=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_dim = int(candidate_dim or hidden_dim)
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rank_position_embedding = nn.Embedding(
            self.window_end_rank, hidden_dim
        )
        self.score_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.exchange = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    def _cross_attention(self, queries, sources):
        projected_queries = self.query_projection(queries)
        projected_keys = self.key_projection(sources)
        attention = torch.softmax(
            torch.einsum("qih,qjh->qij", projected_queries, projected_keys)
            / math.sqrt(self.hidden_dim),
            dim=-1,
        )
        context = torch.einsum(
            "qij,qjh->qih", attention, self.value_projection(sources)
        )
        return context, attention

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "Boundary-exchange path states must have shape "
                "[query, candidate, 3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Boundary-exchange query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Boundary-exchange candidate states do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Boundary-exchange parent scores do not align")
        if candidate_count == 0:
            raise ValueError("Boundary-exchange expert requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_parent_scores = parent_scores.detach()
        ranking_scores = frozen_parent_scores
        valid_count = candidate_count
        if candidate_valid is not None:
            if candidate_valid.shape != (candidate_count,):
                raise ValueError("Boundary-exchange validity mask does not align")
            frozen_valid = candidate_valid.detach().bool()
            valid_count = int(frozen_valid.sum().item())
            if valid_count == 0:
                raise ValueError("Boundary-exchange expert requires a valid candidate")
            ranking_scores = ranking_scores.masked_fill(
                ~frozen_valid.unsqueeze(0),
                torch.finfo(ranking_scores.dtype).min,
            )

        residual = parent_scores.new_zeros(query_count, candidate_count)
        empty_indices = torch.empty(
            query_count, 0, dtype=torch.long, device=parent_scores.device
        )
        empty_attention = parent_scores.new_empty(query_count, 0, 0)
        empty_mask = torch.zeros_like(residual, dtype=torch.bool)
        if valid_count < min(self.challenger_ranks):
            return (
                residual,
                empty_indices,
                empty_indices,
                empty_attention,
                empty_attention,
                empty_mask,
            )

        selected_count = min(self.window_end_rank, valid_count)
        sorted_indices = torch.topk(
            ranking_scores, selected_count, dim=1, sorted=True
        ).indices
        incumbent_positions = torch.arange(
            self.window_start_rank - 1,
            min(10, selected_count),
            device=parent_scores.device,
        )
        challenger_positions = torch.arange(
            10, selected_count, device=parent_scores.device
        )
        incumbent_indices = sorted_indices.index_select(1, incumbent_positions)
        challenger_indices = sorted_indices.index_select(1, challenger_positions)
        window_indices = torch.cat(
            [incumbent_indices, challenger_indices], dim=1
        )
        rank_positions = torch.cat(
            [incumbent_positions, challenger_positions], dim=0
        )

        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        candidate_tokens = self.candidate_token(torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        ))
        gather_index = window_indices.unsqueeze(-1).expand(
            -1, -1, candidate_tokens.shape[-1]
        )
        window_tokens = torch.gather(candidate_tokens, 1, gather_index)
        window_scores = torch.gather(
            frozen_parent_scores, 1, window_indices
        )
        rank10_scores = window_scores[:, 5:6]
        rank11_scores = window_scores[:, 6:7]
        score_features = torch.stack([
            window_scores,
            window_scores - rank10_scores,
            window_scores - rank11_scores,
        ], dim=-1)
        position_tokens = self.rank_position_embedding(rank_positions).unsqueeze(0)
        position_tokens = position_tokens.expand_as(window_tokens)
        tokens = window_tokens + position_tokens + self.score_projection(
            score_features
        )

        incumbent_count = incumbent_indices.shape[1]
        incumbent_tokens = tokens[:, :incumbent_count]
        challenger_tokens = tokens[:, incumbent_count:]
        incumbent_context, incumbent_attention = self._cross_attention(
            incumbent_tokens, challenger_tokens
        )
        challenger_context, challenger_attention = self._cross_attention(
            challenger_tokens, incumbent_tokens
        )
        context = torch.cat(
            [incumbent_context, challenger_context], dim=1
        )
        query_tokens = frozen_condition.unsqueeze(1).expand_as(tokens)
        hidden = self.exchange(torch.cat([
            tokens,
            context,
            tokens - context,
            tokens * context,
            query_tokens,
        ], dim=-1))
        window_residual = self.score_readout(hidden).squeeze(-1)
        window_residual = window_residual - window_residual.mean(
            dim=1, keepdim=True
        )
        residual = residual.scatter(1, window_indices, window_residual)
        window_mask = empty_mask.scatter(
            1, window_indices, torch.ones_like(window_indices, dtype=torch.bool)
        )
        return (
            residual,
            incumbent_indices,
            challenger_indices,
            incumbent_attention,
            challenger_attention,
            window_mask,
        )


class ChallengerBoundaryPromotionResidual(nn.Module):
    """Promote frozen-parent ranks 11-20 against incumbents at ranks 5-10."""

    incumbent_ranks = tuple(range(5, 11))
    challenger_ranks = tuple(range(11, 21))
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rank_position_embedding = nn.Embedding(
            max(self.challenger_ranks), hidden_dim
        )
        self.score_gap_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.promotion = nn.Sequential(
            nn.Linear(6 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "Challenger-promotion path states must have shape "
                "[query, candidate, 3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Challenger-promotion query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Challenger-promotion candidates do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Challenger-promotion parent scores do not align")
        if candidate_count == 0:
            raise ValueError("Challenger-promotion expert requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_parent_scores = parent_scores.detach()
        ranking_scores = frozen_parent_scores
        valid_count = candidate_count
        if candidate_valid is not None:
            if candidate_valid.shape != (candidate_count,):
                raise ValueError("Challenger-promotion validity mask does not align")
            frozen_valid = candidate_valid.detach().bool()
            valid_count = int(frozen_valid.sum().item())
            if valid_count == 0:
                raise ValueError(
                    "Challenger-promotion expert requires a valid candidate"
                )
            ranking_scores = ranking_scores.masked_fill(
                ~frozen_valid.unsqueeze(0),
                torch.finfo(ranking_scores.dtype).min,
            )

        residual = parent_scores.new_zeros(query_count, candidate_count)
        empty_indices = torch.empty(
            query_count, 0, dtype=torch.long, device=parent_scores.device
        )
        empty_attention = parent_scores.new_empty(query_count, 0, 0)
        empty_mask = torch.zeros_like(residual, dtype=torch.bool)
        if valid_count < min(self.challenger_ranks):
            return residual, empty_indices, empty_indices, empty_attention, empty_mask

        selected_count = min(max(self.challenger_ranks), valid_count)
        sorted_indices = torch.topk(
            ranking_scores, selected_count, dim=1, sorted=True
        ).indices
        incumbent_positions = torch.arange(4, 10, device=parent_scores.device)
        challenger_positions = torch.arange(
            10, selected_count, device=parent_scores.device
        )
        incumbent_indices = sorted_indices.index_select(1, incumbent_positions)
        challenger_indices = sorted_indices.index_select(1, challenger_positions)

        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        candidate_tokens = self.candidate_token(torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        ))
        incumbent_tokens = torch.gather(
            candidate_tokens,
            1,
            incumbent_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )
        challenger_tokens = torch.gather(
            candidate_tokens,
            1,
            challenger_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )
        challenger_scores = torch.gather(
            frozen_parent_scores, 1, challenger_indices
        )
        rank10_scores = torch.gather(
            frozen_parent_scores, 1, sorted_indices[:, 9:10]
        )
        rank11_scores = torch.gather(
            frozen_parent_scores, 1, sorted_indices[:, 10:11]
        )
        score_gaps = torch.stack([
            challenger_scores - rank10_scores,
            challenger_scores - rank11_scores,
        ], dim=-1)
        rank_tokens = self.rank_position_embedding(challenger_positions).unsqueeze(0)
        rank_tokens = rank_tokens.expand_as(challenger_tokens)
        challenger_tokens = (
            challenger_tokens
            + rank_tokens
            + self.score_gap_projection(score_gaps)
        )

        attention = torch.softmax(
            torch.einsum(
                "qch,qih->qci",
                self.query_projection(challenger_tokens),
                self.key_projection(incumbent_tokens),
            ) / math.sqrt(self.hidden_dim),
            dim=-1,
        )
        context = torch.einsum(
            "qci,qih->qch", attention, self.value_projection(incumbent_tokens)
        )
        query_tokens = frozen_condition.unsqueeze(1).expand_as(challenger_tokens)
        hidden = self.promotion(torch.cat([
            challenger_tokens,
            context,
            challenger_tokens - context,
            challenger_tokens * context,
            query_tokens,
            self.score_gap_projection(score_gaps),
        ], dim=-1))
        challenger_residual = self.score_readout(hidden).squeeze(-1)
        residual = residual.scatter(1, challenger_indices, challenger_residual)
        challenger_mask = empty_mask.scatter(
            1,
            challenger_indices,
            torch.ones_like(challenger_indices, dtype=torch.bool),
        )
        return (
            residual,
            incumbent_indices,
            challenger_indices,
            attention,
            challenger_mask,
        )


class GatekeeperChallengerPairedMarginResidual(nn.Module):
    """Refine parent rank 10 and ranks 11-20 through fixed pair interactions."""

    gatekeeper_rank = 10
    challenger_ranks = tuple(range(11, 21))
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rank_position_embedding = nn.Embedding(
            max(self.challenger_ranks), hidden_dim
        )
        self.score_gap_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(7 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pair_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.gatekeeper_readout = nn.Linear(hidden_dim, 1, bias=False)
        self.challenger_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.gatekeeper_readout.weight)
        nn.init.zeros_(self.challenger_readout.weight)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != len(
            self.path_labels
        ):
            raise ValueError(
                "Paired-margin path states must have shape "
                "[query, candidate, 3, hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Paired-margin query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Paired-margin candidates do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Paired-margin parent scores do not align")
        if candidate_count == 0:
            raise ValueError("Paired-margin adapter requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_parent_scores = parent_scores.detach()
        ranking_scores = frozen_parent_scores
        valid_count = candidate_count
        if candidate_valid is not None:
            if candidate_valid.shape != (candidate_count,):
                raise ValueError("Paired-margin validity mask does not align")
            frozen_valid = candidate_valid.detach().bool()
            valid_count = int(frozen_valid.sum().item())
            if valid_count == 0:
                raise ValueError("Paired-margin adapter requires a valid candidate")
            ranking_scores = ranking_scores.masked_fill(
                ~frozen_valid.unsqueeze(0),
                torch.finfo(ranking_scores.dtype).min,
            )

        residual = parent_scores.new_zeros(query_count, candidate_count)
        empty_indices = torch.empty(
            query_count, 0, dtype=torch.long, device=parent_scores.device
        )
        empty_pairs = parent_scores.new_empty(query_count, 0, self.hidden_dim)
        empty_weights = parent_scores.new_empty(query_count, 0)
        empty_mask = torch.zeros_like(residual, dtype=torch.bool)
        if valid_count < min(self.challenger_ranks):
            return (
                residual,
                empty_indices,
                empty_indices,
                empty_pairs,
                empty_weights,
                empty_mask,
            )

        selected_count = min(max(self.challenger_ranks), valid_count)
        sorted_indices = torch.topk(
            ranking_scores, selected_count, dim=1, sorted=True
        ).indices
        gatekeeper_indices = sorted_indices[:, self.gatekeeper_rank - 1:self.gatekeeper_rank]
        challenger_positions = torch.arange(
            self.gatekeeper_rank, selected_count, device=parent_scores.device
        )
        challenger_indices = sorted_indices.index_select(1, challenger_positions)

        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        candidate_tokens = self.candidate_token(torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        ))
        gatekeeper_tokens = torch.gather(
            candidate_tokens,
            1,
            gatekeeper_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        ).expand(-1, challenger_indices.shape[1], -1)
        challenger_tokens = torch.gather(
            candidate_tokens,
            1,
            challenger_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )
        gatekeeper_scores = torch.gather(
            frozen_parent_scores, 1, gatekeeper_indices
        ).expand_as(challenger_indices)
        challenger_scores = torch.gather(
            frozen_parent_scores, 1, challenger_indices
        )
        score_gaps = torch.stack([
            challenger_scores - gatekeeper_scores,
            gatekeeper_scores - challenger_scores,
        ], dim=-1)
        rank_tokens = self.rank_position_embedding(challenger_positions).unsqueeze(0)
        rank_tokens = rank_tokens.expand_as(challenger_tokens)
        query_tokens = frozen_condition.unsqueeze(1).expand_as(challenger_tokens)
        gap_tokens = self.score_gap_projection(score_gaps)
        pair_hidden = self.pair_encoder(torch.cat([
            gatekeeper_tokens,
            challenger_tokens,
            challenger_tokens - gatekeeper_tokens,
            challenger_tokens * gatekeeper_tokens,
            query_tokens,
            gap_tokens,
            rank_tokens,
        ], dim=-1))

        pair_weights = torch.softmax(
            self.pair_attention(pair_hidden).squeeze(-1), dim=1
        )
        gatekeeper_context = torch.einsum(
            "qp,qph->qh", pair_weights, pair_hidden
        )
        gatekeeper_residual = self.gatekeeper_readout(
            gatekeeper_context
        ).squeeze(-1).unsqueeze(1)
        challenger_residual = self.challenger_readout(pair_hidden).squeeze(-1)
        residual = residual.scatter(1, gatekeeper_indices, gatekeeper_residual)
        residual = residual.scatter(1, challenger_indices, challenger_residual)
        pair_indices = torch.cat([gatekeeper_indices, challenger_indices], dim=1)
        pair_mask = empty_mask.scatter(
            1, pair_indices, torch.ones_like(pair_indices, dtype=torch.bool)
        )
        return (
            residual,
            gatekeeper_indices,
            challenger_indices,
            pair_hidden,
            pair_weights,
            pair_mask,
        )


class BoundaryPrototypeOrdinalResidual(nn.Module):
    """Apply a shared ordinal boundary direction from two frozen prototypes."""

    incumbent_ranks = (8, 9, 10)
    challenger_ranks = (11, 12, 13)
    active_ranks = tuple(range(6, 17))
    path_labels = ("shallow", "middle", "deep")

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_token = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rank_embedding = nn.Embedding(max(self.active_ranks), hidden_dim)
        self.score_gap_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
        )
        self.prototype_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prototype_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prototype_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(8 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.ordinal_direction = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ordinal_threshold = nn.Linear(hidden_dim, 1, bias=False)
        self.score_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    def _prototype(self, tokens, condition):
        query = self.prototype_query(condition).unsqueeze(1)
        attention = torch.softmax(
            torch.einsum("qih,qjh->qij", query, self.prototype_key(tokens))
            / math.sqrt(self.hidden_dim),
            dim=-1,
        ).squeeze(1)
        prototype = torch.einsum(
            "qp,qph->qh", attention, self.prototype_value(tokens)
        )
        return prototype, attention

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != 3:
            raise ValueError("Ordinal adapter path states must be [query,candidate,3,hidden]")
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape[0] != query_count:
            raise ValueError("Ordinal adapter query states do not align")
        if candidate_state.shape[0] != candidate_count:
            raise ValueError("Ordinal adapter candidates do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Ordinal adapter parent scores do not align")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_scores = parent_scores.detach()
        ranking_scores = frozen_scores
        valid_count = candidate_count
        if candidate_valid is not None:
            frozen_valid = candidate_valid.detach().bool()
            if frozen_valid.shape != (candidate_count,):
                raise ValueError("Ordinal adapter validity mask does not align")
            valid_count = int(frozen_valid.sum().item())
            ranking_scores = ranking_scores.masked_fill(
                ~frozen_valid.unsqueeze(0), torch.finfo(ranking_scores.dtype).min
            )

        residual = parent_scores.new_zeros(query_count, candidate_count)
        empty_indices = torch.empty(
            query_count, 0, dtype=torch.long, device=parent_scores.device
        )
        empty_attention = parent_scores.new_empty(query_count, 0)
        empty_scores = parent_scores.new_empty(query_count, 0)
        empty_mask = torch.zeros_like(residual, dtype=torch.bool)
        if valid_count < max(self.challenger_ranks):
            return (
                residual, empty_indices, empty_indices, empty_indices,
                empty_attention, empty_attention, empty_scores, empty_mask,
            )

        selected_count = min(max(self.active_ranks), valid_count)
        sorted_indices = torch.topk(
            ranking_scores, selected_count, dim=1, sorted=True
        ).indices
        incumbent_positions = torch.tensor([7, 8, 9], device=parent_scores.device)
        challenger_positions = torch.tensor([10, 11, 12], device=parent_scores.device)
        active_positions = torch.arange(5, selected_count, device=parent_scores.device)
        incumbent_indices = sorted_indices.index_select(1, incumbent_positions)
        challenger_indices = sorted_indices.index_select(1, challenger_positions)
        active_indices = sorted_indices.index_select(1, active_positions)

        candidate_matrix = frozen_candidates.unsqueeze(0).expand(query_count, -1, -1)
        candidate_tokens = self.candidate_token(torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        ))
        gather = lambda indices: torch.gather(
            candidate_tokens, 1,
            indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )
        incumbent_tokens = gather(incumbent_indices)
        challenger_tokens = gather(challenger_indices)
        active_tokens = gather(active_indices)
        incumbent_prototype, incumbent_attention = self._prototype(
            incumbent_tokens, frozen_condition
        )
        challenger_prototype, challenger_attention = self._prototype(
            challenger_tokens, frozen_condition
        )

        active_scores = torch.gather(frozen_scores, 1, active_indices)
        rank10_scores = torch.gather(frozen_scores, 1, sorted_indices[:, 9:10])
        rank11_scores = torch.gather(frozen_scores, 1, sorted_indices[:, 10:11])
        gaps = torch.stack([
            active_scores - rank10_scores,
            active_scores - rank11_scores,
            rank10_scores.expand_as(active_scores) - rank11_scores.expand_as(active_scores),
        ], dim=-1)
        gap_tokens = self.score_gap_projection(gaps)
        rank_tokens = self.rank_embedding(active_positions).unsqueeze(0).expand_as(active_tokens)
        incumbent_context = incumbent_prototype.unsqueeze(1).expand_as(active_tokens)
        challenger_context = challenger_prototype.unsqueeze(1).expand_as(active_tokens)
        query_tokens = frozen_condition.unsqueeze(1).expand_as(active_tokens)
        hidden = self.ordinal_encoder(torch.cat([
            active_tokens,
            incumbent_context,
            challenger_context,
            challenger_context - incumbent_context,
            active_tokens - incumbent_context,
            active_tokens - challenger_context,
            query_tokens,
            gap_tokens + rank_tokens,
        ], dim=-1))
        direction = self.ordinal_direction(
            challenger_prototype - incumbent_prototype
        ).unsqueeze(1)
        threshold = self.ordinal_threshold(frozen_condition).unsqueeze(1)
        ordinal_scores = (hidden * direction).sum(dim=-1) / math.sqrt(self.hidden_dim)
        ordinal_scores = ordinal_scores - threshold.squeeze(-1)
        active_residual = self.score_readout(
            hidden * torch.tanh(ordinal_scores).unsqueeze(-1)
        ).squeeze(-1)
        residual = residual.scatter(1, active_indices, active_residual)
        active_mask = empty_mask.scatter(
            1, active_indices, torch.ones_like(active_indices, dtype=torch.bool)
        )
        return (
            residual, incumbent_indices, challenger_indices, active_indices,
            incumbent_attention, challenger_attention, ordinal_scores, active_mask,
        )


class QueryConditionedMultiDepthScoreRouterResidual(nn.Module):
    """Route four independent full-candidate score experts per query."""

    depth_labels = ("candidate", "shallow", "middle", "deep")

    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = min(int(rank), self.hidden_dim)
        self.query_expert_factors = nn.ModuleList([
            nn.Linear(hidden_dim, self.rank, bias=False)
            for _ in self.depth_labels
        ])
        self.state_expert_factors = nn.ModuleList([
            nn.Linear(hidden_dim, self.rank, bias=False)
            for _ in self.depth_labels
        ])
        self.candidate_expert_factors = nn.ModuleList([
            nn.Linear(hidden_dim, self.rank, bias=False)
            for _ in self.depth_labels
        ])
        self.query_gate = nn.Linear(hidden_dim, len(self.depth_labels))
        self.query_gate_factors = nn.ModuleList([
            nn.Linear(hidden_dim, self.rank, bias=False)
            for _ in self.depth_labels
        ])
        self.candidate_gate_factors = nn.ModuleList([
            nn.Linear(hidden_dim, self.rank, bias=False)
            for _ in self.depth_labels
        ])
        self.depth_gain = nn.Parameter(torch.zeros(len(self.depth_labels)))

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != 3:
            raise ValueError(
                "Multi-depth score router path states must be "
                "[query,candidate,3,hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape != (query_count, self.hidden_dim):
            raise ValueError("Multi-depth score router query states do not align")
        if candidate_state.shape != (candidate_count, self.hidden_dim):
            raise ValueError("Multi-depth score router candidates do not align")
        if candidate_count == 0:
            raise ValueError("Multi-depth score router requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        depth_states = (candidate_matrix, *frozen_layers.unbind(dim=2))

        expert_scores = []
        gate_modulations = []
        scale = math.sqrt(self.rank)
        for depth, state in enumerate(depth_states):
            query_factor = self.query_expert_factors[depth](
                frozen_condition
            ).unsqueeze(1)
            state_factor = self.state_expert_factors[depth](state)
            candidate_factor = self.candidate_expert_factors[depth](
                candidate_matrix
            )
            expert_scores.append(
                (query_factor * (state_factor + candidate_factor)).sum(-1)
                / scale
            )
            gate_modulations.append(
                (
                    self.query_gate_factors[depth](frozen_condition).unsqueeze(1)
                    * self.candidate_gate_factors[depth](candidate_matrix)
                ).sum(-1) / scale
            )

        expert_scores = torch.stack(expert_scores, dim=-1)
        gate_modulations = torch.stack(gate_modulations, dim=-1)
        gate_logits = self.query_gate(frozen_condition).unsqueeze(1)
        depth_gates = torch.softmax(gate_logits + gate_modulations, dim=-1)
        depth_contributions = (
            depth_gates * expert_scores * self.depth_gain.view(1, 1, -1)
        )

        if candidate_valid is None:
            valid = torch.ones(
                candidate_count, dtype=torch.bool, device=layer_states.device
            )
        else:
            valid = candidate_valid.detach().bool()
            if valid.shape != (candidate_count,):
                raise ValueError(
                    "Multi-depth score router validity mask does not align"
                )
        valid_matrix = valid.unsqueeze(0).expand(query_count, -1)
        valid_float = valid_matrix.to(depth_contributions.dtype)
        valid_count = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        depth_gates = depth_gates * valid_float.unsqueeze(-1)
        expert_scores = expert_scores * valid_float.unsqueeze(-1)
        depth_contributions = depth_contributions * valid_float.unsqueeze(-1)
        contribution_mean = depth_contributions.sum(
            dim=1, keepdim=True
        ) / valid_count.unsqueeze(-1)
        depth_contributions = (
            depth_contributions - contribution_mean
        ) * valid_float.unsqueeze(-1)
        residual = depth_contributions.sum(dim=-1)
        return (
            residual,
            depth_gates,
            expert_scores,
            depth_contributions,
            valid_matrix,
        )


class DualStreamTrajectoryAgreementCalibratorResidual(nn.Module):
    """Calibrate candidates from main/path agreement and path progress."""

    agreement_labels = (
        "main_deep_hadamard",
        "main_deep_absolute_difference",
        "trajectory_hadamard",
        "trajectory_absolute_difference",
    )
    progress_labels = (
        "main_deep_cosine",
        "trajectory_cosine",
        "shallow_middle_norm",
        "middle_deep_norm",
    )

    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = min(int(rank), self.hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.main_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.deep_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.shallow_middle_projection = nn.Linear(
            hidden_dim, self.rank, bias=False
        )
        self.middle_deep_projection = nn.Linear(
            hidden_dim, self.rank, bias=False
        )
        feature_dim = 4 * self.rank + len(self.progress_labels)
        self.calibrator_down = nn.Linear(feature_dim, self.rank)
        self.calibrator_up = nn.Linear(self.rank, self.rank)
        self.query_modulation = nn.Linear(hidden_dim, self.rank, bias=False)
        self.score_readout = nn.Linear(self.rank, 1, bias=False)
        nn.init.zeros_(self.score_readout.weight)

    @staticmethod
    def _normalize(state):
        return F.normalize(state, p=2, dim=-1, eps=1e-6)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != 3:
            raise ValueError(
                "Trajectory agreement path states must be "
                "[query,candidate,3,hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape != (query_count, self.hidden_dim):
            raise ValueError("Trajectory agreement query states do not align")
        if candidate_state.shape != (candidate_count, self.hidden_dim):
            raise ValueError("Trajectory agreement candidates do not align")
        if candidate_count == 0:
            raise ValueError("Trajectory agreement requires candidates")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        shallow, middle, deep = frozen_layers.unbind(dim=2)
        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        shallow_middle = middle - shallow
        middle_deep = deep - middle

        query_factor = torch.tanh(
            self.query_projection(frozen_condition)
        ).unsqueeze(1)
        main = self.main_projection(candidate_matrix) * (1.0 + query_factor)
        deep_state = self.deep_projection(deep) * (1.0 + query_factor)
        early_delta = self.shallow_middle_projection(shallow_middle)
        late_delta = self.middle_deep_projection(middle_deep)

        main_normalized = self._normalize(main)
        deep_normalized = self._normalize(deep_state)
        early_normalized = self._normalize(early_delta)
        late_normalized = self._normalize(late_delta)
        agreement_features = torch.cat([
            main_normalized * deep_normalized,
            (main_normalized - deep_normalized).abs(),
            early_normalized * late_normalized,
            (early_normalized - late_normalized).abs(),
        ], dim=-1)
        trajectory_progress = torch.stack([
            (main_normalized * deep_normalized).sum(dim=-1),
            (early_normalized * late_normalized).sum(dim=-1),
            shallow_middle.norm(dim=-1),
            middle_deep.norm(dim=-1),
        ], dim=-1)
        progress_scale = trajectory_progress[..., 2:].detach().mean(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        trajectory_progress = torch.cat([
            trajectory_progress[..., :2],
            torch.log1p(trajectory_progress[..., 2:] / progress_scale),
        ], dim=-1)

        hidden = F.gelu(self.calibrator_down(torch.cat([
            agreement_features, trajectory_progress
        ], dim=-1)))
        query_modulation = torch.tanh(
            self.query_modulation(frozen_condition)
        ).unsqueeze(1)
        hidden = F.gelu(self.calibrator_up(hidden * (1.0 + query_modulation)))
        calibration_score = self.score_readout(hidden).squeeze(-1)

        if candidate_valid is None:
            valid = torch.ones(
                candidate_count, dtype=torch.bool, device=layer_states.device
            )
        else:
            valid = candidate_valid.detach().bool()
            if valid.shape != (candidate_count,):
                raise ValueError(
                    "Trajectory agreement validity mask does not align"
                )
        valid_matrix = valid.unsqueeze(0).expand(query_count, -1)
        valid_float = valid_matrix.to(calibration_score.dtype)
        calibration_score = calibration_score * valid_float
        agreement_features = agreement_features * valid_float.unsqueeze(-1)
        trajectory_progress = trajectory_progress * valid_float.unsqueeze(-1)
        return (
            calibration_score,
            agreement_features,
            trajectory_progress,
            calibration_score,
            valid_matrix,
        )


class QueryConditionedMultiAnchorRankLandmarkCalibratorResidual(nn.Module):
    """Calibrate every candidate against fixed multi-scale parent landmarks."""

    landmark_ranks = (1, 5, 10, 20, 50)

    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = min(int(rank), self.hidden_dim)
        self.token_projection = nn.Linear(4 * hidden_dim, self.rank, bias=False)
        self.query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.landmark_rank_embedding = nn.Embedding(
            len(self.landmark_ranks), self.rank
        )
        pair_dim = 4 * self.rank + 1
        self.pair_down = nn.Linear(pair_dim, self.rank)
        self.pair_up = nn.Linear(self.rank, self.rank)
        self.contribution_readout = nn.Linear(self.rank, 1, bias=False)
        self.gate_readout = nn.Linear(3 * self.rank + 1, 1)
        self.final_gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != 3:
            raise ValueError(
                "Multi-anchor landmark path states must be "
                "[query,candidate,3,hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape != (query_count, self.hidden_dim):
            raise ValueError("Multi-anchor landmark query states do not align")
        if candidate_state.shape != (candidate_count, self.hidden_dim):
            raise ValueError("Multi-anchor landmark candidates do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Multi-anchor landmark parent scores do not align")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_scores = parent_scores.detach()
        if candidate_valid is None:
            valid = torch.ones(
                candidate_count, dtype=torch.bool, device=layer_states.device
            )
        else:
            valid = candidate_valid.detach().bool()
            if valid.shape != (candidate_count,):
                raise ValueError(
                    "Multi-anchor landmark validity mask does not align"
                )
        valid_matrix = valid.unsqueeze(0).expand(query_count, -1)
        zero_residual = frozen_scores.new_zeros(frozen_scores.shape)
        empty_indices = torch.zeros(
            (query_count, len(self.landmark_ranks)),
            dtype=torch.long,
            device=frozen_scores.device,
        )
        empty_landmarks = valid_matrix.new_zeros(
            (query_count, len(self.landmark_ranks))
        )
        empty_values = frozen_scores.new_zeros(
            query_count, candidate_count, len(self.landmark_ranks)
        )
        if candidate_count < self.landmark_ranks[-1] or int(valid.sum()) < self.landmark_ranks[-1]:
            return (
                zero_residual,
                empty_indices,
                empty_values,
                empty_values,
                valid_matrix,
                empty_landmarks,
            )

        masked_scores = frozen_scores.masked_fill(~valid_matrix, float("-inf"))
        ranked_indices = torch.argsort(masked_scores, dim=1, descending=True)
        rank_offsets = torch.tensor(
            [rank - 1 for rank in self.landmark_ranks],
            dtype=torch.long,
            device=frozen_scores.device,
        )
        landmark_indices = ranked_indices.index_select(1, rank_offsets)
        landmark_mask = valid_matrix.gather(1, landmark_indices)

        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        token_state = torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        )
        token_factor = self.token_projection(token_state)
        gather_index = landmark_indices.unsqueeze(-1).expand(-1, -1, self.rank)
        landmark_factor = token_factor.gather(1, gather_index)
        candidate_factor = token_factor.unsqueeze(2)
        landmark_factor = landmark_factor.unsqueeze(1)
        difference = candidate_factor - landmark_factor
        product = candidate_factor * landmark_factor
        query_factor = self.query_projection(frozen_condition).unsqueeze(1).unsqueeze(2)
        query_factor = query_factor.expand(-1, candidate_count, len(self.landmark_ranks), -1)
        rank_ids = torch.arange(
            len(self.landmark_ranks), device=frozen_scores.device
        )
        rank_factor = self.landmark_rank_embedding(rank_ids).view(
            1, 1, len(self.landmark_ranks), self.rank
        ).expand(query_count, candidate_count, -1, -1)
        landmark_scores = frozen_scores.gather(1, landmark_indices).unsqueeze(1)
        score_gaps = frozen_scores.unsqueeze(-1) - landmark_scores

        pair_hidden = F.gelu(self.pair_down(torch.cat([
            difference,
            product,
            query_factor,
            rank_factor,
            score_gaps.unsqueeze(-1),
        ], dim=-1)))
        pair_hidden = F.gelu(self.pair_up(pair_hidden))
        raw_contributions = self.contribution_readout(pair_hidden).squeeze(-1)
        gate_logits = self.gate_readout(torch.cat([
            candidate_factor.expand_as(pair_hidden),
            landmark_factor.expand_as(pair_hidden),
            query_factor,
            score_gaps.unsqueeze(-1),
        ], dim=-1)).squeeze(-1)
        pair_mask = valid_matrix.unsqueeze(-1) & landmark_mask.unsqueeze(1)
        gate_logits = gate_logits.masked_fill(~pair_mask, float("-inf"))
        landmark_gates = torch.softmax(gate_logits, dim=-1)
        landmark_gates = torch.where(
            pair_mask, landmark_gates, torch.zeros_like(landmark_gates)
        )
        per_landmark_contributions = (
            landmark_gates * raw_contributions * self.final_gain
        )
        per_landmark_contributions = (
            per_landmark_contributions * pair_mask.to(raw_contributions.dtype)
        )
        residual = per_landmark_contributions.sum(dim=-1)
        residual = residual * valid_matrix.to(residual.dtype)
        return (
            residual,
            landmark_indices,
            landmark_gates,
            per_landmark_contributions,
            valid_matrix,
            landmark_mask,
        )


class QueryTokenTop64CandidateSetTransformerResidual(nn.Module):
    """Jointly contextualize the frozen parent top-64 candidate set."""

    candidate_set_size = 64
    attention_heads = 2

    def __init__(self, hidden_dim, candidate_dim=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.candidate_dim = int(candidate_dim or hidden_dim)
        if self.hidden_dim % self.attention_heads:
            raise ValueError(
                "Top64 candidate-set Transformer hidden size must be divisible "
                "by its fixed attention head count"
            )
        self.candidate_projection = nn.Linear(
            self.candidate_dim + 3 * hidden_dim, hidden_dim
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.score_projection = nn.Linear(1, hidden_dim, bias=False)
        self.rank_embedding = nn.Embedding(self.candidate_set_size, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            self.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual_readout = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.residual_readout.weight)

    def forward(
        self,
        layer_states,
        condition,
        candidate_state,
        parent_scores,
        candidate_valid=None,
    ):
        if layer_states.ndim != 4 or layer_states.shape[2] != 3:
            raise ValueError(
                "Top64 candidate-set path states must be "
                "[query,candidate,3,hidden]"
            )
        query_count, candidate_count = layer_states.shape[:2]
        if condition.shape != (query_count, self.hidden_dim):
            raise ValueError("Top64 candidate-set query states do not align")
        if candidate_state.shape != (candidate_count, self.candidate_dim):
            raise ValueError("Top64 candidate-set candidates do not align")
        if parent_scores.shape != (query_count, candidate_count):
            raise ValueError("Top64 candidate-set parent scores do not align")

        frozen_layers = layer_states.detach()
        frozen_condition = condition.detach()
        frozen_candidates = candidate_state.detach()
        frozen_scores = parent_scores.detach()
        if candidate_valid is None:
            valid = torch.ones(
                candidate_count, dtype=torch.bool, device=layer_states.device
            )
        else:
            valid = candidate_valid.detach().bool()
            if valid.shape != (candidate_count,):
                raise ValueError("Top64 candidate-set validity mask does not align")
        valid_matrix = valid.unsqueeze(0).expand(query_count, -1)
        zero_residual = frozen_scores.new_zeros(frozen_scores.shape)
        empty_indices = torch.zeros(
            (query_count, self.candidate_set_size),
            dtype=torch.long,
            device=frozen_scores.device,
        )
        empty_ranks = torch.arange(
            1, self.candidate_set_size + 1, device=frozen_scores.device
        ).unsqueeze(0).expand(query_count, -1)
        empty_attention = frozen_scores.new_zeros(
            query_count,
            self.attention_heads,
            self.candidate_set_size + 1,
            self.candidate_set_size + 1,
        )
        empty_contributions = frozen_scores.new_zeros(
            query_count, self.candidate_set_size
        )
        empty_selection = valid_matrix.new_zeros(valid_matrix.shape)
        if (
            candidate_count < self.candidate_set_size
            or int(valid.sum()) < self.candidate_set_size
        ):
            return (
                zero_residual,
                empty_indices,
                empty_attention,
                empty_ranks,
                empty_contributions,
                empty_selection,
            )

        masked_scores = frozen_scores.masked_fill(~valid_matrix, float("-inf"))
        selected_indices = torch.topk(
            masked_scores, self.candidate_set_size, dim=1, sorted=True
        ).indices
        selected_scores = frozen_scores.gather(1, selected_indices)
        candidate_matrix = frozen_candidates.unsqueeze(0).expand(
            query_count, -1, -1
        )
        combined_state = torch.cat(
            [candidate_matrix, *frozen_layers.unbind(dim=2)], dim=-1
        )
        gather_index = selected_indices.unsqueeze(-1).expand(
            -1, -1, combined_state.shape[-1]
        )
        selected_state = combined_state.gather(1, gather_index)
        rank_ids = torch.arange(
            self.candidate_set_size, device=frozen_scores.device
        )
        candidate_tokens = (
            self.candidate_projection(selected_state)
            + self.score_projection(selected_scores.unsqueeze(-1))
            + self.rank_embedding(rank_ids).unsqueeze(0)
        )
        query_token = self.query_projection(frozen_condition).unsqueeze(1)
        tokens = self.input_norm(torch.cat([query_token, candidate_tokens], dim=1))
        attended, attention = self.self_attention(
            tokens,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = self.attention_norm(tokens + attended)
        tokens = self.output_norm(tokens + self.feed_forward(tokens))
        selected_contributions = self.residual_readout(tokens[:, 1:]).squeeze(-1)
        residual = zero_residual.scatter(1, selected_indices, selected_contributions)
        selection_mask = valid_matrix.new_zeros(valid_matrix.shape)
        selection_mask.scatter_(1, selected_indices, True)
        rank_positions = empty_ranks
        return (
            residual,
            selected_indices,
            attention,
            rank_positions,
            selected_contributions,
            selection_mask,
        )


class TorchRelationalHeteroConv(nn.Module):
    """DGL-graph-compatible relation convolution executed with PyTorch ops."""

    def __init__(self, relation_names, input_dim, output_dim):
        super().__init__()
        self.relation_names = list(relation_names)
        self.relation_to_index = {
            relation: index
            for index, relation in enumerate(self.relation_names)
        }
        self.linears = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in self.relation_names
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(output_dim))
            for _ in self.relation_names
        )
        self.output_dim = int(output_dim)

    def forward(self, graph, features, edge_weights=None):
        outputs = {}
        for etype in graph.canonical_etypes:
            if graph.num_edges(etype) == 0:
                continue
            src_type, relation, dst_type = etype
            if src_type not in features:
                continue

            src_nodes, dst_nodes = graph.edges(etype=etype)
            device = features[src_type].device
            src_nodes = src_nodes.to(device=device, dtype=torch.long)
            dst_nodes = dst_nodes.to(device=device, dtype=torch.long)
            src_count = graph.num_nodes(src_type)
            dst_count = graph.num_nodes(dst_type)

            src_degree = torch.bincount(
                src_nodes,
                minlength=src_count,
            ).clamp_min(1).to(features[src_type].dtype)
            dst_degree = torch.bincount(
                dst_nodes,
                minlength=dst_count,
            ).clamp_min(1).to(features[src_type].dtype)
            normalization = (
                src_degree.index_select(0, src_nodes).pow(-0.5)
                * dst_degree.index_select(0, dst_nodes).pow(-0.5)
            )
            weight = None
            if edge_weights is not None:
                weight = edge_weights.get(etype)
            if weight is None and "weight" in graph.edges[etype].data:
                weight = graph.edges[etype].data["weight"]
            if weight is not None:
                normalization = normalization * weight.to(
                    device=device,
                    dtype=normalization.dtype,
                ).reshape(-1)

            relation_index = self.relation_to_index[relation]
            linear = self.linears[relation_index]
            messages = linear(
                features[src_type].index_select(0, src_nodes)
            ) * normalization.unsqueeze(-1)
            relation_output = messages.new_zeros((dst_count, self.output_dim))
            relation_output.index_add_(0, dst_nodes, messages)
            relation_output = relation_output + self.biases[relation_index]
            if dst_type in outputs:
                outputs[dst_type] = outputs[dst_type] + relation_output
            else:
                outputs[dst_type] = relation_output
        return outputs


class SharedRelationalHGCN(nn.Module):
    """Relation-specific graph convolution shared by all output modalities."""

    def __init__(
        self,
        node_types,
        canonical_etypes,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        graph_backend="dgl",
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.relation_names = sorted({etype[1] for etype in canonical_etypes})
        if graph_backend not in {"dgl", "torch"}:
            raise ValueError(
                "graph_backend must be either 'dgl' or 'torch'"
            )
        self.graph_backend = graph_backend
        layer_count = max(0, int(num_layers))
        dimensions = []
        if layer_count == 1:
            dimensions.append((input_dim, output_dim))
        elif layer_count > 1:
            dimensions.append((input_dim, hidden_dim))
            dimensions.extend(
                (hidden_dim, hidden_dim)
                for _ in range(layer_count - 2)
            )
            dimensions.append((hidden_dim, output_dim))

        self.layers = nn.ModuleList()
        self.carry_projections = nn.ModuleList()
        for in_dim, out_dim in dimensions:
            if self.graph_backend == "torch":
                self.layers.append(
                    TorchRelationalHeteroConv(
                        self.relation_names,
                        in_dim,
                        out_dim,
                    )
                )
            else:
                self.layers.append(
                    HeteroGraphConv(
                        {
                            relation: GraphConv(
                                in_dim,
                                out_dim,
                                allow_zero_in_degree=True,
                            )
                            for relation in self.relation_names
                        },
                        aggregate="sum",
                    )
                )
            self.carry_projections.append(
                nn.ModuleDict(
                    {
                        node_type: (
                            nn.Identity()
                            if in_dim == out_dim
                            else nn.Linear(in_dim, out_dim, bias=False)
                        )
                        for node_type in self.node_types
                    }
                )
            )

    def forward(self, graph, features, edge_weights=None):
        hidden = features
        relation_counts = {}
        for etype in graph.canonical_etypes:
            relation_counts[etype[1]] = relation_counts.get(etype[1], 0) + 1

        for layer, carry_projection in zip(
            self.layers,
            self.carry_projections,
        ):
            if self.graph_backend == "torch":
                layer_output = layer(
                    graph,
                    hidden,
                    edge_weights=edge_weights,
                )
            else:
                relation_kwargs = {}
                for etype in graph.canonical_etypes:
                    if relation_counts[etype[1]] != 1:
                        continue
                    weight = None
                    if edge_weights is not None:
                        weight = edge_weights.get(etype)
                    if weight is None and "weight" in graph.edges[etype].data:
                        weight = graph.edges[etype].data["weight"]
                    if weight is not None:
                        relation_kwargs[etype[1]] = {
                            "edge_weight": weight.reshape(-1).float()
                        }

                layer_output = layer(
                    graph,
                    hidden,
                    mod_kwargs=relation_kwargs,
                )
            next_hidden = {}
            for node_type in self.node_types:
                if node_type in layer_output:
                    next_hidden[node_type] = F.gelu(layer_output[node_type])
                elif node_type in hidden:
                    next_hidden[node_type] = carry_projection[node_type](
                        hidden[node_type]
                    )
            hidden = next_hidden
        return hidden


class SparseRouterMoE(nn.Module):
    """Top-k router whose tokens execute only their selected experts."""

    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router = nn.Linear(input_dim, self.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.last_aux_loss = None
        self.last_topk_indices = None
        self.last_router_probabilities = None

    def forward(self, inputs):
        original_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        if flat_inputs.shape[0] == 0:
            return inputs.new_empty(*original_shape)

        router_logits = self.router(flat_inputs)
        router_probabilities = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            router_probabilities,
            k=self.top_k,
            dim=-1,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        outputs = flat_inputs.new_zeros((flat_inputs.shape[0], 1))
        for expert_index, expert in enumerate(self.experts):
            assignments = torch.nonzero(topk_indices == expert_index, as_tuple=False)
            if assignments.numel() == 0:
                continue
            token_indices = assignments[:, 0]
            route_slots = assignments[:, 1]
            expert_outputs = expert(flat_inputs.index_select(0, token_indices))
            route_weights = topk_weights[token_indices, route_slots].unsqueeze(-1)
            outputs.index_add_(0, token_indices, expert_outputs * route_weights)

        importance = router_probabilities.mean(dim=0)
        route_load = F.one_hot(topk_indices, num_classes=self.num_experts).float().mean(dim=(0, 1))
        self.last_aux_loss = self.num_experts * torch.sum(importance * route_load)
        self.last_topk_indices = topk_indices.detach()
        self.last_router_probabilities = router_probabilities.detach()
        return outputs.reshape(*original_shape)


class ModalityAdapter(nn.Module):
    """Modality-specific fusion followed by a residual adapter."""

    def __init__(self, hidden_dim, fusion_mode, dropout=0.1):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.concat_projection = None

        if fusion_mode == "hierarchical":
            self.hierarchical_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.hierarchical_gate = None

        if fusion_mode == "weighted":
            self.fusion_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("fusion_logit", None)

        self.norm = nn.LayerNorm(hidden_dim)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, graph_features, transformer_features):
        if self.fusion_mode == "concat":
            fused = self.concat_projection(
                torch.cat([graph_features, transformer_features], dim=-1)
            )
        elif self.fusion_mode == "hierarchical":
            gate = self.hierarchical_gate(
                torch.cat([graph_features, transformer_features], dim=-1)
            )
            fused = graph_features + gate * transformer_features
        elif self.fusion_mode == "sequential":
            fused = graph_features + transformer_features
        else:
            transformer_weight = torch.sigmoid(self.fusion_logit)
            fused = (
                (1.0 - transformer_weight) * graph_features
                + transformer_weight * transformer_features
            )

        fused = self.norm(fused)
        return self.norm(fused + self.adapter(fused))


class RelationAwareSparseTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, attention_mask):
        attended, _ = self.attention(
            sequence,
            sequence,
            sequence,
            attn_mask=attention_mask,
            need_weights=False,
        )
        sequence = self.norm1(sequence + self.dropout(attended))
        return self.norm2(sequence + self.feedforward(sequence))


class QueryCenteredRelationAwareTransformer(nn.Module):
    """
    Selects query-relevant context and restricts ordinary-node attention to
    graph neighbors plus node-type summary tokens.
    """

    def __init__(
        self,
        node_types,
        canonical_etypes,
        hidden_dim,
        num_heads,
        feedforward_dim,
        num_layers,
        context_topk=256,
        max_hops=4,
        hop_penalty=0.2,
        evidence_weight=0.2,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.node_types = list(node_types)
        self.node_type_to_index = {
            node_type: index for index, node_type in enumerate(self.node_types)
        }
        self.relation_names = sorted({etype[1] for etype in canonical_etypes})
        self.relation_to_index = {
            relation: index for index, relation in enumerate(self.relation_names)
        }
        self.hidden_dim = hidden_dim
        self.context_topk = max(1, int(context_topk))
        self.max_hops = max(1, int(max_hops))
        self.hop_penalty = float(hop_penalty)
        self.evidence_weight = float(evidence_weight)

        self.node_type_embeddings = nn.Embedding(len(self.node_types), hidden_dim)
        self.hop_embeddings = nn.Embedding(self.max_hops + 2, hidden_dim)
        self.relation_embeddings = nn.Embedding(
            max(1, len(self.relation_names)),
            hidden_dim,
        )
        self.relation_attention_bias = nn.Parameter(
            torch.zeros(max(1, len(self.relation_names)))
        )
        self.summary_tokens = nn.Parameter(
            torch.empty(len(self.node_types), hidden_dim)
        )
        nn.init.normal_(self.summary_tokens, std=0.02)
        self.layers = nn.ModuleList(
            [
                RelationAwareSparseTransformerBlock(
                    hidden_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(max(0, int(num_layers)))
            ]
        )

    def _flatten_features(self, feature_dict):
        offsets = {}
        tensors = []
        cursor = 0
        for node_type in self.node_types:
            tensor = feature_dict.get(node_type)
            node_count = 0 if tensor is None else tensor.shape[0]
            offsets[node_type] = (cursor, cursor + node_count)
            if tensor is not None and node_count > 0:
                tensors.append(tensor)
            cursor += node_count
        if not tensors:
            return None, offsets
        return torch.cat(tensors, dim=0), offsets

    def _relation_context(self, graph, flat_features, offsets):
        relation_sum = torch.zeros_like(flat_features)
        relation_degree = flat_features.new_zeros((flat_features.shape[0], 1))
        for etype in graph.canonical_etypes:
            if graph.num_edges(etype) == 0:
                continue
            src_type, relation, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            src_offset = offsets[src_type][0]
            dst_offset = offsets[dst_type][0]
            src_indices = src_nodes.to(flat_features.device) + src_offset
            dst_indices = dst_nodes.to(flat_features.device) + dst_offset
            relation_index = self.relation_to_index.get(relation, 0)
            relation_vector = self.relation_embeddings.weight[relation_index].unsqueeze(0)
            src_values = relation_vector.expand(src_indices.shape[0], -1)
            dst_values = relation_vector.expand(dst_indices.shape[0], -1)
            relation_sum.index_add_(0, src_indices, src_values)
            relation_sum.index_add_(0, dst_indices, dst_values)
            ones = relation_degree.new_ones((src_indices.shape[0], 1))
            relation_degree.index_add_(0, src_indices, ones)
            relation_degree.index_add_(0, dst_indices, ones)
        return relation_sum / relation_degree.clamp_min(1.0)

    def _metadata_tensors(self, graph, offsets, device):
        node_type_ids = []
        hop_distances = []
        seed_masks = []
        evidence_scores = []
        for node_type in self.node_types:
            node_count = offsets[node_type][1] - offsets[node_type][0]
            if node_count == 0:
                continue
            node_type_ids.append(
                torch.full(
                    (node_count,),
                    self.node_type_to_index[node_type],
                    dtype=torch.long,
                    device=device,
                )
            )
            node_data = graph.nodes[node_type].data
            hop = node_data[QUERY_HOP_KEY] if QUERY_HOP_KEY in node_data else None
            if hop is None:
                hop = torch.full(
                    (node_count,),
                    self.max_hops + 1,
                    dtype=torch.long,
                    device=device,
                )
            else:
                hop = hop.to(device=device, dtype=torch.long)
            hop_distances.append(hop.clamp(min=0, max=self.max_hops + 1))

            seed = node_data[QUERY_SEED_KEY] if QUERY_SEED_KEY in node_data else None
            if seed is None:
                seed = torch.zeros(node_count, dtype=torch.bool, device=device)
            else:
                seed = seed.to(device=device, dtype=torch.bool)
            seed_masks.append(seed)

            evidence = (
                node_data[EVIDENCE_SCORE_KEY]
                if EVIDENCE_SCORE_KEY in node_data
                else None
            )
            if evidence is None:
                evidence = torch.zeros(node_count, dtype=torch.float32, device=device)
            else:
                evidence = evidence.to(device=device, dtype=torch.float32)
            evidence_scores.append(evidence)

        return (
            torch.cat(node_type_ids),
            torch.cat(hop_distances),
            torch.cat(seed_masks),
            torch.cat(evidence_scores),
        )

    def _select_context(
        self,
        flat_features,
        hop_distances,
        seed_mask,
        evidence_scores,
    ):
        if seed_mask.any():
            query_vector = flat_features[seed_mask].mean(dim=0, keepdim=True)
        else:
            query_vector = flat_features.mean(dim=0, keepdim=True)

        relevance = F.cosine_similarity(
            flat_features,
            query_vector.expand_as(flat_features),
            dim=-1,
        )
        evidence = torch.log1p(evidence_scores.clamp_min(0.0))
        if evidence.numel() > 1:
            evidence = (evidence - evidence.mean()) / evidence.std(unbiased=False).clamp_min(1e-6)
        context_scores = (
            relevance
            - self.hop_penalty * hop_distances.float()
            + self.evidence_weight * evidence
        )

        seed_indices = torch.nonzero(seed_mask, as_tuple=False).flatten()
        non_seed_indices = torch.nonzero(~seed_mask, as_tuple=False).flatten()
        remaining_slots = max(0, self.context_topk - seed_indices.numel())
        if remaining_slots == 0:
            selected_non_seed = non_seed_indices.new_empty(0)
        elif non_seed_indices.numel() > remaining_slots:
            selected_non_seed = non_seed_indices[
                torch.topk(
                    context_scores.index_select(0, non_seed_indices),
                    k=remaining_slots,
                ).indices
            ]
        else:
            selected_non_seed = non_seed_indices

        selected = torch.cat([seed_indices, selected_non_seed])
        if selected.numel() == 0:
            selected = torch.topk(
                context_scores,
                k=min(self.context_topk, context_scores.numel()),
            ).indices
        return torch.unique(selected, sorted=True)

    def _build_attention_mask(self, graph, offsets, selected_indices, device, dtype):
        summary_count = len(self.node_types)
        sequence_length = summary_count + selected_indices.numel()
        attention_mask = torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        attention_mask[:summary_count, :] = 0.0
        attention_mask[:, :summary_count] = 0.0
        attention_mask.fill_diagonal_(0.0)

        flat_to_sequence = torch.full(
            (offsets[self.node_types[-1]][1],),
            -1,
            dtype=torch.long,
            device=device,
        )
        flat_to_sequence[selected_indices] = (
            torch.arange(selected_indices.numel(), device=device) + summary_count
        )

        for etype in graph.canonical_etypes:
            if graph.num_edges(etype) == 0:
                continue
            src_type, relation, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            src_flat = src_nodes.to(device) + offsets[src_type][0]
            dst_flat = dst_nodes.to(device) + offsets[dst_type][0]
            src_sequence = flat_to_sequence[src_flat]
            dst_sequence = flat_to_sequence[dst_flat]
            valid = (src_sequence >= 0) & (dst_sequence >= 0)
            if not valid.any():
                continue
            src_sequence = src_sequence[valid]
            dst_sequence = dst_sequence[valid]
            relation_index = self.relation_to_index.get(relation, 0)
            relation_bias = self.relation_attention_bias[relation_index].to(dtype)
            attention_mask[src_sequence, dst_sequence] = relation_bias
            attention_mask[dst_sequence, src_sequence] = relation_bias
        return attention_mask

    def forward(self, feature_dict, graph):
        if not self.layers:
            return feature_dict

        flat_features, offsets = self._flatten_features(feature_dict)
        if flat_features is None or flat_features.shape[0] == 0:
            return feature_dict

        device = flat_features.device
        node_type_ids, hop_distances, seed_mask, evidence_scores = self._metadata_tensors(
            graph,
            offsets,
            device,
        )
        relation_context = self._relation_context(graph, flat_features, offsets)
        enriched_features = (
            flat_features
            + self.node_type_embeddings(node_type_ids)
            + self.hop_embeddings(hop_distances)
            + relation_context
        )

        selected_indices = self._select_context(
            enriched_features,
            hop_distances,
            seed_mask,
            evidence_scores,
        )
        selected_features = enriched_features.index_select(0, selected_indices)

        summary_features = []
        for node_type in self.node_types:
            start, end = offsets[node_type]
            type_selected = selected_indices[
                (selected_indices >= start) & (selected_indices < end)
            ]
            summary = self.summary_tokens[self.node_type_to_index[node_type]]
            if type_selected.numel() > 0:
                summary = summary + enriched_features.index_select(0, type_selected).mean(dim=0)
            summary_features.append(summary)

        sequence = torch.cat(
            [torch.stack(summary_features), selected_features],
            dim=0,
        ).unsqueeze(0)
        attention_mask = self._build_attention_mask(
            graph,
            offsets,
            selected_indices,
            device,
            sequence.dtype,
        )
        for layer in self.layers:
            sequence = layer(sequence, attention_mask)

        transformed_selected = sequence.squeeze(0)[len(self.node_types):]
        transformed_flat = flat_features.clone()
        transformed_flat.index_copy_(0, selected_indices, transformed_selected)

        transformed = {}
        for node_type in self.node_types:
            start, end = offsets[node_type]
            if end > start:
                transformed[node_type] = transformed_flat[start:end]
        return transformed


class UnifiedDiseaseAssociationModel(nn.Module):
    """
    One shared encoder with modality-specific adapters and four sparse-router
    output heads.
    """

    def __init__(
        self,
        node_feature_embeddings,
        hgcn_hidden,
        hgcn_out,
        num_transformer_layers,
        transformer_heads,
        transformer_ff,
        num_latents,
        ntypes,
        etypes,
        target_node_types,
        num_nodes_dict,
        node_info=None,
        dropout=0.1,
        num_gnn_layers=2,
        freeze_pretrained=False,
        disable_hgcn=False,
        disable_transformer=False,
        disable_moe=False,
        disable_mla=False,
        disable_feature_gating=False,
        fusion_mode="weighted",
        modality_fusion_modes=None,
        scorer_mode="concat_mlp",
        global_bilinear_score_weight=1.0,
        contextual_score_weight=1.0,
        kge_pretrain_scope="target_relations",
        kge_relation_model="distmult",
        kge_relation_rank=16,
        kge_context_input_weight=0.0,
        kge_target_specific_disease_embeddings=False,
        kge_target_disease_shared_weight=0.0,
        kge_candidate_bias=False,
        kge_target_residual_rank=0,
        kge_path_propagation_hops=0,
        kge_path_cross_modal=False,
        kge_path_cross_modal_types=None,
        kge_path_cross_modal_support_types=None,
        biogt_path_propagation_hops=0,
        biogt_path_support_types=None,
        biogt_neural_path_dim=0,
        biogt_relbf_dim=0,
        biogt_relbf_jk_readout=False,
        biogt_relbf_edge_attention=False,
        biogt_relbf_deep_refinement=False,
        biogt_relbf_indicator_residual=False,
        biogt_relbf_indicator_support_gating=False,
        biogt_relbf_indicator_semantic_readout=False,
        biogt_relbf_query_relation_adapter_rank=0,
        biogt_relbf_indicator_query_boundary_rank=0,
        biogt_relbf_indicator_multirule_count=0,
        biogt_relbf_indicator_multirule_readout_rank=0,
        biogt_relbf_indicator_relation_transition_layers=0,
        biogt_relbf_indicator_relation_transition_rank=0,
        biogt_relbf_indicator_query_local_relation_transition_layers=0,
        biogt_relbf_indicator_query_local_relation_transition_rank=0,
        biogt_relbf_path_layer_attention_rank=0,
        biogt_relbf_path_token_transformer=False,
        biogt_relbf_candidate_competition_boundary=False,
        biogt_relbf_parent_top10_boundary_comparator=False,
        biogt_relbf_boundary_window_rank_slot_expert=False,
        biogt_relbf_boundary_exchange=False,
        biogt_relbf_challenger_boundary_promotion=False,
        biogt_relbf_gatekeeper_challenger_paired_margin=False,
        biogt_relbf_boundary_prototype_ordinal_adapter=False,
        biogt_relbf_multi_depth_score_router=False,
        biogt_relbf_dual_stream_trajectory_agreement_calibrator=False,
        biogt_relbf_multi_anchor_rank_landmark_calibrator=False,
        biogt_relbf_query_token_top64_candidate_set_transformer=False,
        sparse_transformer_topk=256,
        sparse_transformer_max_hops=4,
        sparse_transformer_hop_penalty=0.2,
        sparse_transformer_evidence_weight=0.2,
        moe_num_experts=4,
        moe_top_k=2,
        moe_hidden_dim=None,
        embedding_cache_size=2,
        graph_backend="dgl",
        **unused_kwargs,
    ):
        super().__init__()
        del num_latents, disable_mla, disable_feature_gating, unused_kwargs
        self.ntypes = list(ntypes)
        self.canonical_etypes = list(etypes)
        self.target_node_types = set(target_node_types)
        self.target_etypes = {
            etype
            for etype in self.canonical_etypes
            if etype[0] in self.target_node_types or etype[2] in self.target_node_types
        }
        self.node_info = node_info
        self.hidden_dim = int(hgcn_out)
        self.num_gnn_layers = int(num_gnn_layers)
        self.disable_hgcn = bool(disable_hgcn)
        self.disable_transformer = bool(disable_transformer)
        self.disable_moe = bool(disable_moe)
        if graph_backend not in {"dgl", "torch"}:
            raise ValueError(
                "graph_backend must be either 'dgl' or 'torch'"
            )
        self.graph_backend = graph_backend
        self.fusion_mode = "modality_specific"
        self.enable_explainability = False
        self.scorer_mode = str(scorer_mode or "concat_mlp").lower()
        self.kge_pretrain_scope = str(
            kge_pretrain_scope or "target_relations"
        ).lower()
        self.kge_relation_model = str(
            kge_relation_model or "distmult"
        ).lower()
        self.kge_relation_rank = int(kge_relation_rank)
        self.kge_context_input_weight = float(kge_context_input_weight)
        self.kge_target_specific_disease_embeddings = bool(
            kge_target_specific_disease_embeddings
        )
        self.kge_target_disease_shared_weight = float(
            kge_target_disease_shared_weight
        )
        self.kge_candidate_bias = bool(kge_candidate_bias)
        self.kge_target_residual_rank = int(kge_target_residual_rank)
        self.kge_path_propagation_hops = int(
            kge_path_propagation_hops
        )
        self.kge_path_cross_modal = bool(kge_path_cross_modal)
        if isinstance(kge_path_cross_modal_types, str):
            kge_path_cross_modal_types = (
                item.strip()
                for item in kge_path_cross_modal_types.split(",")
            )
        self.kge_path_cross_modal_types = {
            str(item).strip()
            for item in (kge_path_cross_modal_types or [])
            if str(item).strip()
        }
        if isinstance(kge_path_cross_modal_support_types, str):
            parsed_support_types = {}
            for entry in kge_path_cross_modal_support_types.split(";"):
                entry = entry.strip()
                if not entry:
                    continue
                target, separator, supports = entry.partition("=")
                if not separator:
                    raise ValueError(
                        "Cross-modal support entries must use "
                        "'target=support+support'"
                    )
                parsed_support_types[target.strip()] = {
                    item.strip()
                    for item in supports.split("+")
                    if item.strip()
                }
            kge_path_cross_modal_support_types = parsed_support_types
        self.kge_path_cross_modal_support_types = {
            str(target).strip(): {
                str(item).strip()
                for item in supports
                if str(item).strip()
            }
            for target, supports in (
                kge_path_cross_modal_support_types or {}
            ).items()
            if str(target).strip()
        }
        self.biogt_path_propagation_hops = int(
            biogt_path_propagation_hops
        )
        self.biogt_neural_path_dim = int(biogt_neural_path_dim)
        self.biogt_relbf_dim = int(biogt_relbf_dim)
        self.biogt_relbf_jk_readout = bool(biogt_relbf_jk_readout)
        self.biogt_relbf_edge_attention = bool(
            biogt_relbf_edge_attention
        )
        self.biogt_relbf_deep_refinement = bool(
            biogt_relbf_deep_refinement
        )
        self.biogt_relbf_indicator_residual = bool(
            biogt_relbf_indicator_residual
        )
        self.biogt_relbf_indicator_support_gating = bool(
            biogt_relbf_indicator_support_gating
        )
        self.biogt_relbf_indicator_semantic_readout = bool(
            biogt_relbf_indicator_semantic_readout
        )
        self.biogt_relbf_query_relation_adapter_rank = int(
            biogt_relbf_query_relation_adapter_rank
        )
        self.biogt_relbf_indicator_query_boundary_rank = int(
            biogt_relbf_indicator_query_boundary_rank
        )
        self.biogt_relbf_indicator_multirule_count = int(
            biogt_relbf_indicator_multirule_count
        )
        self.biogt_relbf_indicator_multirule_readout_rank = int(
            biogt_relbf_indicator_multirule_readout_rank
        )
        self.biogt_relbf_indicator_relation_transition_layers = int(
            biogt_relbf_indicator_relation_transition_layers
        )
        self.biogt_relbf_indicator_relation_transition_rank = int(
            biogt_relbf_indicator_relation_transition_rank
        )
        self.biogt_relbf_indicator_query_local_relation_transition_layers = int(
            biogt_relbf_indicator_query_local_relation_transition_layers
        )
        self.biogt_relbf_indicator_query_local_relation_transition_rank = int(
            biogt_relbf_indicator_query_local_relation_transition_rank
        )
        self.biogt_relbf_path_layer_attention_rank = int(
            biogt_relbf_path_layer_attention_rank
        )
        self.biogt_relbf_path_token_transformer = bool(
            biogt_relbf_path_token_transformer
        )
        self.biogt_relbf_candidate_competition_boundary = bool(
            biogt_relbf_candidate_competition_boundary
        )
        self.biogt_relbf_parent_top10_boundary_comparator = bool(
            biogt_relbf_parent_top10_boundary_comparator
        )
        self.biogt_relbf_boundary_window_rank_slot_expert = bool(
            biogt_relbf_boundary_window_rank_slot_expert
        )
        self.biogt_relbf_boundary_exchange = bool(
            biogt_relbf_boundary_exchange
        )
        self.biogt_relbf_challenger_boundary_promotion = bool(
            biogt_relbf_challenger_boundary_promotion
        )
        self.biogt_relbf_gatekeeper_challenger_paired_margin = bool(
            biogt_relbf_gatekeeper_challenger_paired_margin
        )
        self.biogt_relbf_boundary_prototype_ordinal_adapter = bool(
            biogt_relbf_boundary_prototype_ordinal_adapter
        )
        self.biogt_relbf_multi_depth_score_router = bool(
            biogt_relbf_multi_depth_score_router
        )
        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator = bool(
            biogt_relbf_dual_stream_trajectory_agreement_calibrator
        )
        self.biogt_relbf_multi_anchor_rank_landmark_calibrator = bool(
            biogt_relbf_multi_anchor_rank_landmark_calibrator
        )
        self.biogt_relbf_query_token_top64_candidate_set_transformer = bool(
            biogt_relbf_query_token_top64_candidate_set_transformer
        )
        if isinstance(biogt_path_support_types, str):
            parsed_biogt_support_types = {}
            for entry in biogt_path_support_types.split(";"):
                entry = entry.strip()
                if not entry:
                    continue
                target, separator, supports = entry.partition("=")
                if not separator:
                    raise ValueError(
                        "BioGT path support entries must use "
                        "'target=support+support'"
                    )
                parsed_biogt_support_types[target.strip()] = {
                    item.strip()
                    for item in supports.split("+")
                    if item.strip()
                }
            biogt_path_support_types = parsed_biogt_support_types
        self.biogt_path_support_types = {
            str(target).strip(): {
                str(item).strip()
                for item in supports
                if str(item).strip()
            }
            for target, supports in (
                biogt_path_support_types or {}
            ).items()
            if str(target).strip()
        }
        if self.kge_target_residual_rank < 0:
            raise ValueError("kge_target_residual_rank must be non-negative")
        if self.kge_path_propagation_hops not in {0, 3, 5}:
            raise ValueError(
                "kge_path_propagation_hops must be 0, 3, or 5"
            )
        if self.biogt_path_propagation_hops not in {0, 3, 5}:
            raise ValueError(
                "biogt_path_propagation_hops must be 0, 3, or 5"
            )
        if self.biogt_neural_path_dim < 0:
            raise ValueError("biogt_neural_path_dim must be non-negative")
        if self.biogt_relbf_dim < 0:
            raise ValueError("biogt_relbf_dim must be non-negative")
        if self.biogt_relbf_query_relation_adapter_rank < 0:
            raise ValueError(
                "biogt_relbf_query_relation_adapter_rank must be non-negative"
            )
        if self.biogt_relbf_indicator_query_boundary_rank < 0:
            raise ValueError(
                "biogt_relbf_indicator_query_boundary_rank must be non-negative"
            )
        if self.biogt_relbf_indicator_multirule_count < 0:
            raise ValueError(
                "biogt_relbf_indicator_multirule_count must be non-negative"
            )
        if self.biogt_relbf_indicator_multirule_readout_rank < 0:
            raise ValueError(
                "biogt_relbf_indicator_multirule_readout_rank must be "
                "non-negative"
            )
        if bool(self.biogt_relbf_indicator_multirule_count) != bool(
            self.biogt_relbf_indicator_multirule_readout_rank
        ):
            raise ValueError(
                "BioGT indicator multi-rule count and readout rank must both "
                "be zero or both be positive"
            )
        if self.biogt_relbf_indicator_relation_transition_layers not in {
            0, 1, 2,
        }:
            raise ValueError(
                "BioGT indicator relation-transition layers must be 0, 1, or 2"
            )
        if self.biogt_relbf_indicator_relation_transition_rank < 0:
            raise ValueError(
                "BioGT indicator relation-transition rank must be non-negative"
            )
        if bool(self.biogt_relbf_indicator_relation_transition_layers) != bool(
            self.biogt_relbf_indicator_relation_transition_rank
        ):
            raise ValueError(
                "BioGT indicator relation-transition layers and rank must "
                "both be zero or both be positive"
            )
        if self.biogt_relbf_indicator_relation_transition_layers > 0 and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
        ):
            raise ValueError(
                "BioGT indicator relation-transition residual requires an "
                "enabled indicator residual and biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_indicator_query_local_relation_transition_layers
            not in {0, 1, 2}
        ):
            raise ValueError(
                "BioGT query-local relation-transition layers must be 0, 1, "
                "or 2"
            )
        if self.biogt_relbf_indicator_query_local_relation_transition_rank < 0:
            raise ValueError(
                "BioGT query-local relation-transition rank must be "
                "non-negative"
            )
        if self.biogt_relbf_path_layer_attention_rank < 0:
            raise ValueError(
                "BioGT RelBF path-layer attention rank must be non-negative"
            )
        if (
            self.biogt_relbf_path_layer_attention_rank > 0
            and self.biogt_relbf_dim <= 0
        ):
            raise ValueError(
                "BioGT RelBF path-layer attention requires "
                "biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_path_token_transformer and self.biogt_relbf_dim <= 0:
            raise ValueError(
                "BioGT path-token Transformer requires biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_path_token_transformer
            and self.biogt_relbf_dim
            % CandidateConditionedPathTokenTransformerResidual.attention_heads
        ):
            raise ValueError(
                "BioGT path-token Transformer requires a RelBF dimension "
                "divisible by its fixed attention head count"
            )
        if (
            self.biogt_relbf_candidate_competition_boundary
            and self.biogt_relbf_dim <= 0
        ):
            raise ValueError(
                "BioGT candidate competition requires biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_parent_top10_boundary_comparator
            and self.biogt_relbf_dim <= 0
        ):
            raise ValueError(
                "BioGT parent top10 boundary comparator requires "
                "biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_boundary_window_rank_slot_expert
            and (
                self.biogt_relbf_dim <= 0
                or not self.biogt_relbf_parent_top10_boundary_comparator
            )
        ):
            raise ValueError(
                "BioGT boundary-window rank-slot expert requires an enabled "
                "parent top10 boundary comparator"
            )
        if self.biogt_relbf_challenger_boundary_promotion and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
        ):
            raise ValueError(
                "BioGT challenger-only promotion requires the enabled gene9 "
                "parent structure without the gene10 exchange branch"
            )
        if self.biogt_relbf_gatekeeper_challenger_paired_margin and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
        ):
            raise ValueError(
                "BioGT paired-margin adapter requires the enabled gene9 "
                "parent without gene10 or gene11 branches"
            )
        if self.biogt_relbf_boundary_prototype_ordinal_adapter and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
        ):
            raise ValueError(
                "BioGT boundary-prototype ordinal adapter requires the enabled "
                "gene9 parent without gene10, gene11, or gene12 branches"
            )
        if self.biogt_relbf_multi_depth_score_router and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
        ):
            raise ValueError(
                "BioGT multi-depth score router requires the enabled gene9 "
                "parent without gene10-gene13 branches"
            )
        if self.biogt_relbf_dual_stream_trajectory_agreement_calibrator and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or self.biogt_relbf_multi_depth_score_router
        ):
            raise ValueError(
                "BioGT dual-stream trajectory agreement calibrator requires "
                "the enabled gene9 parent without gene10-gene14 branches"
            )
        if self.biogt_relbf_multi_anchor_rank_landmark_calibrator and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or self.biogt_relbf_multi_depth_score_router
            or self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
        ):
            raise ValueError(
                "BioGT multi-anchor rank-landmark calibrator requires the "
                "enabled gene9 parent without gene10-gene15 branches"
            )
        if self.biogt_relbf_query_token_top64_candidate_set_transformer and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or self.biogt_relbf_multi_depth_score_router
            or self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
            or self.biogt_relbf_multi_anchor_rank_landmark_calibrator
        ):
            raise ValueError(
                "BioGT query-token top64 candidate-set Transformer requires "
                "the enabled gene9 parent without gene10-gene16 branches"
            )
        if bool(
            self.biogt_relbf_indicator_query_local_relation_transition_layers
        ) != bool(
            self.biogt_relbf_indicator_query_local_relation_transition_rank
        ):
            raise ValueError(
                "BioGT query-local relation-transition layers and rank must "
                "both be zero or both be positive"
            )
        if (
            self.biogt_relbf_indicator_query_local_relation_transition_layers > 0
            and (
                self.biogt_relbf_dim <= 0
                or not self.biogt_relbf_indicator_residual
            )
        ):
            raise ValueError(
                "BioGT query-local relation-transition requires an enabled "
                "indicator residual and biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_indicator_relation_transition_layers > 0
            and self.biogt_relbf_indicator_query_local_relation_transition_layers
            > 0
        ):
            raise ValueError(
                "Global and query-local relation-transition modes are "
                "mutually exclusive"
            )
        if self.biogt_relbf_indicator_multirule_count > 0 and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
        ):
            raise ValueError(
                "BioGT indicator multi-rule residual requires an enabled "
                "indicator residual and biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_indicator_multirule_count > 0 and (
            self.biogt_relbf_indicator_support_gating
            or self.biogt_relbf_indicator_semantic_readout
            or self.biogt_relbf_indicator_query_boundary_rank > 0
        ):
            raise ValueError(
                "BioGT indicator multi-rule residual cannot be combined "
                "with support gating, semantic readout, or query boundary"
            )
        if self.biogt_relbf_indicator_query_boundary_rank > 0 and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
        ):
            raise ValueError(
                "BioGT RelBF indicator query-boundary adapter requires an "
                "enabled indicator residual and biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_query_relation_adapter_rank > 0
            and self.biogt_relbf_dim <= 0
        ):
            raise ValueError(
                "BioGT RelBF query-relation adapters require "
                "biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_jk_readout and self.biogt_relbf_dim <= 0:
            raise ValueError(
                "BioGT RelBF JK readout requires biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_edge_attention and self.biogt_relbf_dim <= 0:
            raise ValueError(
                "BioGT RelBF edge attention requires biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_deep_refinement and self.biogt_relbf_dim <= 0:
            raise ValueError(
                "BioGT RelBF deep refinement requires biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_indicator_residual and self.biogt_relbf_dim <= 0:
            raise ValueError(
                "BioGT RelBF indicator residual requires biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_indicator_support_gating and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
        ):
            raise ValueError(
                "BioGT RelBF indicator support gating requires an enabled "
                "indicator residual and biogt_relbf_dim > 0"
            )
        if self.biogt_relbf_indicator_semantic_readout and (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
        ):
            raise ValueError(
                "BioGT RelBF indicator semantic readout requires an enabled "
                "indicator residual and biogt_relbf_dim > 0"
            )
        if (
            self.biogt_relbf_indicator_semantic_readout
            and self.biogt_relbf_indicator_support_gating
        ):
            raise ValueError(
                "BioGT RelBF indicator semantic readout cannot be combined "
                "with the frozen support-gating axis"
            )
        if self.biogt_relbf_indicator_residual and (
            self.biogt_relbf_jk_readout
            or self.biogt_relbf_edge_attention
            or self.biogt_relbf_deep_refinement
        ):
            raise ValueError(
                "BioGT RelBF indicator residual cannot be combined with the "
                "frozen JK, edge-attention, or deep-refinement axes"
            )
        if self.biogt_relbf_deep_refinement and (
            self.biogt_relbf_jk_readout
            or self.biogt_relbf_edge_attention
        ):
            raise ValueError(
                "BioGT RelBF deep refinement cannot be combined with the "
                "frozen JK or edge-attention axes"
            )
        if self.biogt_neural_path_dim > 0 and self.biogt_relbf_dim > 0:
            raise ValueError(
                "BioGT neural diffusion and RelBF are mutually exclusive"
            )
        if (
            self.biogt_neural_path_dim > 0
            and self.biogt_path_propagation_hops <= 0
        ):
            raise ValueError(
                "BioGT neural diffusion requires path propagation"
            )
        if (
            self.biogt_relbf_dim > 0
            and self.biogt_path_propagation_hops != 5
        ):
            raise ValueError(
                "BioGT RelBF requires the frozen five-hop scalar path parent"
            )
        if (
            self.biogt_relbf_dim > 0
            and self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF is a pure BioGT branch and cannot be combined "
                "with a KGE or bilinear scorer"
            )
        if (
            (
                self.kge_path_cross_modal
                or self.kge_path_cross_modal_types
                or self.kge_path_cross_modal_support_types
            )
            and self.kge_path_propagation_hops <= 0
        ):
            raise ValueError(
                "Cross-modal path propagation requires path propagation"
            )
        unknown_cross_modal_types = (
            self.kge_path_cross_modal_types - self.target_node_types
        )
        if unknown_cross_modal_types:
            raise ValueError(
                "Unknown cross-modal path target types: "
                f"{sorted(unknown_cross_modal_types)}"
            )
        unknown_support_targets = (
            set(self.kge_path_cross_modal_support_types)
            - self.target_node_types
        )
        unknown_support_modalities = (
            set().union(
                *self.kge_path_cross_modal_support_types.values()
            )
            - self.target_node_types
            if self.kge_path_cross_modal_support_types
            else set()
        )
        empty_support_targets = sorted(
            target
            for target, supports in (
                self.kge_path_cross_modal_support_types.items()
            )
            if not supports
        )
        if unknown_support_targets or unknown_support_modalities:
            raise ValueError(
                "Unknown cross-modal path support targets/modalities: "
                f"targets={sorted(unknown_support_targets)}, "
                f"supports={sorted(unknown_support_modalities)}"
            )
        if empty_support_targets:
            raise ValueError(
                "Cross-modal path support sets must be nonempty: "
                f"{empty_support_targets}"
            )
        unknown_biogt_support_targets = (
            set(self.biogt_path_support_types)
            - self.target_node_types
        )
        unknown_biogt_support_modalities = (
            set().union(*self.biogt_path_support_types.values())
            - self.target_node_types
            if self.biogt_path_support_types
            else set()
        )
        empty_biogt_support_targets = sorted(
            target
            for target, supports in self.biogt_path_support_types.items()
            if not supports
        )
        if (
            unknown_biogt_support_targets
            or unknown_biogt_support_modalities
        ):
            raise ValueError(
                "Unknown BioGT path support targets/modalities: "
                f"targets={sorted(unknown_biogt_support_targets)}, "
                f"supports={sorted(unknown_biogt_support_modalities)}"
            )
        if empty_biogt_support_targets:
            raise ValueError(
                "BioGT path support sets must be nonempty: "
                f"{empty_biogt_support_targets}"
            )
        if (
            self.biogt_path_support_types
            and self.biogt_path_propagation_hops <= 0
        ):
            raise ValueError(
                "BioGT path support types require path propagation"
            )
        if (
            self.biogt_path_propagation_hops > 0
            and self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "The BioGT-native path branch requires a pure contextual "
                "scorer (concat_mlp or interaction_mlp)"
            )
        self.global_bilinear_score_weight = float(
            global_bilinear_score_weight
        )
        self.contextual_score_weight = float(contextual_score_weight)
        if self.global_bilinear_score_weight < 0:
            raise ValueError(
                "global_bilinear_score_weight must be non-negative"
            )
        if self.contextual_score_weight < 0:
            raise ValueError("contextual_score_weight must be non-negative")
        if self.scorer_mode not in {
            "concat_mlp",
            "interaction_mlp",
            "bilinear",
            "hybrid_bilinear",
            "hybrid_global_bilinear",
            "hybrid_kge",
        }:
            raise ValueError(
                "scorer_mode must be concat_mlp, interaction_mlp, "
                "bilinear, hybrid_bilinear, hybrid_global_bilinear, "
                "or hybrid_kge"
            )
        if self.kge_pretrain_scope not in {
            "target_relations",
            "all_train_relations",
        }:
            raise ValueError(
                "kge_pretrain_scope must be target_relations or "
                "all_train_relations"
            )
        if self.kge_relation_model not in {
            "distmult",
            "lowrank_bilinear",
        }:
            raise ValueError(
                "kge_relation_model must be distmult or lowrank_bilinear"
            )
        if self.kge_relation_rank <= 0:
            raise ValueError("kge_relation_rank must be positive")
        if self.kge_context_input_weight < 0:
            raise ValueError("kge_context_input_weight must be non-negative")
        if self.kge_target_disease_shared_weight < 0:
            raise ValueError(
                "kge_target_disease_shared_weight must be non-negative"
            )
        if (
            self.kge_target_disease_shared_weight > 0
            and not self.kge_target_specific_disease_embeddings
        ):
            raise ValueError(
                "A shared disease KGE weight requires target-specific "
                "disease embeddings"
            )
        if (
            (
                self.kge_relation_model != "distmult"
                or self.kge_context_input_weight > 0
                or self.kge_target_specific_disease_embeddings
                or self.kge_candidate_bias
                or self.kge_target_residual_rank > 0
                or self.kge_path_propagation_hops > 0
            )
            and self.scorer_mode != "hybrid_kge"
        ):
            raise ValueError(
                "KGE relation and contextual-input controls require "
                "scorer_mode=hybrid_kge"
            )
        if (
            self.kge_target_specific_disease_embeddings
            and "disease" not in self.ntypes
        ):
            raise ValueError(
                "Target-specific disease KGE embeddings require a disease "
                "node type"
            )
        if (
            self.kge_pretrain_scope == "all_train_relations"
            and self.scorer_mode != "hybrid_kge"
        ):
            raise ValueError(
                "all_train_relations pretraining requires "
                "scorer_mode=hybrid_kge"
            )

        self.modality_fusion_modes = parse_modality_fusion_modes(
            modality_fusion_modes,
            self.target_node_types,
            fallback=fusion_mode if fusion_mode in {"weighted", "concat", "hierarchical", "sequential"} else "weighted",
        )

        self.node_embeddings = nn.ModuleDict()
        self.input_adapters = nn.ModuleDict()
        self.embedding_dimensions = {}
        for node_type in self.ntypes:
            initial_weight = None
            if node_feature_embeddings and node_type in node_feature_embeddings:
                candidate = node_feature_embeddings[node_type]
                if isinstance(candidate, nn.Embedding):
                    candidate = candidate.weight.detach()
                if isinstance(candidate, torch.Tensor):
                    initial_weight = candidate.detach().float().cpu()

            if initial_weight is None:
                node_count = int(num_nodes_dict[node_type])
                initial_weight = torch.empty(node_count, self.hidden_dim)
                nn.init.xavier_uniform_(initial_weight)

            node_count, input_dim = initial_weight.shape
            embedding = nn.Embedding(
                node_count,
                input_dim,
                _weight=initial_weight,
                sparse=True,
            )
            embedding.weight.requires_grad = not freeze_pretrained
            self.node_embeddings[node_type] = embedding
            self.embedding_dimensions[node_type] = input_dim
            self.input_adapters[node_type] = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.kge_node_embeddings = nn.ModuleDict()
        self.kge_target_disease_embeddings = nn.ModuleDict()
        self.kge_candidate_biases = nn.ModuleDict()
        if self.scorer_mode == "hybrid_kge":
            for node_type in self.ntypes:
                embedding = nn.Embedding(
                    int(num_nodes_dict[node_type]),
                    self.hidden_dim,
                    sparse=True,
                )
                nn.init.xavier_uniform_(embedding.weight)
                self.kge_node_embeddings[node_type] = embedding
            if self.kge_target_specific_disease_embeddings:
                disease_count = int(num_nodes_dict["disease"])
                for target_type in sorted(self.target_node_types):
                    embedding = nn.Embedding(
                        disease_count,
                        self.hidden_dim,
                        sparse=True,
                    )
                    nn.init.xavier_uniform_(embedding.weight)
                    self.kge_target_disease_embeddings[
                        target_type
                    ] = embedding
            if self.kge_candidate_bias:
                for target_type in sorted(self.target_node_types):
                    embedding = nn.Embedding(
                        int(num_nodes_dict[target_type]),
                        1,
                        sparse=True,
                    )
                    nn.init.zeros_(embedding.weight)
                    self.kge_candidate_biases[target_type] = embedding

        if self.disable_hgcn:
            self.hgnn = None
        else:
            self.hgnn = SharedRelationalHGCN(
                self.ntypes,
                self.canonical_etypes,
                self.hidden_dim,
                hgcn_hidden,
                self.hidden_dim,
                self.num_gnn_layers,
                graph_backend=self.graph_backend,
            )

        self.sparse_transformer = QueryCenteredRelationAwareTransformer(
            self.ntypes,
            self.canonical_etypes,
            self.hidden_dim,
            transformer_heads,
            transformer_ff,
            0 if self.disable_transformer else num_transformer_layers,
            context_topk=sparse_transformer_topk,
            max_hops=sparse_transformer_max_hops,
            hop_penalty=sparse_transformer_hop_penalty,
            evidence_weight=sparse_transformer_evidence_weight,
            dropout=dropout,
        )

        self.modality_adapters = nn.ModuleDict(
            {
                target_type: ModalityAdapter(
                    self.hidden_dim,
                    self.modality_fusion_modes[target_type],
                    dropout=dropout,
                )
                for target_type in sorted(self.target_node_types)
            }
        )

        self.bilinear_relations = nn.ParameterDict()
        self.bilinear_biases = nn.ParameterDict()
        self.kge_target_left_factors = nn.ParameterDict()
        self.kge_target_right_factors = nn.ParameterDict()
        self.kge_target_residual_left_factors = nn.ParameterDict()
        self.kge_target_residual_right_factors = nn.ParameterDict()
        self.kge_path_query_gates = nn.ModuleDict()
        self.biogt_path_query_gates = nn.ModuleDict()
        self.biogt_neural_path_query_projections = nn.ModuleDict()
        self.biogt_neural_path_candidate_projections = nn.ModuleDict()
        self.biogt_neural_path_relation_transforms = nn.ModuleDict()
        self.biogt_neural_path_state_norms = nn.ModuleDict()
        self.biogt_relbf_query_projections = nn.ModuleDict()
        self.biogt_relbf_candidate_projections = nn.ModuleDict()
        self.biogt_relbf_relation_tokens = nn.ParameterDict()
        self.biogt_relbf_output_films = nn.ModuleDict()
        self.biogt_relbf_readouts = nn.ModuleDict()
        self.biogt_relbf_jk_readouts = nn.ModuleDict()
        self.biogt_relbf_edge_attention_queries = nn.ModuleDict()
        self.biogt_relbf_message_transforms = nn.ModuleDict()
        self.biogt_relbf_pna_updates = nn.ModuleList()
        self.biogt_relbf_state_norms = nn.ModuleList()
        self.biogt_relbf_deep_message_transforms = nn.ModuleDict()
        self.biogt_relbf_deep_pna_updates = nn.ModuleDict()
        self.biogt_relbf_deep_state_norms = nn.ModuleDict()
        self.biogt_relbf_deep_output_films = nn.ModuleDict()
        self.biogt_relbf_deep_readouts = nn.ModuleDict()
        self.biogt_relbf_indicator_tokens = nn.ParameterDict()
        self.biogt_relbf_indicator_message_transforms = nn.ModuleDict()
        self.biogt_relbf_indicator_pna_updates = nn.ModuleDict()
        self.biogt_relbf_indicator_state_norms = nn.ModuleDict()
        self.biogt_relbf_indicator_readouts = nn.ModuleDict()
        self.biogt_relbf_indicator_support_gates = nn.ModuleDict()
        self.biogt_relbf_indicator_semantic_readouts = nn.ModuleDict()
        self.biogt_relbf_path_layer_attention_residuals = nn.ModuleDict()
        self.biogt_relbf_path_token_transformer_residuals = nn.ModuleDict()
        self.biogt_relbf_candidate_competition_boundary_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_parent_top10_boundary_comparator_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_boundary_window_rank_slot_expert_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_boundary_exchange_residuals = nn.ModuleDict()
        self.biogt_relbf_challenger_boundary_promotion_residuals = nn.ModuleDict()
        self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals = nn.ModuleDict()
        self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals = nn.ModuleDict()
        self.biogt_relbf_multi_depth_score_router_residuals = nn.ModuleDict()
        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_query_relation_adapters = nn.ModuleDict()
        self.biogt_relbf_indicator_query_boundary_adapters = nn.ModuleDict()
        self.biogt_relbf_indicator_multirule_residuals = nn.ModuleDict()
        self.biogt_relbf_indicator_relation_transition_residuals = (
            nn.ModuleDict()
        )
        self.biogt_relbf_indicator_query_local_relation_transition_residuals = (
            nn.ModuleDict()
        )
        if self.scorer_mode in {
            "bilinear",
            "hybrid_bilinear",
            "hybrid_global_bilinear",
            "hybrid_kge",
        }:
            for target_type in sorted(self.target_node_types):
                relation = nn.Parameter(torch.empty(self.hidden_dim))
                nn.init.xavier_uniform_(relation.unsqueeze(0))
                self.bilinear_relations[target_type] = relation
                self.bilinear_biases[target_type] = nn.Parameter(
                    torch.zeros(1)
                )
                if (
                    self.scorer_mode == "hybrid_kge"
                    and self.kge_relation_model == "lowrank_bilinear"
                ):
                    left = nn.Parameter(
                        torch.empty(
                            self.hidden_dim,
                            self.kge_relation_rank,
                        )
                    )
                    right = nn.Parameter(
                        torch.empty(
                            self.hidden_dim,
                            self.kge_relation_rank,
                        )
                    )
                    nn.init.xavier_uniform_(left)
                    nn.init.xavier_uniform_(right)
                    self.kge_target_left_factors[target_type] = left
                    self.kge_target_right_factors[target_type] = right
                if (
                    self.scorer_mode == "hybrid_kge"
                    and self.kge_target_residual_rank > 0
                ):
                    residual_left = nn.Parameter(
                        torch.zeros(
                            self.hidden_dim,
                            self.kge_target_residual_rank,
                        )
                    )
                    residual_right = nn.Parameter(
                        torch.empty(
                            self.hidden_dim,
                            self.kge_target_residual_rank,
                        )
                    )
                    nn.init.xavier_uniform_(residual_right)
                    self.kge_target_residual_left_factors[
                        target_type
                    ] = residual_left
                    self.kge_target_residual_right_factors[
                        target_type
                    ] = residual_right
                if (
                    self.scorer_mode == "hybrid_kge"
                    and self.kge_path_propagation_hops > 0
                ):
                    hop_count = (
                        1
                        if self.kge_path_propagation_hops == 3
                        else 2
                    )
                    gate = nn.Linear(
                        self.hidden_dim,
                        hop_count,
                        bias=True,
                    )
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
                    self.kge_path_query_gates[target_type] = gate
        if self.biogt_path_propagation_hops > 0:
            for target_type in sorted(self.target_node_types):
                supports = self._biogt_path_support_modalities(
                    target_type
                )
                feature_count = len(supports)
                if self.biogt_path_propagation_hops == 5:
                    feature_count += len(supports) ** 2
                gate = nn.Linear(
                    self.hidden_dim,
                    feature_count,
                    bias=True,
                )
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                self.biogt_path_query_gates[target_type] = gate
                if self.biogt_neural_path_dim > 0:
                    query_projection = nn.Linear(
                        self.hidden_dim,
                        self.biogt_neural_path_dim,
                        bias=False,
                    )
                    nn.init.xavier_uniform_(query_projection.weight)
                    self.biogt_neural_path_query_projections[
                        target_type
                    ] = query_projection

                    candidate_projection = nn.Linear(
                        self.hidden_dim,
                        self.biogt_neural_path_dim,
                        bias=False,
                    )
                    nn.init.zeros_(candidate_projection.weight)
                    self.biogt_neural_path_candidate_projections[
                        target_type
                    ] = candidate_projection

                    transforms = nn.ModuleDict()
                    for support in supports:
                        transform = nn.Linear(
                            self.biogt_neural_path_dim,
                            self.biogt_neural_path_dim,
                            bias=False,
                        )
                        nn.init.eye_(transform.weight)
                        transforms[support] = transform
                    self.biogt_neural_path_relation_transforms[
                        target_type
                    ] = transforms

                    cycle_count = (
                        1
                        if self.biogt_path_propagation_hops == 3
                        else 2
                    )
                    self.biogt_neural_path_state_norms[
                        target_type
                    ] = nn.ModuleList([
                        nn.LayerNorm(self.biogt_neural_path_dim)
                        for _ in range(cycle_count)
                    ])
                if self.biogt_relbf_dim > 0:
                    query_projection = nn.Linear(
                        self.hidden_dim,
                        self.biogt_relbf_dim,
                        bias=False,
                    )
                    nn.init.xavier_uniform_(query_projection.weight)
                    self.biogt_relbf_query_projections[
                        target_type
                    ] = query_projection

                    candidate_projection = nn.Linear(
                        self.hidden_dim,
                        self.biogt_relbf_dim,
                        bias=False,
                    )
                    nn.init.xavier_uniform_(candidate_projection.weight)
                    self.biogt_relbf_candidate_projections[
                        target_type
                    ] = candidate_projection

                    relation_token = nn.Parameter(
                        torch.empty(self.biogt_relbf_dim)
                    )
                    nn.init.normal_(
                        relation_token,
                        mean=0.0,
                        std=self.biogt_relbf_dim ** -0.5,
                    )
                    self.biogt_relbf_relation_tokens[
                        target_type
                    ] = relation_token

                    films = nn.ModuleList([
                        nn.Linear(
                            self.biogt_relbf_dim,
                            2 * self.biogt_relbf_dim,
                        )
                        for _ in range(3)
                    ])
                    self.biogt_relbf_output_films[target_type] = films

                    readout = nn.Sequential(
                        nn.Linear(
                            3 * self.biogt_relbf_dim,
                            self.biogt_relbf_dim,
                        ),
                        nn.GELU(),
                        nn.Linear(self.biogt_relbf_dim, 1),
                    )
                    nn.init.zeros_(readout[-1].weight)
                    nn.init.zeros_(readout[-1].bias)
                    self.biogt_relbf_readouts[target_type] = readout
                    if self.biogt_relbf_jk_readout:
                        jk_readout = nn.Sequential(
                            nn.Linear(
                                4 * self.biogt_relbf_dim,
                                self.biogt_relbf_dim,
                            ),
                            nn.GELU(),
                            nn.Linear(self.biogt_relbf_dim, 1),
                        )
                        nn.init.zeros_(jk_readout[-1].weight)
                        nn.init.zeros_(jk_readout[-1].bias)
                        self.biogt_relbf_jk_readouts[
                            target_type
                        ] = jk_readout
                    if self.biogt_relbf_path_layer_attention_rank > 0:
                        self.biogt_relbf_path_layer_attention_residuals[
                            target_type
                        ] = CandidateConditionedPathLayerAttentionResidual(
                            self.biogt_relbf_dim,
                            self.biogt_relbf_path_layer_attention_rank,
                        )
                    if self.biogt_relbf_path_token_transformer:
                        self.biogt_relbf_path_token_transformer_residuals[
                            target_type
                        ] = CandidateConditionedPathTokenTransformerResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_candidate_competition_boundary:
                        self.biogt_relbf_candidate_competition_boundary_residuals[
                            target_type
                        ] = CandidateCompetitionBoundaryResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_parent_top10_boundary_comparator:
                        self.biogt_relbf_parent_top10_boundary_comparator_residuals[
                            target_type
                        ] = ParentTopBoundarySlotComparatorResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_boundary_window_rank_slot_expert:
                        self.biogt_relbf_boundary_window_rank_slot_expert_residuals[
                            target_type
                        ] = BoundaryWindowRankSlotExpertResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_boundary_exchange:
                        self.biogt_relbf_boundary_exchange_residuals[
                            target_type
                        ] = BoundaryExchangeResidual(self.biogt_relbf_dim)
                    if self.biogt_relbf_challenger_boundary_promotion:
                        self.biogt_relbf_challenger_boundary_promotion_residuals[
                            target_type
                        ] = ChallengerBoundaryPromotionResidual(self.biogt_relbf_dim)
                    if self.biogt_relbf_gatekeeper_challenger_paired_margin:
                        self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals[
                            target_type
                        ] = GatekeeperChallengerPairedMarginResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_boundary_prototype_ordinal_adapter:
                        self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals[
                            target_type
                        ] = BoundaryPrototypeOrdinalResidual(self.biogt_relbf_dim)
                    if self.biogt_relbf_multi_depth_score_router:
                        self.biogt_relbf_multi_depth_score_router_residuals[
                            target_type
                        ] = QueryConditionedMultiDepthScoreRouterResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_dual_stream_trajectory_agreement_calibrator:
                        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals[
                            target_type
                        ] = DualStreamTrajectoryAgreementCalibratorResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_multi_anchor_rank_landmark_calibrator:
                        self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals[
                            target_type
                        ] = QueryConditionedMultiAnchorRankLandmarkCalibratorResidual(
                            self.biogt_relbf_dim
                        )
                    if self.biogt_relbf_query_token_top64_candidate_set_transformer:
                        self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals[
                            target_type
                        ] = QueryTokenTop64CandidateSetTransformerResidual(
                            self.biogt_relbf_dim,
                            self.hidden_dim,
                        )
                    if self.biogt_relbf_edge_attention:
                        support_attention = nn.ModuleDict()
                        for support in sorted(self.target_node_types):
                            projections = nn.ModuleList([
                                nn.Linear(
                                    self.biogt_relbf_dim,
                                    self.biogt_relbf_dim,
                                    bias=False,
                                )
                                for _ in range(3)
                            ])
                            for projection in projections:
                                nn.init.zeros_(projection.weight)
                            support_attention[support] = projections
                        self.biogt_relbf_edge_attention_queries[
                            target_type
                        ] = support_attention
                    if self.biogt_relbf_deep_refinement:
                        deep_transforms = nn.ModuleDict()
                        for support in supports:
                            deep_transforms[support] = nn.ModuleList([
                                nn.Linear(
                                    self.biogt_relbf_dim,
                                    self.biogt_relbf_dim,
                                    bias=False,
                                )
                                for _ in range(2)
                            ])
                        self.biogt_relbf_deep_message_transforms[
                            target_type
                        ] = deep_transforms
                        self.biogt_relbf_deep_pna_updates[target_type] = (
                            nn.ModuleList([
                                nn.Linear(
                                    12 * self.biogt_relbf_dim,
                                    self.biogt_relbf_dim,
                                )
                                for _ in range(2)
                            ])
                        )
                        self.biogt_relbf_deep_state_norms[target_type] = (
                            nn.ModuleList([
                                nn.LayerNorm(self.biogt_relbf_dim)
                                for _ in range(2)
                            ])
                        )
                        self.biogt_relbf_deep_output_films[target_type] = (
                            nn.ModuleList([
                                nn.Linear(
                                    self.biogt_relbf_dim,
                                    2 * self.biogt_relbf_dim,
                                )
                                for _ in range(2)
                            ])
                        )
                        deep_readout = nn.Sequential(
                            nn.Linear(
                                4 * self.biogt_relbf_dim,
                                self.biogt_relbf_dim,
                            ),
                            nn.GELU(),
                            nn.Linear(self.biogt_relbf_dim, 1),
                        )
                        nn.init.zeros_(deep_readout[-1].weight)
                        nn.init.zeros_(deep_readout[-1].bias)
                        self.biogt_relbf_deep_readouts[
                            target_type
                        ] = deep_readout
                    if self.biogt_relbf_indicator_residual:
                        indicator_token = nn.Parameter(
                            torch.empty(self.biogt_relbf_dim)
                        )
                        nn.init.normal_(
                            indicator_token,
                            mean=0.0,
                            std=self.biogt_relbf_dim ** -0.5,
                        )
                        self.biogt_relbf_indicator_tokens[
                            target_type
                        ] = indicator_token
                        indicator_transforms = nn.ModuleDict()
                        for support in supports:
                            indicator_transforms[support] = nn.ModuleList([
                                nn.Linear(
                                    self.biogt_relbf_dim,
                                    self.biogt_relbf_dim,
                                    bias=False,
                                )
                                for _ in range(3)
                            ])
                        self.biogt_relbf_indicator_message_transforms[
                            target_type
                        ] = indicator_transforms
                        self.biogt_relbf_indicator_pna_updates[
                            target_type
                        ] = nn.ModuleList([
                            nn.Linear(
                                12 * self.biogt_relbf_dim,
                                self.biogt_relbf_dim,
                            )
                            for _ in range(3)
                        ])
                        self.biogt_relbf_indicator_state_norms[
                            target_type
                        ] = nn.ModuleList([
                            nn.LayerNorm(self.biogt_relbf_dim)
                            for _ in range(3)
                        ])
                        indicator_readout = nn.Sequential(
                            nn.Linear(
                                3 * self.biogt_relbf_dim,
                                self.biogt_relbf_dim,
                            ),
                            nn.GELU(),
                            nn.Linear(self.biogt_relbf_dim, 1),
                        )
                        nn.init.zeros_(indicator_readout[-1].weight)
                        nn.init.zeros_(indicator_readout[-1].bias)
                        self.biogt_relbf_indicator_readouts[
                            target_type
                        ] = indicator_readout
                        if self.biogt_relbf_indicator_query_boundary_rank > 0:
                            self.biogt_relbf_indicator_query_boundary_adapters[
                                target_type
                            ] = QueryConditionedIndicatorBoundaryAdapter(
                                self.biogt_relbf_dim,
                                self.biogt_relbf_indicator_query_boundary_rank,
                            )
                        if self.biogt_relbf_indicator_multirule_count > 0:
                            self.biogt_relbf_indicator_multirule_residuals[
                                target_type
                            ] = QueryConditionedIndicatorMultiRuleResidual(
                                self.biogt_relbf_dim,
                                self.biogt_relbf_indicator_multirule_count,
                                self.biogt_relbf_indicator_multirule_readout_rank,
                            )
                        if (
                            self.biogt_relbf_indicator_relation_transition_layers
                            > 0
                        ):
                            relation_count = 2 * len(supports)
                            self.biogt_relbf_indicator_relation_transition_residuals[
                                target_type
                            ] = QueryConditionedRelationTransitionResidual(
                                self.biogt_relbf_dim,
                                relation_count,
                                self.biogt_relbf_indicator_relation_transition_rank,
                                self.biogt_relbf_indicator_relation_transition_layers,
                            )
                        if (
                            self.biogt_relbf_indicator_query_local_relation_transition_layers
                            > 0
                        ):
                            relation_count = 2 * len(supports)
                            self.biogt_relbf_indicator_query_local_relation_transition_residuals[
                                target_type
                            ] = QueryLocalRelationTransitionResidual(
                                self.biogt_relbf_dim,
                                relation_count,
                                self.biogt_relbf_indicator_query_local_relation_transition_rank,
                                self.biogt_relbf_indicator_query_local_relation_transition_layers,
                            )
                        if self.biogt_relbf_indicator_semantic_readout:
                            semantic_readout = nn.Sequential(
                                nn.Linear(
                                    4 * self.biogt_relbf_dim,
                                    self.biogt_relbf_dim,
                                ),
                                nn.GELU(),
                                nn.Linear(self.biogt_relbf_dim, 1),
                            )
                            nn.init.zeros_(semantic_readout[-1].weight)
                            nn.init.zeros_(semantic_readout[-1].bias)
                            self.biogt_relbf_indicator_semantic_readouts[
                                target_type
                            ] = semantic_readout
                        if self.biogt_relbf_indicator_support_gating:
                            support_gates = nn.ModuleDict()
                            for support in supports:
                                layer_gates = nn.ModuleList([
                                    nn.Linear(
                                        self.biogt_relbf_dim,
                                        1,
                                        bias=True,
                                    )
                                    for _ in range(3)
                                ])
                                for gate in layer_gates:
                                    nn.init.zeros_(gate.weight)
                                    nn.init.zeros_(gate.bias)
                                support_gates[support] = layer_gates
                            self.biogt_relbf_indicator_support_gates[
                                target_type
                            ] = support_gates
                    if self.biogt_relbf_query_relation_adapter_rank > 0:
                        target_adapters = nn.ModuleDict()
                        for support in supports:
                            target_adapters[support] = nn.ModuleList([
                                QueryConditionedLowRankRelationAdapter(
                                    self.biogt_relbf_dim,
                                    self.biogt_relbf_query_relation_adapter_rank,
                                )
                                for _ in range(3)
                            ])
                        self.biogt_relbf_query_relation_adapters[
                            target_type
                        ] = target_adapters

            if self.biogt_relbf_dim > 0:
                for support in sorted(self.target_node_types):
                    self.biogt_relbf_message_transforms[
                        support
                    ] = nn.ModuleList([
                        nn.Linear(
                            self.biogt_relbf_dim,
                            self.biogt_relbf_dim,
                            bias=False,
                        )
                        for _ in range(3)
                    ])
                self.biogt_relbf_pna_updates = nn.ModuleList([
                    nn.Linear(
                        12 * self.biogt_relbf_dim,
                        self.biogt_relbf_dim,
                    )
                    for _ in range(3)
                ])
                self.biogt_relbf_state_norms = nn.ModuleList([
                    nn.LayerNorm(self.biogt_relbf_dim)
                    for _ in range(3)
                ])
        self.kge_pretrain_relation_keys = {
            etype: f"relation_{index:03d}"
            for index, etype in enumerate(sorted(self.canonical_etypes))
        }
        self.kge_pretrain_relations = nn.ParameterDict()
        self.kge_pretrain_biases = nn.ParameterDict()
        self.kge_pretrain_left_factors = nn.ParameterDict()
        self.kge_pretrain_right_factors = nn.ParameterDict()
        if (
            self.scorer_mode == "hybrid_kge"
            and self.kge_pretrain_scope == "all_train_relations"
        ):
            for relation_key in self.kge_pretrain_relation_keys.values():
                relation = nn.Parameter(torch.empty(self.hidden_dim))
                nn.init.xavier_uniform_(relation.unsqueeze(0))
                self.kge_pretrain_relations[relation_key] = relation
                self.kge_pretrain_biases[relation_key] = nn.Parameter(
                    torch.zeros(1)
                )
                if self.kge_relation_model == "lowrank_bilinear":
                    left = nn.Parameter(
                        torch.empty(
                            self.hidden_dim,
                            self.kge_relation_rank,
                        )
                    )
                    right = nn.Parameter(
                        torch.empty(
                            self.hidden_dim,
                            self.kge_relation_rank,
                        )
                    )
                    nn.init.xavier_uniform_(left)
                    nn.init.xavier_uniform_(right)
                    self.kge_pretrain_left_factors[relation_key] = left
                    self.kge_pretrain_right_factors[relation_key] = right

        pair_dim = self.hidden_dim * (
            4
            if self.scorer_mode in {
                "interaction_mlp",
                "hybrid_bilinear",
                "hybrid_global_bilinear",
                "hybrid_kge",
            }
            else 2
        )
        hidden_dim = int(moe_hidden_dim or self.hidden_dim * 2)
        if self.scorer_mode == "bilinear":
            self.expert_predictors = None
            self.dense_predictors = None
        elif self.disable_moe:
            self.expert_predictors = None
            self.dense_predictors = nn.ModuleDict(
                {
                    target_type: nn.Sequential(
                        nn.Linear(pair_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    for target_type in sorted(self.target_node_types)
                }
            )
        else:
            self.expert_predictors = nn.ModuleDict(
                {
                    target_type: SparseRouterMoE(
                        pair_dim,
                        hidden_dim,
                        num_experts=moe_num_experts,
                        top_k=moe_top_k,
                        dropout=dropout,
                    )
                    for target_type in sorted(self.target_node_types)
                }
            )
            self.dense_predictors = None

        self._last_moe_aux_loss = None
        self._last_global_pos_scores = {}
        self._last_global_neg_scores = {}
        self._embedding_cache = OrderedDict()
        self._embedding_cache_size = max(1, int(embedding_cache_size))
        self._cache_generation = 0
        self.independent_branch_frozen = False
        self.shared_kge_frozen_for_target_disease_finetune = False
        self.target_disease_finetune_types = []
        self.target_disease_initialized_from_shared_checkpoint = False
        self.kge_frozen_for_candidate_bias_finetune = False
        self.candidate_bias_finetune_types = []
        self.candidate_bias_initialized_from_checkpoint = False
        self.kge_frozen_for_target_residual_finetune = False
        self.target_residual_finetune_types = []
        self.target_residual_initialized_from_checkpoint = False
        self.kge_frozen_for_path_propagation_finetune = False
        self.path_propagation_finetune_types = []
        self.path_propagation_initialized_from_checkpoint = False
        self._kge_path_adjacencies = {}
        self._kge_path_graph_metadata = {}
        self._biogt_path_adjacencies = {}
        self._biogt_neural_path_adjacencies = {}
        self._biogt_neural_path_candidate_ids = {}
        self._biogt_neural_path_candidate_lookup = {}
        self._biogt_path_graph_metadata = {}
        self._biogt_relbf_relation_transition_adjacencies = {}
        self._biogt_relbf_relation_transition_labels = {}
        self._biogt_relbf_relation_transition_indices = {}
        self._biogt_relbf_relation_transition_attributions = {}
        self._biogt_relbf_query_local_relation_transition_attributions = {}
        self._biogt_relbf_path_layer_attention_attributions = {}
        self._biogt_relbf_path_token_transformer_attributions = {}
        self._biogt_relbf_candidate_competition_boundary_attributions = {}
        self._biogt_relbf_parent_top10_boundary_comparator_attributions = {}
        self._biogt_relbf_boundary_window_rank_slot_expert_attributions = {}
        self._biogt_relbf_boundary_exchange_attributions = {}
        self._biogt_relbf_challenger_boundary_promotion_attributions = {}
        self._biogt_relbf_gatekeeper_challenger_paired_margin_attributions = {}
        self._biogt_relbf_boundary_prototype_ordinal_adapter_attributions = {}
        self._biogt_relbf_multi_depth_score_router_attributions = {}
        self._biogt_relbf_dual_stream_trajectory_agreement_calibrator_attributions = {}
        self._biogt_relbf_multi_anchor_rank_landmark_calibrator_attributions = {}
        self._biogt_relbf_query_token_top64_candidate_set_transformer_attributions = {}
        self.biogt_path_initialized_from_checkpoint = False
        self.biogt_neural_path_initialized_from_checkpoint = False
        self.biogt_relbf_initialized_from_checkpoint = False
        self.biogt_relbf_jk_initialized_from_checkpoint = False
        self.biogt_relbf_edge_attention_initialized_from_checkpoint = False
        self.biogt_relbf_deep_initialized_from_checkpoint = False
        self.biogt_relbf_indicator_initialized_from_checkpoint = False
        self.biogt_relbf_indicator_support_gating_initialized_from_checkpoint = (
            False
        )
        self.biogt_relbf_indicator_semantic_initialized_from_checkpoint = False
        self.biogt_relbf_query_relation_adapter_initialized_from_checkpoint = (
            False
        )
        self.biogt_relbf_indicator_query_boundary_initialized_from_checkpoint = (
            False
        )
        self.biogt_relbf_indicator_multirule_initialized_from_checkpoint = False
        self.biogt_relbf_indicator_relation_transition_initialized_from_checkpoint = (
            False
        )
        self.biogt_relbf_indicator_query_local_relation_transition_initialized_from_checkpoint = (
            False
        )
        self.biogt_relbf_path_layer_attention_initialized_from_checkpoint = False
        self.biogt_relbf_path_token_transformer_initialized_from_checkpoint = False
        self.biogt_relbf_candidate_competition_boundary_initialized_from_checkpoint = False
        self.biogt_relbf_parent_top10_boundary_comparator_initialized_from_checkpoint = False
        self.biogt_relbf_boundary_window_rank_slot_expert_initialized_from_checkpoint = False
        self.biogt_relbf_boundary_exchange_initialized_from_checkpoint = False
        self.biogt_relbf_challenger_boundary_promotion_initialized_from_checkpoint = False
        self.biogt_relbf_gatekeeper_challenger_paired_margin_initialized_from_checkpoint = False
        self.biogt_relbf_boundary_prototype_ordinal_adapter_initialized_from_checkpoint = False
        self.biogt_relbf_multi_depth_score_router_initialized_from_checkpoint = False
        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_initialized_from_checkpoint = False
        self.biogt_relbf_multi_anchor_rank_landmark_calibrator_initialized_from_checkpoint = False
        self.biogt_relbf_query_token_top64_candidate_set_transformer_initialized_from_checkpoint = False
        self.biogt_relbf_deep_migration_provenance = "none"
        self.biogt_relbf_jk_migration_provenance = "none"
        self.biogt_relbf_target_only_finetune = False
        self.biogt_relbf_jk_only_finetune = False
        self.biogt_relbf_edge_attention_only_finetune = False
        self.biogt_relbf_deep_only_finetune = False
        self.biogt_relbf_indicator_only_finetune = False
        self.biogt_relbf_indicator_support_gate_only_finetune = False
        self.biogt_relbf_indicator_semantic_only_finetune = False
        self.biogt_relbf_query_relation_adapter_only_finetune = False
        self.biogt_relbf_indicator_query_boundary_only_finetune = False
        self.biogt_relbf_indicator_multirule_only_finetune = False
        self.biogt_relbf_indicator_relation_transition_only_finetune = False
        self.biogt_relbf_indicator_query_local_relation_transition_only_finetune = (
            False
        )
        self.biogt_relbf_path_layer_attention_only_finetune = False
        self.biogt_relbf_path_token_transformer_only_finetune = False
        self.biogt_relbf_candidate_competition_boundary_only_finetune = False
        self.biogt_relbf_parent_top10_boundary_comparator_only_finetune = False
        self.biogt_relbf_boundary_window_rank_slot_expert_only_finetune = False
        self.biogt_relbf_boundary_exchange_only_finetune = False
        self.biogt_relbf_challenger_boundary_promotion_only_finetune = False
        self.biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune = False
        self.biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune = False
        self.biogt_relbf_multi_depth_score_router_only_finetune = False
        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune = False
        self.biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune = False
        self.biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune = False
        self.biogt_base_frozen_for_path_finetune = False
        self.biogt_path_finetune_types = []

    def set_target_etypes(self, target_etypes):
        self.target_etypes = set(target_etypes)

    def freeze_independent_bilinear_branch(self):
        if self.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Independent branch freezing requires scorer_mode=hybrid_kge"
            )
        frozen_parameters = []
        for embedding in self.kge_node_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_target_disease_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_candidate_biases.values():
            frozen_parameters.append(embedding.weight)
        frozen_parameters.extend(self.bilinear_relations.values())
        frozen_parameters.extend(self.bilinear_biases.values())
        frozen_parameters.extend(self.kge_pretrain_relations.values())
        frozen_parameters.extend(self.kge_pretrain_biases.values())
        frozen_parameters.extend(self.kge_target_left_factors.values())
        frozen_parameters.extend(self.kge_target_right_factors.values())
        frozen_parameters.extend(
            self.kge_target_residual_left_factors.values()
        )
        frozen_parameters.extend(
            self.kge_target_residual_right_factors.values()
        )
        frozen_parameters.extend(self.kge_path_query_gates.parameters())
        frozen_parameters.extend(self.kge_pretrain_left_factors.values())
        frozen_parameters.extend(self.kge_pretrain_right_factors.values())
        for parameter in frozen_parameters:
            parameter.requires_grad_(False)
        self.independent_branch_frozen = True
        return sum(parameter.numel() for parameter in frozen_parameters)

    def configure_train_only_path_graph(self, graph, target_etypes):
        if self.kge_path_propagation_hops <= 0:
            self._kge_path_adjacencies = {}
            self._kge_path_graph_metadata = {}
            return {}
        if self.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Path propagation requires scorer_mode=hybrid_kge"
            )
        if "disease" not in graph.ntypes:
            raise ValueError(
                "Path propagation requires a disease node type"
            )
        device = next(self.parameters()).device
        configured = {}
        for modality in sorted(self.target_node_types):
            pairs = set()
            for etype in target_etypes:
                if {etype[0], etype[2]} != {"disease", modality}:
                    continue
                src_ids, dst_ids = graph.edges(etype=etype)
                if etype[0] == "disease":
                    disease_ids, candidate_ids = src_ids, dst_ids
                else:
                    disease_ids, candidate_ids = dst_ids, src_ids
                pairs.update(zip(
                    disease_ids.cpu().tolist(),
                    candidate_ids.cpu().tolist(),
                ))
            if not pairs:
                raise ValueError(
                    f"No train-only disease-{modality} path edges found"
                )
            ordered_pairs = sorted(pairs)
            disease_ids = torch.tensor(
                [pair[0] for pair in ordered_pairs],
                dtype=torch.long,
                device=device,
            )
            candidate_ids = torch.tensor(
                [pair[1] for pair in ordered_pairs],
                dtype=torch.long,
                device=device,
            )
            disease_degree = torch.bincount(
                disease_ids,
                minlength=self.kge_node_embeddings[
                    "disease"
                ].num_embeddings,
            ).float().clamp_min_(1.0)
            candidate_degree = torch.bincount(
                candidate_ids,
                minlength=self.kge_node_embeddings[
                    modality
                ].num_embeddings,
            ).float().clamp_min_(1.0)
            values = (
                disease_degree.index_select(0, disease_ids).rsqrt()
                * candidate_degree.index_select(
                    0,
                    candidate_ids,
                ).rsqrt()
            )
            adjacency = torch.sparse_coo_tensor(
                torch.stack([disease_ids, candidate_ids]),
                values,
                size=(
                    disease_degree.numel(),
                    candidate_degree.numel(),
                ),
                device=device,
            ).coalesce()
            self._kge_path_adjacencies[modality] = adjacency
            configured[modality] = {
                "source": "train_graph_only",
                "edge_count": int(adjacency._nnz()),
                "disease_count": int(adjacency.shape[0]),
                "candidate_count": int(adjacency.shape[1]),
                "normalization": "symmetric_degree",
                "scored_hops": (
                    [3]
                    if self.kge_path_propagation_hops == 3
                    else [3, 5]
                ),
            }
        for modality, metadata in configured.items():
            uses_cross_modal = self._path_uses_cross_modal(modality)
            metadata["propagation_mode"] = (
                "cross_modal_target_relations"
                if uses_cross_modal
                else "within_modality"
            )
            metadata["support_modalities"] = (
                self._path_support_modalities(modality)
                if uses_cross_modal
                else []
            )
        self._kge_path_graph_metadata = configured
        return configured

    def _biogt_path_support_modalities(self, modality):
        configured = self.biogt_path_support_types.get(modality)
        if configured is not None:
            return sorted(configured)
        return sorted(self.target_node_types)

    def configure_train_only_biogt_path_graph(
        self,
        graph,
        target_etypes,
    ):
        if self.biogt_path_propagation_hops <= 0:
            self._biogt_path_adjacencies = {}
            self._biogt_path_graph_metadata = {}
            return {}
        if self.scorer_mode == "hybrid_kge":
            raise ValueError(
                "BioGT-native path propagation cannot use hybrid_kge"
            )
        if "disease" not in graph.ntypes:
            raise ValueError(
                "BioGT-native path propagation requires a disease node type"
            )

        device = next(self.parameters()).device
        configured = {}
        disease_count = self.node_embeddings["disease"].num_embeddings
        for modality in sorted(self.target_node_types):
            pairs = set()
            for etype in target_etypes:
                if {etype[0], etype[2]} != {"disease", modality}:
                    continue
                src_ids, dst_ids = graph.edges(etype=etype)
                if etype[0] == "disease":
                    disease_ids, candidate_ids = src_ids, dst_ids
                else:
                    disease_ids, candidate_ids = dst_ids, src_ids
                pairs.update(zip(
                    disease_ids.cpu().tolist(),
                    candidate_ids.cpu().tolist(),
                ))
            if not pairs:
                raise ValueError(
                    f"No train-only disease-{modality} BioGT path edges found"
                )
            ordered_pairs = sorted(pairs)
            disease_ids = torch.tensor(
                [pair[0] for pair in ordered_pairs],
                dtype=torch.long,
                device=device,
            )
            candidate_ids = torch.tensor(
                [pair[1] for pair in ordered_pairs],
                dtype=torch.long,
                device=device,
            )
            candidate_count = self.node_embeddings[
                modality
            ].num_embeddings
            disease_degree = torch.bincount(
                disease_ids,
                minlength=disease_count,
            ).float().clamp_min_(1.0)
            candidate_degree = torch.bincount(
                candidate_ids,
                minlength=candidate_count,
            ).float().clamp_min_(1.0)
            values = (
                disease_degree.index_select(0, disease_ids).rsqrt()
                * candidate_degree.index_select(
                    0,
                    candidate_ids,
                ).rsqrt()
            )
            adjacency = torch.sparse_coo_tensor(
                torch.stack([disease_ids, candidate_ids]),
                values,
                size=(disease_count, candidate_count),
                device=device,
            ).coalesce()
            self._biogt_path_adjacencies[modality] = adjacency
            supports = self._biogt_path_support_modalities(modality)
            configured[modality] = {
                "source": "train_graph_only",
                "edge_count": int(adjacency._nnz()),
                "disease_count": int(adjacency.shape[0]),
                "candidate_count": int(adjacency.shape[1]),
                "normalization": "symmetric_degree",
                "propagation_mode": "query_conditioned_typed_metapath",
                "support_modalities": supports,
                "scored_hops": (
                    [3]
                    if self.biogt_path_propagation_hops == 3
                    else [3, 5]
                ),
                "metapath_count": (
                    len(supports)
                    if self.biogt_path_propagation_hops == 3
                    else len(supports) + len(supports) ** 2
                ),
            }
        self._biogt_path_graph_metadata = configured
        return configured

    def configure_train_only_biogt_neural_path_graph(
        self,
        graph,
        target_etypes,
        candidate_nodes_by_etype,
    ):
        if (
            self.biogt_neural_path_dim <= 0
            and self.biogt_relbf_dim <= 0
        ):
            self._biogt_neural_path_adjacencies = {}
            self._biogt_neural_path_candidate_ids = {}
            self._biogt_neural_path_candidate_lookup = {}
            return {}

        device = next(self.parameters()).device
        disease_count = self.node_embeddings["disease"].num_embeddings
        configured = {}
        for modality in sorted(self.target_node_types):
            relation_candidates = []
            pairs = set()
            for etype in target_etypes:
                if {etype[0], etype[2]} != {"disease", modality}:
                    continue
                candidates = (candidate_nodes_by_etype or {}).get(etype)
                if candidates is not None:
                    relation_candidates.extend(
                        torch.as_tensor(
                            candidates,
                            dtype=torch.long,
                        ).cpu().tolist()
                    )
                src_ids, dst_ids = graph.edges(etype=etype)
                if etype[0] == "disease":
                    disease_ids, candidate_ids = src_ids, dst_ids
                else:
                    disease_ids, candidate_ids = dst_ids, src_ids
                pairs.update(zip(
                    disease_ids.cpu().tolist(),
                    candidate_ids.cpu().tolist(),
                ))
            if not relation_candidates:
                raise ValueError(
                    "BioGT neural diffusion requires strict relation "
                    f"candidates for {modality}"
                )
            candidate_ids = torch.tensor(
                sorted(set(relation_candidates)),
                dtype=torch.long,
                device=device,
            )
            full_candidate_count = self.node_embeddings[
                modality
            ].num_embeddings
            lookup = torch.full(
                (full_candidate_count,),
                -1,
                dtype=torch.long,
                device=device,
            )
            lookup[candidate_ids] = torch.arange(
                candidate_ids.numel(),
                dtype=torch.long,
                device=device,
            )

            ordered_pairs = sorted(pairs)
            disease_ids = torch.tensor(
                [pair[0] for pair in ordered_pairs],
                dtype=torch.long,
                device=device,
            )
            original_candidate_ids = torch.tensor(
                [pair[1] for pair in ordered_pairs],
                dtype=torch.long,
                device=device,
            )
            compact_candidate_ids = lookup.index_select(
                0,
                original_candidate_ids,
            )
            in_scope = compact_candidate_ids >= 0
            disease_ids = disease_ids[in_scope]
            compact_candidate_ids = compact_candidate_ids[in_scope]
            if disease_ids.numel() == 0:
                raise ValueError(
                    "BioGT neural diffusion has no train edges inside the "
                    f"strict candidate universe for {modality}"
                )

            values = torch.ones(
                disease_ids.numel(),
                dtype=torch.float32,
                device=device,
            )
            adjacency = torch.sparse_coo_tensor(
                torch.stack([disease_ids, compact_candidate_ids]),
                values,
                size=(disease_count, candidate_ids.numel()),
                device=device,
            ).coalesce()
            self._biogt_neural_path_adjacencies[modality] = adjacency
            self._biogt_neural_path_candidate_ids[modality] = candidate_ids
            self._biogt_neural_path_candidate_lookup[modality] = lookup
            configured[modality] = {
                "source": "train_graph_only",
                "edge_count": int(adjacency._nnz()),
                "disease_count": int(adjacency.shape[0]),
                "candidate_count": int(adjacency.shape[1]),
                "normalization": "binary_query_maskable",
                "propagation_mode": (
                    (
                        "relational_bellman_ford_pna_deep_residual"
                        if self.biogt_relbf_deep_refinement
                        else "relational_bellman_ford_pna"
                    )
                    if self.biogt_relbf_dim > 0
                    else "relation_conditioned_query_neural_diffusion"
                ),
                "hidden_dim": (
                    self.biogt_relbf_dim
                    if self.biogt_relbf_dim > 0
                    else self.biogt_neural_path_dim
                ),
                "layers": (
                    (5 if self.biogt_relbf_deep_refinement else 3)
                    if self.biogt_relbf_dim > 0
                    else self.biogt_path_propagation_hops
                ),
                "query_output_relation_masked": (
                    "train_only_dynamic_forward_and_inverse"
                    if self.biogt_relbf_dim > 0
                    else True
                ),
                "evaluation_topology": "complete_train_graph",
                "support_modalities": (
                    self._biogt_path_support_modalities(modality)
                ),
                "edge_attention": (
                    "query_output_relation_x_support_relation_sigmoid_v1"
                    if self.biogt_relbf_edge_attention
                    else "disabled"
                ),
                "edge_attention_initial_weight": (
                    1.0 if self.biogt_relbf_edge_attention else None
                ),
                "deep_refinement": (
                    "relation_specific_two_layer_pna_residual_v1"
                    if self.biogt_relbf_deep_refinement
                    else "disabled"
                ),
                "deep_refinement_path": (
                    "target_to_disease_to_target"
                    if self.biogt_relbf_deep_refinement
                    else None
                ),
                "deep_refinement_parent_layers": (
                    3 if self.biogt_relbf_deep_refinement else None
                ),
                "indicator_residual": (
                    "relation_only_query_conditioned_bellman_ford_pna_v1"
                    if self.biogt_relbf_indicator_residual
                    else "disabled"
                ),
                "indicator_source_boundary": (
                    "target_relation_token_at_query_disease"
                    if self.biogt_relbf_indicator_residual
                    else None
                ),
                "indicator_parent_path": (
                    "disease_conditioned_three_layer_relbf"
                    if self.biogt_relbf_indicator_residual
                    else None
                ),
                "indicator_support_gating": (
                    "parent_condition_per_support_per_layer_2sigmoid_v1"
                    if self.biogt_relbf_indicator_support_gating
                    else "disabled"
                ),
                "indicator_support_gate_initial_weight": (
                    1.0
                    if self.biogt_relbf_indicator_support_gating
                    else None
                ),
                "indicator_support_gate_modalities": (
                    self._biogt_path_support_modalities(modality)
                    if self.biogt_relbf_indicator_support_gating
                    else []
                ),
                "indicator_semantic_readout": (
                    "indicator_candidate_parent_condition_4way_residual_v1"
                    if self.biogt_relbf_indicator_semantic_readout
                    else "disabled"
                ),
                "indicator_semantic_parent_condition": (
                    "frozen_disease_conditioned_relbf_query_projection"
                    if self.biogt_relbf_indicator_semantic_readout
                    else "disabled"
                ),
                "query_relation_adapter": (
                    "support_relation_layer_low_rank_query_gated_delta_v1"
                    if self.biogt_relbf_query_relation_adapter_rank > 0
                    else "disabled"
                ),
                "query_relation_adapter_rank": (
                    self.biogt_relbf_query_relation_adapter_rank
                ),
                "query_relation_adapter_source": "train_graph_only",
                "query_relation_adapter_attribution": (
                    "relation_support_layer_delta_preserved_in_relbf_paths"
                    if self.biogt_relbf_query_relation_adapter_rank > 0
                    else "disabled"
                ),
                "indicator_query_boundary": (
                    "low_rank_frozen_parent_condition_source_delta_v1"
                    if self.biogt_relbf_indicator_query_boundary_rank > 0
                    else "disabled"
                ),
                "indicator_query_boundary_rank": (
                    self.biogt_relbf_indicator_query_boundary_rank
                ),
                "indicator_query_boundary_source": "train_graph_only",
                "indicator_query_boundary_condition": (
                    "detached_parent_relbf_condition"
                    if self.biogt_relbf_indicator_query_boundary_rank > 0
                    else "disabled"
                ),
                "indicator_query_boundary_attribution": (
                    "query_disease_source_delta_preserved_in_indicator_paths"
                    if self.biogt_relbf_indicator_query_boundary_rank > 0
                    else "disabled"
                ),
                "indicator_multirule": (
                    "independent_rule_source_boundaries_shared_indicator_pna_v1"
                    if self.biogt_relbf_indicator_multirule_count > 0
                    else "disabled"
                ),
                "indicator_multirule_count": (
                    self.biogt_relbf_indicator_multirule_count
                ),
                "indicator_multirule_gate": (
                    "softmax_detached_parent_relbf_condition"
                    if self.biogt_relbf_indicator_multirule_count > 0
                    else "disabled"
                ),
                "indicator_multirule_rule_labels": [
                    f"rule_{index}"
                    for index in range(
                        self.biogt_relbf_indicator_multirule_count
                    )
                ],
                "indicator_multirule_gate_attribution": {
                    f"rule_{index}": index
                    for index in range(
                        self.biogt_relbf_indicator_multirule_count
                    )
                },
                "indicator_multirule_source": "train_graph_only",
                "indicator_multirule_attribution": (
                    "per_rule_gate_and_three_layer_indicator_path_state"
                    if self.biogt_relbf_indicator_multirule_count > 0
                    else "disabled"
                ),
                "indicator_relation_transition": (
                    "pending_train_graph_configuration"
                    if self.biogt_relbf_indicator_relation_transition_layers > 0
                    else "disabled"
                ),
                "indicator_relation_transition_source": (
                    "train_graph_only"
                    if self.biogt_relbf_indicator_relation_transition_layers > 0
                    else None
                ),
            }

        if (
            self.biogt_relbf_indicator_relation_transition_layers > 0
            or self.biogt_relbf_indicator_query_local_relation_transition_layers
            > 0
        ):
            for modality, metadata in configured.items():
                transition_metadata = (
                    self._configure_train_only_biogt_relation_transition_graph(
                        modality
                    )
                )
                metadata.update(transition_metadata)
        else:
            self._biogt_relbf_relation_transition_adjacencies = {}
            self._biogt_relbf_relation_transition_labels = {}
            self._biogt_relbf_relation_transition_indices = {}
            self._biogt_relbf_relation_transition_attributions = {}
            self._biogt_relbf_query_local_relation_transition_attributions = {}

        for modality, metadata in configured.items():
            self._biogt_path_graph_metadata.setdefault(
                modality,
                {},
            ).update({
                "neural_diffusion": metadata,
            })
        return configured

    def _configure_train_only_biogt_relation_transition_graph(self, modality):
        supports = self._biogt_path_support_modalities(modality)
        labels = []
        for support in supports:
            labels.extend([
                f"disease_to_{support}",
                f"{support}_to_disease",
            ])
        indices = {label: index for index, label in enumerate(labels)}
        transition_counts = {}
        disease_degrees = {}
        for support in supports:
            adjacency = self._biogt_neural_path_adjacencies[support].coalesce()
            disease_ids = adjacency.indices()[0]
            disease_degrees[support] = torch.bincount(
                disease_ids,
                minlength=adjacency.shape[0],
            ).to(dtype=torch.float32)
            source = indices[f"disease_to_{support}"]
            destination = indices[f"{support}_to_disease"]
            transition_counts[(source, destination)] = float(adjacency._nnz())

        for source_support in supports:
            source = indices[f"{source_support}_to_disease"]
            for destination_support in supports:
                destination = indices[f"disease_to_{destination_support}"]
                shared_walks = torch.dot(
                    disease_degrees[source_support],
                    disease_degrees[destination_support],
                ).item()
                if shared_walks > 0:
                    transition_counts[(source, destination)] = float(
                        shared_walks
                    )

        ordered = sorted(transition_counts.items())
        rows = torch.tensor(
            [edge[0][0] for edge in ordered],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        columns = torch.tensor(
            [edge[0][1] for edge in ordered],
            dtype=torch.long,
            device=rows.device,
        )
        values = torch.tensor(
            [edge[1] for edge in ordered],
            dtype=torch.float32,
            device=rows.device,
        )
        row_totals = torch.zeros(
            len(labels),
            dtype=values.dtype,
            device=values.device,
        ).scatter_add(0, rows, values)
        normalized_values = values / row_totals.index_select(0, rows).clamp_min(
            1.0
        )
        adjacency = torch.sparse_coo_tensor(
            torch.stack([rows, columns]),
            normalized_values,
            size=(len(labels), len(labels)),
            device=values.device,
        ).coalesce()
        self._biogt_relbf_relation_transition_adjacencies[modality] = adjacency
        self._biogt_relbf_relation_transition_labels[modality] = labels
        self._biogt_relbf_relation_transition_indices[modality] = indices
        self._biogt_relbf_relation_transition_attributions.pop(modality, None)
        transitions = []
        for (source, destination), count in ordered:
            transitions.append({
                "source_relation": labels[source],
                "destination_relation": labels[destination],
                "train_two_hop_walk_count": int(count),
                "normalized_weight": float(
                    normalized_values[len(transitions)].item()
                ),
            })
        return {
            "indicator_relation_transition": (
                "query_boundary_sparse_relation_message_residual_v1"
            ),
            "indicator_relation_transition_source": "train_graph_only",
            "indicator_relation_transition_layers": (
                self.biogt_relbf_indicator_relation_transition_layers
            ),
            "indicator_relation_transition_rank": (
                self.biogt_relbf_indicator_relation_transition_rank
            ),
            "indicator_relation_transition_labels": labels,
            "indicator_relation_transition_boundary": (
                f"disease_to_{modality}"
            ),
            "indicator_relation_transition_adjacency": transitions,
            "indicator_relation_transition_nnz": int(adjacency._nnz()),
            "indicator_relation_transition_normalization": "row_l1",
            "indicator_relation_transition_attribution": (
                "query_specific_relation_message_multiplier"
            ),
            "indicator_query_local_relation_transition": (
                "query_local_incidence_modulated_global_train_prior_v1"
                if self.biogt_relbf_indicator_query_local_relation_transition_layers
                > 0
                else "disabled"
            ),
            "indicator_query_local_relation_transition_source": (
                "compact_train_graph_query_incidence"
                if self.biogt_relbf_indicator_query_local_relation_transition_layers
                > 0
                else None
            ),
            "indicator_query_local_target_mask": (
                "training_dynamic_forward_and_inverse"
                if self.biogt_relbf_indicator_query_local_relation_transition_layers
                > 0
                else None
            ),
        }

    def _biogt_relbf_query_local_relation_incidence(
        self,
        modality,
        query_ids,
    ):
        labels = self._biogt_relbf_relation_transition_labels[modality]
        indices = self._biogt_relbf_relation_transition_indices[modality]
        incidence = query_ids.new_zeros(
            (query_ids.numel(), len(labels)),
            dtype=torch.float32,
        )
        sources = {}
        for support in self._biogt_path_support_modalities(modality):
            adjacency = self._biogt_neural_path_adjacencies[support].coalesce()
            disease_ids = adjacency.indices()[0]
            degree = torch.bincount(
                disease_ids,
                minlength=adjacency.shape[0],
            ).to(device=query_ids.device, dtype=incidence.dtype)
            local_degree = degree.index_select(0, query_ids)
            masked = bool(self.training and support == modality)
            if masked:
                local_degree = torch.zeros_like(local_degree)
            for label in (
                f"disease_to_{support}",
                f"{support}_to_disease",
            ):
                incidence[:, indices[label]] = local_degree
                sources[label] = {
                    "support_modality": support,
                    "source": "compact_train_graph_query_degree",
                    "target_relation_dynamic_masked": masked,
                }
        return incidence, sources

    def get_biogt_relbf_relation_transition_attribution(self, modality):
        labels = self._biogt_relbf_relation_transition_labels.get(modality)
        if labels is None:
            return None
        weights = self._biogt_relbf_relation_transition_attributions.get(
            modality
        )
        return {
            "source": "train_graph_only",
            "relation_labels": list(labels),
            "mean_query_conditioned_weights": (
                None
                if weights is None
                else {
                    label: float(weight)
                    for label, weight in zip(labels, weights.tolist())
                }
            ),
            "query_local": self._biogt_relbf_query_local_relation_transition_attributions.get(
                modality
            ),
        }

    def get_biogt_relbf_path_layer_attention_attribution(self, modality):
        """Return the latest candidate-specific RelBF layer attention."""
        attribution = self._biogt_relbf_path_layer_attention_attributions.get(
            modality
        )
        if attribution is None:
            return None
        return {
            "source": attribution["source"],
            "layer_labels": list(attribution["layer_labels"]),
            "query_ids": list(attribution["query_ids"]),
            "candidate_ids": list(attribution["candidate_ids"]),
            "attention": [
                [list(weights) for weights in query_weights]
                for query_weights in attribution["attention"]
            ],
        }

    def get_biogt_relbf_path_token_transformer_attribution(self, modality):
        """Return candidate-token attention over RelBF path tokens."""
        attribution = (
            self._biogt_relbf_path_token_transformer_attributions.get(modality)
        )
        if attribution is None:
            return None
        return {
            "source": attribution["source"],
            "path_labels": list(attribution["path_labels"]),
            "query_ids": list(attribution["query_ids"]),
            "candidate_ids": list(attribution["candidate_ids"]),
            "attention": [
                [list(weights) for weights in query_weights]
                for query_weights in attribution["attention"]
            ],
        }

    def get_biogt_relbf_candidate_competition_boundary_attribution(
        self, modality
    ):
        """Return parent-selected anchors and candidate-to-anchor attention."""
        attribution = (
            self._biogt_relbf_candidate_competition_boundary_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {
            "source": attribution["source"],
            "query_ids": list(attribution["query_ids"]),
            "candidate_ids": list(attribution["candidate_ids"]),
            "anchor_candidate_ids": [
                list(ids) for ids in attribution["anchor_candidate_ids"]
            ],
            "attention": [
                [list(weights) for weights in query_weights]
                for query_weights in attribution["attention"]
            ],
        }

    def get_biogt_relbf_parent_top10_boundary_comparator_attribution(
        self, modality
    ):
        """Return frozen parent rank-10/11 slots and signed score gaps."""
        attribution = (
            self._biogt_relbf_parent_top10_boundary_comparator_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {
            "source": attribution["source"],
            "boundary_ranks": list(attribution["boundary_ranks"]),
            "query_ids": list(attribution["query_ids"]),
            "candidate_ids": list(attribution["candidate_ids"]),
            "boundary_candidate_ids": [
                list(ids) for ids in attribution["boundary_candidate_ids"]
            ],
            "boundary_scores": [
                list(scores) for scores in attribution["boundary_scores"]
            ],
            "score_gaps": [
                [list(gaps) for gaps in query_gaps]
                for query_gaps in attribution["score_gaps"]
            ],
        }

    def get_biogt_relbf_boundary_window_rank_slot_expert_attribution(
        self, modality
    ):
        """Return the latest parent-ranked boundary-window explanation."""
        attribution = (
            self._biogt_relbf_boundary_window_rank_slot_expert_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {
            key: value
            for key, value in attribution.items()
        }

    def get_biogt_relbf_boundary_exchange_attribution(self, modality):
        """Return the latest incumbent/challenger exchange explanation."""
        attribution = self._biogt_relbf_boundary_exchange_attributions.get(
            modality
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_challenger_boundary_promotion_attribution(
        self, modality
    ):
        """Return challenger-to-incumbent promotion evidence."""
        attribution = (
            self._biogt_relbf_challenger_boundary_promotion_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_gatekeeper_challenger_paired_margin_attribution(
        self, modality
    ):
        """Return rank-10/challenger pair identities and aggregation weights."""
        attribution = (
            self._biogt_relbf_gatekeeper_challenger_paired_margin_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_boundary_prototype_ordinal_adapter_attribution(
        self, modality
    ):
        attribution = self._biogt_relbf_boundary_prototype_ordinal_adapter_attributions.get(
            modality
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_multi_depth_score_router_attribution(self, modality):
        attribution = self._biogt_relbf_multi_depth_score_router_attributions.get(
            modality
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_dual_stream_trajectory_agreement_calibrator_attribution(
        self, modality
    ):
        attribution = (
            self._biogt_relbf_dual_stream_trajectory_agreement_calibrator_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_multi_anchor_rank_landmark_calibrator_attribution(
        self, modality
    ):
        attribution = (
            self._biogt_relbf_multi_anchor_rank_landmark_calibrator_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    def get_biogt_relbf_query_token_top64_candidate_set_transformer_attribution(
        self, modality
    ):
        attribution = (
            self._biogt_relbf_query_token_top64_candidate_set_transformer_attributions.get(
                modality
            )
        )
        if attribution is None:
            return None
        return {key: value for key, value in attribution.items()}

    @staticmethod
    def _standardize_path_feature(values):
        values = torch.log1p(values.clamp_min(0.0))
        values = values - values.mean(dim=1, keepdim=True)
        return values / values.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)

    def _biogt_path_disease_transition(
        self,
        disease_state,
        support_modality,
        query_ids=None,
        mask_query_relation=False,
    ):
        adjacency = self._biogt_path_adjacencies[support_modality]
        if mask_query_relation:
            target_state = self._biogt_path_masked_transition(
                disease_state,
                adjacency,
                query_ids,
                direction="disease_to_target",
            )
            return self._biogt_path_masked_transition(
                target_state,
                adjacency,
                query_ids,
                direction="target_to_disease",
            )
        target_state = torch.sparse.mm(
            adjacency.transpose(0, 1),
            disease_state.transpose(0, 1),
        )
        return torch.sparse.mm(
            adjacency,
            target_state,
        ).transpose(0, 1)

    @staticmethod
    def _biogt_path_masked_transition(
        state,
        adjacency,
        query_ids,
        direction,
    ):
        adjacency = adjacency.coalesce()
        disease_index, target_index = adjacency.indices()
        batch_size = state.shape[0]
        keep = disease_index.unsqueeze(0) != query_ids.unsqueeze(1)
        keep_values = keep.to(state.dtype)
        disease_degree = state.new_zeros(
            (batch_size, adjacency.shape[0])
        ).scatter_add(
            1,
            disease_index.unsqueeze(0).expand(batch_size, -1),
            keep_values,
        )
        target_degree = state.new_zeros(
            (batch_size, adjacency.shape[1])
        ).scatter_add(
            1,
            target_index.unsqueeze(0).expand(batch_size, -1),
            keep_values,
        )
        edge_weights = keep_values * (
            disease_degree.index_select(1, disease_index)
            * target_degree.index_select(1, target_index)
        ).clamp_min(1.0).rsqrt()
        if direction == "disease_to_target":
            source_index = disease_index
            destination_index = target_index
            destination_count = adjacency.shape[1]
        elif direction == "target_to_disease":
            source_index = target_index
            destination_index = disease_index
            destination_count = adjacency.shape[0]
        else:
            raise ValueError(
                f"Unknown BioGT path transition direction: {direction}"
            )
        messages = state.index_select(1, source_index)
        messages = messages * edge_weights
        result = state.new_zeros((batch_size, destination_count))
        result.scatter_add_(
            1,
            destination_index.unsqueeze(0).expand(batch_size, -1),
            messages,
        )
        return result

    def _biogt_path_target_transition(
        self,
        disease_state,
        output_modality,
        query_ids,
        mask_query_relation,
    ):
        adjacency = self._biogt_path_adjacencies[output_modality]
        if mask_query_relation:
            return self._biogt_path_masked_transition(
                disease_state,
                adjacency,
                query_ids,
                direction="disease_to_target",
            )
        return torch.sparse.mm(
            adjacency.transpose(0, 1),
            disease_state.transpose(0, 1),
        ).transpose(0, 1)

    def _biogt_path_query_features(self, query_ids):
        return self.input_adapters["disease"](
            self.node_embeddings["disease"](query_ids)
        )

    @staticmethod
    def _mask_batched_query_state(state, query_ids):
        masked = state.clone()
        rows = torch.arange(query_ids.numel(), device=state.device)
        masked[rows, query_ids] = 0.0
        return masked

    def _biogt_neural_path_transition(
        self,
        state,
        support_modality,
        query_ids,
        mask_query_relation,
    ):
        adjacency = self._biogt_neural_path_adjacencies[
            support_modality
        ]
        if mask_query_relation:
            state = self._mask_batched_query_state(state, query_ids)
        batch_size, disease_count, hidden_dim = state.shape
        flattened = (
            state.permute(1, 0, 2)
            .reshape(disease_count, batch_size * hidden_dim)
        )
        target_state = torch.sparse.mm(
            adjacency.transpose(0, 1),
            flattened,
        )
        disease_state = torch.sparse.mm(
            adjacency,
            target_state,
        )
        result = (
            disease_state.reshape(
                disease_count,
                batch_size,
                hidden_dim,
            )
            .permute(1, 0, 2)
        )
        if mask_query_relation:
            result = self._mask_batched_query_state(result, query_ids)
        return result

    def _biogt_relbf_pna_aggregate(
        self,
        sources,
        destination_count,
        output_modality,
        layer_index,
        condition,
        query_ids,
        deep_refinement=False,
        indicator_residual=False,
        indicator_gate_condition=None,
        indicator_relation_transition_weights=None,
    ):
        batch_size = condition.shape[0]
        hidden_dim = self.biogt_relbf_dim
        device = condition.device
        totals = []
        squared_totals = []
        maxima = []
        minima = []
        counts = []
        if indicator_residual:
            gamma = condition.new_ones(
                (batch_size, 1, hidden_dim)
            )
            beta = condition.new_zeros(
                (batch_size, 1, hidden_dim)
            )
        else:
            films = (
                self.biogt_relbf_deep_output_films[output_modality]
                if deep_refinement
                else self.biogt_relbf_output_films[output_modality]
            )
            gamma, beta = films[layer_index](condition).chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)

        for support, state, direction in sources:
            adjacency = self._biogt_neural_path_adjacencies[
                support
            ].coalesce()
            disease_index, target_index = adjacency.indices()
            if direction == "disease_to_target":
                source_index = disease_index
                destination_index = target_index
            elif direction == "target_to_disease":
                source_index = target_index
                destination_index = disease_index
            else:
                raise ValueError(
                    f"Unknown RelBF direction: {direction}"
                )
            messages = state.index_select(1, source_index)
            adapter_input = messages
            if indicator_residual:
                transforms = (
                    self.biogt_relbf_indicator_message_transforms[
                        output_modality
                    ][support]
                )
            else:
                transforms = (
                    self.biogt_relbf_deep_message_transforms[
                        output_modality
                    ][support]
                    if deep_refinement
                    else self.biogt_relbf_message_transforms[support]
                )
            messages = transforms[layer_index](messages)
            if (
                not deep_refinement
                and not indicator_residual
                and self.biogt_relbf_query_relation_adapter_rank > 0
            ):
                messages = messages + (
                    self.biogt_relbf_query_relation_adapters[
                        output_modality
                    ][support][layer_index](adapter_input, condition)
                )
            messages = gamma * messages + beta
            if (
                indicator_residual
                and self.biogt_relbf_indicator_support_gating
            ):
                if indicator_gate_condition is None:
                    raise ValueError(
                        "Indicator support gating requires the frozen parent "
                        "RelBF condition"
                    )
                gate = self.biogt_relbf_indicator_support_gates[
                    output_modality
                ][support][layer_index](indicator_gate_condition)
                messages = messages * (
                    2.0 * torch.sigmoid(gate)
                ).unsqueeze(1)
            if (
                indicator_residual
                and indicator_relation_transition_weights is not None
            ):
                relation_label = (
                    f"disease_to_{support}"
                    if direction == "disease_to_target"
                    else f"{support}_to_disease"
                )
                relation_index = (
                    self._biogt_relbf_relation_transition_indices[
                        output_modality
                    ][relation_label]
                )
                messages = messages * (
                    indicator_relation_transition_weights[
                        :, relation_index
                    ].view(batch_size, 1, 1)
                )
            if self.biogt_relbf_edge_attention and not deep_refinement:
                attention_query = (
                    self.biogt_relbf_edge_attention_queries[
                        output_modality
                    ][support][layer_index](condition)
                )
                attention_logits = (
                    F.normalize(messages, dim=-1)
                    * attention_query.unsqueeze(1)
                ).sum(dim=-1) / math.sqrt(hidden_dim)
                attention_weights = 2.0 * torch.sigmoid(
                    attention_logits
                )
                messages = messages * attention_weights.unsqueeze(-1)
            keep = torch.ones(
                (batch_size, source_index.numel()),
                dtype=torch.bool,
                device=device,
            )
            if self.training and support == output_modality:
                keep = disease_index.unsqueeze(0) != query_ids.unsqueeze(1)
            weighted = messages * keep.unsqueeze(-1).to(messages.dtype)
            scatter_index = destination_index.view(1, -1, 1).expand(
                batch_size,
                -1,
                hidden_dim,
            )
            relation_total = condition.new_zeros(
                (batch_size, destination_count, hidden_dim)
            ).scatter_add(1, scatter_index, weighted)
            relation_squared = condition.new_zeros(
                (batch_size, destination_count, hidden_dim)
            ).scatter_add(1, scatter_index, weighted.square())
            totals.append(relation_total)
            squared_totals.append(relation_squared)

            max_messages = messages.masked_fill(
                ~keep.unsqueeze(-1),
                -torch.inf,
            )
            min_messages = messages.masked_fill(
                ~keep.unsqueeze(-1),
                torch.inf,
            )
            relation_maximum = torch.full_like(
                relation_total,
                -torch.inf,
            ).scatter_reduce(
                1,
                scatter_index,
                max_messages,
                reduce="amax",
                include_self=True,
            )
            relation_minimum = torch.full_like(
                relation_total,
                torch.inf,
            ).scatter_reduce(
                1,
                scatter_index,
                min_messages,
                reduce="amin",
                include_self=True,
            )
            maxima.append(relation_maximum)
            minima.append(relation_minimum)
            relation_count = condition.new_zeros(
                (batch_size, destination_count, 1)
            ).scatter_add(
                1,
                destination_index.view(1, -1, 1).expand(
                    batch_size,
                    -1,
                    1,
                ),
                keep.unsqueeze(-1).to(condition.dtype),
            )
            counts.append(relation_count)

        total = torch.stack(totals).sum(dim=0)
        squared = torch.stack(squared_totals).sum(dim=0)
        maximum = torch.stack(maxima).amax(dim=0)
        minimum = torch.stack(minima).amin(dim=0)
        count = torch.stack(counts).sum(dim=0)

        safe_count = count.clamp_min(1.0)
        mean = total / safe_count
        variance = (squared / safe_count - mean.square()).clamp_min(0.0)
        std = torch.sqrt(variance + 1e-6)
        empty = count == 0
        maximum = maximum.masked_fill(empty, 0.0)
        minimum = minimum.masked_fill(empty, 0.0)
        features = torch.cat([mean, maximum, minimum, std], dim=-1)

        log_degree = (count + 1.0).log()
        degree_mean = log_degree.mean(dim=1, keepdim=True).clamp_min(1e-2)
        degree_scale = log_degree / degree_mean
        scalers = torch.cat([
            torch.ones_like(degree_scale),
            degree_scale,
            degree_scale.clamp_min(1e-2).reciprocal(),
        ], dim=-1)
        pna_features = (
            features.unsqueeze(-1) * scalers.unsqueeze(-2)
        ).flatten(-2)
        if indicator_residual:
            updates = self.biogt_relbf_indicator_pna_updates[
                output_modality
            ]
        else:
            updates = (
                self.biogt_relbf_deep_pna_updates[output_modality]
                if deep_refinement
                else self.biogt_relbf_pna_updates
            )
        return updates[layer_index](pna_features)

    def _biogt_relbf_candidate_scores(
        self,
        modality,
        query_ids,
        candidate_ids,
        query_features,
        candidate_features=None,
    ):
        if self.biogt_relbf_dim <= 0:
            return query_features.new_zeros(
                (query_ids.numel(), candidate_ids.numel())
            )
        if modality not in self._biogt_neural_path_adjacencies:
            raise RuntimeError(
                "Train-only compact BioGT RelBF graph was not configured "
                "before scoring"
            )

        relation_token = self.biogt_relbf_relation_tokens[modality]
        condition = torch.tanh(
            self.biogt_relbf_query_projections[modality](query_features)
            + relation_token.unsqueeze(0)
        )
        indicator_relation_transition_weights = None
        if self.biogt_relbf_indicator_relation_transition_layers > 0:
            if modality not in (
                self._biogt_relbf_relation_transition_adjacencies
            ):
                raise RuntimeError(
                    "Train-only BioGT relation-transition graph was not "
                    "configured before scoring"
                )
            relation_transition = (
                self.biogt_relbf_indicator_relation_transition_residuals[
                    modality
                ]
            )
            relation_indices = (
                self._biogt_relbf_relation_transition_indices[modality]
            )
            indicator_relation_transition_weights = relation_transition(
                condition,
                self._biogt_relbf_relation_transition_adjacencies[modality],
                relation_indices[f"disease_to_{modality}"],
            )
            self._biogt_relbf_relation_transition_attributions[modality] = (
                indicator_relation_transition_weights.detach().mean(dim=0).cpu()
            )
        elif (
            self.biogt_relbf_indicator_query_local_relation_transition_layers > 0
        ):
            if modality not in (
                self._biogt_relbf_relation_transition_adjacencies
            ):
                raise RuntimeError(
                    "Train-only BioGT query-local relation-transition graph "
                    "was not configured before scoring"
                )
            relation_transition = (
                self.biogt_relbf_indicator_query_local_relation_transition_residuals[
                    modality
                ]
            )
            relation_indices = (
                self._biogt_relbf_relation_transition_indices[modality]
            )
            local_incidence, local_sources = (
                self._biogt_relbf_query_local_relation_incidence(
                    modality,
                    query_ids,
                )
            )
            (
                indicator_relation_transition_weights,
                normalized_incidence,
                local_gates,
            ) = relation_transition(
                condition,
                self._biogt_relbf_relation_transition_adjacencies[modality],
                relation_indices[f"disease_to_{modality}"],
                local_incidence,
            )
            labels = self._biogt_relbf_relation_transition_labels[modality]
            self._biogt_relbf_query_local_relation_transition_attributions[
                modality
            ] = {
                "source": "compact_train_graph_query_incidence",
                "query_ids": [int(value) for value in query_ids.detach().cpu()],
                "relations": {
                    label: {
                        **local_sources[label],
                        "normalized_incidence": [
                            float(value)
                            for value in normalized_incidence[
                                :, relation_indices[label]
                            ].detach().cpu()
                        ],
                        "mean_local_gate": float(
                            local_gates[
                                :, relation_indices[label]
                            ].detach().mean().cpu()
                        ),
                        "mean_aggregate_multiplier": float(
                            indicator_relation_transition_weights[
                                :, relation_indices[label]
                            ].detach().mean().cpu()
                        ),
                    }
                    for label in labels
                },
            }
        disease_count = self._biogt_neural_path_adjacencies[
            modality
        ].shape[0]
        disease_boundary = condition.new_zeros(
            (query_ids.numel(), disease_count, self.biogt_relbf_dim)
        )
        rows = torch.arange(query_ids.numel(), device=condition.device)
        disease_boundary[rows, query_ids] = condition

        supports = self._biogt_path_support_modalities(modality)
        target_states = {}
        for support in supports:
            target_update = self._biogt_relbf_pna_aggregate(
                [(
                    support,
                    disease_boundary,
                    "disease_to_target",
                )],
                self._biogt_neural_path_adjacencies[support].shape[1],
                modality,
                0,
                condition,
                query_ids,
            )
            target_states[support] = F.gelu(
                self.biogt_relbf_state_norms[0](target_update)
            )

        disease_update = self._biogt_relbf_pna_aggregate(
            [
                (support, target_states[support], "target_to_disease")
                for support in supports
            ],
            disease_count,
            modality,
            1,
            condition,
            query_ids,
        )
        disease_state = F.gelu(
            self.biogt_relbf_state_norms[1](
                disease_boundary + disease_update
            )
        )
        output_update = self._biogt_relbf_pna_aggregate(
            [(
                modality,
                disease_state,
                "disease_to_target",
            )],
            self._biogt_neural_path_adjacencies[modality].shape[1],
            modality,
            2,
            condition,
            query_ids,
        )
        output_state = F.gelu(
            self.biogt_relbf_state_norms[2](
                target_states[modality] + output_update
            )
        )
        refined_output_state = None
        if self.biogt_relbf_deep_refinement:
            deep_target_states = dict(target_states)
            deep_target_states[modality] = output_state
            deep_disease_update = self._biogt_relbf_pna_aggregate(
                [
                    (
                        support,
                        deep_target_states[support],
                        "target_to_disease",
                    )
                    for support in supports
                ],
                disease_count,
                modality,
                0,
                condition,
                query_ids,
                deep_refinement=True,
            )
            refined_disease_state = F.gelu(
                self.biogt_relbf_deep_state_norms[modality][0](
                    disease_state + deep_disease_update
                )
            )
            deep_output_update = self._biogt_relbf_pna_aggregate(
                [(
                    modality,
                    refined_disease_state,
                    "disease_to_target",
                )],
                self._biogt_neural_path_adjacencies[modality].shape[1],
                modality,
                1,
                condition,
                query_ids,
                deep_refinement=True,
            )
            refined_output_state = F.gelu(
                self.biogt_relbf_deep_state_norms[modality][1](
                    output_state + deep_output_update
                )
            )

        indicator_output_state = None
        multirule_output_states = []
        if self.biogt_relbf_indicator_residual:
            indicator_token = self.biogt_relbf_indicator_tokens[modality]
            indicator_condition = indicator_token.unsqueeze(0).expand(
                query_ids.numel(), -1
            )
            if self.biogt_relbf_indicator_query_boundary_rank > 0:
                indicator_condition = (
                    indicator_condition
                    + self.biogt_relbf_indicator_query_boundary_adapters[
                        modality
                    ](condition)
                )
            indicator_boundary = indicator_condition.new_zeros(
                (query_ids.numel(), disease_count, self.biogt_relbf_dim)
            )
            indicator_boundary[rows, query_ids] = indicator_condition
            indicator_targets = {}
            for support in supports:
                indicator_update = self._biogt_relbf_pna_aggregate(
                    [(
                        support,
                        indicator_boundary,
                        "disease_to_target",
                    )],
                    self._biogt_neural_path_adjacencies[support].shape[1],
                    modality,
                    0,
                    indicator_condition,
                    query_ids,
                    indicator_residual=True,
                    indicator_gate_condition=condition,
                    indicator_relation_transition_weights=(
                        indicator_relation_transition_weights
                    ),
                )
                indicator_targets[support] = F.gelu(
                    self.biogt_relbf_indicator_state_norms[modality][0](
                        indicator_update
                    )
                )
            indicator_disease_update = self._biogt_relbf_pna_aggregate(
                [
                    (
                        support,
                        indicator_targets[support],
                        "target_to_disease",
                    )
                    for support in supports
                ],
                disease_count,
                modality,
                1,
                indicator_condition,
                query_ids,
                indicator_residual=True,
                indicator_gate_condition=condition,
                indicator_relation_transition_weights=(
                    indicator_relation_transition_weights
                ),
            )
            indicator_disease_state = F.gelu(
                self.biogt_relbf_indicator_state_norms[modality][1](
                    indicator_boundary + indicator_disease_update
                )
            )
            indicator_output_update = self._biogt_relbf_pna_aggregate(
                [(
                    modality,
                    indicator_disease_state,
                    "disease_to_target",
                )],
                self._biogt_neural_path_adjacencies[modality].shape[1],
                modality,
                2,
                indicator_condition,
                query_ids,
                indicator_residual=True,
                indicator_gate_condition=condition,
                indicator_relation_transition_weights=(
                    indicator_relation_transition_weights
                ),
            )
            indicator_output_state = F.gelu(
                self.biogt_relbf_indicator_state_norms[modality][2](
                    indicator_targets[modality]
                    + indicator_output_update
                )
            )
            if self.biogt_relbf_indicator_multirule_count > 0:
                multirule = self.biogt_relbf_indicator_multirule_residuals[
                    modality
                ]
                for rule_index in range(multirule.rule_count):
                    rule_condition = multirule.rule_tokens[
                        rule_index
                    ].unsqueeze(0).expand(query_ids.numel(), -1)
                    rule_boundary = rule_condition.new_zeros(
                        (
                            query_ids.numel(),
                            disease_count,
                            self.biogt_relbf_dim,
                        )
                    )
                    rule_boundary[rows, query_ids] = rule_condition
                    rule_targets = {}
                    for support in supports:
                        rule_update = self._biogt_relbf_pna_aggregate(
                            [(
                                support,
                                rule_boundary,
                                "disease_to_target",
                            )],
                            self._biogt_neural_path_adjacencies[
                                support
                            ].shape[1],
                            modality,
                            0,
                            rule_condition,
                            query_ids,
                            indicator_residual=True,
                            indicator_gate_condition=condition,
                        )
                        rule_targets[support] = F.gelu(
                            self.biogt_relbf_indicator_state_norms[
                                modality
                            ][0](rule_update)
                        )
                    rule_disease_update = self._biogt_relbf_pna_aggregate(
                        [
                            (
                                support,
                                rule_targets[support],
                                "target_to_disease",
                            )
                            for support in supports
                        ],
                        disease_count,
                        modality,
                        1,
                        rule_condition,
                        query_ids,
                        indicator_residual=True,
                        indicator_gate_condition=condition,
                    )
                    rule_disease_state = F.gelu(
                        self.biogt_relbf_indicator_state_norms[modality][1](
                            rule_boundary + rule_disease_update
                        )
                    )
                    rule_output_update = self._biogt_relbf_pna_aggregate(
                        [(
                            modality,
                            rule_disease_state,
                            "disease_to_target",
                        )],
                        self._biogt_neural_path_adjacencies[
                            modality
                        ].shape[1],
                        modality,
                        2,
                        rule_condition,
                        query_ids,
                        indicator_residual=True,
                        indicator_gate_condition=condition,
                    )
                    multirule_output_states.append(F.gelu(
                        self.biogt_relbf_indicator_state_norms[modality][2](
                            rule_targets[modality] + rule_output_update
                        )
                    ))

        if candidate_features is not None:
            if candidate_features.shape[0] != candidate_ids.numel():
                raise ValueError(
                    "RelBF candidate features must align with candidate IDs"
                )
            selected_features = candidate_features
        else:
            selected_features = self.input_adapters[modality](
                self.node_embeddings[modality](candidate_ids)
            ).detach()
        candidate_state = self.biogt_relbf_candidate_projections[
            modality
        ](selected_features)
        candidate_state = torch.tanh(candidate_state)
        relation_state = relation_token.view(1, 1, -1).expand(
            query_ids.numel(),
            candidate_state.shape[0],
            -1,
        )
        lookup = self._biogt_neural_path_candidate_lookup[modality]
        positions = lookup.index_select(0, candidate_ids)
        valid = positions >= 0
        selected_output_state = output_state.index_select(
            1,
            positions.clamp_min(0),
        )
        selected_shallow_state = target_states[modality].index_select(
            1,
            positions.clamp_min(0),
        )
        selected_middle_state = output_update.index_select(
            1,
            positions.clamp_min(0),
        )
        readout_input = torch.cat([
            selected_output_state,
            selected_output_state * candidate_state.unsqueeze(0),
            candidate_state.unsqueeze(0) * (
                condition.unsqueeze(1) + relation_state
            ),
        ], dim=-1)
        compact_scores = self.biogt_relbf_readouts[modality](
            readout_input
        ).squeeze(-1)
        if self.biogt_relbf_path_layer_attention_rank > 0:
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            path_layer_residual, path_layer_attention = (
                self.biogt_relbf_path_layer_attention_residuals[modality](
                    layer_states,
                    condition,
                    candidate_state,
                )
            )
            compact_scores = compact_scores + path_layer_residual
            self._biogt_relbf_path_layer_attention_attributions[modality] = {
                "source": "train_graph_relbf_path_states",
                "layer_labels": list(
                    CandidateConditionedPathLayerAttentionResidual.layer_labels
                ),
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "attention": path_layer_attention.detach().cpu().tolist(),
                "shape": list(path_layer_attention.shape),
                "mean_attention_by_layer": path_layer_attention.detach().mean(
                    dim=(0, 1)
                ).cpu().tolist(),
            }
        if self.biogt_relbf_path_token_transformer:
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            token_residual, path_attention = (
                self.biogt_relbf_path_token_transformer_residuals[modality](
                    layer_states, condition, candidate_state
                )
            )
            compact_scores = compact_scores + token_residual
            labels = CandidateConditionedPathTokenTransformerResidual.path_labels
            self._biogt_relbf_path_token_transformer_attributions[modality] = {
                "source": "train_graph_candidate_conditioned_path_tokens",
                "path_labels": list(labels),
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "attention": path_attention.detach().cpu().tolist(),
                "shape": list(path_attention.shape),
                "mean_attention_by_path": path_attention.detach().mean(
                    dim=(0, 1)
                ).cpu().tolist(),
            }
        if self.biogt_relbf_jk_readout:
            candidate_matrix = candidate_state.unsqueeze(0)
            deep_delta = selected_output_state - selected_shallow_state
            jk_input = torch.cat([
                selected_shallow_state,
                selected_shallow_state * candidate_matrix,
                deep_delta,
                deep_delta * candidate_matrix,
            ], dim=-1)
            compact_scores = compact_scores + (
                self.biogt_relbf_jk_readouts[modality](jk_input)
                .squeeze(-1)
            )
        if refined_output_state is not None:
            selected_refined_state = refined_output_state.index_select(
                1,
                positions.clamp_min(0),
            )
            refinement_delta = selected_refined_state - selected_output_state
            candidate_matrix = candidate_state.unsqueeze(0)
            deep_input = torch.cat([
                selected_refined_state,
                selected_refined_state * candidate_matrix,
                refinement_delta,
                refinement_delta * candidate_matrix,
            ], dim=-1)
            compact_scores = compact_scores + (
                self.biogt_relbf_deep_readouts[modality](deep_input)
                .squeeze(-1)
            )
        if indicator_output_state is not None:
            selected_indicator_state = indicator_output_state.index_select(
                1,
                positions.clamp_min(0),
            )
            indicator_relation = self.biogt_relbf_indicator_tokens[
                modality
            ].view(1, 1, -1).expand_as(selected_indicator_state)
            indicator_input = torch.cat([
                selected_indicator_state,
                selected_indicator_state * indicator_relation,
                indicator_relation,
            ], dim=-1)
            compact_scores = compact_scores + (
                self.biogt_relbf_indicator_readouts[modality](
                    indicator_input
                ).squeeze(-1)
            )
            if self.biogt_relbf_indicator_semantic_readout:
                candidate_matrix = candidate_state.unsqueeze(0)
                frozen_condition = condition.detach().unsqueeze(1)
                semantic_input = torch.cat([
                    selected_indicator_state,
                    selected_indicator_state * candidate_matrix,
                    selected_indicator_state * frozen_condition,
                    candidate_matrix * frozen_condition,
                ], dim=-1)
                compact_scores = compact_scores + (
                    self.biogt_relbf_indicator_semantic_readouts[modality](
                        semantic_input
                    ).squeeze(-1)
                )
        if multirule_output_states:
            multirule = self.biogt_relbf_indicator_multirule_residuals[
                modality
            ]
            rule_scores = []
            for rule_index, rule_output_state in enumerate(
                multirule_output_states
            ):
                selected_rule_state = rule_output_state.index_select(
                    1,
                    positions.clamp_min(0),
                )
                rule_relation = multirule.rule_tokens[
                    rule_index
                ].view(1, 1, -1).expand_as(selected_rule_state)
                rule_input = torch.cat([
                    selected_rule_state,
                    selected_rule_state * rule_relation,
                    rule_relation,
                ], dim=-1)
                rule_scores.append(
                    multirule.rule_readouts[rule_index](rule_input)
                    .squeeze(-1)
                )
            gate_weights = multirule.gate_weights(condition).unsqueeze(-1)
            compact_scores = compact_scores + (
                torch.stack(rule_scores, dim=1) * gate_weights
            ).sum(dim=1)
        if self.biogt_relbf_candidate_competition_boundary:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            competition_residual, anchor_indices, anchor_attention = (
                self.biogt_relbf_candidate_competition_boundary_residuals[
                    modality
                ](
                    layer_states,
                    condition,
                    candidate_state,
                    parent_compact_scores,
                    valid,
                )
            )
            compact_scores = compact_scores + competition_residual
            anchor_candidate_ids = candidate_ids.index_select(
                0, anchor_indices.reshape(-1)
            ).reshape_as(anchor_indices)
            self._biogt_relbf_candidate_competition_boundary_attributions[
                modality
            ] = {
                "source": "frozen_parent_compact_scores_top32_train_graph_tokens",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "anchor_candidate_ids": anchor_candidate_ids.detach().cpu().tolist(),
                "attention": anchor_attention.detach().cpu().tolist(),
                "shape": list(anchor_attention.shape),
                "mean_attention_by_anchor_rank": anchor_attention.detach().mean(
                    dim=(0, 1)
                ).cpu().tolist(),
            }
        if self.biogt_relbf_parent_top10_boundary_comparator:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                boundary_residual,
                boundary_indices,
                boundary_scores,
                score_gaps,
            ) = self.biogt_relbf_parent_top10_boundary_comparator_residuals[
                modality
            ](
                layer_states,
                condition,
                candidate_state,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + boundary_residual
            boundary_candidate_ids = candidate_ids.index_select(
                0, boundary_indices.reshape(-1)
            ).reshape_as(boundary_indices)
            self._biogt_relbf_parent_top10_boundary_comparator_attributions[
                modality
            ] = {
                "source": "frozen_parent_rank10_rank11_train_graph_tokens",
                "boundary_ranks": list(
                    ParentTopBoundarySlotComparatorResidual.boundary_ranks
                ),
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "boundary_candidate_ids": (
                    boundary_candidate_ids.detach().cpu().tolist()
                ),
                "boundary_scores": boundary_scores.detach().cpu().tolist(),
                "score_gaps": score_gaps.detach().cpu().tolist(),
                "shape": list(score_gaps.shape),
            }
        if self.biogt_relbf_boundary_window_rank_slot_expert:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                window_residual,
                window_indices,
                rank_positions,
                boundary_indices,
                boundary_scores,
                score_gaps,
                window_mask,
            ) = self.biogt_relbf_boundary_window_rank_slot_expert_residuals[
                modality
            ](
                layer_states,
                condition,
                candidate_state,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + window_residual
            window_candidate_ids = candidate_ids.index_select(
                0, window_indices.reshape(-1)
            ).reshape_as(window_indices)
            boundary_candidate_ids = candidate_ids.index_select(
                0, boundary_indices.reshape(-1)
            ).reshape_as(boundary_indices)
            self._biogt_relbf_boundary_window_rank_slot_expert_attributions[
                modality
            ] = {
                "source": "gene8_parent_compact_scores_rank5_20_window",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "window_candidate_ids": (
                    window_candidate_ids.detach().cpu().tolist()
                ),
                "rank_positions": rank_positions.detach().cpu().tolist(),
                "boundary_candidate_ids": (
                    boundary_candidate_ids.detach().cpu().tolist()
                ),
                "boundary_scores": boundary_scores.detach().cpu().tolist(),
                "score_gaps": score_gaps.detach().cpu().tolist(),
                "window_mask": window_mask.detach().cpu().tolist(),
                "nonzero_mask": window_residual.detach().ne(0).cpu().tolist(),
                "residual": window_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_boundary_exchange:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                exchange_residual,
                incumbent_indices,
                challenger_indices,
                incumbent_attention,
                challenger_attention,
                exchange_mask,
            ) = self.biogt_relbf_boundary_exchange_residuals[modality](
                layer_states,
                condition,
                candidate_state,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + exchange_residual
            incumbent_candidate_ids = candidate_ids.index_select(
                0, incumbent_indices.reshape(-1)
            ).reshape_as(incumbent_indices)
            challenger_candidate_ids = candidate_ids.index_select(
                0, challenger_indices.reshape(-1)
            ).reshape_as(challenger_indices)
            self._biogt_relbf_boundary_exchange_attributions[modality] = {
                "source": "gene9_parent_rank5_20_incumbent_challenger_exchange",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "incumbent_candidate_ids": (
                    incumbent_candidate_ids.detach().cpu().tolist()
                ),
                "challenger_candidate_ids": (
                    challenger_candidate_ids.detach().cpu().tolist()
                ),
                "incumbent_to_challenger_attention": (
                    incumbent_attention.detach().cpu().tolist()
                ),
                "challenger_to_incumbent_attention": (
                    challenger_attention.detach().cpu().tolist()
                ),
                "window_mask": exchange_mask.detach().cpu().tolist(),
                "residual": exchange_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_challenger_boundary_promotion:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                promotion_residual,
                incumbent_indices,
                challenger_indices,
                challenger_attention,
                challenger_mask,
            ) = self.biogt_relbf_challenger_boundary_promotion_residuals[
                modality
            ](
                layer_states,
                condition,
                candidate_state,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + promotion_residual
            incumbent_candidate_ids = candidate_ids.index_select(
                0, incumbent_indices.reshape(-1)
            ).reshape_as(incumbent_indices)
            challenger_candidate_ids = candidate_ids.index_select(
                0, challenger_indices.reshape(-1)
            ).reshape_as(challenger_indices)
            self._biogt_relbf_challenger_boundary_promotion_attributions[
                modality
            ] = {
                "source": "gene9_parent_rank11_20_challenger_only_promotion",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "incumbent_candidate_ids": (
                    incumbent_candidate_ids.detach().cpu().tolist()
                ),
                "challenger_candidate_ids": (
                    challenger_candidate_ids.detach().cpu().tolist()
                ),
                "challenger_to_incumbent_attention": (
                    challenger_attention.detach().cpu().tolist()
                ),
                "challenger_mask": challenger_mask.detach().cpu().tolist(),
                "residual": promotion_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_gatekeeper_challenger_paired_margin:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                paired_residual,
                gatekeeper_indices,
                challenger_indices,
                pair_hidden,
                pair_weights,
                pair_mask,
            ) = self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals[
                modality
            ](
                layer_states,
                condition,
                candidate_state,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + paired_residual
            gatekeeper_candidate_ids = candidate_ids.index_select(
                0, gatekeeper_indices.reshape(-1)
            ).reshape_as(gatekeeper_indices)
            challenger_candidate_ids = candidate_ids.index_select(
                0, challenger_indices.reshape(-1)
            ).reshape_as(challenger_indices)
            self._biogt_relbf_gatekeeper_challenger_paired_margin_attributions[
                modality
            ] = {
                "source": "gene9_parent_rank10_11_20_paired_margin",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "gatekeeper_candidate_ids": (
                    gatekeeper_candidate_ids.detach().cpu().tolist()
                ),
                "challenger_candidate_ids": (
                    challenger_candidate_ids.detach().cpu().tolist()
                ),
                "pair_features": pair_hidden.detach().cpu().tolist(),
                "pair_weights": pair_weights.detach().cpu().tolist(),
                "pair_mask": pair_mask.detach().cpu().tolist(),
                "residual": paired_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_boundary_prototype_ordinal_adapter:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                ordinal_residual, incumbent_indices, challenger_indices,
                active_indices, incumbent_attention, challenger_attention,
                ordinal_scores, ordinal_mask,
            ) = self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals[
                modality
            ](layer_states, condition, candidate_state, parent_compact_scores, valid)
            compact_scores = compact_scores + ordinal_residual
            def candidate_ids_for(indices):
                return candidate_ids.index_select(0, indices.reshape(-1)).reshape_as(indices)
            self._biogt_relbf_boundary_prototype_ordinal_adapter_attributions[modality] = {
                "source": "gene9_parent_boundary_prototype_ordinal_rank6_16",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "incumbent_prototype_candidate_ids": candidate_ids_for(incumbent_indices).detach().cpu().tolist(),
                "challenger_prototype_candidate_ids": candidate_ids_for(challenger_indices).detach().cpu().tolist(),
                "active_candidate_ids": candidate_ids_for(active_indices).detach().cpu().tolist(),
                "incumbent_prototype_attention": incumbent_attention.detach().cpu().tolist(),
                "challenger_prototype_attention": challenger_attention.detach().cpu().tolist(),
                "ordinal_score": ordinal_scores.detach().cpu().tolist(),
                "mask": ordinal_mask.detach().cpu().tolist(),
                "residual": ordinal_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_multi_depth_score_router:
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                router_residual,
                depth_gates,
                expert_scores,
                depth_contributions,
                router_mask,
            ) = self.biogt_relbf_multi_depth_score_router_residuals[modality](
                layer_states, condition, candidate_state, valid
            )
            compact_scores = compact_scores + router_residual
            self._biogt_relbf_multi_depth_score_router_attributions[modality] = {
                "source": "gene9_parent_full_candidate_query_multidepth_score_router",
                "depth_labels": list(
                    QueryConditionedMultiDepthScoreRouterResidual.depth_labels
                ),
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "depth_gates": depth_gates.detach().cpu().tolist(),
                "per_depth_expert_scores": expert_scores.detach().cpu().tolist(),
                "per_depth_contributions": depth_contributions.detach().cpu().tolist(),
                "mask": router_mask.detach().cpu().tolist(),
                "residual": router_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_dual_stream_trajectory_agreement_calibrator:
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                agreement_residual,
                agreement_features,
                trajectory_progress,
                calibration_score,
                agreement_mask,
            ) = self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals[
                modality
            ](layer_states, condition, candidate_state, valid)
            compact_scores = compact_scores + agreement_residual
            self._biogt_relbf_dual_stream_trajectory_agreement_calibrator_attributions[
                modality
            ] = {
                "source": "gene9_parent_full_candidate_dual_stream_trajectory_agreement",
                "agreement_labels": list(
                    DualStreamTrajectoryAgreementCalibratorResidual.agreement_labels
                ),
                "trajectory_progress_labels": list(
                    DualStreamTrajectoryAgreementCalibratorResidual.progress_labels
                ),
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "agreement_features": agreement_features.detach().cpu().tolist(),
                "trajectory_progress": trajectory_progress.detach().cpu().tolist(),
                "calibration_score": calibration_score.detach().cpu().tolist(),
                "mask": agreement_mask.detach().cpu().tolist(),
                "residual": agreement_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_multi_anchor_rank_landmark_calibrator:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                landmark_residual,
                landmark_indices,
                landmark_gates,
                landmark_contributions,
                landmark_candidate_mask,
                landmark_mask,
            ) = self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals[
                modality
            ](
                layer_states,
                condition,
                candidate_state,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + landmark_residual
            landmark_candidate_ids = candidate_ids.index_select(
                0, landmark_indices.reshape(-1)
            ).reshape_as(landmark_indices)
            self._biogt_relbf_multi_anchor_rank_landmark_calibrator_attributions[
                modality
            ] = {
                "source": "gene9_parent_full_candidate_multi_anchor_rank_landmark",
                "landmark_ranks": list(
                    QueryConditionedMultiAnchorRankLandmarkCalibratorResidual.landmark_ranks
                ),
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "landmark_candidate_ids": (
                    landmark_candidate_ids.detach().cpu().tolist()
                ),
                "landmark_gates": landmark_gates.detach().cpu().tolist(),
                "per_landmark_contributions": (
                    landmark_contributions.detach().cpu().tolist()
                ),
                "candidate_mask": landmark_candidate_mask.detach().cpu().tolist(),
                "landmark_mask": landmark_mask.detach().cpu().tolist(),
                "residual": landmark_residual.detach().cpu().tolist(),
            }
        if self.biogt_relbf_query_token_top64_candidate_set_transformer:
            parent_compact_scores = compact_scores.detach()
            layer_states = torch.stack([
                selected_shallow_state,
                selected_middle_state,
                selected_output_state,
            ], dim=2)
            (
                set_residual,
                selected_indices,
                set_attention,
                rank_positions,
                selected_contributions,
                selection_mask,
            ) = self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals[
                modality
            ](
                layer_states,
                condition,
                selected_features,
                parent_compact_scores,
                valid,
            )
            compact_scores = compact_scores + set_residual
            selected_candidate_ids = candidate_ids.index_select(
                0, selected_indices.reshape(-1)
            ).reshape_as(selected_indices)
            self._biogt_relbf_query_token_top64_candidate_set_transformer_attributions[
                modality
            ] = {
                "source": "gene9_parent_query_token_top64_candidate_set_transformer",
                "query_ids": query_ids.detach().cpu().tolist(),
                "candidate_ids": candidate_ids.detach().cpu().tolist(),
                "selected_candidate_ids": selected_candidate_ids.detach().cpu().tolist(),
                "attention": set_attention.detach().cpu().tolist(),
                "rank_positions": rank_positions.detach().cpu().tolist(),
                "selected_contributions": selected_contributions.detach().cpu().tolist(),
                "selection_mask": selection_mask.detach().cpu().tolist(),
                "residual": set_residual.detach().cpu().tolist(),
            }
        return compact_scores * valid.to(compact_scores.dtype).unsqueeze(0)

    def _biogt_neural_path_candidate_scores(
        self,
        modality,
        query_ids,
        candidate_ids,
        query_features,
    ):
        if self.biogt_neural_path_dim <= 0:
            return query_features.new_zeros(
                (query_ids.numel(), candidate_ids.numel())
            )
        if modality not in self._biogt_neural_path_adjacencies:
            raise RuntimeError(
                "Train-only compact BioGT neural graph was not configured "
                "before scoring"
            )

        boundary = torch.tanh(
            self.biogt_neural_path_query_projections[modality](
                query_features
            )
        )
        disease_count = self._biogt_neural_path_adjacencies[
            modality
        ].shape[0]
        state = boundary.new_zeros(
            (
                query_ids.numel(),
                disease_count,
                self.biogt_neural_path_dim,
            )
        )
        rows = torch.arange(query_ids.numel(), device=state.device)
        state[rows, query_ids] = boundary

        supports = self._biogt_path_support_modalities(modality)
        cycle_count = (
            1
            if self.biogt_path_propagation_hops == 3
            else 2
        )
        for cycle in range(cycle_count):
            messages = []
            for support in supports:
                propagated = self._biogt_neural_path_transition(
                    state,
                    support,
                    query_ids,
                    mask_query_relation=(support == modality),
                )
                messages.append(
                    self.biogt_neural_path_relation_transforms[
                        modality
                    ][support](propagated)
                )
            aggregate = sum(messages) / math.sqrt(len(messages))
            state = F.gelu(
                self.biogt_neural_path_state_norms[modality][cycle](
                    aggregate
                )
            )

        output_adjacency = self._biogt_neural_path_adjacencies[
            modality
        ]
        state = self._mask_batched_query_state(state, query_ids)
        batch_size, disease_count, hidden_dim = state.shape
        flattened = (
            state.permute(1, 0, 2)
            .reshape(disease_count, batch_size * hidden_dim)
        )
        target_state = torch.sparse.mm(
            output_adjacency.transpose(0, 1),
            flattened,
        )
        target_state = (
            target_state.reshape(
                output_adjacency.shape[1],
                batch_size,
                hidden_dim,
            )
            .permute(1, 0, 2)
        )

        compact_candidate_ids = self._biogt_neural_path_candidate_ids[
            modality
        ]
        candidate_features = self.input_adapters[modality](
            self.node_embeddings[modality](compact_candidate_ids)
        ).detach()
        candidate_features = (
            self.biogt_neural_path_candidate_projections[modality](
                candidate_features
            )
        )
        compact_scores = (
            target_state * candidate_features.unsqueeze(0)
        ).sum(dim=-1) / math.sqrt(self.biogt_neural_path_dim)

        lookup = self._biogt_neural_path_candidate_lookup[modality]
        positions = lookup.index_select(0, candidate_ids)
        valid = positions >= 0
        gathered = compact_scores.index_select(
            1,
            positions.clamp_min(0),
        )
        return gathered * valid.to(gathered.dtype).unsqueeze(0)

    def _biogt_path_candidate_matrix(
        self,
        modality,
        query_ids,
        query_features=None,
    ):
        if self.biogt_path_propagation_hops <= 0:
            candidate_count = self.node_embeddings[
                modality
            ].num_embeddings
            return next(self.parameters()).new_zeros(
                (query_ids.numel(), candidate_count)
            )
        output_adjacency = self._biogt_path_adjacencies.get(modality)
        if output_adjacency is None:
            raise RuntimeError(
                "Train-only BioGT path graph was not configured before "
                "scoring"
            )
        if query_features is None:
            query_features = self._biogt_path_query_features(query_ids)
        seed = query_features.new_zeros(
            (query_ids.numel(), output_adjacency.shape[0])
        )
        seed.scatter_(1, query_ids.unsqueeze(1), 1.0)

        supports = self._biogt_path_support_modalities(modality)
        mask_query_relation = self.training and self.biogt_relbf_dim > 0
        first_states = {
            support: self._biogt_path_disease_transition(
                seed,
                support,
                query_ids=query_ids,
                mask_query_relation=(
                    mask_query_relation and support == modality
                ),
            )
            for support in supports
        }
        path_features = []
        for support in supports:
            target_state = self._biogt_path_target_transition(
                first_states[support],
                modality,
                query_ids,
                mask_query_relation,
            )
            path_features.append(
                self._standardize_path_feature(target_state)
            )
        if self.biogt_path_propagation_hops == 5:
            for first_support in supports:
                for second_support in supports:
                    disease_state = self._biogt_path_disease_transition(
                        first_states[first_support],
                        second_support,
                        query_ids=query_ids,
                        mask_query_relation=(
                            mask_query_relation
                            and second_support == modality
                        ),
                    )
                    target_state = self._biogt_path_target_transition(
                        disease_state,
                        modality,
                        query_ids,
                        mask_query_relation,
                    )
                    path_features.append(
                        self._standardize_path_feature(target_state)
                    )

        gates = torch.tanh(
            self.biogt_path_query_gates[modality](query_features)
        )
        return sum(
            gates[:, index:index + 1] * feature
            for index, feature in enumerate(path_features)
        )

    def score_biogt_path_candidate_matrix(
        self,
        etype,
        query_ids,
        candidate_ids,
        detach_query_features=False,
        relbf_query_features=None,
        relbf_candidate_features=None,
    ):
        if self.biogt_path_propagation_hops <= 0:
            raise ValueError(
                "BioGT path candidate scoring requires path propagation"
            )
        modality = self._target_modality(etype)
        if modality is None:
            raise ValueError(f"No target modality is available for {etype}")
        candidate_is_source = etype[0] == modality
        query_type = etype[2] if candidate_is_source else etype[0]
        if query_type != "disease":
            raise ValueError(
                "BioGT path candidate scoring requires disease queries"
            )
        device = next(self.parameters()).device
        query_ids = torch.as_tensor(
            query_ids,
            dtype=torch.long,
            device=device,
        )
        candidate_ids = torch.as_tensor(
            candidate_ids,
            dtype=torch.long,
            device=device,
        )
        query_features = self._biogt_path_query_features(query_ids)
        if detach_query_features:
            query_features = query_features.detach()
        matrix = self._biogt_path_candidate_matrix(
            modality,
            query_ids,
            query_features=query_features,
        )
        scores = matrix.index_select(1, candidate_ids)
        if self.biogt_neural_path_dim > 0:
            scores = scores + self._biogt_neural_path_candidate_scores(
                modality,
                query_ids,
                candidate_ids,
                query_features,
            )
        if self.biogt_relbf_dim > 0:
            contextual_query = (
                query_features
                if relbf_query_features is None
                else torch.as_tensor(
                    relbf_query_features,
                    device=device,
                    dtype=query_features.dtype,
                )
            )
            contextual_candidates = (
                None
                if relbf_candidate_features is None
                else torch.as_tensor(
                    relbf_candidate_features,
                    device=device,
                    dtype=query_features.dtype,
                )
            )
            if detach_query_features:
                contextual_query = contextual_query.detach()
                if contextual_candidates is not None:
                    contextual_candidates = contextual_candidates.detach()
            scores = scores + self._biogt_relbf_candidate_scores(
                modality,
                query_ids,
                candidate_ids,
                contextual_query,
                contextual_candidates,
            )
        return scores

    def score_biogt_path_pairs(
        self,
        etype,
        src_ids,
        dst_ids,
        src_features=None,
        dst_features=None,
    ):
        modality = self._target_modality(etype)
        if (
            self.biogt_path_propagation_hops <= 0
            or modality is None
        ):
            return next(self.parameters()).new_zeros(
                torch.as_tensor(src_ids).shape
            )
        device = next(self.parameters()).device
        src_ids = torch.as_tensor(src_ids, dtype=torch.long, device=device)
        dst_ids = torch.as_tensor(dst_ids, dtype=torch.long, device=device)
        candidate_is_source = etype[0] == modality
        query_ids = dst_ids if candidate_is_source else src_ids
        candidate_ids = src_ids if candidate_is_source else dst_ids
        unique_queries, query_inverse = torch.unique(
            query_ids,
            sorted=True,
            return_inverse=True,
        )
        relbf_query_features = None
        relbf_candidate_features = None
        if self.biogt_relbf_dim > 0:
            if (src_features is None) != (dst_features is None):
                raise ValueError(
                    "RelBF pair scoring requires both source and "
                    "destination features"
                )
            if src_features is not None:
                src_features = torch.as_tensor(
                    src_features,
                    device=device,
                )
                dst_features = torch.as_tensor(
                    dst_features,
                    device=device,
                )
                edge_query_features = (
                    dst_features if candidate_is_source else src_features
                )
                relbf_candidate_features = (
                    src_features if candidate_is_source else dst_features
                )
                first_positions = torch.stack([
                    torch.nonzero(
                        query_inverse == index,
                        as_tuple=False,
                    )[0, 0]
                    for index in range(unique_queries.numel())
                ])
                relbf_query_features = edge_query_features.index_select(
                    0,
                    first_positions,
                )
        matrix = self.score_biogt_path_candidate_matrix(
            etype,
            unique_queries,
            candidate_ids,
            relbf_query_features=relbf_query_features,
            relbf_candidate_features=relbf_candidate_features,
        )
        return matrix[
            query_inverse,
            torch.arange(candidate_ids.numel(), device=device),
        ]

    def initialize_biogt_path_from_checkpoint(self):
        if (
            self.biogt_path_propagation_hops <= 0
            or not self.biogt_path_query_gates
            or self.scorer_mode == "hybrid_kge"
        ):
            raise ValueError(
                "BioGT path checkpoint migration requires an enabled pure "
                "BioGT path branch"
            )
        with torch.no_grad():
            for gate in self.biogt_path_query_gates.values():
                gate.weight.zero_()
                gate.bias.zero_()
        self.biogt_path_initialized_from_checkpoint = True
        return len(self.biogt_path_query_gates)

    def initialize_biogt_neural_path_from_checkpoint(self):
        if (
            self.biogt_neural_path_dim <= 0
            or not self.biogt_neural_path_candidate_projections
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT neural diffusion checkpoint migration requires an "
                "enabled pure BioGT branch"
            )
        with torch.no_grad():
            for projection in (
                self.biogt_neural_path_candidate_projections.values()
            ):
                projection.weight.zero_()
        self.biogt_neural_path_initialized_from_checkpoint = True
        return len(self.biogt_neural_path_candidate_projections)

    def initialize_biogt_relbf_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_readouts
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF checkpoint migration requires an enabled "
                "pure BioGT RelBF branch"
            )
        with torch.no_grad():
            for readout in self.biogt_relbf_readouts.values():
                readout[-1].weight.zero_()
                readout[-1].bias.zero_()
        self.biogt_relbf_initialized_from_checkpoint = True
        return len(self.biogt_relbf_readouts)

    def initialize_biogt_relbf_jk_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_jk_readout
            or not self.biogt_relbf_jk_readouts
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF JK checkpoint migration requires an enabled "
                "pure BioGT RelBF JK readout"
            )
        with torch.no_grad():
            for readout in self.biogt_relbf_jk_readouts.values():
                readout[-1].weight.zero_()
                readout[-1].bias.zero_()
        self.biogt_relbf_jk_initialized_from_checkpoint = True
        self.biogt_relbf_jk_migration_provenance = (
            "strict_zero_init_from_relbf_checkpoint"
        )
        return len(self.biogt_relbf_jk_readouts)

    def initialize_biogt_relbf_edge_attention_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_edge_attention
            or not self.biogt_relbf_edge_attention_queries
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF edge-attention checkpoint migration requires "
                "an enabled pure BioGT RelBF edge-attention branch"
            )
        with torch.no_grad():
            for target_attention in (
                self.biogt_relbf_edge_attention_queries.values()
            ):
                for projections in target_attention.values():
                    for projection in projections:
                        projection.weight.zero_()
        self.biogt_relbf_edge_attention_initialized_from_checkpoint = True
        return sum(
            len(projections)
            for target_attention in (
                self.biogt_relbf_edge_attention_queries.values()
            )
            for projections in target_attention.values()
        )

    def initialize_biogt_relbf_deep_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_deep_refinement
            or not self.biogt_relbf_deep_readouts
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF deep-refinement checkpoint migration requires "
                "an enabled pure BioGT deep-refinement branch"
            )
        with torch.no_grad():
            for readout in self.biogt_relbf_deep_readouts.values():
                readout[-1].weight.zero_()
                readout[-1].bias.zero_()
        self.biogt_relbf_deep_initialized_from_checkpoint = True
        self.biogt_relbf_deep_migration_provenance = (
            "strict_zero_residual_from_three_layer_relbf_checkpoint"
        )
        return len(self.biogt_relbf_deep_readouts)

    def initialize_biogt_relbf_indicator_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or not self.biogt_relbf_indicator_readouts
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF indicator checkpoint migration requires an "
                "enabled pure BioGT relation-only indicator branch"
            )
        with torch.no_grad():
            for readout in self.biogt_relbf_indicator_readouts.values():
                readout[-1].weight.zero_()
                readout[-1].bias.zero_()
        self.biogt_relbf_indicator_initialized_from_checkpoint = True
        return len(self.biogt_relbf_indicator_readouts)

    def initialize_biogt_relbf_indicator_support_gating_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or not self.biogt_relbf_indicator_support_gating
            or not self.biogt_relbf_indicator_support_gates
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "BioGT RelBF indicator support-gating checkpoint migration "
                "requires an enabled pure BioGT indicator gate branch"
            )
        gate_count = 0
        with torch.no_grad():
            for support_gates in (
                self.biogt_relbf_indicator_support_gates.values()
            ):
                for layer_gates in support_gates.values():
                    for gate in layer_gates:
                        gate.weight.zero_()
                        gate.bias.zero_()
                        gate_count += 1
        self.biogt_relbf_indicator_support_gating_initialized_from_checkpoint = (
            True
        )
        return gate_count

    def initialize_biogt_relbf_indicator_semantic_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or not self.biogt_relbf_indicator_semantic_readout
            or not self.biogt_relbf_indicator_semantic_readouts
            or self.biogt_relbf_indicator_support_gating
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT RelBF indicator semantic checkpoint migration "
                "requires an enabled pure BioGT semantic readout branch"
            )
        with torch.no_grad():
            for readout in (
                self.biogt_relbf_indicator_semantic_readouts.values()
            ):
                readout[-1].weight.zero_()
                readout[-1].bias.zero_()
        self.biogt_relbf_indicator_semantic_initialized_from_checkpoint = True
        return len(self.biogt_relbf_indicator_semantic_readouts)

    def initialize_biogt_relbf_query_relation_adapter_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or self.biogt_relbf_query_relation_adapter_rank <= 0
            or not self.biogt_relbf_query_relation_adapters
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT RelBF query-relation adapter checkpoint migration "
                "requires an enabled pure BioGT adapter branch"
            )
        adapter_count = 0
        with torch.no_grad():
            for target_adapters in (
                self.biogt_relbf_query_relation_adapters.values()
            ):
                for layer_adapters in target_adapters.values():
                    for adapter in layer_adapters:
                        adapter.output_up.weight.zero_()
                        adapter_count += 1
        self.biogt_relbf_query_relation_adapter_initialized_from_checkpoint = (
            True
        )
        return adapter_count

    def initialize_biogt_relbf_indicator_query_boundary_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or self.biogt_relbf_indicator_query_boundary_rank <= 0
            or not self.biogt_relbf_indicator_query_boundary_adapters
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT RelBF indicator query-boundary checkpoint migration "
                "requires an enabled pure BioGT adapter branch"
            )
        with torch.no_grad():
            for adapter in (
                self.biogt_relbf_indicator_query_boundary_adapters.values()
            ):
                adapter.output_up.weight.zero_()
        self.biogt_relbf_indicator_query_boundary_initialized_from_checkpoint = (
            True
        )
        return len(self.biogt_relbf_indicator_query_boundary_adapters)

    def initialize_biogt_relbf_indicator_multirule_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or self.biogt_relbf_indicator_multirule_count <= 0
            or not self.biogt_relbf_indicator_multirule_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT indicator multi-rule checkpoint migration requires "
                "an enabled pure BioGT multi-rule branch"
            )
        readout_count = 0
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_indicator_multirule_residuals.values()
            ):
                for readout in residual.rule_readouts:
                    readout[-1].weight.zero_()
                    readout_count += 1
        self.biogt_relbf_indicator_multirule_initialized_from_checkpoint = True
        return readout_count

    def initialize_biogt_relbf_indicator_relation_transition_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or self.biogt_relbf_indicator_relation_transition_layers <= 0
            or not self.biogt_relbf_indicator_relation_transition_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT indicator relation-transition checkpoint migration "
                "requires an enabled pure BioGT relation-transition branch"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_indicator_relation_transition_residuals.values()
            ):
                residual.readout_up.weight.zero_()
        self.biogt_relbf_indicator_relation_transition_initialized_from_checkpoint = (
            True
        )
        return len(
            self.biogt_relbf_indicator_relation_transition_residuals
        )

    def initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_indicator_residual
            or self.biogt_relbf_indicator_query_local_relation_transition_layers
            <= 0
            or not self.biogt_relbf_indicator_query_local_relation_transition_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT query-local relation-transition checkpoint migration "
                "requires an enabled pure BioGT query-local branch"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_indicator_query_local_relation_transition_residuals.values()
            ):
                residual.readout_up.weight.zero_()
        self.biogt_relbf_indicator_query_local_relation_transition_initialized_from_checkpoint = (
            True
        )
        return len(
            self.biogt_relbf_indicator_query_local_relation_transition_residuals
        )

    def initialize_biogt_relbf_path_layer_attention_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or self.biogt_relbf_path_layer_attention_rank <= 0
            or not self.biogt_relbf_path_layer_attention_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT RelBF path-layer attention checkpoint migration "
                "requires an enabled pure BioGT attention branch"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_path_layer_attention_residuals.values()
            ):
                residual.evidence_up.weight.zero_()
        self.biogt_relbf_path_layer_attention_initialized_from_checkpoint = True
        return len(self.biogt_relbf_path_layer_attention_residuals)

    def initialize_biogt_relbf_path_token_transformer_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_path_token_transformer
            or not self.biogt_relbf_path_token_transformer_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT path-token Transformer checkpoint migration requires "
                "an enabled pure BioGT branch"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_path_token_transformer_residuals.values()
            ):
                residual.score_readout.weight.zero_()
        self.biogt_relbf_path_token_transformer_initialized_from_checkpoint = True
        return len(self.biogt_relbf_path_token_transformer_residuals)

    def initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_candidate_competition_boundary
            or not self.biogt_relbf_candidate_competition_boundary_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT candidate-competition checkpoint migration requires "
                "an enabled pure BioGT branch"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_candidate_competition_boundary_residuals.values()
            ):
                residual.score_readout.weight.zero_()
        self.biogt_relbf_candidate_competition_boundary_initialized_from_checkpoint = True
        return len(
            self.biogt_relbf_candidate_competition_boundary_residuals
        )

    def initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_parent_top10_boundary_comparator_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT parent top10 boundary-comparator checkpoint migration "
                "requires an enabled pure BioGT branch"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_parent_top10_boundary_comparator_residuals.values()
            ):
                residual.score_readout.weight.zero_()
        self.biogt_relbf_parent_top10_boundary_comparator_initialized_from_checkpoint = True
        return len(
            self.biogt_relbf_parent_top10_boundary_comparator_residuals
        )

    def initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or not self.biogt_relbf_boundary_window_rank_slot_expert_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT boundary-window rank-slot expert checkpoint migration "
                "requires the enabled gene8 comparator parent structure"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_boundary_window_rank_slot_expert_residuals.values()
            ):
                residual.score_readout.weight.zero_()
        self.biogt_relbf_boundary_window_rank_slot_expert_initialized_from_checkpoint = True
        return len(
            self.biogt_relbf_boundary_window_rank_slot_expert_residuals
        )

    def initialize_biogt_relbf_boundary_exchange_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or not self.biogt_relbf_boundary_exchange
            or not self.biogt_relbf_boundary_exchange_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT boundary-exchange checkpoint migration requires the "
                "enabled gene9 parent structure"
            )
        with torch.no_grad():
            for residual in self.biogt_relbf_boundary_exchange_residuals.values():
                residual.score_readout.weight.zero_()
        self.biogt_relbf_boundary_exchange_initialized_from_checkpoint = True
        return len(self.biogt_relbf_boundary_exchange_residuals)

    def initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or not self.biogt_relbf_challenger_boundary_promotion
            or not self.biogt_relbf_challenger_boundary_promotion_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT challenger-only promotion migration requires the "
                "enabled gene9 parent structure without gene10"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_challenger_boundary_promotion_residuals.values()
            ):
                residual.score_readout.weight.zero_()
        self.biogt_relbf_challenger_boundary_promotion_initialized_from_checkpoint = True
        return len(self.biogt_relbf_challenger_boundary_promotion_residuals)

    def initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or not self.biogt_relbf_gatekeeper_challenger_paired_margin
            or not self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT paired-margin migration requires the enabled gene9 "
                "parent structure without gene10 or gene11"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals.values()
            ):
                residual.gatekeeper_readout.weight.zero_()
                residual.challenger_readout.weight.zero_()
        self.biogt_relbf_gatekeeper_challenger_paired_margin_initialized_from_checkpoint = True
        return len(self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals)

    def initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or not self.biogt_relbf_boundary_prototype_ordinal_adapter
            or not self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT ordinal-adapter migration requires the enabled gene9 "
                "parent without gene10, gene11, or gene12"
            )
        with torch.no_grad():
            for residual in self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals.values():
                residual.score_readout.weight.zero_()
        self.biogt_relbf_boundary_prototype_ordinal_adapter_initialized_from_checkpoint = True
        return len(self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals)

    def initialize_biogt_relbf_multi_depth_score_router_from_checkpoint(self):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or not self.biogt_relbf_multi_depth_score_router
            or not self.biogt_relbf_multi_depth_score_router_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT multi-depth score-router migration requires gene9"
            )
        with torch.no_grad():
            for residual in self.biogt_relbf_multi_depth_score_router_residuals.values():
                residual.depth_gain.zero_()
        self.biogt_relbf_multi_depth_score_router_initialized_from_checkpoint = True
        return len(self.biogt_relbf_multi_depth_score_router_residuals)

    def initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or self.biogt_relbf_multi_depth_score_router
            or not self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
            or not self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT dual-stream trajectory-agreement migration requires gene9"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals.values()
            ):
                residual.score_readout.weight.zero_()
        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_initialized_from_checkpoint = True
        return len(
            self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals
        )

    def initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or self.biogt_relbf_multi_depth_score_router
            or self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
            or not self.biogt_relbf_multi_anchor_rank_landmark_calibrator
            or not self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT multi-anchor rank-landmark migration requires gene9"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals.values()
            ):
                residual.final_gain.zero_()
        self.biogt_relbf_multi_anchor_rank_landmark_calibrator_initialized_from_checkpoint = True
        return len(
            self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals
        )

    def initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint(
        self,
    ):
        if (
            self.biogt_relbf_dim <= 0
            or not self.biogt_relbf_parent_top10_boundary_comparator
            or not self.biogt_relbf_boundary_window_rank_slot_expert
            or self.biogt_relbf_boundary_exchange
            or self.biogt_relbf_challenger_boundary_promotion
            or self.biogt_relbf_gatekeeper_challenger_paired_margin
            or self.biogt_relbf_boundary_prototype_ordinal_adapter
            or self.biogt_relbf_multi_depth_score_router
            or self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
            or self.biogt_relbf_multi_anchor_rank_landmark_calibrator
            or not self.biogt_relbf_query_token_top64_candidate_set_transformer
            or not self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals
            or self.scorer_mode not in {"concat_mlp", "interaction_mlp"}
        ):
            raise ValueError(
                "BioGT query-token top64 candidate-set migration requires gene9"
            )
        with torch.no_grad():
            for residual in (
                self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals.values()
            ):
                residual.residual_readout.weight.zero_()
        self.biogt_relbf_query_token_top64_candidate_set_transformer_initialized_from_checkpoint = True
        return len(
            self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals
        )

    def freeze_biogt_base_for_path_finetune(
        self,
        train_target_types=None,
        relbf_target_only=False,
        relbf_jk_only=False,
        relbf_edge_attention_only=False,
        relbf_deep_only=False,
        relbf_indicator_only=False,
        relbf_indicator_support_gate_only=False,
        relbf_indicator_semantic_only=False,
        relbf_query_relation_adapter_only=False,
        relbf_indicator_query_boundary_only=False,
        relbf_indicator_multirule_only=False,
        relbf_indicator_relation_transition_only=False,
        relbf_indicator_query_local_relation_transition_only=False,
        relbf_path_layer_attention_only=False,
        relbf_path_token_transformer_only=False,
        relbf_candidate_competition_boundary_only=False,
        relbf_parent_top10_boundary_comparator_only=False,
        relbf_boundary_window_rank_slot_expert_only=False,
        relbf_boundary_exchange_only=False,
        relbf_challenger_boundary_promotion_only=False,
        relbf_gatekeeper_challenger_paired_margin_only=False,
        relbf_boundary_prototype_ordinal_adapter_only=False,
        relbf_multi_depth_score_router_only=False,
        relbf_dual_stream_trajectory_agreement_calibrator_only=False,
        relbf_multi_anchor_rank_landmark_calibrator_only=False,
        relbf_query_token_top64_candidate_set_transformer_only=False,
    ):
        if (
            self.biogt_path_propagation_hops <= 0
            or not self.biogt_path_query_gates
            or self.scorer_mode not in {
                "concat_mlp",
                "interaction_mlp",
            }
        ):
            raise ValueError(
                "Pure BioGT path fine-tuning requires an enabled contextual "
                "path branch"
            )
        available_target_types = set(
            self.biogt_path_query_gates.keys()
        )
        requested_target_types = (
            available_target_types
            if train_target_types is None
            else {
                str(target_type).strip()
                for target_type in train_target_types
                if str(target_type).strip()
            }
        )
        if not requested_target_types:
            raise ValueError(
                "Pure BioGT path fine-tuning requires a target type"
            )
        unknown = requested_target_types - available_target_types
        if unknown:
            raise ValueError(
                "Unknown pure BioGT path fine-tuning types: "
                f"{sorted(unknown)}"
            )
        if (
            self.biogt_relbf_dim > 0
            and requested_target_types != available_target_types
            and not (
                relbf_target_only
                or relbf_jk_only
                or relbf_edge_attention_only
                or relbf_deep_only
                or relbf_indicator_only
                or relbf_indicator_support_gate_only
                or relbf_indicator_semantic_only
                or relbf_query_relation_adapter_only
                or relbf_indicator_query_boundary_only
                or relbf_indicator_multirule_only
                or relbf_indicator_relation_transition_only
                or relbf_indicator_query_local_relation_transition_only
                or relbf_path_layer_attention_only
                or relbf_path_token_transformer_only
                or relbf_candidate_competition_boundary_only
                or relbf_parent_top10_boundary_comparator_only
                or relbf_boundary_window_rank_slot_expert_only
                or relbf_boundary_exchange_only
                or relbf_challenger_boundary_promotion_only
                or relbf_gatekeeper_challenger_paired_margin_only
                or relbf_boundary_prototype_ordinal_adapter_only
                or relbf_multi_depth_score_router_only
                or relbf_dual_stream_trajectory_agreement_calibrator_only
                or relbf_multi_anchor_rank_landmark_calibrator_only
                or relbf_query_token_top64_candidate_set_transformer_only
            )
        ):
            raise ValueError(
                "BioGT RelBF shares PNA propagation parameters and therefore "
                "requires joint fine-tuning of all target modalities"
            )

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for target_type in requested_target_types:
            if self.biogt_relbf_dim > 0:
                if relbf_query_token_top64_candidate_set_transformer_only:
                    if not self.biogt_relbf_query_token_top64_candidate_set_transformer:
                        raise ValueError(
                            "RelBF query-token top64 candidate-set-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                if relbf_dual_stream_trajectory_agreement_calibrator_only:
                    if not self.biogt_relbf_dual_stream_trajectory_agreement_calibrator:
                        raise ValueError(
                            "RelBF trajectory-agreement-only fine-tuning requires an enabled branch"
                        )
                    for parameter in self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                if relbf_multi_anchor_rank_landmark_calibrator_only:
                    if not self.biogt_relbf_multi_anchor_rank_landmark_calibrator:
                        raise ValueError(
                            "RelBF multi-anchor landmark-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                if relbf_multi_depth_score_router_only:
                    if not self.biogt_relbf_multi_depth_score_router:
                        raise ValueError(
                            "RelBF multi-depth-router-only fine-tuning requires an enabled branch"
                        )
                    for parameter in self.biogt_relbf_multi_depth_score_router_residuals[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                if relbf_boundary_prototype_ordinal_adapter_only:
                    if not self.biogt_relbf_boundary_prototype_ordinal_adapter:
                        raise ValueError(
                            "RelBF ordinal-adapter-only fine-tuning requires an enabled branch"
                        )
                    for parameter in self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                if relbf_gatekeeper_challenger_paired_margin_only:
                    if not self.biogt_relbf_gatekeeper_challenger_paired_margin:
                        raise ValueError(
                            "RelBF paired-margin-only fine-tuning requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_challenger_boundary_promotion_only:
                    if not self.biogt_relbf_challenger_boundary_promotion:
                        raise ValueError(
                            "RelBF challenger-promotion-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_challenger_boundary_promotion_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_boundary_exchange_only:
                    if not self.biogt_relbf_boundary_exchange:
                        raise ValueError(
                            "RelBF boundary-exchange-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in self.biogt_relbf_boundary_exchange_residuals[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                if relbf_boundary_window_rank_slot_expert_only:
                    if not self.biogt_relbf_boundary_window_rank_slot_expert:
                        raise ValueError(
                            "RelBF boundary-window rank-slot-expert-only "
                            "fine-tuning requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_boundary_window_rank_slot_expert_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_parent_top10_boundary_comparator_only:
                    if not self.biogt_relbf_parent_top10_boundary_comparator:
                        raise ValueError(
                            "RelBF parent-top10-boundary-comparator-only "
                            "fine-tuning requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_parent_top10_boundary_comparator_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_candidate_competition_boundary_only:
                    if not self.biogt_relbf_candidate_competition_boundary:
                        raise ValueError(
                            "RelBF candidate-competition-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_candidate_competition_boundary_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_path_token_transformer_only:
                    if not self.biogt_relbf_path_token_transformer:
                        raise ValueError(
                            "RelBF path-token-Transformer-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_path_token_transformer_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_path_layer_attention_only:
                    if self.biogt_relbf_path_layer_attention_rank <= 0:
                        raise ValueError(
                            "RelBF path-layer-attention-only fine-tuning "
                            "requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_path_layer_attention_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_query_local_relation_transition_only:
                    if (
                        self.biogt_relbf_indicator_query_local_relation_transition_layers
                        <= 0
                    ):
                        raise ValueError(
                            "RelBF query-local relation-transition-only "
                            "fine-tuning requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_indicator_query_local_relation_transition_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_relation_transition_only:
                    if (
                        self.biogt_relbf_indicator_relation_transition_layers
                        <= 0
                    ):
                        raise ValueError(
                            "RelBF indicator relation-transition-only "
                            "fine-tuning requires an enabled branch"
                        )
                    for parameter in (
                        self.biogt_relbf_indicator_relation_transition_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_multirule_only:
                    if self.biogt_relbf_indicator_multirule_count <= 0:
                        raise ValueError(
                            "RelBF indicator multi-rule-only fine-tuning "
                            "requires an enabled multi-rule branch"
                        )
                    for parameter in (
                        self.biogt_relbf_indicator_multirule_residuals[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_query_boundary_only:
                    if self.biogt_relbf_indicator_query_boundary_rank <= 0:
                        raise ValueError(
                            "RelBF indicator query-boundary-only fine-tuning "
                            "requires an enabled adapter branch"
                        )
                    for parameter in (
                        self.biogt_relbf_indicator_query_boundary_adapters[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_query_relation_adapter_only:
                    if self.biogt_relbf_query_relation_adapter_rank <= 0:
                        raise ValueError(
                            "RelBF query-relation-adapter-only fine-tuning "
                            "requires an enabled adapter branch"
                        )
                    for parameter in (
                        self.biogt_relbf_query_relation_adapters[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_semantic_only:
                    if not self.biogt_relbf_indicator_semantic_readout:
                        raise ValueError(
                            "RelBF indicator semantic-only fine-tuning "
                            "requires the semantic readout branch"
                        )
                    for parameter in (
                        self.biogt_relbf_indicator_semantic_readouts[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_support_gate_only:
                    if not self.biogt_relbf_indicator_support_gating:
                        raise ValueError(
                            "RelBF indicator support-gate-only fine-tuning "
                            "requires the support-gating branch"
                        )
                    for parameter in (
                        self.biogt_relbf_indicator_support_gates[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_indicator_only:
                    if not self.biogt_relbf_indicator_residual:
                        raise ValueError(
                            "RelBF indicator-only fine-tuning requires the "
                            "relation-only indicator residual branch"
                        )
                    indicator_modules = (
                        self.biogt_relbf_indicator_message_transforms[
                            target_type
                        ],
                        self.biogt_relbf_indicator_pna_updates[target_type],
                        self.biogt_relbf_indicator_state_norms[target_type],
                        self.biogt_relbf_indicator_readouts[target_type],
                    )
                    for module in indicator_modules:
                        for parameter in module.parameters():
                            parameter.requires_grad_(True)
                    self.biogt_relbf_indicator_tokens[
                        target_type
                    ].requires_grad_(True)
                    continue
                if relbf_deep_only:
                    if not self.biogt_relbf_deep_refinement:
                        raise ValueError(
                            "RelBF deep-only fine-tuning requires the deep "
                            "refinement branch"
                        )
                    deep_modules = (
                        self.biogt_relbf_deep_message_transforms[target_type],
                        self.biogt_relbf_deep_pna_updates[target_type],
                        self.biogt_relbf_deep_state_norms[target_type],
                        self.biogt_relbf_deep_output_films[target_type],
                        self.biogt_relbf_deep_readouts[target_type],
                    )
                    for module in deep_modules:
                        for parameter in module.parameters():
                            parameter.requires_grad_(True)
                    continue
                if relbf_edge_attention_only:
                    if not self.biogt_relbf_edge_attention:
                        raise ValueError(
                            "RelBF edge-attention-only fine-tuning requires "
                            "the edge-attention branch"
                        )
                    for parameter in (
                        self.biogt_relbf_edge_attention_queries[
                            target_type
                        ].parameters()
                    ):
                        parameter.requires_grad_(True)
                    continue
                if relbf_jk_only:
                    if not self.biogt_relbf_jk_readout:
                        raise ValueError(
                            "RelBF JK-only fine-tuning requires the JK "
                            "readout"
                        )
                    for parameter in self.biogt_relbf_jk_readouts[
                        target_type
                    ].parameters():
                        parameter.requires_grad_(True)
                    continue
                relbf_modules = (
                    self.biogt_relbf_query_projections[target_type],
                    self.biogt_relbf_candidate_projections[target_type],
                    self.biogt_relbf_output_films[target_type],
                    self.biogt_relbf_readouts[target_type],
                )
                for module in relbf_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
                self.biogt_relbf_relation_tokens[
                    target_type
                ].requires_grad_(True)
                if not relbf_target_only:
                    for module in (
                        self.biogt_relbf_message_transforms,
                        self.biogt_relbf_pna_updates,
                        self.biogt_relbf_state_norms,
                    ):
                        for parameter in module.parameters():
                            parameter.requires_grad_(True)
            elif self.biogt_neural_path_dim <= 0:
                for parameter in self.biogt_path_query_gates[
                    target_type
                ].parameters():
                    parameter.requires_grad_(True)
            else:
                neural_modules = (
                    self.biogt_neural_path_query_projections[target_type],
                    self.biogt_neural_path_candidate_projections[
                        target_type
                    ],
                    self.biogt_neural_path_relation_transforms[
                        target_type
                    ],
                    self.biogt_neural_path_state_norms[target_type],
                )
                for module in neural_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        self.biogt_base_frozen_for_path_finetune = True
        self.biogt_relbf_target_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_target_only
        )
        self.biogt_relbf_jk_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_jk_only
        )
        self.biogt_relbf_edge_attention_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_edge_attention_only
        )
        self.biogt_relbf_deep_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_deep_only
        )
        self.biogt_relbf_indicator_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_indicator_only
        )
        self.biogt_relbf_indicator_support_gate_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_indicator_support_gate_only
        )
        self.biogt_relbf_indicator_semantic_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_indicator_semantic_only
        )
        self.biogt_relbf_query_relation_adapter_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_query_relation_adapter_only
        )
        self.biogt_relbf_indicator_query_boundary_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_indicator_query_boundary_only
        )
        self.biogt_relbf_indicator_multirule_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_indicator_multirule_only
        )
        self.biogt_relbf_indicator_relation_transition_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_indicator_relation_transition_only
        )
        self.biogt_relbf_indicator_query_local_relation_transition_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_indicator_query_local_relation_transition_only
        )
        self.biogt_relbf_path_layer_attention_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_path_layer_attention_only
        )
        self.biogt_relbf_path_token_transformer_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_path_token_transformer_only
        )
        self.biogt_relbf_candidate_competition_boundary_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_candidate_competition_boundary_only
        )
        self.biogt_relbf_parent_top10_boundary_comparator_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_parent_top10_boundary_comparator_only
        )
        self.biogt_relbf_boundary_window_rank_slot_expert_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_boundary_window_rank_slot_expert_only
        )
        self.biogt_relbf_boundary_exchange_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_boundary_exchange_only
        )
        self.biogt_relbf_challenger_boundary_promotion_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_challenger_boundary_promotion_only
        )
        self.biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_gatekeeper_challenger_paired_margin_only
        )
        self.biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_boundary_prototype_ordinal_adapter_only
        )
        self.biogt_relbf_multi_depth_score_router_only_finetune = bool(
            self.biogt_relbf_dim > 0 and relbf_multi_depth_score_router_only
        )
        self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_dual_stream_trajectory_agreement_calibrator_only
        )
        self.biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_multi_anchor_rank_landmark_calibrator_only
        )
        self.biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune = bool(
            self.biogt_relbf_dim > 0
            and relbf_query_token_top64_candidate_set_transformer_only
        )
        self.biogt_path_finetune_types = sorted(
            requested_target_types
        )
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not parameter.requires_grad
        )

    def _path_uses_cross_modal(self, modality):
        return (
            self.kge_path_cross_modal
            or modality in self.kge_path_cross_modal_types
            or modality in self.kge_path_cross_modal_support_types
        )

    def _path_support_modalities(self, modality):
        configured = self.kge_path_cross_modal_support_types.get(
            modality
        )
        if configured is not None:
            return sorted(configured)
        return sorted(self._kge_path_adjacencies)

    def _path_disease_transition(self, disease_state, modality):
        if not self._path_uses_cross_modal(modality):
            adjacency = self._kge_path_adjacencies[modality]
            target_state = torch.sparse.mm(
                adjacency.transpose(0, 1),
                disease_state.transpose(0, 1),
            )
            return torch.sparse.mm(
                adjacency,
                target_state,
            ).transpose(0, 1)

        contributions = []
        for support_modality in self._path_support_modalities(modality):
            adjacency = self._kge_path_adjacencies[support_modality]
            target_state = torch.sparse.mm(
                adjacency.transpose(0, 1),
                disease_state.transpose(0, 1),
            )
            contributions.append(
                torch.sparse.mm(adjacency, target_state).transpose(0, 1)
            )
        if not contributions:
            raise RuntimeError(
                "Cross-modal path propagation has no support modalities"
            )
        return torch.stack(contributions, dim=0).mean(dim=0)

    def _path_propagation_candidate_matrix(
        self,
        modality,
        query_ids,
        query_features,
    ):
        if self.kge_path_propagation_hops <= 0:
            return query_features.new_zeros((
                query_ids.numel(),
                self.kge_node_embeddings[modality].num_embeddings,
            ))
        adjacency = self._kge_path_adjacencies.get(modality)
        if adjacency is None:
            raise RuntimeError(
                "Train-only path graph was not configured before scoring"
            )
        seed = query_features.new_zeros((
            query_ids.numel(),
            adjacency.shape[0],
        ))
        seed.scatter_(1, query_ids.unsqueeze(1), 1.0)
        path_features = []
        disease_state = seed
        for hop in (3, 5):
            disease_state = self._path_disease_transition(
                disease_state,
                modality,
            )
            target_state = torch.sparse.mm(
                adjacency.transpose(0, 1),
                disease_state.transpose(0, 1),
            ).transpose(0, 1)
            centered = torch.log1p(target_state.clamp_min(0.0))
            centered = centered - centered.mean(dim=1, keepdim=True)
            centered = centered / centered.std(
                dim=1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)
            path_features.append(centered)
            if hop >= self.kge_path_propagation_hops:
                break
        gates = torch.tanh(
            self.kge_path_query_gates[modality](query_features)
        )
        return sum(
            gates[:, index:index + 1] * feature
            for index, feature in enumerate(path_features)
        )

    def _path_propagation_pair_scores(
        self,
        etype,
        src_ids,
        dst_ids,
    ):
        modality = self._target_modality(etype)
        if (
            self.kge_path_propagation_hops <= 0
            or modality is None
            or etype not in self.target_etypes
        ):
            return next(self.parameters()).new_zeros(src_ids.shape)
        candidate_is_source = etype[0] == modality
        query_ids = dst_ids if candidate_is_source else src_ids
        candidate_ids = src_ids if candidate_is_source else dst_ids
        unique_queries, inverse = torch.unique(
            query_ids,
            sorted=True,
            return_inverse=True,
        )
        source_features, target_features = self._kge_pair_features(
            etype,
            (
                candidate_ids[:1].expand(unique_queries.numel())
                if candidate_is_source
                else unique_queries
            ),
            (
                unique_queries
                if candidate_is_source
                else candidate_ids[:1].expand(unique_queries.numel())
            ),
        )
        query_features = (
            target_features if candidate_is_source else source_features
        )
        path_matrix = self._path_propagation_candidate_matrix(
            modality,
            unique_queries,
            query_features,
        )
        return path_matrix[inverse, candidate_ids]

    def initialize_path_propagation_from_checkpoint(self):
        if (
            self.scorer_mode != "hybrid_kge"
            or self.kge_path_propagation_hops <= 0
            or not self.kge_path_query_gates
        ):
            raise ValueError(
                "Path-propagation checkpoint migration requires an enabled "
                "hybrid_kge path branch"
            )
        with torch.no_grad():
            for gate in self.kge_path_query_gates.values():
                gate.weight.zero_()
                gate.bias.zero_()
        self.path_propagation_initialized_from_checkpoint = True
        return len(self.kge_path_query_gates)

    def freeze_kge_for_path_propagation_finetune(
        self,
        train_target_types=None,
    ):
        if (
            self.scorer_mode != "hybrid_kge"
            or self.kge_path_propagation_hops <= 0
            or not self.kge_path_query_gates
        ):
            raise ValueError(
                "Path-propagation fine-tuning requires an enabled "
                "hybrid_kge path branch"
            )
        available_target_types = set(self.kge_path_query_gates.keys())
        requested_target_types = (
            available_target_types
            if train_target_types is None
            else {
                str(target_type).strip()
                for target_type in train_target_types
                if str(target_type).strip()
            }
        )
        if not requested_target_types:
            raise ValueError(
                "Path-propagation fine-tuning requires a target type"
            )
        unknown = requested_target_types - available_target_types
        if unknown:
            raise ValueError(
                "Unknown path-propagation fine-tuning types: "
                f"{sorted(unknown)}"
            )
        frozen_parameters = []
        for embedding in self.kge_node_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_target_disease_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_candidate_biases.values():
            frozen_parameters.append(embedding.weight)
        frozen_parameters.extend(self.bilinear_relations.values())
        frozen_parameters.extend(self.bilinear_biases.values())
        frozen_parameters.extend(self.kge_pretrain_relations.values())
        frozen_parameters.extend(self.kge_pretrain_biases.values())
        frozen_parameters.extend(self.kge_target_left_factors.values())
        frozen_parameters.extend(self.kge_target_right_factors.values())
        frozen_parameters.extend(
            self.kge_target_residual_left_factors.values()
        )
        frozen_parameters.extend(
            self.kge_target_residual_right_factors.values()
        )
        frozen_parameters.extend(self.kge_pretrain_left_factors.values())
        frozen_parameters.extend(self.kge_pretrain_right_factors.values())
        for target_type in available_target_types - requested_target_types:
            frozen_parameters.extend(
                self.kge_path_query_gates[target_type].parameters()
            )
        for parameter in frozen_parameters:
            parameter.requires_grad_(False)
        self.kge_frozen_for_path_propagation_finetune = True
        self.path_propagation_finetune_types = sorted(
            requested_target_types
        )
        return sum(parameter.numel() for parameter in frozen_parameters)

    def freeze_kge_for_target_residual_finetune(
        self,
        train_target_types=None,
    ):
        if (
            self.scorer_mode != "hybrid_kge"
            or self.kge_target_residual_rank <= 0
            or not self.kge_target_residual_left_factors
        ):
            raise ValueError(
                "Target-residual fine-tuning requires hybrid_kge target "
                "residual factors"
            )
        available_target_types = set(
            self.kge_target_residual_left_factors.keys()
        )
        requested_target_types = (
            available_target_types
            if train_target_types is None
            else {
                str(target_type).strip()
                for target_type in train_target_types
                if str(target_type).strip()
            }
        )
        if not requested_target_types:
            raise ValueError(
                "Target-residual fine-tuning requires at least one target type"
            )
        unknown_target_types = (
            requested_target_types - available_target_types
        )
        if unknown_target_types:
            raise ValueError(
                "Unknown target-residual fine-tuning types: "
                f"{sorted(unknown_target_types)}"
            )
        frozen_parameters = []
        for embedding in self.kge_node_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_target_disease_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_candidate_biases.values():
            frozen_parameters.append(embedding.weight)
        frozen_parameters.extend(self.bilinear_relations.values())
        frozen_parameters.extend(self.bilinear_biases.values())
        frozen_parameters.extend(self.kge_pretrain_relations.values())
        frozen_parameters.extend(self.kge_pretrain_biases.values())
        frozen_parameters.extend(self.kge_target_left_factors.values())
        frozen_parameters.extend(self.kge_target_right_factors.values())
        frozen_parameters.extend(self.kge_pretrain_left_factors.values())
        frozen_parameters.extend(self.kge_pretrain_right_factors.values())
        for target_type in available_target_types:
            if target_type not in requested_target_types:
                frozen_parameters.append(
                    self.kge_target_residual_left_factors[target_type]
                )
                frozen_parameters.append(
                    self.kge_target_residual_right_factors[target_type]
                )
        for parameter in frozen_parameters:
            parameter.requires_grad_(False)
        self.kge_frozen_for_target_residual_finetune = True
        self.target_residual_finetune_types = sorted(
            requested_target_types
        )
        return sum(parameter.numel() for parameter in frozen_parameters)

    def initialize_target_residual_from_checkpoint(self):
        if (
            self.scorer_mode != "hybrid_kge"
            or self.kge_target_residual_rank <= 0
            or not self.kge_target_residual_left_factors
        ):
            raise ValueError(
                "Target-residual checkpoint migration requires hybrid_kge "
                "target residual factors"
            )
        with torch.no_grad():
            for left in self.kge_target_residual_left_factors.values():
                left.zero_()
            for right in self.kge_target_residual_right_factors.values():
                nn.init.xavier_uniform_(right)
        self.target_residual_initialized_from_checkpoint = True
        return len(self.kge_target_residual_left_factors)

    def freeze_kge_for_candidate_bias_finetune(
        self,
        train_target_types=None,
    ):
        if (
            self.scorer_mode != "hybrid_kge"
            or not self.kge_candidate_bias
            or not self.kge_candidate_biases
        ):
            raise ValueError(
                "Candidate-bias fine-tuning requires hybrid_kge candidate "
                "bias tables"
            )
        available_target_types = set(self.kge_candidate_biases.keys())
        requested_target_types = (
            available_target_types
            if train_target_types is None
            else {
                str(target_type).strip()
                for target_type in train_target_types
                if str(target_type).strip()
            }
        )
        if not requested_target_types:
            raise ValueError(
                "Candidate-bias fine-tuning requires at least one target type"
            )
        unknown_target_types = (
            requested_target_types - available_target_types
        )
        if unknown_target_types:
            raise ValueError(
                "Unknown candidate-bias fine-tuning types: "
                f"{sorted(unknown_target_types)}"
            )
        frozen_parameters = []
        for embedding in self.kge_node_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for embedding in self.kge_target_disease_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for target_type, embedding in self.kge_candidate_biases.items():
            if target_type not in requested_target_types:
                frozen_parameters.append(embedding.weight)
        frozen_parameters.extend(self.bilinear_relations.values())
        frozen_parameters.extend(self.bilinear_biases.values())
        frozen_parameters.extend(self.kge_pretrain_relations.values())
        frozen_parameters.extend(self.kge_pretrain_biases.values())
        frozen_parameters.extend(self.kge_target_left_factors.values())
        frozen_parameters.extend(self.kge_target_right_factors.values())
        frozen_parameters.extend(
            self.kge_target_residual_left_factors.values()
        )
        frozen_parameters.extend(
            self.kge_target_residual_right_factors.values()
        )
        frozen_parameters.extend(self.kge_pretrain_left_factors.values())
        frozen_parameters.extend(self.kge_pretrain_right_factors.values())
        for parameter in frozen_parameters:
            parameter.requires_grad_(False)
        self.kge_frozen_for_candidate_bias_finetune = True
        self.candidate_bias_finetune_types = sorted(
            requested_target_types
        )
        return sum(parameter.numel() for parameter in frozen_parameters)

    def initialize_candidate_biases_from_checkpoint(self):
        if (
            self.scorer_mode != "hybrid_kge"
            or not self.kge_candidate_bias
            or not self.kge_candidate_biases
        ):
            raise ValueError(
                "Candidate-bias checkpoint migration requires hybrid_kge "
                "candidate bias tables"
            )
        with torch.no_grad():
            for embedding in self.kge_candidate_biases.values():
                embedding.weight.zero_()
        self.candidate_bias_initialized_from_checkpoint = True
        return len(self.kge_candidate_biases)

    def freeze_shared_kge_for_target_disease_finetune(
        self,
        train_target_types=None,
    ):
        if (
            self.scorer_mode != "hybrid_kge"
            or not self.kge_target_specific_disease_embeddings
            or not self.kge_target_disease_embeddings
        ):
            raise ValueError(
                "Shared KGE freezing requires target-specific hybrid_kge "
                "disease embeddings"
            )
        available_target_types = set(
            self.kge_target_disease_embeddings.keys()
        )
        requested_target_types = (
            available_target_types
            if train_target_types is None
            else {
                str(target_type).strip()
                for target_type in train_target_types
                if str(target_type).strip()
            }
        )
        if not requested_target_types:
            raise ValueError(
                "Target-disease fine-tuning requires at least one target type"
            )
        unknown_target_types = (
            requested_target_types - available_target_types
        )
        if unknown_target_types:
            raise ValueError(
                "Unknown target-disease fine-tuning types: "
                f"{sorted(unknown_target_types)}"
            )
        frozen_parameters = []
        for embedding in self.kge_node_embeddings.values():
            frozen_parameters.append(embedding.weight)
        for target_type, embedding in (
            self.kge_target_disease_embeddings.items()
        ):
            if target_type not in requested_target_types:
                frozen_parameters.append(embedding.weight)
        for embedding in self.kge_candidate_biases.values():
            frozen_parameters.append(embedding.weight)
        frozen_parameters.extend(self.bilinear_relations.values())
        frozen_parameters.extend(self.bilinear_biases.values())
        frozen_parameters.extend(self.kge_pretrain_relations.values())
        frozen_parameters.extend(self.kge_pretrain_biases.values())
        frozen_parameters.extend(self.kge_target_left_factors.values())
        frozen_parameters.extend(self.kge_target_right_factors.values())
        frozen_parameters.extend(
            self.kge_target_residual_left_factors.values()
        )
        frozen_parameters.extend(
            self.kge_target_residual_right_factors.values()
        )
        frozen_parameters.extend(self.kge_pretrain_left_factors.values())
        frozen_parameters.extend(self.kge_pretrain_right_factors.values())
        for parameter in frozen_parameters:
            parameter.requires_grad_(False)
        self.shared_kge_frozen_for_target_disease_finetune = True
        self.target_disease_finetune_types = sorted(
            requested_target_types
        )
        return sum(parameter.numel() for parameter in frozen_parameters)

    def initialize_target_disease_embeddings_from_shared(self):
        if (
            self.scorer_mode != "hybrid_kge"
            or not self.kge_target_specific_disease_embeddings
            or not self.kge_target_disease_embeddings
        ):
            raise ValueError(
                "Shared-checkpoint disease initialization requires "
                "target-specific hybrid_kge disease embeddings"
            )
        shared_weight = self.kge_node_embeddings["disease"].weight
        with torch.no_grad():
            for embedding in self.kge_target_disease_embeddings.values():
                if embedding.weight.shape != shared_weight.shape:
                    raise ValueError(
                        "Target-specific and shared disease embedding shapes "
                        "must match"
                    )
                embedding.weight.copy_(shared_weight)
        self.target_disease_initialized_from_shared_checkpoint = True
        return len(self.kge_target_disease_embeddings)

    def sparse_embedding_parameters(self):
        return [
            embedding.weight
            for embedding in self.node_embeddings.values()
            if embedding.weight.requires_grad
        ] + [
            embedding.weight
            for embedding in self.kge_node_embeddings.values()
            if embedding.weight.requires_grad
        ] + [
            embedding.weight
            for embedding in self.kge_target_disease_embeddings.values()
            if embedding.weight.requires_grad
        ] + [
            embedding.weight
            for embedding in self.kge_candidate_biases.values()
            if embedding.weight.requires_grad
        ]

    def dense_parameters(self):
        sparse_ids = {id(parameter) for parameter in self.sparse_embedding_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in sparse_ids
        ]

    def clear_embedding_cache(self):
        self._embedding_cache.clear()
        self._cache_generation += 1

    def _global_node_ids(self, graph, node_type):
        node_data = graph.nodes[node_type].data
        if dgl.NID in node_data:
            return node_data[dgl.NID].long()
        return torch.arange(
            graph.num_nodes(node_type),
            device=graph.device,
            dtype=torch.long,
        )

    def lookup_node_features(self, graph, initial_features=None):
        device = next(self.parameters()).device
        prepared = {}
        for node_type in self.ntypes:
            node_count = graph.num_nodes(node_type)
            if node_count == 0:
                prepared[node_type] = torch.empty(
                    (0, self.hidden_dim),
                    device=device,
                )
                continue

            global_ids = self._global_node_ids(graph, node_type)
            if initial_features is not None and node_type in initial_features:
                raw_features = initial_features[node_type]
                if raw_features.shape[0] != node_count:
                    raw_features = raw_features.index_select(
                        0,
                        global_ids.to(raw_features.device),
                    )
                raw_features = raw_features.to(device=device, dtype=torch.float32)
            else:
                raw_features = self.node_embeddings[node_type](
                    global_ids.to(device)
                )
            prepared[node_type] = self.input_adapters[node_type](raw_features)
            if (
                self.scorer_mode == "hybrid_kge"
                and self.kge_context_input_weight > 0
                and (
                    node_type in self.target_node_types
                    or node_type == "disease"
                )
            ):
                kge_features = self.kge_node_embeddings[node_type](
                    global_ids.to(device)
                )
                prepared[node_type] = (
                    prepared[node_type]
                    + self.kge_context_input_weight
                    * F.layer_norm(
                        kge_features,
                        (self.hidden_dim,),
                    )
                )
        return prepared

    def encode_branches(self, graph, initial_features=None, edge_weights=None):
        adapted_features = self.lookup_node_features(graph, initial_features)
        if self.hgnn is None:
            graph_features = adapted_features
        else:
            if edge_weights is None:
                edge_weights = {
                    etype: graph.edges[etype].data["weight"]
                    for etype in graph.canonical_etypes
                    if "weight" in graph.edges[etype].data
                }
            graph_features = self.hgnn(
                graph,
                dict(adapted_features),
                edge_weights=edge_weights,
            )
            for node_type, adapted in adapted_features.items():
                if node_type not in graph_features:
                    graph_features[node_type] = adapted

        if self.disable_transformer:
            transformer_features = graph_features
        else:
            transformer_features = self.sparse_transformer(
                graph_features,
                graph,
            )
            for node_type, graph_feature in graph_features.items():
                if node_type not in transformer_features:
                    transformer_features[node_type] = graph_feature

        return {
            "input": adapted_features,
            "graph": graph_features,
            "transformer": transformer_features,
        }

    def modality_embeddings(self, branches, modality, node_types=None):
        adapter = self.modality_adapters[modality]
        requested_node_types = (
            set(branches["graph"])
            if node_types is None
            else set(node_types)
        )
        return {
            node_type: adapter(
                branches["graph"][node_type],
                branches["transformer"][node_type],
            )
            for node_type in branches["graph"]
            if node_type in requested_node_types
        }

    def get_embeddings(
        self,
        graph,
        initial_features=None,
        edge_weights=None,
        return_branch_outputs=False,
    ):
        branches = self.encode_branches(
            graph,
            initial_features=initial_features,
            edge_weights=edge_weights,
        )
        generic = {
            node_type: 0.5 * (
                branches["graph"][node_type]
                + branches["transformer"][node_type]
            )
            for node_type in branches["graph"]
        }
        if return_branch_outputs:
            return generic, branches
        return generic

    def _target_modality(self, etype):
        if etype[2] in self.target_node_types:
            return etype[2]
        if etype[0] in self.target_node_types:
            return etype[0]
        return None

    def _global_bilinear_scores(
        self,
        modality,
        global_pair_features,
    ):
        source_features, target_features = global_pair_features.chunk(
            2,
            dim=-1,
        )
        if (
            self.scorer_mode == "hybrid_kge"
            and self.kge_relation_model == "lowrank_bilinear"
        ):
            return self._lowrank_bilinear_scores(
                source_features,
                target_features,
                self.kge_target_left_factors[modality],
                self.kge_target_right_factors[modality],
                self.bilinear_biases[modality],
            )
        return (
            source_features
            * self.bilinear_relations[modality]
            * target_features
        ).sum(dim=-1) / math.sqrt(self.hidden_dim) + self.bilinear_biases[
            modality
        ]

    def _lowrank_bilinear_scores(
        self,
        source_features,
        target_features,
        left_factor,
        right_factor,
        bias,
    ):
        source_projection = source_features @ left_factor
        target_projection = target_features @ right_factor
        return (
            source_projection * target_projection
        ).sum(dim=-1) / math.sqrt(self.kge_relation_rank) + bias

    def score_global_pairs(self, etype, src_ids, dst_ids):
        modality = self._target_modality(etype)
        if modality is None:
            raise ValueError(f"No target modality is available for {etype}")
        device = next(self.parameters()).device
        src_ids = torch.as_tensor(
            src_ids,
            dtype=torch.long,
            device=device,
        )
        dst_ids = torch.as_tensor(
            dst_ids,
            dtype=torch.long,
            device=device,
        )
        if src_ids.numel() != dst_ids.numel():
            raise ValueError("src_ids and dst_ids must have equal lengths")
        if self.scorer_mode == "hybrid_kge":
            source_features, target_features = self._kge_pair_features(
                etype,
                src_ids,
                dst_ids,
            )
        else:
            source_features = self.input_adapters[etype[0]](
                self.node_embeddings[etype[0]](src_ids)
            )
            target_features = self.input_adapters[etype[2]](
                self.node_embeddings[etype[2]](dst_ids)
            )
        scores = self._global_bilinear_scores(
            modality,
            torch.cat([source_features, target_features], dim=-1),
        )
        scores = scores + self._target_residual_scores(
            etype,
            source_features,
            target_features,
        )
        scores = scores + self._candidate_bias_for_pairs(
            etype,
            src_ids,
            dst_ids,
        )
        return scores + self._path_propagation_pair_scores(
            etype,
            src_ids,
            dst_ids,
        )

    def _candidate_bias_for_pairs(self, etype, src_ids, dst_ids):
        modality = self._target_modality(etype)
        if (
            not self.kge_candidate_bias
            or etype not in self.target_etypes
            or modality not in self.kge_candidate_biases
        ):
            return next(self.parameters()).new_zeros(src_ids.shape)
        candidate_ids = src_ids if etype[0] == modality else dst_ids
        return self.kge_candidate_biases[modality](
            candidate_ids
        ).squeeze(-1)

    def _target_residual_scores(
        self,
        etype,
        source_features,
        target_features,
    ):
        modality = self._target_modality(etype)
        if (
            self.kge_target_residual_rank <= 0
            or etype not in self.target_etypes
            or modality not in self.kge_target_residual_left_factors
        ):
            return source_features.new_zeros(source_features.shape[0])
        source_projection = (
            source_features
            @ self.kge_target_residual_left_factors[modality]
        )
        target_projection = (
            target_features
            @ self.kge_target_residual_right_factors[modality]
        )
        return (
            source_projection * target_projection
        ).sum(dim=-1) / math.sqrt(self.kge_target_residual_rank)

    def _kge_pair_features(self, etype, src_ids, dst_ids):
        source_features = self.kge_node_embeddings[etype[0]](src_ids)
        target_features = self.kge_node_embeddings[etype[2]](dst_ids)
        modality = self._target_modality(etype)
        if (
            self.kge_target_specific_disease_embeddings
            and etype in self.target_etypes
            and modality in self.kge_target_disease_embeddings
        ):
            disease_embedding = self.kge_target_disease_embeddings[modality]
            if etype[0] == "disease":
                source_features = self._mix_target_disease_features(
                    source_features,
                    disease_embedding(src_ids),
                )
            if etype[2] == "disease":
                target_features = self._mix_target_disease_features(
                    target_features,
                    disease_embedding(dst_ids),
                )
        return source_features, target_features

    def _mix_target_disease_features(
        self,
        shared_features,
        target_features,
    ):
        shared_weight = self.kge_target_disease_shared_weight
        if shared_weight == 0:
            return target_features
        return (
            target_features + shared_weight * shared_features
        ) / math.sqrt(1.0 + shared_weight ** 2)

    def score_kge_candidate_matrix(self, etype, query_ids, candidate_ids):
        if self.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Candidate-matrix scoring requires scorer_mode=hybrid_kge"
            )
        modality = self._target_modality(etype)
        if modality is None or modality not in {etype[0], etype[2]}:
            raise ValueError(f"No target modality is available for {etype}")
        device = next(self.parameters()).device
        query_ids = torch.as_tensor(
            query_ids,
            dtype=torch.long,
            device=device,
        )
        candidate_ids = torch.as_tensor(
            candidate_ids,
            dtype=torch.long,
            device=device,
        )
        candidate_is_source = etype[0] == modality
        query_type = etype[2] if candidate_is_source else etype[0]
        if (
            self.kge_target_specific_disease_embeddings
            and etype in self.target_etypes
            and query_type == "disease"
        ):
            query_features = self._mix_target_disease_features(
                self.kge_node_embeddings[query_type](query_ids),
                self.kge_target_disease_embeddings[modality](query_ids),
            )
        else:
            query_features = self.kge_node_embeddings[query_type](query_ids)
        candidate_features = self.kge_node_embeddings[modality](
            candidate_ids
        )
        if self.kge_relation_model == "lowrank_bilinear":
            left_factor = self.kge_target_left_factors[modality]
            right_factor = self.kge_target_right_factors[modality]
            if candidate_is_source:
                source_projection = candidate_features @ left_factor
                target_projection = query_features @ right_factor
                scores = target_projection @ source_projection.transpose(0, 1)
            else:
                source_projection = query_features @ left_factor
                target_projection = candidate_features @ right_factor
                scores = source_projection @ target_projection.transpose(0, 1)
            scores = (
                scores / math.sqrt(self.kge_relation_rank)
                + self.bilinear_biases[modality]
            )
        else:
            relation = self.bilinear_relations[modality]
            scores = (
                query_features * relation
            ) @ candidate_features.transpose(0, 1)
            scores = (
                scores / math.sqrt(self.hidden_dim)
                + self.bilinear_biases[modality]
            )
        if self.kge_candidate_bias:
            scores = scores + self.kge_candidate_biases[modality](
                candidate_ids
            ).squeeze(-1).unsqueeze(0)
        if self.kge_target_residual_rank > 0:
            left_factor = self.kge_target_residual_left_factors[
                modality
            ]
            right_factor = self.kge_target_residual_right_factors[
                modality
            ]
            if candidate_is_source:
                source_projection = candidate_features @ left_factor
                target_projection = query_features @ right_factor
                residual_scores = (
                    target_projection
                    @ source_projection.transpose(0, 1)
                )
            else:
                source_projection = query_features @ left_factor
                target_projection = candidate_features @ right_factor
                residual_scores = (
                    source_projection
                    @ target_projection.transpose(0, 1)
                )
            scores = scores + (
                residual_scores
                / math.sqrt(self.kge_target_residual_rank)
            )
        if self.kge_path_propagation_hops > 0:
            path_scores = self._path_propagation_candidate_matrix(
                modality,
                query_ids,
                query_features,
            )
            scores = scores + path_scores.index_select(
                1,
                candidate_ids,
            )
        return scores

    def score_kge_pretraining_pairs(self, etype, src_ids, dst_ids):
        if self.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Relation-wide KGE pretraining requires scorer_mode=hybrid_kge"
            )
        device = next(self.parameters()).device
        src_ids = torch.as_tensor(
            src_ids,
            dtype=torch.long,
            device=device,
        )
        dst_ids = torch.as_tensor(
            dst_ids,
            dtype=torch.long,
            device=device,
        )
        if src_ids.numel() != dst_ids.numel():
            raise ValueError("src_ids and dst_ids must have equal lengths")
        source_features, target_features = self._kge_pair_features(
            etype,
            src_ids,
            dst_ids,
        )
        modality = self._target_modality(etype)
        if etype in self.target_etypes and modality is not None:
            if self.kge_relation_model == "lowrank_bilinear":
                scores = self._lowrank_bilinear_scores(
                    source_features,
                    target_features,
                    self.kge_target_left_factors[modality],
                    self.kge_target_right_factors[modality],
                    self.bilinear_biases[modality],
                )
            else:
                relation = self.bilinear_relations[modality]
                bias = self.bilinear_biases[modality]
                scores = (
                    source_features
                    * relation
                    * target_features
                ).sum(dim=-1) / math.sqrt(self.hidden_dim) + bias
            scores = scores + self._target_residual_scores(
                etype,
                source_features,
                target_features,
            )
            return scores + self._candidate_bias_for_pairs(
                etype,
                src_ids,
                dst_ids,
            )
        else:
            relation_key = self.kge_pretrain_relation_keys.get(etype)
            if relation_key not in self.kge_pretrain_relations:
                raise ValueError(
                    f"No train-relation KGE parameter is available for {etype}"
                )
            relation = self.kge_pretrain_relations[relation_key]
            bias = self.kge_pretrain_biases[relation_key]
            if self.kge_relation_model == "lowrank_bilinear":
                return self._lowrank_bilinear_scores(
                    source_features,
                    target_features,
                    self.kge_pretrain_left_factors[relation_key],
                    self.kge_pretrain_right_factors[relation_key],
                    bias,
                )
        return (
            source_features
            * relation
            * target_features
        ).sum(dim=-1) / math.sqrt(self.hidden_dim) + bias

    def _predict_pairs(
        self,
        modality,
        pair_features,
        global_pair_features=None,
        independent_scores=None,
    ):
        source_features, target_features = pair_features.chunk(2, dim=-1)
        bilinear_scores = None
        if self.scorer_mode in {
            "bilinear",
            "hybrid_bilinear",
            "hybrid_global_bilinear",
            "hybrid_kge",
        }:
            if self.scorer_mode == "hybrid_kge":
                if independent_scores is None:
                    raise ValueError(
                        "hybrid_kge requires independent KGE scores"
                    )
                bilinear_scores = independent_scores
            elif self.scorer_mode == "hybrid_global_bilinear":
                if global_pair_features is None:
                    raise ValueError(
                        "hybrid_global_bilinear requires global pair features"
                    )
                source_features, target_features = (
                    global_pair_features.chunk(2, dim=-1)
                )
            if bilinear_scores is None:
                bilinear_scores = self._global_bilinear_scores(
                    modality,
                    torch.cat(
                        [source_features, target_features],
                        dim=-1,
                    ),
                )
            if self.scorer_mode == "bilinear":
                return bilinear_scores, None

        if self.scorer_mode in {
            "interaction_mlp",
            "hybrid_bilinear",
            "hybrid_global_bilinear",
            "hybrid_kge",
        }:
            source_features, target_features = pair_features.chunk(2, dim=-1)
            pair_features = torch.cat(
                [
                    source_features,
                    target_features,
                    source_features * target_features,
                    torch.abs(source_features - target_features),
                ],
                dim=-1,
            )
        if self.disable_moe:
            scores = self.dense_predictors[modality](
                pair_features
            ).reshape(-1)
            if bilinear_scores is not None:
                if self.scorer_mode in {
                    "hybrid_global_bilinear",
                    "hybrid_kge",
                }:
                    scores = (
                        self.contextual_score_weight * scores
                        + self.global_bilinear_score_weight
                        * bilinear_scores
                    )
                else:
                    scores = scores + bilinear_scores
            return scores, None
        predictor = self.expert_predictors[modality]
        scores = predictor(pair_features).reshape(-1)
        if bilinear_scores is not None:
            if self.scorer_mode in {
                "hybrid_global_bilinear",
                "hybrid_kge",
            }:
                scores = (
                    self.contextual_score_weight * scores
                    + self.global_bilinear_score_weight * bilinear_scores
                )
            else:
                scores = scores + bilinear_scores
        return scores, predictor.last_aux_loss

    def _score_graph(self, graph_to_score, modality_cache, active_target_etypes):
        scores = {}
        global_scores = {}
        auxiliary_losses = []
        if graph_to_score is None:
            return scores, auxiliary_losses, global_scores

        for etype in graph_to_score.canonical_etypes:
            if etype not in active_target_etypes or graph_to_score.num_edges(etype) == 0:
                continue
            modality = self._target_modality(etype)
            if modality is None:
                continue
            modality_cache.setdefault(modality, {})
            missing_node_types = {
                etype[0],
                etype[2],
            } - set(modality_cache[modality])
            if missing_node_types:
                modality_cache[modality].update(
                    self.modality_embeddings(
                        modality_cache["branches"],
                        modality,
                        node_types=missing_node_types,
                    )
                )
            embeddings = modality_cache[modality]
            src_nodes, dst_nodes = graph_to_score.edges(etype=etype)
            score_device = embeddings[etype[0]].device
            src_nodes = src_nodes.to(device=score_device, dtype=torch.long)
            dst_nodes = dst_nodes.to(device=score_device, dtype=torch.long)
            pair_features = torch.cat(
                [
                    embeddings[etype[0]].index_select(0, src_nodes.long()),
                    embeddings[etype[2]].index_select(0, dst_nodes.long()),
                ],
                dim=-1,
            )
            src_features, dst_features = pair_features.chunk(2, dim=-1)
            global_pair_features = None
            independent_scores = None
            if self.scorer_mode == "hybrid_global_bilinear":
                input_features = modality_cache["branches"]["input"]
                global_pair_features = torch.cat(
                    [
                        input_features[etype[0]].index_select(
                            0,
                            src_nodes.long(),
                        ),
                        input_features[etype[2]].index_select(
                            0,
                            dst_nodes.long(),
                        ),
                    ],
                    dim=-1,
                )
                global_scores[etype] = self._global_bilinear_scores(
                    modality,
                    global_pair_features,
                )
            elif self.scorer_mode == "hybrid_kge":
                src_global_nodes = self._global_node_ids(
                    graph_to_score,
                    etype[0],
                )
                dst_global_nodes = self._global_node_ids(
                    graph_to_score,
                    etype[2],
                )
                src_global_ids = src_global_nodes.index_select(
                    0,
                    src_nodes.to(src_global_nodes.device),
                )
                dst_global_ids = dst_global_nodes.index_select(
                    0,
                    dst_nodes.to(dst_global_nodes.device),
                )
                independent_scores = self.score_global_pairs(
                    etype,
                    src_global_ids,
                    dst_global_ids,
                )
                global_scores[etype] = independent_scores
            edge_scores, auxiliary_loss = self._predict_pairs(
                modality,
                pair_features,
                global_pair_features=global_pair_features,
                independent_scores=independent_scores,
            )
            if self.biogt_path_propagation_hops > 0:
                src_global_nodes = self._global_node_ids(
                    graph_to_score,
                    etype[0],
                )
                dst_global_nodes = self._global_node_ids(
                    graph_to_score,
                    etype[2],
                )
                src_global_ids = src_global_nodes.index_select(
                    0,
                    src_nodes.to(src_global_nodes.device),
                )
                dst_global_ids = dst_global_nodes.index_select(
                    0,
                    dst_nodes.to(dst_global_nodes.device),
                )
                edge_scores = (
                    edge_scores
                    + self.score_biogt_path_pairs(
                        etype,
                        src_global_ids,
                        dst_global_ids,
                        src_features=src_features,
                        dst_features=dst_features,
                    )
                )
            scores[etype] = edge_scores
            if auxiliary_loss is not None:
                auxiliary_losses.append(auxiliary_loss)
        return scores, auxiliary_losses, global_scores

    def forward(
        self,
        graph,
        input_features,
        pos_graph,
        neg_graph,
        edge_weights=None,
        target_etypes_override=None,
    ):
        if isinstance(graph, list):
            raise TypeError(
                "UnifiedDiseaseAssociationModel expects an induced context graph, "
                "not DGL message-flow blocks."
            )
        _, branches = self.get_embeddings(
            graph,
            initial_features=input_features,
            edge_weights=edge_weights,
            return_branch_outputs=True,
        )
        modality_cache = {"branches": branches}
        active_target_etypes = set(
            target_etypes_override
            if target_etypes_override is not None
            else self.target_etypes
        )
        pos_scores, pos_aux, global_pos_scores = self._score_graph(
            pos_graph,
            modality_cache,
            active_target_etypes,
        )
        neg_scores, neg_aux, global_neg_scores = self._score_graph(
            neg_graph,
            modality_cache,
            active_target_etypes,
        )
        all_aux = pos_aux + neg_aux
        if all_aux:
            self._last_moe_aux_loss = torch.stack(all_aux).mean()
        else:
            self._last_moe_aux_loss = next(self.parameters()).new_zeros(())
        self._last_global_pos_scores = global_pos_scores
        self._last_global_neg_scores = global_neg_scores
        return pos_scores, neg_scores

    def get_moe_aux_loss(self):
        if self._last_moe_aux_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._last_moe_aux_loss

    def get_global_bilinear_scores(self):
        return (
            self._last_global_pos_scores,
            self._last_global_neg_scores,
        )

    def _mark_query_nodes(self, graph, query_nodes):
        graph = graph.local_var()
        for node_type in graph.ntypes:
            seed_mask = torch.zeros(
                graph.num_nodes(node_type),
                dtype=torch.bool,
                device=graph.device,
            )
            for node_id in query_nodes.get(node_type, []):
                node_id = int(node_id)
                if 0 <= node_id < seed_mask.numel():
                    seed_mask[node_id] = True
            graph.nodes[node_type].data[QUERY_SEED_KEY] = seed_mask
            graph.nodes[node_type].data[QUERY_HOP_KEY] = torch.where(
                seed_mask,
                torch.zeros_like(seed_mask, dtype=torch.long),
                torch.full_like(
                    seed_mask,
                    self.sparse_transformer.max_hops + 1,
                    dtype=torch.long,
                ),
            )
        return graph

    def cache_embeddings(
        self,
        graph,
        cache_key,
        initial_features=None,
        query_nodes=None,
        required_etypes=None,
        force=False,
    ):
        required_etypes = tuple(
            sorted(required_etypes or self.target_etypes)
        )
        full_cache_key = (
            self._cache_generation,
            id(graph),
            cache_key,
            required_etypes,
        )
        if not force and full_cache_key in self._embedding_cache:
            self._embedding_cache.move_to_end(full_cache_key)
            return self._embedding_cache[full_cache_key]

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            graph_on_device = (
                graph
                if self.graph_backend == "torch"
                else graph.to(device)
            )
            if query_nodes:
                graph_on_device = self._mark_query_nodes(
                    graph_on_device,
                    query_nodes,
                )
            branches = self.encode_branches(
                graph_on_device,
                initial_features=initial_features,
            )
            required_types_by_modality = {}
            for etype in required_etypes:
                modality = self._target_modality(etype)
                if modality is None:
                    continue
                required_types_by_modality.setdefault(modality, set()).update(
                    [etype[0], etype[2]]
                )
            modality_cache = {}
            for modality, required_node_types in sorted(
                required_types_by_modality.items()
            ):
                modality_cache[modality] = {
                    node_type: tensor.detach().cpu()
                    for node_type, tensor in self.modality_embeddings(
                        branches,
                        modality,
                        node_types=required_node_types,
                    ).items()
                }

        if was_training:
            self.train()
        cached = {
            "modalities": modality_cache,
            "inputs": {
                node_type: tensor.detach().cpu()
                for node_type, tensor in branches["input"].items()
            },
        }
        self._embedding_cache[full_cache_key] = cached
        self._embedding_cache.move_to_end(full_cache_key)
        while len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        return cached

    def score_cached_pairs(
        self,
        cached_embeddings,
        etype,
        src_ids,
        dst_ids,
        chunk_size=4096,
        include_biogt_path=True,
    ):
        modality = self._target_modality(etype)
        if modality is None:
            return torch.empty(0)
        modality_embeddings = cached_embeddings["modalities"][modality]
        src_ids = torch.as_tensor(src_ids, dtype=torch.long)
        dst_ids = torch.as_tensor(dst_ids, dtype=torch.long)
        if src_ids.numel() != dst_ids.numel():
            raise ValueError("src_ids and dst_ids must have equal lengths")

        device = next(self.parameters()).device
        score_chunks = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start in range(0, src_ids.numel(), max(1, int(chunk_size))):
                end = min(start + max(1, int(chunk_size)), src_ids.numel())
                src_features = modality_embeddings[etype[0]].index_select(
                    0,
                    src_ids[start:end],
                ).to(device)
                dst_features = modality_embeddings[etype[2]].index_select(
                    0,
                    dst_ids[start:end],
                ).to(device)
                global_pair_features = None
                independent_scores = None
                if self.scorer_mode == "hybrid_global_bilinear":
                    global_inputs = cached_embeddings["inputs"]
                    global_pair_features = torch.cat(
                        [
                            global_inputs[etype[0]].index_select(
                                0,
                                src_ids[start:end],
                            ),
                            global_inputs[etype[2]].index_select(
                                0,
                                dst_ids[start:end],
                            ),
                        ],
                        dim=-1,
                    ).to(device)
                elif self.scorer_mode == "hybrid_kge":
                    independent_scores = self.score_global_pairs(
                        etype,
                        src_ids[start:end],
                        dst_ids[start:end],
                    )
                scores, _ = self._predict_pairs(
                    modality,
                    torch.cat([src_features, dst_features], dim=-1),
                    global_pair_features=global_pair_features,
                    independent_scores=independent_scores,
                )
                if (
                    include_biogt_path
                    and self.biogt_path_propagation_hops > 0
                ):
                    scores = scores + self.score_biogt_path_pairs(
                        etype,
                        src_ids[start:end],
                        dst_ids[start:end],
                        src_features=src_features,
                        dst_features=dst_features,
                    )
                score_chunks.append(scores.detach().cpu())
        if was_training:
            self.train()
        return torch.cat(score_chunks) if score_chunks else torch.empty(0)

    def predict_links(
        self,
        graph,
        initial_features,
        source_ntype,
        source_dgl_ids,
        target_nodes_dict_with_ids,
        score_chunk_size=4096,
    ):
        source_ids = [int(node_id) for node_id in source_dgl_ids]
        cache_key = (
            "predict",
            id(graph),
            source_ntype,
            tuple(source_ids),
        )
        cached = self.cache_embeddings(
            graph,
            cache_key=cache_key,
            initial_features=initial_features,
            query_nodes={source_ntype: source_ids},
            required_etypes=[
                (source_ntype, "candidate", target_type)
                for target_type in target_nodes_dict_with_ids
                if target_type in self.target_node_types
            ],
        )

        device = next(self.parameters()).device
        predictions = {}
        for target_type, target_ids_raw in target_nodes_dict_with_ids.items():
            target_ids = [int(node_id) for node_id in target_ids_raw]
            if target_type not in self.target_node_types:
                predictions[target_type] = torch.empty(
                    (len(source_ids), 0),
                    device=device,
                )
                continue

            rows = []
            for source_id in source_ids:
                repeated_sources = torch.full(
                    (len(target_ids),),
                    source_id,
                    dtype=torch.long,
                )
                scores = self.score_cached_pairs(
                    cached,
                    (source_ntype, "candidate", target_type),
                    repeated_sources,
                    torch.tensor(target_ids, dtype=torch.long),
                    chunk_size=score_chunk_size,
                )
                rows.append(scores)
            predictions[target_type] = (
                torch.stack(rows).to(device)
                if rows
                else torch.empty((0, len(target_ids)), device=device)
            )
        return predictions

    def export_config(self):
        first_head = (
            next(iter(self.expert_predictors.values()))
            if self.expert_predictors is not None
            else None
        )
        return {
            "architecture": "unified_sparse_v1",
            "target_node_types": sorted(self.target_node_types),
            "modality_fusion_modes": dict(self.modality_fusion_modes),
            "scorer_mode": self.scorer_mode,
            "global_bilinear_score_weight": (
                self.global_bilinear_score_weight
            ),
            "contextual_score_weight": self.contextual_score_weight,
            "kge_pretrain_scope": self.kge_pretrain_scope,
            "kge_relation_model": self.kge_relation_model,
            "kge_relation_rank": self.kge_relation_rank,
            "kge_context_input_weight": self.kge_context_input_weight,
            "kge_target_specific_disease_embeddings": (
                self.kge_target_specific_disease_embeddings
            ),
            "kge_target_disease_shared_weight": (
                self.kge_target_disease_shared_weight
            ),
            "kge_candidate_bias": self.kge_candidate_bias,
            "kge_candidate_bias_count": len(
                self.kge_candidate_biases
            ),
            "kge_target_residual_rank": self.kge_target_residual_rank,
            "kge_target_residual_count": len(
                self.kge_target_residual_left_factors
            ),
            "kge_path_propagation_hops": (
                self.kge_path_propagation_hops
            ),
            "kge_path_cross_modal": self.kge_path_cross_modal,
            "kge_path_cross_modal_types": sorted(
                self.kge_path_cross_modal_types
            ),
            "kge_path_cross_modal_support_types": {
                target: sorted(supports)
                for target, supports in sorted(
                    self.kge_path_cross_modal_support_types.items()
                )
            },
            "kge_path_propagation_count": len(
                self.kge_path_query_gates
            ),
            "kge_path_graph": dict(self._kge_path_graph_metadata),
            "biogt_path_propagation_hops": (
                self.biogt_path_propagation_hops
            ),
            "biogt_neural_path_dim": self.biogt_neural_path_dim,
            "biogt_relbf_dim": self.biogt_relbf_dim,
            "biogt_relbf_jk_readout": self.biogt_relbf_jk_readout,
            "biogt_relbf_edge_attention": (
                self.biogt_relbf_edge_attention
            ),
            "biogt_relbf_deep_refinement": (
                self.biogt_relbf_deep_refinement
            ),
            "biogt_relbf_indicator_residual": (
                self.biogt_relbf_indicator_residual
            ),
            "biogt_relbf_indicator_support_gating": (
                self.biogt_relbf_indicator_support_gating
            ),
            "biogt_relbf_indicator_semantic_readout": (
                self.biogt_relbf_indicator_semantic_readout
            ),
            "biogt_relbf_query_relation_adapter_rank": (
                self.biogt_relbf_query_relation_adapter_rank
            ),
            "biogt_relbf_query_relation_adapter_mode": (
                "relation_support_layer_low_rank_query_gated_delta_v1"
                if self.biogt_relbf_query_relation_adapter_rank > 0
                else "disabled"
            ),
            "biogt_relbf_query_relation_adapter_identity_init": (
                "zero_output_projection"
                if self.biogt_relbf_query_relation_adapter_rank > 0
                else "disabled"
            ),
            "biogt_relbf_query_relation_adapter_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_query_relation_adapters.parameters()
                )
            ),
            "biogt_relbf_indicator_query_boundary_rank": (
                self.biogt_relbf_indicator_query_boundary_rank
            ),
            "biogt_relbf_indicator_query_boundary_mode": (
                "low_rank_frozen_parent_condition_source_delta_v1"
                if self.biogt_relbf_indicator_query_boundary_rank > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_query_boundary_identity_init": (
                "zero_output_projection"
                if self.biogt_relbf_indicator_query_boundary_rank > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_query_boundary_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_indicator_query_boundary_adapters.parameters()
                )
            ),
            "biogt_relbf_indicator_multirule_count": (
                self.biogt_relbf_indicator_multirule_count
            ),
            "biogt_relbf_indicator_multirule_readout_rank": (
                self.biogt_relbf_indicator_multirule_readout_rank
            ),
            "biogt_relbf_indicator_multirule_mode": (
                "query_gated_independent_rule_indicator_residual_v1"
                if self.biogt_relbf_indicator_multirule_count > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_multirule_gate_condition": (
                "detached_parent_relbf_condition"
                if self.biogt_relbf_indicator_multirule_count > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_multirule_identity_init": (
                "zero_final_rule_readouts"
                if self.biogt_relbf_indicator_multirule_count > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_multirule_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_indicator_multirule_residuals.parameters()
                )
            ),
            "biogt_relbf_indicator_relation_transition_layers": (
                self.biogt_relbf_indicator_relation_transition_layers
            ),
            "biogt_relbf_indicator_relation_transition_rank": (
                self.biogt_relbf_indicator_relation_transition_rank
            ),
            "biogt_relbf_indicator_relation_transition_mode": (
                "train_only_sparse_query_relation_message_residual_v1"
                if self.biogt_relbf_indicator_relation_transition_layers > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_relation_transition_identity_init": (
                "zero_output_projection_one_multiplier"
                if self.biogt_relbf_indicator_relation_transition_layers > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_relation_transition_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_indicator_relation_transition_residuals.parameters()
                )
            ),
            "biogt_relbf_indicator_relation_transition_attributions": {
                modality: self.get_biogt_relbf_relation_transition_attribution(
                    modality
                )
                for modality in sorted(
                    self._biogt_relbf_relation_transition_labels
                )
            },
            "biogt_relbf_indicator_query_local_relation_transition_layers": (
                self.biogt_relbf_indicator_query_local_relation_transition_layers
            ),
            "biogt_relbf_indicator_query_local_relation_transition_rank": (
                self.biogt_relbf_indicator_query_local_relation_transition_rank
            ),
            "biogt_relbf_indicator_query_local_relation_transition_mode": (
                "train_only_query_local_incidence_global_prior_residual_v1"
                if self.biogt_relbf_indicator_query_local_relation_transition_layers
                > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_query_local_relation_transition_identity_init": (
                "zero_output_projection_one_multiplier"
                if self.biogt_relbf_indicator_query_local_relation_transition_layers
                > 0
                else "disabled"
            ),
            "biogt_relbf_indicator_query_local_relation_transition_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_indicator_query_local_relation_transition_residuals.parameters()
                )
            ),
            "biogt_relbf_readout": (
                "final_plus_jk_residual_v1"
                if self.biogt_relbf_jk_readout
                else (
                    "final_v1"
                    if self.biogt_relbf_dim > 0
                    else "disabled"
                )
            ),
            "biogt_relbf_layers": (
                (5 if self.biogt_relbf_deep_refinement else 3)
                if self.biogt_relbf_dim > 0
                else 0
            ),
            "biogt_relbf_aggregate": (
                "pna_mean_max_min_std_degree_scaled"
                if self.biogt_relbf_dim > 0
                else "disabled"
            ),
            "biogt_relbf_easy_edge_policy": (
                "train_dynamic_forward_and_inverse_eval_full_train_graph"
                if self.biogt_relbf_dim > 0
                else "disabled"
            ),
            "biogt_relbf_edge_attention_mode": (
                "query_output_relation_x_support_relation_sigmoid_v1"
                if self.biogt_relbf_edge_attention
                else "disabled"
            ),
            "biogt_relbf_edge_attention_identity_init": (
                "2sigmoid_zero_equals_one"
                if self.biogt_relbf_edge_attention
                else "disabled"
            ),
            "biogt_relbf_edge_attention_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_edge_attention_queries.parameters()
                )
            ),
            "biogt_relbf_deep_refinement_mode": (
                "relation_specific_target_disease_target_pna_residual_v1"
                if self.biogt_relbf_deep_refinement
                else "disabled"
            ),
            "biogt_relbf_deep_identity_init": (
                "zero_final_residual_readout"
                if self.biogt_relbf_deep_refinement
                else "disabled"
            ),
            "biogt_relbf_deep_parameter_count": sum(
                parameter.numel()
                for module in (
                    self.biogt_relbf_deep_message_transforms,
                    self.biogt_relbf_deep_pna_updates,
                    self.biogt_relbf_deep_state_norms,
                    self.biogt_relbf_deep_output_films,
                    self.biogt_relbf_deep_readouts,
                )
                for parameter in module.parameters()
            ),
            "biogt_relbf_indicator_mode": (
                "relation_token_boundary_bellman_ford_pna_residual_v1"
                if self.biogt_relbf_indicator_residual
                else "disabled"
            ),
            "biogt_relbf_indicator_identity_init": (
                "zero_final_residual_readout"
                if self.biogt_relbf_indicator_residual
                else "disabled"
            ),
            "biogt_relbf_indicator_parameter_count": sum(
                parameter.numel()
                for module in (
                    self.biogt_relbf_indicator_tokens,
                    self.biogt_relbf_indicator_message_transforms,
                    self.biogt_relbf_indicator_pna_updates,
                    self.biogt_relbf_indicator_state_norms,
                    self.biogt_relbf_indicator_readouts,
                )
                for parameter in module.parameters()
            ),
            "biogt_relbf_indicator_support_gate_mode": (
                "parent_condition_per_output_support_layer_2sigmoid_v1"
                if self.biogt_relbf_indicator_support_gating
                else "disabled"
            ),
            "biogt_relbf_indicator_support_gate_identity_init": (
                "2sigmoid_zero_equals_one"
                if self.biogt_relbf_indicator_support_gating
                else "disabled"
            ),
            "biogt_relbf_indicator_support_gate_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_indicator_support_gates.parameters()
                )
            ),
            "biogt_relbf_indicator_support_gate_modalities": {
                target: sorted(gates.keys())
                for target, gates in sorted(
                    self.biogt_relbf_indicator_support_gates.items()
                )
            },
            "biogt_relbf_indicator_semantic_mode": (
                "indicator_candidate_parent_condition_4way_residual_v1"
                if self.biogt_relbf_indicator_semantic_readout
                else "disabled"
            ),
            "biogt_relbf_indicator_semantic_identity_init": (
                "zero_final_residual_readout"
                if self.biogt_relbf_indicator_semantic_readout
                else "disabled"
            ),
            "biogt_relbf_indicator_semantic_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_indicator_semantic_readouts.parameters()
                )
            ),
            "biogt_relbf_path_layer_attention_rank": (
                self.biogt_relbf_path_layer_attention_rank
            ),
            "biogt_relbf_path_layer_attention_mode": (
                "candidate_conditioned_three_layer_attention_residual_v1"
                if self.biogt_relbf_path_layer_attention_rank > 0
                else "disabled"
            ),
            "biogt_relbf_path_layer_attention_identity_init": (
                "zero_output_projection"
                if self.biogt_relbf_path_layer_attention_rank > 0
                else "disabled"
            ),
            "biogt_relbf_path_layer_attention_source": (
                "train_graph_relbf_path_states_with_detached_hgcn_"
                "transformer_candidate"
                if self.biogt_relbf_path_layer_attention_rank > 0
                else None
            ),
            "biogt_relbf_path_layer_attention_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    self.biogt_relbf_path_layer_attention_residuals.parameters()
                )
            ),
            "biogt_relbf_path_layer_attention_attributions": {
                modality: (
                    None
                    if modality not in (
                        self._biogt_relbf_path_layer_attention_attributions
                    )
                    else {
                        "source": self._biogt_relbf_path_layer_attention_attributions[
                            modality
                        ]["source"],
                        "layer_labels": list(
                            self._biogt_relbf_path_layer_attention_attributions[
                                modality
                            ]["layer_labels"]
                        ),
                        "shape": list(
                            self._biogt_relbf_path_layer_attention_attributions[
                                modality
                            ]["shape"]
                        ),
                        "mean_attention_by_layer": {
                            label: float(value)
                            for label, value in zip(
                                self._biogt_relbf_path_layer_attention_attributions[
                                    modality
                                ]["layer_labels"],
                                self._biogt_relbf_path_layer_attention_attributions[
                                    modality
                                ]["mean_attention_by_layer"],
                            )
                        },
                    }
                )
                for modality in sorted(
                    self.biogt_relbf_path_layer_attention_residuals
                )
            },
            "biogt_relbf_path_token_transformer": self.biogt_relbf_path_token_transformer,
            "biogt_relbf_path_token_transformer_mode": (
                "candidate_conditioned_five_token_micro_transformer_v1"
                if self.biogt_relbf_path_token_transformer
                else "disabled"
            ),
            "biogt_relbf_path_token_transformer_heads": (
                CandidateConditionedPathTokenTransformerResidual.attention_heads
                if self.biogt_relbf_path_token_transformer else 0
            ),
            "biogt_relbf_path_token_transformer_identity_init": (
                "zero_score_readout"
                if self.biogt_relbf_path_token_transformer else "disabled"
            ),
            "biogt_relbf_path_token_transformer_source": (
                "train_graph_relbf_tokens_with_detached_hgcn_transformer"
                if self.biogt_relbf_path_token_transformer else None
            ),
            "biogt_relbf_path_token_transformer_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_path_token_transformer_residuals.parameters()
            ),
            "biogt_relbf_path_token_transformer_attributions": {
                modality: (
                    None
                    if modality not in self._biogt_relbf_path_token_transformer_attributions
                    else {
                        "source": self._biogt_relbf_path_token_transformer_attributions[
                            modality
                        ]["source"],
                        "path_labels": list(
                            self._biogt_relbf_path_token_transformer_attributions[
                                modality
                            ]["path_labels"]
                        ),
                        "shape": list(
                            self._biogt_relbf_path_token_transformer_attributions[
                                modality
                            ]["shape"]
                        ),
                        "mean_attention_by_path": {
                            label: float(value)
                            for label, value in zip(
                                self._biogt_relbf_path_token_transformer_attributions[
                                    modality
                                ]["path_labels"],
                                self._biogt_relbf_path_token_transformer_attributions[
                                    modality
                                ]["mean_attention_by_path"],
                            )
                        },
                    }
                )
                for modality in sorted(
                    self.biogt_relbf_path_token_transformer_residuals
                )
            },
            "biogt_relbf_candidate_competition_boundary": (
                self.biogt_relbf_candidate_competition_boundary
            ),
            "biogt_relbf_candidate_competition_boundary_mode": (
                "parent_top32_linear_cross_attention_v1"
                if self.biogt_relbf_candidate_competition_boundary
                else "disabled"
            ),
            "biogt_relbf_candidate_competition_boundary_anchor_count": (
                CandidateCompetitionBoundaryResidual.anchor_count
                if self.biogt_relbf_candidate_competition_boundary
                else 0
            ),
            "biogt_relbf_candidate_competition_boundary_identity_init": (
                "zero_score_readout"
                if self.biogt_relbf_candidate_competition_boundary
                else "disabled"
            ),
            "biogt_relbf_candidate_competition_boundary_source": (
                "frozen_parent_compact_scores_and_train_graph_relbf_tokens"
                if self.biogt_relbf_candidate_competition_boundary
                else None
            ),
            "biogt_relbf_candidate_competition_boundary_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_candidate_competition_boundary_residuals.parameters()
            ),
            "biogt_relbf_candidate_competition_boundary_attributions": {
                modality: (
                    None
                    if modality not in self._biogt_relbf_candidate_competition_boundary_attributions
                    else {
                        "source": self._biogt_relbf_candidate_competition_boundary_attributions[
                            modality
                        ]["source"],
                        "shape": list(
                            self._biogt_relbf_candidate_competition_boundary_attributions[
                                modality
                            ]["shape"]
                        ),
                        "anchor_candidate_ids": [
                            list(ids)
                            for ids in self._biogt_relbf_candidate_competition_boundary_attributions[
                                modality
                            ]["anchor_candidate_ids"]
                        ],
                        "mean_attention_by_anchor_rank": list(
                            self._biogt_relbf_candidate_competition_boundary_attributions[
                                modality
                            ]["mean_attention_by_anchor_rank"]
                        ),
                    }
                )
                for modality in sorted(
                    self.biogt_relbf_candidate_competition_boundary_residuals
                )
            },
            "biogt_relbf_parent_top10_boundary_comparator": (
                self.biogt_relbf_parent_top10_boundary_comparator
            ),
            "biogt_relbf_parent_top10_boundary_comparator_mode": (
                "frozen_parent_rank10_rank11_slot_comparator_v1"
                if self.biogt_relbf_parent_top10_boundary_comparator
                else "disabled"
            ),
            "biogt_relbf_parent_top10_boundary_comparator_ranks": (
                list(ParentTopBoundarySlotComparatorResidual.boundary_ranks)
                if self.biogt_relbf_parent_top10_boundary_comparator
                else []
            ),
            "biogt_relbf_parent_top10_boundary_comparator_identity_init": (
                "zero_score_readout"
                if self.biogt_relbf_parent_top10_boundary_comparator
                else "disabled"
            ),
            "biogt_relbf_parent_top10_boundary_comparator_source": (
                "detached_parent_scores_hgcn_transformer_and_train_graph_relbf"
                if self.biogt_relbf_parent_top10_boundary_comparator
                else None
            ),
            "biogt_relbf_parent_top10_boundary_comparator_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_parent_top10_boundary_comparator_residuals.parameters()
            ),
            "biogt_relbf_parent_top10_boundary_comparator_attributions": {
                modality: (
                    None
                    if modality not in self._biogt_relbf_parent_top10_boundary_comparator_attributions
                    else {
                        "source": self._biogt_relbf_parent_top10_boundary_comparator_attributions[
                            modality
                        ]["source"],
                        "boundary_ranks": list(
                            self._biogt_relbf_parent_top10_boundary_comparator_attributions[
                                modality
                            ]["boundary_ranks"]
                        ),
                        "shape": list(
                            self._biogt_relbf_parent_top10_boundary_comparator_attributions[
                                modality
                            ]["shape"]
                        ),
                        "boundary_candidate_ids": [
                            list(ids)
                            for ids in self._biogt_relbf_parent_top10_boundary_comparator_attributions[
                                modality
                            ]["boundary_candidate_ids"]
                        ],
                        "boundary_scores": [
                            list(scores)
                            for scores in self._biogt_relbf_parent_top10_boundary_comparator_attributions[
                                modality
                            ]["boundary_scores"]
                        ],
                    }
                )
                for modality in sorted(
                    self.biogt_relbf_parent_top10_boundary_comparator_residuals
                )
            },
            "biogt_relbf_boundary_window_rank_slot_expert": (
                self.biogt_relbf_boundary_window_rank_slot_expert
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_mode": (
                "gene8_parent_rank5_20_window_expert_v1"
                if self.biogt_relbf_boundary_window_rank_slot_expert
                else "disabled"
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_ranks": (
                [
                    BoundaryWindowRankSlotExpertResidual.window_start_rank,
                    BoundaryWindowRankSlotExpertResidual.window_end_rank,
                ]
                if self.biogt_relbf_boundary_window_rank_slot_expert
                else []
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_identity_init": (
                "zero_score_readout"
                if self.biogt_relbf_boundary_window_rank_slot_expert
                else "disabled"
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_boundary_window_rank_slot_expert_residuals.parameters()
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_attributions": {
                modality: self.get_biogt_relbf_boundary_window_rank_slot_expert_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_boundary_window_rank_slot_expert_residuals
                )
            },
            "biogt_relbf_boundary_exchange": self.biogt_relbf_boundary_exchange,
            "biogt_relbf_boundary_exchange_mode": (
                "gene9_parent_rank5_20_centered_cross_set_exchange_v1"
                if self.biogt_relbf_boundary_exchange
                else "disabled"
            ),
            "biogt_relbf_boundary_exchange_ranks": (
                [
                    BoundaryExchangeResidual.window_start_rank,
                    BoundaryExchangeResidual.window_end_rank,
                ]
                if self.biogt_relbf_boundary_exchange
                else []
            ),
            "biogt_relbf_boundary_exchange_groups": (
                {
                    "incumbents": list(BoundaryExchangeResidual.incumbent_ranks),
                    "challengers": list(BoundaryExchangeResidual.challenger_ranks),
                }
                if self.biogt_relbf_boundary_exchange
                else {}
            ),
            "biogt_relbf_boundary_exchange_identity_init": (
                "zero_score_readout_centered_residual"
                if self.biogt_relbf_boundary_exchange
                else "disabled"
            ),
            "biogt_relbf_boundary_exchange_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_boundary_exchange_residuals.parameters()
            ),
            "biogt_relbf_boundary_exchange_attributions": {
                modality: self.get_biogt_relbf_boundary_exchange_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_boundary_exchange_residuals
                )
            },
            "biogt_relbf_challenger_boundary_promotion": (
                self.biogt_relbf_challenger_boundary_promotion
            ),
            "biogt_relbf_challenger_boundary_promotion_mode": (
                "gene9_parent_rank11_20_challenger_only_cross_attention_v1"
                if self.biogt_relbf_challenger_boundary_promotion
                else "disabled"
            ),
            "biogt_relbf_challenger_boundary_promotion_groups": (
                {
                    "incumbents": list(
                        ChallengerBoundaryPromotionResidual.incumbent_ranks
                    ),
                    "challengers": list(
                        ChallengerBoundaryPromotionResidual.challenger_ranks
                    ),
                }
                if self.biogt_relbf_challenger_boundary_promotion
                else {}
            ),
            "biogt_relbf_challenger_boundary_promotion_identity_init": (
                "zero_score_readout_challengers_only"
                if self.biogt_relbf_challenger_boundary_promotion
                else "disabled"
            ),
            "biogt_relbf_challenger_boundary_promotion_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_challenger_boundary_promotion_residuals.parameters()
            ),
            "biogt_relbf_challenger_boundary_promotion_attributions": {
                modality: self.get_biogt_relbf_challenger_boundary_promotion_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_challenger_boundary_promotion_residuals
                )
            },
            "biogt_relbf_gatekeeper_challenger_paired_margin": (
                self.biogt_relbf_gatekeeper_challenger_paired_margin
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_mode": (
                "gene9_parent_rank10_11_20_independent_paired_margin_v1"
                if self.biogt_relbf_gatekeeper_challenger_paired_margin
                else "disabled"
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_groups": (
                {
                    "gatekeeper": GatekeeperChallengerPairedMarginResidual.gatekeeper_rank,
                    "challengers": list(
                        GatekeeperChallengerPairedMarginResidual.challenger_ranks
                    ),
                }
                if self.biogt_relbf_gatekeeper_challenger_paired_margin
                else {}
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_identity_init": (
                "independent_zero_gatekeeper_and_challenger_readouts"
                if self.biogt_relbf_gatekeeper_challenger_paired_margin
                else "disabled"
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals.parameters()
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_attributions": {
                modality: self.get_biogt_relbf_gatekeeper_challenger_paired_margin_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_gatekeeper_challenger_paired_margin_residuals
                )
            },
            "biogt_relbf_boundary_prototype_ordinal_adapter": self.biogt_relbf_boundary_prototype_ordinal_adapter,
            "biogt_relbf_boundary_prototype_ordinal_adapter_mode": (
                "gene9_parent_rank6_16_boundary_prototype_ordinal_v1"
                if self.biogt_relbf_boundary_prototype_ordinal_adapter else "disabled"
            ),
            "biogt_relbf_boundary_prototype_ordinal_adapter_groups": (
                {
                    "incumbent_prototype": list(BoundaryPrototypeOrdinalResidual.incumbent_ranks),
                    "challenger_prototype": list(BoundaryPrototypeOrdinalResidual.challenger_ranks),
                    "active": list(BoundaryPrototypeOrdinalResidual.active_ranks),
                } if self.biogt_relbf_boundary_prototype_ordinal_adapter else {}
            ),
            "biogt_relbf_boundary_prototype_ordinal_adapter_parameter_count": sum(
                parameter.numel() for parameter in
                self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals.parameters()
            ),
            "biogt_relbf_boundary_prototype_ordinal_adapter_attributions": {
                modality: self.get_biogt_relbf_boundary_prototype_ordinal_adapter_attribution(modality)
                for modality in sorted(self.biogt_relbf_boundary_prototype_ordinal_adapter_residuals)
            },
            "biogt_relbf_multi_depth_score_router": self.biogt_relbf_multi_depth_score_router,
            "biogt_relbf_multi_depth_score_router_mode": (
                "gene9_parent_full_candidate_query_multidepth_score_router_v1"
                if self.biogt_relbf_multi_depth_score_router else "disabled"
            ),
            "biogt_relbf_multi_depth_score_router_depths": list(
                QueryConditionedMultiDepthScoreRouterResidual.depth_labels
            ) if self.biogt_relbf_multi_depth_score_router else [],
            "biogt_relbf_multi_depth_score_router_parameter_count": sum(
                parameter.numel() for parameter in
                self.biogt_relbf_multi_depth_score_router_residuals.parameters()
            ),
            "biogt_relbf_multi_depth_score_router_attributions": {
                modality: self.get_biogt_relbf_multi_depth_score_router_attribution(modality)
                for modality in sorted(self.biogt_relbf_multi_depth_score_router_residuals)
            },
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator": (
                self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
            ),
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator_mode": (
                "gene9_parent_full_candidate_dual_stream_trajectory_agreement_v1"
                if self.biogt_relbf_dual_stream_trajectory_agreement_calibrator
                else "disabled"
            ),
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals.parameters()
            ),
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator_attributions": {
                modality: self.get_biogt_relbf_dual_stream_trajectory_agreement_calibrator_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals
                )
            },
            "biogt_relbf_multi_anchor_rank_landmark_calibrator": (
                self.biogt_relbf_multi_anchor_rank_landmark_calibrator
            ),
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_mode": (
                "gene9_parent_full_candidate_multi_anchor_rank_landmark_v1"
                if self.biogt_relbf_multi_anchor_rank_landmark_calibrator
                else "disabled"
            ),
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_landmark_ranks": (
                list(
                    QueryConditionedMultiAnchorRankLandmarkCalibratorResidual.landmark_ranks
                )
                if self.biogt_relbf_multi_anchor_rank_landmark_calibrator
                else []
            ),
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals.parameters()
            ),
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_attributions": {
                modality: self.get_biogt_relbf_multi_anchor_rank_landmark_calibrator_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals
                )
            },
            "biogt_relbf_query_token_top64_candidate_set_transformer": (
                self.biogt_relbf_query_token_top64_candidate_set_transformer
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_mode": (
                "gene9_parent_query_token_top64_candidate_set_transformer_v1"
                if self.biogt_relbf_query_token_top64_candidate_set_transformer
                else "disabled"
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_set_size": (
                QueryTokenTop64CandidateSetTransformerResidual.candidate_set_size
                if self.biogt_relbf_query_token_top64_candidate_set_transformer
                else 0
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_heads": (
                QueryTokenTop64CandidateSetTransformerResidual.attention_heads
                if self.biogt_relbf_query_token_top64_candidate_set_transformer
                else 0
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_parameter_count": sum(
                parameter.numel()
                for parameter in self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals.parameters()
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_attributions": {
                modality: self.get_biogt_relbf_query_token_top64_candidate_set_transformer_attribution(
                    modality
                )
                for modality in sorted(
                    self.biogt_relbf_query_token_top64_candidate_set_transformer_residuals
                )
            },
            "biogt_path_support_types": {
                target: sorted(supports)
                for target, supports in sorted(
                    self.biogt_path_support_types.items()
                )
            },
            "biogt_path_propagation_count": len(
                self.biogt_path_query_gates
            ),
            "biogt_path_graph": dict(
                self._biogt_path_graph_metadata
            ),
            "biogt_path_initialized_from_checkpoint": (
                self.biogt_path_initialized_from_checkpoint
            ),
            "biogt_neural_path_initialized_from_checkpoint": (
                self.biogt_neural_path_initialized_from_checkpoint
            ),
            "biogt_relbf_initialized_from_checkpoint": (
                self.biogt_relbf_initialized_from_checkpoint
            ),
            "biogt_relbf_jk_initialized_from_checkpoint": (
                self.biogt_relbf_jk_initialized_from_checkpoint
            ),
            "biogt_relbf_edge_attention_initialized_from_checkpoint": (
                self.biogt_relbf_edge_attention_initialized_from_checkpoint
            ),
            "biogt_relbf_deep_initialized_from_checkpoint": (
                self.biogt_relbf_deep_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_support_gating_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_support_gating_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_semantic_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_semantic_initialized_from_checkpoint
            ),
            "biogt_relbf_query_relation_adapter_initialized_from_checkpoint": (
                self.biogt_relbf_query_relation_adapter_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_query_boundary_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_query_boundary_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_multirule_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_multirule_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_relation_transition_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_relation_transition_initialized_from_checkpoint
            ),
            "biogt_relbf_indicator_query_local_relation_transition_initialized_from_checkpoint": (
                self.biogt_relbf_indicator_query_local_relation_transition_initialized_from_checkpoint
            ),
            "biogt_relbf_path_layer_attention_initialized_from_checkpoint": (
                self.biogt_relbf_path_layer_attention_initialized_from_checkpoint
            ),
            "biogt_relbf_path_token_transformer_initialized_from_checkpoint": (
                self.biogt_relbf_path_token_transformer_initialized_from_checkpoint
            ),
            "biogt_relbf_candidate_competition_boundary_initialized_from_checkpoint": (
                self.biogt_relbf_candidate_competition_boundary_initialized_from_checkpoint
            ),
            "biogt_relbf_parent_top10_boundary_comparator_initialized_from_checkpoint": (
                self.biogt_relbf_parent_top10_boundary_comparator_initialized_from_checkpoint
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_initialized_from_checkpoint": (
                self.biogt_relbf_boundary_window_rank_slot_expert_initialized_from_checkpoint
            ),
            "biogt_relbf_boundary_exchange_initialized_from_checkpoint": (
                self.biogt_relbf_boundary_exchange_initialized_from_checkpoint
            ),
            "biogt_relbf_challenger_boundary_promotion_initialized_from_checkpoint": (
                self.biogt_relbf_challenger_boundary_promotion_initialized_from_checkpoint
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_initialized_from_checkpoint": (
                self.biogt_relbf_gatekeeper_challenger_paired_margin_initialized_from_checkpoint
            ),
            "biogt_relbf_boundary_prototype_ordinal_adapter_initialized_from_checkpoint": (
                self.biogt_relbf_boundary_prototype_ordinal_adapter_initialized_from_checkpoint
            ),
            "biogt_relbf_multi_depth_score_router_initialized_from_checkpoint": (
                self.biogt_relbf_multi_depth_score_router_initialized_from_checkpoint
            ),
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator_initialized_from_checkpoint": (
                self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_initialized_from_checkpoint
            ),
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_initialized_from_checkpoint": (
                self.biogt_relbf_multi_anchor_rank_landmark_calibrator_initialized_from_checkpoint
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_initialized_from_checkpoint": (
                self.biogt_relbf_query_token_top64_candidate_set_transformer_initialized_from_checkpoint
            ),
            "biogt_relbf_deep_migration_provenance": (
                self.biogt_relbf_deep_migration_provenance
            ),
            "biogt_relbf_jk_migration_provenance": (
                self.biogt_relbf_jk_migration_provenance
            ),
            "biogt_relbf_target_only_finetune": (
                self.biogt_relbf_target_only_finetune
            ),
            "biogt_relbf_jk_only_finetune": (
                self.biogt_relbf_jk_only_finetune
            ),
            "biogt_relbf_edge_attention_only_finetune": (
                self.biogt_relbf_edge_attention_only_finetune
            ),
            "biogt_relbf_deep_only_finetune": (
                self.biogt_relbf_deep_only_finetune
            ),
            "biogt_relbf_indicator_only_finetune": (
                self.biogt_relbf_indicator_only_finetune
            ),
            "biogt_relbf_indicator_support_gate_only_finetune": (
                self.biogt_relbf_indicator_support_gate_only_finetune
            ),
            "biogt_relbf_indicator_semantic_only_finetune": (
                self.biogt_relbf_indicator_semantic_only_finetune
            ),
            "biogt_relbf_query_relation_adapter_only_finetune": (
                self.biogt_relbf_query_relation_adapter_only_finetune
            ),
            "biogt_relbf_indicator_query_boundary_only_finetune": (
                self.biogt_relbf_indicator_query_boundary_only_finetune
            ),
            "biogt_relbf_indicator_multirule_only_finetune": (
                self.biogt_relbf_indicator_multirule_only_finetune
            ),
            "biogt_relbf_indicator_relation_transition_only_finetune": (
                self.biogt_relbf_indicator_relation_transition_only_finetune
            ),
            "biogt_relbf_indicator_query_local_relation_transition_only_finetune": (
                self.biogt_relbf_indicator_query_local_relation_transition_only_finetune
            ),
            "biogt_relbf_path_layer_attention_only_finetune": (
                self.biogt_relbf_path_layer_attention_only_finetune
            ),
            "biogt_relbf_path_token_transformer_only_finetune": (
                self.biogt_relbf_path_token_transformer_only_finetune
            ),
            "biogt_relbf_candidate_competition_boundary_only_finetune": (
                self.biogt_relbf_candidate_competition_boundary_only_finetune
            ),
            "biogt_relbf_parent_top10_boundary_comparator_only_finetune": (
                self.biogt_relbf_parent_top10_boundary_comparator_only_finetune
            ),
            "biogt_relbf_boundary_window_rank_slot_expert_only_finetune": (
                self.biogt_relbf_boundary_window_rank_slot_expert_only_finetune
            ),
            "biogt_relbf_boundary_exchange_only_finetune": (
                self.biogt_relbf_boundary_exchange_only_finetune
            ),
            "biogt_relbf_challenger_boundary_promotion_only_finetune": (
                self.biogt_relbf_challenger_boundary_promotion_only_finetune
            ),
            "biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune": (
                self.biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune
            ),
            "biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune": (
                self.biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune
            ),
            "biogt_relbf_multi_depth_score_router_only_finetune": (
                self.biogt_relbf_multi_depth_score_router_only_finetune
            ),
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune": (
                self.biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune
            ),
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune": (
                self.biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune
            ),
            "biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune": (
                self.biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune
            ),
            "biogt_base_frozen_for_path_finetune": (
                self.biogt_base_frozen_for_path_finetune
            ),
            "biogt_path_finetune_types": list(
                self.biogt_path_finetune_types
            ),
            "kge_target_disease_embedding_count": len(
                self.kge_target_disease_embeddings
            ),
            "kge_pretrain_relation_count": len(
                self.kge_pretrain_relations
            ),
            "independent_branch_frozen": self.independent_branch_frozen,
            "shared_kge_frozen_for_target_disease_finetune": (
                self.shared_kge_frozen_for_target_disease_finetune
            ),
            "target_disease_finetune_types": list(
                self.target_disease_finetune_types
            ),
            "target_disease_initialized_from_shared_checkpoint": (
                self.target_disease_initialized_from_shared_checkpoint
            ),
            "kge_frozen_for_candidate_bias_finetune": (
                self.kge_frozen_for_candidate_bias_finetune
            ),
            "candidate_bias_finetune_types": list(
                self.candidate_bias_finetune_types
            ),
            "candidate_bias_initialized_from_checkpoint": (
                self.candidate_bias_initialized_from_checkpoint
            ),
            "kge_frozen_for_target_residual_finetune": (
                self.kge_frozen_for_target_residual_finetune
            ),
            "target_residual_finetune_types": list(
                self.target_residual_finetune_types
            ),
            "target_residual_initialized_from_checkpoint": (
                self.target_residual_initialized_from_checkpoint
            ),
            "kge_frozen_for_path_propagation_finetune": (
                self.kge_frozen_for_path_propagation_finetune
            ),
            "path_propagation_finetune_types": list(
                self.path_propagation_finetune_types
            ),
            "path_propagation_initialized_from_checkpoint": (
                self.path_propagation_initialized_from_checkpoint
            ),
            "hidden_dim": self.hidden_dim,
            "num_gnn_layers": self.num_gnn_layers,
            "graph_backend": self.graph_backend,
            "moe_num_experts": first_head.num_experts if first_head else 0,
            "moe_top_k": first_head.top_k if first_head else 0,
            "sparse_transformer_topk": self.sparse_transformer.context_topk,
            "sparse_transformer_max_hops": self.sparse_transformer.max_hops,
        }
