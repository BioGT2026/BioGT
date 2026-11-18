import logging
import math
from dataclasses import dataclass

import dgl
import torch

from Model.unified_model import (
    EVIDENCE_SCORE_KEY,
    QUERY_HOP_KEY,
    QUERY_SEED_KEY,
)


def target_modality_for_etype(etype, target_types):
    target_types = set(target_types)
    if etype[2] in target_types:
        return etype[2]
    if etype[0] in target_types:
        return etype[0]
    return None


@dataclass
class EdgeBatch:
    etype: tuple
    eids: torch.Tensor
    pos_src: torch.Tensor
    pos_dst: torch.Tensor
    neg_src: torch.Tensor
    neg_dst: torch.Tensor
    negatives_per_positive: int


class KnownPositiveIndex:
    """Memory-efficient filtered-ranking index over all known graph edges."""

    def __init__(self, graph, relevant_type_pairs):
        self._packed_by_type_pair = {}
        self._num_dst_nodes = {}
        for src_type, dst_type in sorted(set(relevant_type_pairs)):
            packed_parts = []
            num_dst_nodes = graph.num_nodes(dst_type)
            for etype in graph.canonical_etypes:
                if etype[0] != src_type or etype[2] != dst_type:
                    continue
                src_nodes, dst_nodes = graph.edges(etype=etype)
                if src_nodes.numel() == 0:
                    continue
                packed_parts.append(
                    src_nodes.cpu().long() * num_dst_nodes + dst_nodes.cpu().long()
                )
            packed = (
                torch.unique(torch.cat(packed_parts), sorted=True)
                if packed_parts
                else torch.empty(0, dtype=torch.long)
            )
            key = (src_type, dst_type)
            self._packed_by_type_pair[key] = packed
            self._num_dst_nodes[key] = num_dst_nodes

    @classmethod
    def for_target_etypes(cls, graph, target_etypes):
        return cls(
            graph,
            {(etype[0], etype[2]) for etype in target_etypes},
        )

    def contains(self, etype, src_nodes, dst_nodes):
        key = (etype[0], etype[2])
        known = self._packed_by_type_pair.get(key)
        if known is None or known.numel() == 0:
            return torch.zeros(src_nodes.numel(), dtype=torch.bool)
        packed = (
            src_nodes.cpu().long() * self._num_dst_nodes[key]
            + dst_nodes.cpu().long()
        )
        positions = torch.searchsorted(known, packed)
        valid = positions < known.numel()
        result = torch.zeros_like(valid)
        result[valid] = known[positions[valid]] == packed[valid]
        return result


def sample_filtered_negatives(
    graph,
    etype,
    pos_src,
    pos_dst,
    negatives_per_positive,
    target_types,
    known_positive_index,
    generator,
    candidate_nodes_by_etype=None,
    max_attempts=32,
):
    negatives_per_positive = max(1, int(negatives_per_positive))
    negative_src = pos_src.cpu().long().repeat_interleave(negatives_per_positive)
    negative_dst = pos_dst.cpu().long().repeat_interleave(negatives_per_positive)
    corrupt_destination = etype[2] in set(target_types)
    candidate_count = negative_src.numel()
    candidate_nodes = (candidate_nodes_by_etype or {}).get(etype)
    if candidate_nodes is not None:
        candidate_nodes = candidate_nodes.cpu().long()
        if candidate_nodes.numel() == 0:
            raise RuntimeError(f"No ranking candidates are available for {etype}")

    def draw(count):
        limit = (
            candidate_nodes.numel()
            if candidate_nodes is not None
            else graph.num_nodes(etype[2] if corrupt_destination else etype[0])
        )
        sampled = torch.randint(limit, (count,), generator=generator)
        return (
            candidate_nodes.index_select(0, sampled)
            if candidate_nodes is not None
            else sampled
        )

    if corrupt_destination:
        negative_dst = draw(candidate_count)
    else:
        negative_src = draw(candidate_count)

    original_src = pos_src.cpu().long().repeat_interleave(negatives_per_positive)
    original_dst = pos_dst.cpu().long().repeat_interleave(negatives_per_positive)
    for _ in range(max_attempts):
        known_mask = known_positive_index.contains(
            etype,
            negative_src,
            negative_dst,
        )
        unchanged_mask = (negative_src == original_src) & (negative_dst == original_dst)
        invalid = known_mask | unchanged_mask
        if not invalid.any():
            break
        invalid_count = int(invalid.sum().item())
        if corrupt_destination:
            negative_dst[invalid] = draw(invalid_count)
        else:
            negative_src[invalid] = draw(invalid_count)

    remaining_invalid = (
        known_positive_index.contains(etype, negative_src, negative_dst)
        | ((negative_src == original_src) & (negative_dst == original_dst))
    )
    if remaining_invalid.any():
        fallback_candidates = (
            candidate_nodes.tolist()
            if candidate_nodes is not None
            else range(
                graph.num_nodes(etype[2])
                if corrupt_destination
                else graph.num_nodes(etype[0])
            )
        )
        for invalid_index in torch.nonzero(
            remaining_invalid,
            as_tuple=False,
        ).flatten().tolist():
            replacement_found = False
            for candidate in fallback_candidates:
                candidate_src = (
                    negative_src[invalid_index]
                    if corrupt_destination
                    else torch.tensor(candidate)
                )
                candidate_dst = (
                    torch.tensor(candidate)
                    if corrupt_destination
                    else negative_dst[invalid_index]
                )
                if (
                    candidate_src == original_src[invalid_index]
                    and candidate_dst == original_dst[invalid_index]
                ):
                    continue
                if not known_positive_index.contains(
                    etype,
                    candidate_src.reshape(1),
                    candidate_dst.reshape(1),
                )[0]:
                    negative_src[invalid_index] = candidate_src
                    negative_dst[invalid_index] = candidate_dst
                    replacement_found = True
                    break
            if not replacement_found:
                raise RuntimeError(
                    f"No filtered negative exists for positive edge in {etype}"
                )
    return negative_src, negative_dst


def relation_candidate_nodes(graph, target_etypes, target_types):
    """Return the exact relation-scoped ranking entities for each target edge."""
    target_types = set(target_types)
    candidates = {}
    for etype in target_etypes:
        src, dst = graph.edges(etype=etype)
        if etype[2] in target_types:
            values = dst
        elif etype[0] in target_types:
            values = src
        else:
            continue
        candidates[etype] = torch.unique(values.cpu().long(), sorted=True)
    return candidates


class BalancedTargetEdgeBatcher:
    """Samples target relations with optional deterministic weighted balancing."""

    def __init__(
        self,
        graph,
        target_etypes,
        target_types,
        batch_size,
        steps_per_epoch=0,
        seed=42,
        modality_weights=None,
    ):
        self.graph = graph
        self.target_types = set(target_types)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.etypes_by_modality = {}
        for etype in target_etypes:
            if graph.num_edges(etype) == 0:
                continue
            modality = target_modality_for_etype(etype, self.target_types)
            if modality is not None:
                self.etypes_by_modality.setdefault(modality, []).append(etype)
        self.modalities = sorted(self.etypes_by_modality)
        if not self.modalities:
            raise ValueError("No non-empty target edge types are available for mini-batch training")
        self.modality_weights = None
        if modality_weights:
            self.modality_weights = torch.tensor(
                [
                    float(modality_weights.get(modality, 1.0))
                    for modality in self.modalities
                ],
                dtype=torch.float64,
            )
            if (
                not torch.isfinite(self.modality_weights).all()
                or bool((self.modality_weights <= 0).any())
            ):
                raise ValueError(
                    "All explicit modality sampling weights must be finite "
                    "and positive"
                )

        total_edges = sum(
            graph.num_edges(etype)
            for etypes in self.etypes_by_modality.values()
            for etype in etypes
        )
        automatic_steps = max(
            len(self.modalities),
            math.ceil(total_edges / self.batch_size),
        )
        self.steps_per_epoch = (
            automatic_steps
            if int(steps_per_epoch) <= 0
            else max(len(self.modalities), int(steps_per_epoch))
        )

    def iter_epoch(self, epoch):
        generator = torch.Generator().manual_seed(self.seed + int(epoch) * 100003)
        modality_offset = int(epoch) % len(self.modalities)
        for step in range(self.steps_per_epoch):
            if (
                self.modality_weights is None
                or step < len(self.modalities)
            ):
                modality_index = (
                    step + modality_offset
                ) % len(self.modalities)
            else:
                modality_index = int(torch.multinomial(
                    self.modality_weights,
                    1,
                    replacement=True,
                    generator=generator,
                ).item())
            modality = self.modalities[modality_index]
            etypes = self.etypes_by_modality[modality]
            etype_index = int(
                torch.randint(
                    len(etypes),
                    (1,),
                    generator=generator,
                ).item()
            )
            etype = etypes[etype_index]
            edge_count = self.graph.num_edges(etype)
            if edge_count >= self.batch_size:
                eids = torch.randperm(
                    edge_count,
                    generator=generator,
                )[: self.batch_size]
            else:
                eids = torch.randint(
                    edge_count,
                    (self.batch_size,),
                    generator=generator,
                )
            pos_src, pos_dst = self.graph.find_edges(eids, etype=etype)
            yield etype, eids, pos_src.cpu(), pos_dst.cpu()

    def __len__(self):
        return self.steps_per_epoch


class FullCoverageTargetEdgeBatcher:
    """Visits every target-relation training edge exactly once per epoch."""

    def __init__(
        self,
        graph,
        target_etypes,
        target_types,
        batch_size,
        seed=42,
    ):
        self.graph = graph
        self.target_types = set(target_types)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.target_etypes = [
            etype
            for etype in sorted(target_etypes)
            if (
                graph.num_edges(etype) > 0
                and target_modality_for_etype(
                    etype,
                    self.target_types,
                )
                is not None
            )
        ]
        if not self.target_etypes:
            raise ValueError(
                "No non-empty target edge types are available for "
                "full-coverage training"
            )
        self.steps_per_epoch = sum(
            math.ceil(graph.num_edges(etype) / self.batch_size)
            for etype in self.target_etypes
        )

    def iter_epoch(self, epoch):
        generator = torch.Generator().manual_seed(
            self.seed + int(epoch) * 100003
        )
        batches = []
        for etype_index, etype in enumerate(self.target_etypes):
            edge_count = self.graph.num_edges(etype)
            permutation = torch.randperm(
                edge_count,
                generator=generator,
            )
            for start in range(0, edge_count, self.batch_size):
                batches.append(
                    (
                        etype_index,
                        start,
                        etype,
                        permutation[start:start + self.batch_size],
                    )
                )
        order = torch.randperm(
            len(batches),
            generator=generator,
        ).tolist()
        for batch_index in order:
            _, _, etype, eids = batches[batch_index]
            pos_src, pos_dst = self.graph.find_edges(
                eids,
                etype=etype,
            )
            yield etype, eids, pos_src.cpu(), pos_dst.cpu()

    def __len__(self):
        return self.steps_per_epoch


class RelationCoverageEdgeBatcher:
    """Visits every selected train relation with an optional per-relation cap."""

    def __init__(
        self,
        graph,
        etypes,
        batch_size,
        max_edges_per_relation=0,
        seed=42,
    ):
        self.graph = graph
        self.batch_size = max(1, int(batch_size))
        self.max_edges_per_relation = max(0, int(max_edges_per_relation))
        self.seed = int(seed)
        self.etypes = [
            etype
            for etype in sorted(set(etypes))
            if graph.num_edges(etype) > 0
        ]
        if not self.etypes:
            raise ValueError(
                "No non-empty edge types are available for relation coverage"
            )
        self.available_edges = sum(
            graph.num_edges(etype)
            for etype in self.etypes
        )
        self.selected_edges_by_etype = {
            etype: min(
                graph.num_edges(etype),
                self.max_edges_per_relation,
            )
            if self.max_edges_per_relation > 0
            else graph.num_edges(etype)
            for etype in self.etypes
        }
        self.covered_edges_per_epoch = sum(
            self.selected_edges_by_etype.values()
        )
        self.steps_per_epoch = sum(
            math.ceil(edge_count / self.batch_size)
            for edge_count in self.selected_edges_by_etype.values()
        )

    def iter_epoch(self, epoch):
        generator = torch.Generator().manual_seed(
            self.seed + int(epoch) * 100003
        )
        batches = []
        for etype_index, etype in enumerate(self.etypes):
            selected_count = self.selected_edges_by_etype[etype]
            permutation = torch.randperm(
                self.graph.num_edges(etype),
                generator=generator,
            )[:selected_count]
            for start in range(0, selected_count, self.batch_size):
                batches.append(
                    (
                        etype_index,
                        start,
                        etype,
                        permutation[start:start + self.batch_size],
                    )
                )
        order = torch.randperm(
            len(batches),
            generator=generator,
        ).tolist()
        for batch_index in order:
            _, _, etype, eids = batches[batch_index]
            pos_src, pos_dst = self.graph.find_edges(
                eids,
                etype=etype,
            )
            yield etype, eids, pos_src.cpu(), pos_dst.cpu()

    def __len__(self):
        return self.steps_per_epoch


class QueryContextSampler:
    """Samples a multi-hop context and relabels it into a compact induced graph."""

    def __init__(self, fanouts, seed=42):
        self.fanouts = [max(1, int(fanout)) for fanout in fanouts]
        if not self.fanouts:
            raise ValueError("At least one context fanout is required")
        self.seed = int(seed)

    @staticmethod
    def _add_seed(seed_sets, node_type, node_ids):
        seed_sets[node_type].update(int(node_id) for node_id in node_ids.tolist())

    def _collect_context_nodes(self, graph, seed_nodes, random_seed):
        torch.manual_seed(int(random_seed))
        try:
            dgl.seed(int(random_seed))
        except AttributeError:
            pass

        context = {node_type: set() for node_type in graph.ntypes}
        hop_by_node = {node_type: {} for node_type in graph.ntypes}
        frontier = {}
        for node_type in graph.ntypes:
            values = {
                int(node_id)
                for node_id in seed_nodes.get(
                    node_type,
                    torch.empty(0, dtype=torch.long),
                ).tolist()
            }
            context[node_type].update(values)
            hop_by_node[node_type].update({node_id: 0 for node_id in values})
            frontier[node_type] = values

        for hop, fanout in enumerate(self.fanouts, start=1):
            frontier_tensors = {
                node_type: torch.tensor(sorted(node_ids), dtype=torch.long)
                for node_type, node_ids in frontier.items()
                if node_ids
            }
            if not frontier_tensors:
                break

            sampled_graphs = []
            for edge_direction in ("in", "out"):
                try:
                    sampled_graphs.append(
                        dgl.sampling.sample_neighbors(
                            graph,
                            frontier_tensors,
                            fanout,
                            edge_dir=edge_direction,
                        )
                    )
                except Exception as exc:
                    logging.debug(
                        "Context sampling direction '%s' failed at hop %d: %s",
                        edge_direction,
                        hop,
                        exc,
                    )

            next_frontier = {node_type: set() for node_type in graph.ntypes}
            for sampled_graph in sampled_graphs:
                for etype in sampled_graph.canonical_etypes:
                    if sampled_graph.num_edges(etype) == 0:
                        continue
                    src_nodes, dst_nodes = sampled_graph.edges(etype=etype)
                    for node_type, node_ids in (
                        (etype[0], src_nodes.tolist()),
                        (etype[2], dst_nodes.tolist()),
                    ):
                        for node_id in node_ids:
                            node_id = int(node_id)
                            if node_id not in context[node_type]:
                                context[node_type].add(node_id)
                                hop_by_node[node_type][node_id] = hop
                                next_frontier[node_type].add(node_id)
            frontier = next_frontier
        return context, hop_by_node

    @staticmethod
    def _annotate_context_graph(context_graph, hop_by_node, seed_nodes, max_hop):
        seed_lookup = {
            node_type: set(node_ids.tolist())
            for node_type, node_ids in seed_nodes.items()
        }
        evidence_sum = {
            node_type: torch.zeros(context_graph.num_nodes(node_type))
            for node_type in context_graph.ntypes
        }
        evidence_degree = {
            node_type: torch.zeros(context_graph.num_nodes(node_type))
            for node_type in context_graph.ntypes
        }

        for etype in context_graph.canonical_etypes:
            if context_graph.num_edges(etype) == 0:
                continue
            src_nodes, dst_nodes = context_graph.edges(etype=etype)
            edge_data = context_graph.edges[etype].data
            weight = edge_data["weight"] if "weight" in edge_data else None
            if weight is None:
                weight = torch.ones(src_nodes.numel())
            else:
                weight = weight.detach().cpu().float().reshape(-1).abs()
            evidence_sum[etype[0]].index_add_(0, src_nodes.cpu(), weight)
            evidence_sum[etype[2]].index_add_(0, dst_nodes.cpu(), weight)
            ones = torch.ones(src_nodes.numel())
            evidence_degree[etype[0]].index_add_(0, src_nodes.cpu(), ones)
            evidence_degree[etype[2]].index_add_(0, dst_nodes.cpu(), ones)

        for node_type in context_graph.ntypes:
            global_ids = context_graph.nodes[node_type].data[dgl.NID].cpu().long()
            context_graph.nodes[node_type].data[QUERY_HOP_KEY] = torch.tensor(
                [
                    hop_by_node[node_type].get(int(node_id), max_hop + 1)
                    for node_id in global_ids.tolist()
                ],
                dtype=torch.long,
            )
            context_graph.nodes[node_type].data[QUERY_SEED_KEY] = torch.tensor(
                [
                    int(node_id) in seed_lookup.get(node_type, set())
                    for node_id in global_ids.tolist()
                ],
                dtype=torch.bool,
            )
            context_graph.nodes[node_type].data[EVIDENCE_SCORE_KEY] = (
                evidence_sum[node_type]
                / evidence_degree[node_type].clamp_min(1.0)
            )

    @staticmethod
    def _global_to_local(context_graph):
        return {
            node_type: {
                int(global_id): local_id
                for local_id, global_id in enumerate(
                    context_graph.nodes[node_type].data[dgl.NID].tolist()
                )
            }
            for node_type in context_graph.ntypes
        }

    def build_batch_graphs(
        self,
        graph,
        etype,
        pos_src,
        pos_dst,
        neg_src,
        neg_dst,
        random_seed,
    ):
        seed_sets = {node_type: set() for node_type in graph.ntypes}
        self._add_seed(seed_sets, etype[0], pos_src)
        self._add_seed(seed_sets, etype[2], pos_dst)
        self._add_seed(seed_sets, etype[0], neg_src)
        self._add_seed(seed_sets, etype[2], neg_dst)
        seed_nodes = {
            node_type: torch.tensor(sorted(node_ids), dtype=torch.long)
            for node_type, node_ids in seed_sets.items()
            if node_ids
        }

        context, hop_by_node = self._collect_context_nodes(
            graph,
            seed_nodes,
            random_seed,
        )
        subgraph_nodes = {
            node_type: torch.tensor(
                sorted(context[node_type]),
                dtype=torch.long,
            )
            for node_type in graph.ntypes
        }
        context_graph = dgl.node_subgraph(
            graph,
            subgraph_nodes,
            relabel_nodes=True,
            store_ids=True,
        )
        self._annotate_context_graph(
            context_graph,
            hop_by_node,
            seed_nodes,
            len(self.fanouts),
        )

        local_lookup = self._global_to_local(context_graph)
        pos_src_local = torch.tensor(
            [local_lookup[etype[0]][int(node_id)] for node_id in pos_src.tolist()],
            dtype=torch.long,
        )
        pos_dst_local = torch.tensor(
            [local_lookup[etype[2]][int(node_id)] for node_id in pos_dst.tolist()],
            dtype=torch.long,
        )
        neg_src_local = torch.tensor(
            [local_lookup[etype[0]][int(node_id)] for node_id in neg_src.tolist()],
            dtype=torch.long,
        )
        neg_dst_local = torch.tensor(
            [local_lookup[etype[2]][int(node_id)] for node_id in neg_dst.tolist()],
            dtype=torch.long,
        )
        num_nodes_dict = {
            node_type: context_graph.num_nodes(node_type)
            for node_type in context_graph.ntypes
        }
        pos_graph = dgl.heterograph(
            {etype: (pos_src_local, pos_dst_local)},
            num_nodes_dict=num_nodes_dict,
        )
        neg_graph = dgl.heterograph(
            {etype: (neg_src_local, neg_dst_local)},
            num_nodes_dict=num_nodes_dict,
        )
        for node_type in context_graph.ntypes:
            global_ids = context_graph.nodes[node_type].data[dgl.NID]
            pos_graph.nodes[node_type].data[dgl.NID] = global_ids.clone()
            neg_graph.nodes[node_type].data[dgl.NID] = global_ids.clone()
        return context_graph, pos_graph, neg_graph


class HardNegativePool:
    def __init__(self):
        self._pool = {}
        self.last_refresh_epoch = None

    def get(self, etype, eid):
        return self._pool.get((etype, int(eid)))

    def refresh(
        self,
        model,
        graph,
        target_etypes,
        target_types,
        known_positive_index,
        epoch,
        max_edges,
        candidates_per_positive,
        negatives_per_positive,
        margin,
        score_chunk_size,
        seed,
        candidate_nodes_by_etype=None,
    ):
        target_etypes = [
            etype for etype in target_etypes if graph.num_edges(etype) > 0
        ]
        if not target_etypes or max_edges <= 0:
            return 0

        generator = torch.Generator().manual_seed(int(seed) + int(epoch) * 7919)
        per_etype_limit = max(1, int(max_edges) // len(target_etypes))
        cached_embeddings = model.cache_embeddings(
            graph,
            cache_key=("hard-negative", int(epoch)),
            required_etypes=target_etypes,
            force=True,
        )
        refreshed = 0
        new_pool = {}
        for etype in target_etypes:
            edge_count = graph.num_edges(etype)
            selected_count = min(edge_count, per_etype_limit)
            selected_eids = torch.randperm(
                edge_count,
                generator=generator,
            )[:selected_count]
            pos_src, pos_dst = graph.find_edges(selected_eids, etype=etype)
            candidate_src, candidate_dst = sample_filtered_negatives(
                graph,
                etype,
                pos_src,
                pos_dst,
                candidates_per_positive,
                target_types,
                known_positive_index,
                generator,
                candidate_nodes_by_etype,
            )
            positive_scores = model.score_cached_pairs(
                cached_embeddings,
                etype,
                pos_src,
                pos_dst,
                chunk_size=score_chunk_size,
            )
            candidate_scores = model.score_cached_pairs(
                cached_embeddings,
                etype,
                candidate_src,
                candidate_dst,
                chunk_size=score_chunk_size,
            ).reshape(selected_count, candidates_per_positive)
            candidate_src = candidate_src.reshape(
                selected_count,
                candidates_per_positive,
            )
            candidate_dst = candidate_dst.reshape(
                selected_count,
                candidates_per_positive,
            )

            for row, eid in enumerate(selected_eids.tolist()):
                row_scores = candidate_scores[row]
                semi_hard = row_scores < (positive_scores[row] - float(margin))
                preferred_scores = row_scores.masked_fill(~semi_hard, float("-inf"))
                take_count = min(negatives_per_positive, candidates_per_positive)
                if int(semi_hard.sum().item()) >= take_count:
                    chosen = torch.topk(preferred_scores, k=take_count).indices
                else:
                    chosen = torch.topk(row_scores, k=take_count).indices
                new_pool[(etype, int(eid))] = (
                    candidate_src[row].index_select(0, chosen).clone(),
                    candidate_dst[row].index_select(0, chosen).clone(),
                )
                refreshed += 1

        self._pool = new_pool
        self.last_refresh_epoch = int(epoch)
        return refreshed

    def mix_with_random(
        self,
        etype,
        eids,
        random_src,
        random_dst,
        negatives_per_positive,
        hard_ratio,
    ):
        negatives_per_positive = int(negatives_per_positive)
        mixed_src = random_src.reshape(-1, negatives_per_positive).clone()
        mixed_dst = random_dst.reshape(-1, negatives_per_positive).clone()
        hard_count = min(
            negatives_per_positive,
            max(0, int(math.ceil(negatives_per_positive * float(hard_ratio)))),
        )
        if hard_count == 0:
            return random_src, random_dst

        for row, eid in enumerate(eids.tolist()):
            hard_pair = self.get(etype, eid)
            if hard_pair is None:
                continue
            hard_src, hard_dst = hard_pair
            count = min(hard_count, hard_src.numel())
            mixed_src[row, :count] = hard_src[:count]
            mixed_dst[row, :count] = hard_dst[:count]
        return mixed_src.reshape(-1), mixed_dst.reshape(-1)


def build_fixed_edge_batches(
    graph,
    target_etypes,
    target_types,
    known_positive_index,
    batch_size,
    negatives_per_positive,
    seed,
    max_edges=0,
    candidate_nodes_by_etype=None,
):
    generator = torch.Generator().manual_seed(int(seed))
    fixed_batches = []
    for etype in target_etypes:
        edge_count = graph.num_edges(etype)
        if edge_count == 0:
            continue
        eids = torch.arange(edge_count)
        if int(max_edges) > 0 and edge_count > int(max_edges):
            eids = torch.randperm(
                edge_count,
                generator=generator,
            )[: int(max_edges)]
        for start in range(0, eids.numel(), max(1, int(batch_size))):
            batch_eids = eids[start : start + max(1, int(batch_size))]
            pos_src, pos_dst = graph.find_edges(batch_eids, etype=etype)
            neg_src, neg_dst = sample_filtered_negatives(
                graph,
                etype,
                pos_src,
                pos_dst,
                negatives_per_positive,
                target_types,
                known_positive_index,
                generator,
                candidate_nodes_by_etype,
            )
            fixed_batches.append(
                EdgeBatch(
                    etype=etype,
                    eids=batch_eids.cpu(),
                    pos_src=pos_src.cpu(),
                    pos_dst=pos_dst.cpu(),
                    neg_src=neg_src,
                    neg_dst=neg_dst,
                    negatives_per_positive=int(negatives_per_positive),
                )
            )
    return fixed_batches
