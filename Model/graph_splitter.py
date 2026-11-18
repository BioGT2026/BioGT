"""Utilities for splitting DGL heterographs into train/validation/test subsets.

This module centralises edge splitting logic so different strategies (random,
category-aware, k-fold, etc.) can be invoked consistently from the
`DGLGraphCreator` and other callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import dgl
import numpy as np

CanonicalEType = Tuple[str, str, str]


def _empty_int_array() -> np.ndarray:
    """Return a reusable empty array of dtype int64."""
    return np.empty(0, dtype=np.int64)


@dataclass
class SplitResult:
    """Container for the outcome of an edge split operation."""

    train_eids: Dict[CanonicalEType, np.ndarray]
    val_eids: Dict[CanonicalEType, np.ndarray]
    test_eids: Dict[CanonicalEType, np.ndarray]
    folds: Optional[List[Dict[CanonicalEType, np.ndarray]]] = None
    metadata: Optional[Dict[str, object]] = None


class BaseEdgeSplitter:
    """Shared utilities for edge splitters."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng


class RandomEdgeSplitter(BaseEdgeSplitter):
    """Random per-edge-type splitting that mirrors the previous behaviour."""

    def __init__(
        self,
        val_size: float,
        test_size: float,
        random_state: Optional[int] = None,
        ensure_minimum: bool = True,
    ) -> None:
        if val_size < 0 or test_size < 0:
            raise ValueError("val_size and test_size must be non-negative")
        if val_size + test_size >= 1.0:
            raise ValueError("The sum of val_size and test_size must be less than 1.0")
        super().__init__(random_state=random_state)
        self.val_size = val_size
        self.test_size = test_size
        self.ensure_minimum = ensure_minimum

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        train: Dict[CanonicalEType, np.ndarray] = {}
        val: Dict[CanonicalEType, np.ndarray] = {}
        test: Dict[CanonicalEType, np.ndarray] = {}

        for etype in graph.canonical_etypes:
            num_edges = graph.num_edges(etype)
            if num_edges == 0:
                empty = _empty_int_array()
                train[etype] = empty
                val[etype] = empty
                test[etype] = empty
                continue

            eids = np.arange(num_edges, dtype=np.int64)
            self.rng.shuffle(eids)

            num_val = int(num_edges * self.val_size) if self.val_size > 0 else 0
            num_test = int(num_edges * self.test_size) if self.test_size > 0 else 0

            if self.ensure_minimum and num_edges >= 3:
                if self.val_size > 0 and num_val == 0:
                    num_val = 1
                if self.test_size > 0 and num_test == 0 and (num_edges - num_val) >= 2:
                    num_test = 1
                if num_val + num_test > num_edges:
                    num_test = max(0, num_edges - num_val)

            val_split = eids[:num_val].copy() if num_val > 0 else _empty_int_array()
            test_split = (
                eids[num_val : num_val + num_test].copy()
                if num_test > 0
                else _empty_int_array()
            )
            train_split = eids[num_val + num_test :].copy()

            train[etype] = train_split if train_split.size > 0 else _empty_int_array()
            val[etype] = val_split
            test[etype] = test_split

        return SplitResult(train_eids=train, val_eids=val, test_eids=test)


def _node_split_counts(
    num_nodes: int,
    val_size: float,
    test_size: float,
) -> Tuple[int, int]:
    num_val = int(num_nodes * val_size) if val_size > 0 else 0
    num_test = int(num_nodes * test_size) if test_size > 0 else 0

    if val_size > 0 and num_val == 0 and num_nodes >= 2:
        num_val = 1
    if test_size > 0 and num_test == 0 and (num_nodes - num_val) >= 2:
        num_test = 1
    if num_val + num_test > num_nodes:
        num_test = max(0, num_nodes - num_val)

    return num_val, num_test


def _edge_ids_for_node_assignments(
    graph: dgl.DGLHeteroGraph,
    node_assignments: Mapping[str, np.ndarray],
) -> Tuple[
    Dict[CanonicalEType, np.ndarray],
    Dict[CanonicalEType, np.ndarray],
    Dict[CanonicalEType, np.ndarray],
    Dict[CanonicalEType, np.ndarray],
]:
    split_eids = {
        "train": {},
        "val": {},
        "test": {},
    }
    excluded: Dict[CanonicalEType, np.ndarray] = {}

    for etype in graph.canonical_etypes:
        src_type, _, dst_type = etype
        src_nodes, dst_nodes = graph.edges(etype=etype)
        src_np = src_nodes.detach().cpu().numpy()
        dst_np = dst_nodes.detach().cpu().numpy()
        src_splits = node_assignments[src_type][src_np]
        dst_splits = node_assignments[dst_type][dst_np]

        same_split = src_splits == dst_splits
        for split_name in ("train", "val", "test"):
            eids = np.flatnonzero(same_split & (src_splits == split_name)).astype(
                np.int64,
                copy=False,
            )
            split_eids[split_name][etype] = (
                eids if eids.size else _empty_int_array()
            )

        cross_eids = np.flatnonzero(~same_split).astype(np.int64, copy=False)
        excluded[etype] = cross_eids if cross_eids.size else _empty_int_array()

    return (
        split_eids["train"],
        split_eids["val"],
        split_eids["test"],
        excluded,
    )


class NodeRandomSplitter(BaseEdgeSplitter):
    """Partition every typed node, then retain only within-split edges."""

    def __init__(
        self,
        val_size: float,
        test_size: float,
        random_state: Optional[int] = None,
    ) -> None:
        if val_size < 0 or test_size < 0:
            raise ValueError("val_size and test_size must be non-negative")
        if val_size + test_size >= 1.0:
            raise ValueError("The sum of val_size and test_size must be less than 1.0")
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.val_size = val_size
        self.test_size = test_size

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        rng = np.random.default_rng(self.random_state)
        assignments: Dict[str, np.ndarray] = {}
        node_splits = {
            "train": {},
            "val": {},
            "test": {},
        }

        for ntype in sorted(graph.ntypes):
            node_ids = np.arange(graph.num_nodes(ntype), dtype=np.int64)
            rng.shuffle(node_ids)
            num_val, num_test = _node_split_counts(
                node_ids.size,
                self.val_size,
                self.test_size,
            )
            val_nodes = node_ids[:num_val].copy()
            test_nodes = node_ids[num_val : num_val + num_test].copy()
            train_nodes = node_ids[num_val + num_test :].copy()

            labels = np.full(node_ids.size, "train", dtype="<U5")
            labels[val_nodes] = "val"
            labels[test_nodes] = "test"
            assignments[ntype] = labels
            node_splits["train"][ntype] = train_nodes
            node_splits["val"][ntype] = val_nodes
            node_splits["test"][ntype] = test_nodes

        train, val, test, excluded = _edge_ids_for_node_assignments(
            graph,
            assignments,
        )
        metadata = {
            "strategy": "node_random",
            "node_splits": node_splits,
            "excluded_cross_edges": excluded,
            "excluded_cross_edge_counts": {
                etype: int(eids.size) for etype, eids in excluded.items()
            },
            "excluded_cross_edge_total": int(
                sum(eids.size for eids in excluded.values())
            ),
        }
        return SplitResult(
            train_eids=train,
            val_eids=val,
            test_eids=test,
            metadata=metadata,
        )


class NodeKFoldSplitter(BaseEdgeSplitter):
    """Partition typed nodes into folds and induce train/val/test edge sets."""

    def __init__(
        self,
        n_splits: int = 5,
        fold_index: int = 0,
        val_fold_index: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if n_splits < 3:
            raise ValueError(
                "n_splits must be at least 3 so test, validation, and training folds are disjoint"
            )
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits
        self.fold_index = fold_index % n_splits
        self.val_fold_index = (
            (self.fold_index + 1) % n_splits
            if val_fold_index is None
            else val_fold_index % n_splits
        )
        if self.val_fold_index == self.fold_index:
            raise ValueError("test and validation fold indices must be disjoint")

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        rng = np.random.default_rng(self.random_state)
        node_folds: List[Dict[str, np.ndarray]] = [
            {} for _ in range(self.n_splits)
        ]
        node_splits = {
            "train": {},
            "val": {},
            "test": {},
        }
        assignments: Dict[str, np.ndarray] = {}
        fold_assignments: Dict[str, np.ndarray] = {}

        for ntype in sorted(graph.ntypes):
            node_ids = np.arange(graph.num_nodes(ntype), dtype=np.int64)
            rng.shuffle(node_ids)
            typed_folds = np.array_split(node_ids, self.n_splits)
            fold_labels = np.empty(node_ids.size, dtype=np.int64)
            for fold_idx, fold_nodes in enumerate(typed_folds):
                fold_nodes = fold_nodes.astype(np.int64, copy=True)
                node_folds[fold_idx][ntype] = fold_nodes
                fold_labels[fold_nodes] = fold_idx

            labels = np.full(node_ids.size, "train", dtype="<U5")
            labels[fold_labels == self.val_fold_index] = "val"
            labels[fold_labels == self.fold_index] = "test"
            assignments[ntype] = labels
            fold_assignments[ntype] = fold_labels
            node_splits["train"][ntype] = np.flatnonzero(
                labels == "train"
            ).astype(np.int64, copy=False)
            node_splits["val"][ntype] = np.flatnonzero(labels == "val").astype(
                np.int64,
                copy=False,
            )
            node_splits["test"][ntype] = np.flatnonzero(
                labels == "test"
            ).astype(np.int64, copy=False)

        train, val, test, excluded = _edge_ids_for_node_assignments(
            graph,
            assignments,
        )

        edge_folds: List[Dict[CanonicalEType, np.ndarray]] = [
            {} for _ in range(self.n_splits)
        ]
        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            src_np = src_nodes.detach().cpu().numpy()
            dst_np = dst_nodes.detach().cpu().numpy()
            src_folds = fold_assignments[src_type][src_np]
            dst_folds = fold_assignments[dst_type][dst_np]
            for fold_idx in range(self.n_splits):
                fold_eids = np.flatnonzero(
                    (src_folds == fold_idx) & (dst_folds == fold_idx)
                ).astype(np.int64, copy=False)
                edge_folds[fold_idx][etype] = (
                    fold_eids if fold_eids.size else _empty_int_array()
                )

        metadata = {
            "strategy": "node_kfold",
            "n_splits": self.n_splits,
            "test_fold_index": self.fold_index,
            "val_fold_index": self.val_fold_index,
            "train_fold_indices": [
                idx
                for idx in range(self.n_splits)
                if idx not in {self.fold_index, self.val_fold_index}
            ],
            "node_folds": node_folds,
            "node_splits": node_splits,
            "excluded_cross_edges": excluded,
            "excluded_cross_edge_counts": {
                etype: int(eids.size) for etype, eids in excluded.items()
            },
            "excluded_cross_edge_total": int(
                sum(eids.size for eids in excluded.values())
            ),
        }
        return SplitResult(
            train_eids=train,
            val_eids=val,
            test_eids=test,
            folds=edge_folds,
            metadata=metadata,
        )


def resolve_typed_node_ids(
    requested_ids: Iterable[str],
    graph: dgl.DGLHeteroGraph,
    node_id_map: Mapping[object, object],
    node_dglid_to_orig: Mapping[str, Mapping[int, object]],
) -> Tuple[Dict[str, Set[int]], List[str]]:
    """Resolve original IDs to per-type DGL IDs, with optional ``type:id`` input."""

    requested = [str(value).strip() for value in requested_ids if str(value).strip()]
    type_names = set(graph.ntypes)
    available: Dict[str, Dict[str, int]] = {ntype: {} for ntype in graph.ntypes}

    for ntype in graph.ntypes:
        for dgl_id, original_id in node_dglid_to_orig.get(ntype, {}).items():
            available[ntype][str(original_id)] = int(dgl_id)

        typed_id_map = node_id_map.get(ntype) if isinstance(node_id_map, Mapping) else None
        if isinstance(typed_id_map, Mapping):
            for original_id, dgl_id in typed_id_map.items():
                available[ntype].setdefault(str(original_id), int(dgl_id))

    matched: Dict[str, Set[int]] = {ntype: set() for ntype in graph.ntypes}
    unmatched: List[str] = []

    for raw_id in requested:
        requested_type: Optional[str] = None
        original_id = raw_id
        for delimiter in ("\t", ",", ":"):
            prefix, separator, suffix = raw_id.partition(delimiter)
            if separator and prefix in type_names and suffix:
                requested_type = prefix
                original_id = suffix
                break

        candidate_types = [requested_type] if requested_type else sorted(graph.ntypes)
        found = False
        for ntype in candidate_types:
            dgl_id = available.get(ntype, {}).get(original_id)
            if dgl_id is None:
                continue
            matched[ntype].add(dgl_id)
            found = True
        if not found:
            unmatched.append(raw_id)

    return matched, unmatched


class SpecificIDEdgeSplitter(BaseEdgeSplitter):
    """Place only edges incident to matched typed nodes in the test split."""

    def __init__(
        self,
        typed_test_nodes: Mapping[str, Iterable[int]],
        val_size: float,
        random_state: Optional[int] = None,
    ) -> None:
        if val_size < 0 or val_size >= 1:
            raise ValueError("val_size must be in [0, 1)")
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.val_size = val_size
        self.typed_test_nodes = {
            ntype: np.fromiter(
                sorted({int(node_id) for node_id in node_ids}),
                dtype=np.int64,
            )
            for ntype, node_ids in typed_test_nodes.items()
        }

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        rng = np.random.default_rng(self.random_state)
        train: Dict[CanonicalEType, np.ndarray] = {}
        val: Dict[CanonicalEType, np.ndarray] = {}
        test: Dict[CanonicalEType, np.ndarray] = {}

        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            src_np = src_nodes.detach().cpu().numpy()
            dst_np = dst_nodes.detach().cpu().numpy()
            src_matches = self.typed_test_nodes.get(
                src_type,
                _empty_int_array(),
            )
            dst_matches = self.typed_test_nodes.get(
                dst_type,
                _empty_int_array(),
            )
            test_mask = np.isin(src_np, src_matches) | np.isin(
                dst_np,
                dst_matches,
            )
            test_eids = np.flatnonzero(test_mask).astype(np.int64, copy=False)
            remaining = np.flatnonzero(~test_mask).astype(np.int64, copy=False)
            rng.shuffle(remaining)

            target_val_size = int(graph.num_edges(etype) * self.val_size)
            num_val = min(target_val_size, remaining.size)
            val_eids = remaining[:num_val].copy()
            train_eids = remaining[num_val:].copy()

            train[etype] = train_eids if train_eids.size else _empty_int_array()
            val[etype] = val_eids if val_eids.size else _empty_int_array()
            test[etype] = test_eids if test_eids.size else _empty_int_array()

        return SplitResult(
            train_eids=train,
            val_eids=val,
            test_eids=test,
            metadata={
                "strategy": "specific_ids",
                "matched_typed_nodes": self.typed_test_nodes,
            },
        )


class ExternalEdgePairSplitter(BaseEdgeSplitter):
    """Hold out an explicit set of typed endpoint pairs as external test edges."""

    def __init__(
        self,
        typed_test_pairs: Iterable[Tuple[str, int, str, int]],
        val_size: float,
        random_state: Optional[int] = None,
    ) -> None:
        if val_size < 0 or val_size >= 1:
            raise ValueError("val_size must be in [0, 1)")
        super().__init__(random_state=random_state)
        self.random_state = random_state
        self.val_size = val_size
        self.typed_test_pairs = {
            (str(src_type), int(src_id), str(dst_type), int(dst_id))
            for src_type, src_id, dst_type, dst_id in typed_test_pairs
        }

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        rng = np.random.default_rng(self.random_state)
        train: Dict[CanonicalEType, np.ndarray] = {}
        val: Dict[CanonicalEType, np.ndarray] = {}
        test: Dict[CanonicalEType, np.ndarray] = {}
        matched_pairs: Set[Tuple[str, int, str, int]] = set()

        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            endpoint_pairs = list(zip(src_nodes.tolist(), dst_nodes.tolist()))
            test_mask = np.fromiter(
                (
                    (src_type, int(src_id), dst_type, int(dst_id))
                    in self.typed_test_pairs
                    for src_id, dst_id in endpoint_pairs
                ),
                dtype=bool,
                count=len(endpoint_pairs),
            )
            test_eids = np.flatnonzero(test_mask).astype(np.int64, copy=False)
            for eid in test_eids:
                src_id, dst_id = endpoint_pairs[int(eid)]
                matched_pairs.add(
                    (src_type, int(src_id), dst_type, int(dst_id))
                )

            remaining = np.flatnonzero(~test_mask).astype(np.int64, copy=False)
            rng.shuffle(remaining)
            num_val = min(int(remaining.size * self.val_size), remaining.size)
            if self.val_size > 0 and remaining.size >= 2 and num_val == 0:
                num_val = 1

            val_eids = remaining[:num_val].copy()
            train_eids = remaining[num_val:].copy()
            train[etype] = train_eids if train_eids.size else _empty_int_array()
            val[etype] = val_eids if val_eids.size else _empty_int_array()
            test[etype] = test_eids if test_eids.size else _empty_int_array()

        unmatched_pairs = self.typed_test_pairs - matched_pairs
        if unmatched_pairs:
            logging.warning(
                "External split matched %d/%d directed typed pair(s); %d were absent.",
                len(matched_pairs),
                len(self.typed_test_pairs),
                len(unmatched_pairs),
            )
        return SplitResult(
            train_eids=train,
            val_eids=val,
            test_eids=test,
            metadata={
                "strategy": "external_edges",
                "requested_directed_pair_count": len(self.typed_test_pairs),
                "matched_directed_pair_count": len(matched_pairs),
                "unmatched_directed_pair_count": len(unmatched_pairs),
                "test_edge_count": sum(len(eids) for eids in test.values()),
            },
        )


class CategoryEdgeSplitter(RandomEdgeSplitter):
    """Splitter that forces designated node types into specific datasets."""

    def __init__(
        self,
        val_size: float,
        test_size: float,
        category_assignments: Mapping[str, str],
        random_state: Optional[int] = None,
        ensure_minimum: bool = True,
    ) -> None:
        super().__init__(
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            ensure_minimum=ensure_minimum,
        )
        self.category_assignments = {
            k.strip(): v.strip().lower()
            for k, v in category_assignments.items()
            if v.strip().lower() in {"train", "val", "test"}
        }

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        base_result = super().split(graph)

        if not self.category_assignments:
            return base_result

        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            target_split = self.category_assignments.get(src_type) or self.category_assignments.get(dst_type)
            if not target_split:
                continue

            all_eids = np.arange(graph.num_edges(etype), dtype=np.int64)
            if target_split == "train":
                base_result.train_eids[etype] = all_eids
                base_result.val_eids[etype] = _empty_int_array()
                base_result.test_eids[etype] = _empty_int_array()
            elif target_split == "val":
                base_result.train_eids[etype] = _empty_int_array()
                base_result.val_eids[etype] = all_eids
                base_result.test_eids[etype] = _empty_int_array()
            else:  # target_split == "test"
                base_result.train_eids[etype] = _empty_int_array()
                base_result.val_eids[etype] = _empty_int_array()
                base_result.test_eids[etype] = all_eids

            logging.info(
                "Category-aware splitting assigned all %d edge(s) of %s to %s set.",
                all_eids.size,
                etype,
                target_split,
            )

        return base_result


class DiseaseCentricSplitter(RandomEdgeSplitter):
    """Assign entire disease nodes to specific splits so their incident edges stay together."""

    def __init__(
        self,
        val_size: float,
        test_size: float,
        disease_node_type: str = "disease",
        random_state: Optional[int] = None,
        ensure_minimum: bool = True,
    ) -> None:
        super().__init__(
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            ensure_minimum=ensure_minimum,
        )
        self.disease_node_type = disease_node_type

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        if self.disease_node_type not in graph.ntypes:
            logging.warning(
                "Disease-centric split requested but node type '%s' was not found. Falling back to random split.",
                self.disease_node_type,
            )
            return super().split(graph)

        num_disease_nodes = graph.num_nodes(self.disease_node_type)
        if num_disease_nodes == 0:
            logging.warning(
                "Disease-centric split requested but there are no nodes of type '%s'. Falling back to random split.",
                self.disease_node_type,
            )
            return super().split(graph)

        disease_ids = np.arange(num_disease_nodes, dtype=np.int64)
        self.rng.shuffle(disease_ids)

        num_val_nodes = int(num_disease_nodes * self.val_size) if self.val_size > 0 else 0
        num_test_nodes = int(num_disease_nodes * self.test_size) if self.test_size > 0 else 0

        if self.ensure_minimum:
            if self.val_size > 0 and num_val_nodes == 0 and num_disease_nodes >= 2:
                num_val_nodes = 1
            if (
                self.test_size > 0
                and num_test_nodes == 0
                and (num_disease_nodes - num_val_nodes) >= 2
            ):
                num_test_nodes = 1
            if num_val_nodes + num_test_nodes > num_disease_nodes:
                num_test_nodes = max(0, num_disease_nodes - num_val_nodes)

        val_nodes = set(disease_ids[:num_val_nodes])
        test_nodes = set(disease_ids[num_val_nodes : num_val_nodes + num_test_nodes])
        train_nodes = set(disease_ids[num_val_nodes + num_test_nodes :])

        def assignment_for_node(node_id: int) -> str:
            if node_id in test_nodes:
                return "test"
            if node_id in val_nodes:
                return "val"
            return "train"

        base_result = super().split(graph)

        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            if src_type != self.disease_node_type and dst_type != self.disease_node_type:
                continue

            train_set = set(map(int, base_result.train_eids.get(etype, _empty_int_array())))
            val_set = set(map(int, base_result.val_eids.get(etype, _empty_int_array())))
            test_set = set(map(int, base_result.test_eids.get(etype, _empty_int_array())))

            src_nodes, dst_nodes = graph.edges(etype=etype)
            src_np = src_nodes.numpy()
            dst_np = dst_nodes.numpy()

            for eid, (src_id, dst_id) in enumerate(zip(src_np, dst_np)):
                edge_assignments: List[str] = []
                if src_type == self.disease_node_type:
                    edge_assignments.append(assignment_for_node(int(src_id)))
                if dst_type == self.disease_node_type:
                    edge_assignments.append(assignment_for_node(int(dst_id)))

                if not edge_assignments:
                    continue

                if "test" in edge_assignments:
                    target = "test"
                elif "val" in edge_assignments:
                    target = "val"
                else:
                    target = "train"

                if target == "test" and "val" in edge_assignments:
                    logging.warning(
                        "Edge %s in %s touched disease nodes assigned to different splits (val/test). Prioritising 'test'.",
                        eid,
                        etype,
                    )
                elif target == "val" and "train" in edge_assignments:
                    logging.debug(
                        "Edge %s in %s involves disease nodes split across train/val. Assigning to 'val'.",
                        eid,
                        etype,
                    )

                train_set.discard(eid)
                val_set.discard(eid)
                test_set.discard(eid)

                if target == "train":
                    train_set.add(eid)
                elif target == "val":
                    val_set.add(eid)
                else:
                    test_set.add(eid)

            base_result.train_eids[etype] = (
                np.fromiter(sorted(train_set), dtype=np.int64)
                if train_set
                else _empty_int_array()
            )
            base_result.val_eids[etype] = (
                np.fromiter(sorted(val_set), dtype=np.int64)
                if val_set
                else _empty_int_array()
            )
            base_result.test_eids[etype] = (
                np.fromiter(sorted(test_set), dtype=np.int64)
                if test_set
                else _empty_int_array()
            )

        logging.info(
            "Disease-centric split assigned %d disease node(s) to train, %d to val, %d to test.",
            len(train_nodes),
            len(val_nodes),
            len(test_nodes),
        )

        return base_result


class IndependentCohortSplitter(RandomEdgeSplitter):
    """Hold out target-label edges for disjoint source-node cohorts.

    This is intended for independent disease-cohort validation in link
    prediction: validation and test contain target association edges for
    held-out cohort nodes, while non-target graph context stays in train.
    """

    def __init__(
        self,
        val_size: float,
        test_size: float,
        cohort_node_type: str = "disease",
        target_node_types: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        holdout_scope: str = "target_edges",
        ensure_minimum: bool = True,
        explicit_test_nodes: Optional[Iterable[int]] = None,
        explicit_val_nodes: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__(
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            ensure_minimum=ensure_minimum,
        )
        self.cohort_node_type = cohort_node_type
        self.target_node_types: Set[str] = {
            str(t).strip()
            for t in (target_node_types or [])
            if str(t).strip()
        }
        self.holdout_scope = (holdout_scope or "target_edges").strip().lower()
        self.explicit_test_nodes = (
            {int(node_id) for node_id in explicit_test_nodes}
            if explicit_test_nodes is not None
            else None
        )
        self.explicit_val_nodes = (
            {int(node_id) for node_id in explicit_val_nodes}
            if explicit_val_nodes is not None
            else None
        )

    def _is_target_etype(self, etype: CanonicalEType) -> bool:
        src_type, _, dst_type = etype
        if self.holdout_scope == "all_incident":
            return src_type == self.cohort_node_type or dst_type == self.cohort_node_type

        if src_type == self.cohort_node_type and dst_type in self.target_node_types:
            return True
        if dst_type == self.cohort_node_type and src_type in self.target_node_types:
            return True
        return False

    def _select_cohort_nodes(self, graph: dgl.DGLHeteroGraph) -> Tuple[Set[int], Set[int], Set[int]]:
        num_nodes = graph.num_nodes(self.cohort_node_type)
        all_nodes = np.arange(num_nodes, dtype=np.int64)

        if self.explicit_test_nodes is not None:
            test_nodes = {
                node_id for node_id in self.explicit_test_nodes if 0 <= node_id < num_nodes
            }
            val_nodes = {
                node_id
                for node_id in (self.explicit_val_nodes or set())
                if 0 <= node_id < num_nodes and node_id not in test_nodes
            }
        else:
            shuffled = all_nodes.copy()
            self.rng.shuffle(shuffled)

            num_val = int(num_nodes * self.val_size) if self.val_size > 0 else 0
            num_test = int(num_nodes * self.test_size) if self.test_size > 0 else 0
            if self.ensure_minimum:
                if self.val_size > 0 and num_val == 0 and num_nodes >= 2:
                    num_val = 1
                if self.test_size > 0 and num_test == 0 and (num_nodes - num_val) >= 2:
                    num_test = 1
                if num_val + num_test > num_nodes:
                    num_test = max(0, num_nodes - num_val)

            val_nodes = set(map(int, shuffled[:num_val]))
            test_nodes = set(map(int, shuffled[num_val : num_val + num_test]))

        train_nodes = set(map(int, all_nodes)) - val_nodes - test_nodes
        return train_nodes, val_nodes, test_nodes

    def _split_eids_randomly(self, all_eids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly split one edge type while preserving non-target edge handling."""
        num_edges = all_eids.size
        if num_edges == 0:
            empty = _empty_int_array()
            return empty, empty, empty

        shuffled = all_eids.copy()
        self.rng.shuffle(shuffled)

        num_val = int(num_edges * self.val_size) if self.val_size > 0 else 0
        num_test = int(num_edges * self.test_size) if self.test_size > 0 else 0

        if self.ensure_minimum and num_edges >= 3:
            if self.val_size > 0 and num_val == 0:
                num_val = 1
            if self.test_size > 0 and num_test == 0 and (num_edges - num_val) >= 2:
                num_test = 1
            if num_val + num_test > num_edges:
                num_test = max(0, num_edges - num_val)

        val_split = shuffled[:num_val].copy() if num_val > 0 else _empty_int_array()
        test_split = (
            shuffled[num_val : num_val + num_test].copy()
            if num_test > 0
            else _empty_int_array()
        )
        train_split = shuffled[num_val + num_test :].copy()
        return train_split, val_split, test_split

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        if self.cohort_node_type not in graph.ntypes:
            logging.warning(
                "Independent-cohort split requested but cohort node type '%s' was not found. Falling back to random split.",
                self.cohort_node_type,
            )
            return super().split(graph)

        if self.holdout_scope != "all_incident" and not self.target_node_types:
            logging.warning(
                "Independent-cohort split requested target-edge scope without target node types. Falling back to random split.",
            )
            return super().split(graph)

        _, val_nodes, test_nodes = self._select_cohort_nodes(graph)
        train: Dict[CanonicalEType, np.ndarray] = {}
        val: Dict[CanonicalEType, np.ndarray] = {}
        test: Dict[CanonicalEType, np.ndarray] = {}

        target_edge_types_found = 0
        heldout_val_edges = 0
        heldout_test_edges = 0

        for etype in graph.canonical_etypes:
            num_edges = graph.num_edges(etype)
            all_eids = np.arange(num_edges, dtype=np.int64)
            if num_edges == 0:
                empty = _empty_int_array()
                train[etype] = empty
                val[etype] = empty
                test[etype] = empty
                continue

            if not self._is_target_etype(etype):
                train[etype] = all_eids
                val[etype] = _empty_int_array()
                test[etype] = _empty_int_array()
                continue

            target_edge_types_found += 1
            src_type, _, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            src_np = src_nodes.numpy()
            dst_np = dst_nodes.numpy()

            if src_type == self.cohort_node_type:
                cohort_endpoint = src_np
            elif dst_type == self.cohort_node_type:
                cohort_endpoint = dst_np
            else:
                cohort_endpoint = np.array([], dtype=np.int64)

            val_mask = np.isin(cohort_endpoint, list(val_nodes))
            test_mask = np.isin(cohort_endpoint, list(test_nodes))
            val_mask &= ~test_mask
            train_mask = ~(val_mask | test_mask)

            train[etype] = all_eids[train_mask].copy()
            val[etype] = all_eids[val_mask].copy()
            test[etype] = all_eids[test_mask].copy()

            needs_val = self.val_size > 0 and val[etype].size == 0
            needs_test = self.test_size > 0 and test[etype].size == 0
            if (needs_val or needs_test) and num_edges >= 3:
                logging.warning(
                    "Independent-cohort split for %s produced val=%d, test=%d; falling back to random split for this sparse target edge type.",
                    etype,
                    val[etype].size,
                    test[etype].size,
                )
                train[etype], val[etype], test[etype] = self._split_eids_randomly(all_eids)

            heldout_val_edges += int(val[etype].size)
            heldout_test_edges += int(test[etype].size)

            logging.info(
                "Independent-cohort split for %s: train=%d, val=%d, test=%d.",
                etype,
                train[etype].size,
                val[etype].size,
                test[etype].size,
            )

        if target_edge_types_found == 0 or (heldout_val_edges + heldout_test_edges) == 0:
            logging.warning(
                "Independent-cohort split found no held-out target edges (target edge types=%d, val edges=%d, test edges=%d). Falling back to random split.",
                target_edge_types_found,
                heldout_val_edges,
                heldout_test_edges,
            )
            return super().split(graph)

        logging.info(
            "Independent-cohort split selected %d validation and %d test cohort node(s), holding out %d validation and %d test target edge(s).",
            len(val_nodes),
            len(test_nodes),
            heldout_val_edges,
            heldout_test_edges,
        )
        return SplitResult(train_eids=train, val_eids=val, test_eids=test)


class TypeExclusionSplitter(RandomEdgeSplitter):
    """Handle edges that involve specified node types by forcing or dropping them."""

    def __init__(
        self,
        val_size: float,
        test_size: float,
        exclude_node_types: Sequence[str],
        action: str = "train",
        random_state: Optional[int] = None,
        ensure_minimum: bool = True,
    ) -> None:
        super().__init__(
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            ensure_minimum=ensure_minimum,
        )
        valid_actions = {"train", "drop"}
        if action not in valid_actions:
            raise ValueError(f"Unsupported exclusion action '{action}'. Expected one of {valid_actions}.")
        self.action = action
        self.exclude_node_types = {t.strip() for t in exclude_node_types if t and t.strip()}

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        result = super().split(graph)

        if not self.exclude_node_types:
            return result

        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            if src_type not in self.exclude_node_types and dst_type not in self.exclude_node_types:
                continue

            all_eids = np.arange(graph.num_edges(etype), dtype=np.int64)
            logging.info(
                "Type exclusion strategy processed %d edge(s) of %s (action=%s).",
                all_eids.size,
                etype,
                self.action,
            )

            if self.action == "train":
                result.train_eids[etype] = all_eids if all_eids.size > 0 else _empty_int_array()
            else:  # drop
                result.train_eids[etype] = _empty_int_array()

            result.val_eids[etype] = _empty_int_array()
            result.test_eids[etype] = _empty_int_array()

        return result


class KFoldEdgeSplitter(BaseEdgeSplitter):
    """Generate k-fold splits and expose one fold as test (and optionally validation)."""

    def __init__(
        self,
        n_splits: int = 5,
        fold_index: int = 0,
        val_fold_index: Optional[int] = None,
        val_fraction: float = 0.0,
        random_state: Optional[int] = None,
        ensure_minimum: bool = True,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2 for k-fold splitting")
        if val_fraction < 0:
            raise ValueError("val_fraction must be non-negative")
        super().__init__(random_state=random_state)
        self.n_splits = n_splits
        self.fold_index = fold_index % n_splits
        self.val_fold_index = val_fold_index
        self.val_fraction = val_fraction
        self.ensure_minimum = ensure_minimum

    def split(self, graph: dgl.DGLHeteroGraph) -> SplitResult:
        folds: List[Dict[CanonicalEType, np.ndarray]] = [dict() for _ in range(self.n_splits)]
        train: Dict[CanonicalEType, np.ndarray] = {}
        val: Dict[CanonicalEType, np.ndarray] = {}
        test: Dict[CanonicalEType, np.ndarray] = {}

        default_val_idx = None
        if self.n_splits > 1:
            default_val_idx = (self.fold_index + 1) % self.n_splits

        for etype in graph.canonical_etypes:
            num_edges = graph.num_edges(etype)
            if num_edges == 0:
                empty = _empty_int_array()
                for fold_dict in folds:
                    fold_dict[etype] = empty
                train[etype] = empty
                val[etype] = empty
                test[etype] = empty
                continue

            eids = np.arange(num_edges, dtype=np.int64)
            self.rng.shuffle(eids)
            fold_arrays = np.array_split(eids, self.n_splits)
            for idx, fold_arr in enumerate(fold_arrays):
                folds[idx][etype] = fold_arr.copy() if fold_arr.size > 0 else _empty_int_array()

            test_fold_idx = self.fold_index % self.n_splits
            test_eids = (
                fold_arrays[test_fold_idx].copy()
                if fold_arrays[test_fold_idx].size > 0
                else _empty_int_array()
            )

            candidate_val_idx = (
                self.val_fold_index if self.val_fold_index is not None else default_val_idx
            )
            if candidate_val_idx is not None and candidate_val_idx == test_fold_idx:
                candidate_val_idx = None  # avoid collision; will fallback below

            remaining_indices = [i for i in range(self.n_splits) if i != test_fold_idx]
            val_eids = _empty_int_array()
            train_eids = _empty_int_array()

            if remaining_indices:
                if candidate_val_idx is not None and candidate_val_idx in remaining_indices:
                    val_idx = candidate_val_idx
                    val_eids = (
                        fold_arrays[val_idx].copy()
                        if fold_arrays[val_idx].size > 0
                        else _empty_int_array()
                    )
                    train_indices = [i for i in remaining_indices if i != val_idx]
                    if train_indices:
                        concatenated = np.concatenate([fold_arrays[i] for i in train_indices])
                        train_eids = (
                            concatenated if concatenated.size > 0 else _empty_int_array()
                        )
                    else:
                        train_eids = _empty_int_array()
                else:
                    combined = (
                        np.concatenate([fold_arrays[i] for i in remaining_indices])
                        if remaining_indices
                        else _empty_int_array()
                    )
                    if combined.size == 0:
                        train_eids = _empty_int_array()
                        val_eids = _empty_int_array()
                    else:
                        val_count = (
                            int(combined.size * self.val_fraction)
                            if self.val_fraction > 0
                            else 0
                        )
                        if (
                            self.ensure_minimum
                            and self.val_fraction > 0
                            and val_count == 0
                            and combined.size > 0
                        ):
                            val_count = 1
                        val_count = min(val_count, combined.size)
                        if val_count > 0:
                            self.rng.shuffle(combined)
                            val_eids = combined[:val_count].copy()
                            train_eids = combined[val_count:].copy()
                        else:
                            train_eids = combined
                            val_eids = _empty_int_array()
            # Ensure arrays are int64 even if empty
            train[etype] = train_eids if train_eids.size > 0 else _empty_int_array()
            val[etype] = val_eids if val_eids.size > 0 else _empty_int_array()
            test[etype] = test_eids if test_eids.size > 0 else _empty_int_array()

        return SplitResult(train_eids=train, val_eids=val, test_eids=test, folds=folds)


__all__ = [
    "SplitResult",
    "RandomEdgeSplitter",
    "NodeRandomSplitter",
    "NodeKFoldSplitter",
    "SpecificIDEdgeSplitter",
    "resolve_typed_node_ids",
    "CategoryEdgeSplitter",
    "DiseaseCentricSplitter",
    "IndependentCohortSplitter",
    "TypeExclusionSplitter",
    "KFoldEdgeSplitter",
]
