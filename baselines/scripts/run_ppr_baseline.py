#!/usr/bin/env python
"""Personalized PageRank / random-walk-with-restart baseline."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from baseline_io import (
    average_tie_rank,
    evaluation_known_tail_filter,
    load_dataset,
    load_split,
    load_target_specs,
    macro_average_metrics,
    metrics_payload_by_target,
    target_lookups,
    write_json,
)

try:
    import numpy as np
    from scipy.sparse import csr_matrix
except ImportError:  # pragma: no cover - fallback for minimal environments.
    np = None
    csr_matrix = None


ROOT = Path(__file__).resolve().parents[1]


def ppr_for_source(adj: dict[int, list[int]], source: int, alpha: float, max_iter: int, tol: float):
    rank = {source: 1.0}
    teleport = 1.0 - alpha
    for _ in range(max_iter):
        new_rank = defaultdict(float)
        new_rank[source] += teleport
        for node, score in rank.items():
            neigh = adj.get(node)
            if not neigh:
                new_rank[source] += alpha * score
                continue
            share = alpha * score / len(neigh)
            for dst in neigh:
                new_rank[dst] += share
        delta = sum(abs(new_rank.get(k, 0.0) - rank.get(k, 0.0)) for k in set(new_rank) | set(rank))
        rank = dict(new_rank)
        if delta < tol:
            break
    return rank


def batched_ppr_ranks(
    train_triples: list[tuple[int, int, int]],
    test_triples: list[tuple[int, int, int]],
    known: dict[tuple[int, int], set[int]],
    num_entities: int,
    candidates_by_relation: dict[int, list[int]],
    relation_to_target: dict[int, str],
    alpha: float,
    max_iter: int,
    tol: float,
    directed: bool,
    batch_size: int,
) -> dict[str, list[float]]:
    if np is None or csr_matrix is None:
        raise ImportError("numpy/scipy are required for batched PPR")

    rows = []
    cols = []
    for head, _, tail in train_triples:
        rows.append(head)
        cols.append(tail)
        if not directed:
            rows.append(tail)
            cols.append(head)

    rows_np = np.asarray(rows, dtype=np.int64)
    cols_np = np.asarray(cols, dtype=np.int64)
    degrees = np.bincount(rows_np, minlength=num_entities).astype(np.float32)
    data = (1.0 / np.maximum(degrees[rows_np], 1.0)).astype(np.float32)
    transition = csr_matrix((data, (rows_np, cols_np)), shape=(num_entities, num_entities), dtype=np.float32)

    by_head = defaultdict(list)
    for query_idx, (head, rel, tail) in enumerate(test_triples):
        by_head[head].append((query_idx, rel, tail))

    ranks_out: dict[str, list[float]] = defaultdict(list)
    candidate_arrays = {
        relation: np.asarray(candidate_ids, dtype=np.int64)
        for relation, candidate_ids in candidates_by_relation.items()
    }
    heads = list(by_head)
    teleport = np.float32(1.0 - alpha)
    alpha32 = np.float32(alpha)
    batch_size = max(1, batch_size)

    for batch_start in range(0, len(heads), batch_size):
        batch_heads = heads[batch_start : batch_start + batch_size]
        batch_len = len(batch_heads)
        head_positions = {head: pos for pos, head in enumerate(batch_heads)}
        ranks = np.zeros((batch_len, num_entities), dtype=np.float32)
        batch_indices = np.arange(batch_len)
        ranks[batch_indices, np.asarray(batch_heads, dtype=np.int64)] = 1.0

        for _ in range(max_iter):
            new_ranks = transition.T.dot(ranks.T).T.astype(np.float32, copy=False)
            new_ranks *= alpha32
            new_ranks[batch_indices, np.asarray(batch_heads, dtype=np.int64)] += teleport
            delta = np.abs(new_ranks - ranks).sum(axis=1).max()
            ranks = new_ranks
            if float(delta) < tol:
                break

        for head in batch_heads:
            row_idx = head_positions[head]
            scores = ranks[row_idx]
            for query_idx, rel, tail in by_head[head]:
                target_score = scores[tail]
                candidate_array = candidate_arrays[rel]
                filtered = known[(head, rel)] - {tail}
                if filtered:
                    eligible = candidate_array[
                        ~np.isin(candidate_array, np.fromiter(filtered, dtype=np.int64))
                    ]
                else:
                    eligible = candidate_array
                candidate_scores = scores[eligible]
                ranks_out[relation_to_target[rel]].append(average_tie_rank(
                    int(np.count_nonzero(candidate_scores > target_score)),
                    int(np.count_nonzero(np.isclose(candidate_scores, target_score))),
                ))
        print(f"processed_ppr_heads={min(batch_start + batch_len, len(heads))}/{len(heads)}", flush=True)

    return ranks_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--alpha-grid", default="0.75,0.85,0.95")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data_path = args.data_path or ROOT / "data" / args.dataset / "kge"
    ent2id, _, rel2id, _, splits = load_dataset(
        data_path,
        ("train", "valid"),
    )
    target_specs = load_target_specs(data_path, ent2id, rel2id)
    relation_to_target, candidates_by_relation = target_lookups(target_specs)
    adj = defaultdict(list)
    for head, _, tail in splits["train"]:
        adj[head].append(tail)
        if not args.directed:
            adj[tail].append(head)

    known = evaluation_known_tail_filter(splits)

    def evaluate_rows(rows, alpha):
        try:
            return batched_ppr_ranks(
                splits["train"],
                rows,
                known,
                len(ent2id),
                candidates_by_relation,
                relation_to_target,
                alpha,
                args.max_iter,
                args.tol,
                args.directed,
                args.batch_size,
            )
        except ImportError:
            ranks_local = defaultdict(list)
            cache = {}
            for head, rel, tail in rows:
                scores = cache.get(head)
                if scores is None:
                    scores = ppr_for_source(
                        adj,
                        head,
                        alpha,
                        args.max_iter,
                        args.tol,
                    )
                    cache[head] = scores
                candidate_entities = set(candidates_by_relation[rel])
                filtered = candidate_entities - (known[(head, rel)] - {tail})
                target_score = scores.get(tail, 0.0)
                filtered_scores = [
                    scores.get(entity, 0.0)
                    for entity in filtered
                ]
                rank = average_tie_rank(
                    sum(score > target_score for score in filtered_scores),
                    sum(
                        math.isclose(score, target_score)
                        for score in filtered_scores
                    ),
                )
                ranks_local[relation_to_target[rel]].append(rank)
            return ranks_local

    alpha_values = [
        float(item)
        for item in args.alpha_grid.split(",")
        if item.strip()
    ] or [args.alpha]
    validation_rows = splits["valid"][: args.max_eval_triples or None]
    validation_grid = {}
    best_alpha = None
    best_validation_mrr = -math.inf
    for alpha in alpha_values:
        validation_ranks = evaluate_rows(validation_rows, alpha)
        _, validation_by_target = metrics_payload_by_target(validation_ranks)
        validation_macro = macro_average_metrics(validation_by_target)
        validation_grid[str(alpha)] = validation_macro
        validation_mrr = float(validation_macro.get("MRR", -math.inf))
        if validation_mrr > best_validation_mrr:
            best_alpha = alpha
            best_validation_mrr = validation_mrr

    if best_alpha is None:
        raise RuntimeError("PPR alpha selection produced no validation result")
    splits["test"] = load_split(data_path, "test", ent2id, rel2id)
    test = splits["test"][: args.max_eval_triples or None]
    ranks = evaluate_rows(test, best_alpha)

    metrics, metrics_by_target = metrics_payload_by_target(ranks)
    payload = {
        "baseline": "PPR/RWR",
        "dataset": args.dataset,
        "data_path": str(data_path),
        "alpha": best_alpha,
        "validation_alpha_grid": validation_grid,
        "directed": args.directed,
        "batch_size": args.batch_size,
        "candidate_entity_count_by_target": {
            target: len(spec["candidate_ids"])
            for target, spec in target_specs.items()
        },
        "tie_policy": "average",
        "split_protocol": {
            "graph": "train_only",
            "parameter_selection": "valid_only",
            "test_queries": "test_only",
            "test_filter": "train_plus_valid_without_other_test_labels",
        },
        "metrics": metrics,
        "metrics_macro": macro_average_metrics(metrics_by_target),
        "metrics_by_target": metrics_by_target,
    }
    out = args.out or ROOT / "runs" / "ppr" / args.dataset / "metrics.json"
    write_json(out, payload)
    print(payload)


if __name__ == "__main__":
    main()
