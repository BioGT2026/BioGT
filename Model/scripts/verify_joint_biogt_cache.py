#!/usr/bin/env python3
"""Verify that a BioGT graph cache matches the joint baseline split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import torch


TARGETS = ("protein", "gene", "rna", "metabolite")


def source_target_pairs(graph, target):
    pairs = set()
    for etype in graph.canonical_etypes:
        if graph.num_edges(etype) == 0:
            continue
        src_type, _, dst_type = etype
        src, dst = graph.edges(etype=etype)
        if src_type == "disease" and dst_type == target:
            pairs.update(zip(src.tolist(), dst.tolist()))
        elif src_type == target and dst_type == "disease":
            pairs.update(
                (disease_id, target_id)
                for target_id, disease_id in zip(
                    src.tolist(),
                    dst.tolist(),
                )
            )
    return pairs


def _load_feature_cache(feature_cache: Path) -> dict:
    path = (
        feature_cache
        if feature_cache.suffix == ".pt"
        else feature_cache / "features.pt"
    )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Feature cache is not a mapping: {path}")
    return payload


def _verify_feature_cache(
    feature_cache: Path,
    graph,
    expected_feature_dim: int,
) -> dict:
    payload = _load_feature_cache(feature_cache)
    schema_version = payload.get("feature_schema_version")
    if schema_version != 2:
        raise ValueError(
            f"Feature cache schema={schema_version!r}, expected=2"
        )
    state_dict = payload.get("node_features_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Feature cache has no node_features_state_dict")

    embedding_shapes = {}
    errors = []
    for node_type in graph.ntypes:
        key = f"{node_type}.weight"
        weight = state_dict.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            errors.append(f"missing 2D tensor {key}")
            continue
        shape = list(weight.shape)
        embedding_shapes[node_type] = shape
        if shape[0] != graph.num_nodes(node_type):
            errors.append(
                f"{key} rows={shape[0]}, graph nodes="
                f"{graph.num_nodes(node_type)}"
            )
        if shape[1] != expected_feature_dim:
            errors.append(
                f"{key} dim={shape[1]}, expected={expected_feature_dim}"
            )
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": schema_version,
        "expected_feature_dim": expected_feature_dim,
        "embedding_shapes": embedding_shapes,
    }


def verify(
    graph_cache: Path,
    baseline_summary: Path,
    feature_cache: Path | None = None,
    expected_feature_dim: int = 64,
) -> dict:
    graphs, _ = dgl.load_graphs(str(graph_cache / "graphs.bin"))
    if len(graphs) != 4:
        raise ValueError(f"Expected four cached graphs, found {len(graphs)}")
    full_graph, train_graph, valid_graph, test_graph = graphs
    expected = json.loads(baseline_summary.read_text(encoding="utf-8"))
    split_graphs = {
        "train": train_graph,
        "valid": valid_graph,
        "test": test_graph,
    }
    split_pairs = {
        split: {
            target: source_target_pairs(graph, target)
            for target in TARGETS
        }
        for split, graph in split_graphs.items()
    }
    split_counts = {
        split: sum(len(pairs) for pairs in target_pairs.values())
        for split, target_pairs in split_pairs.items()
    }
    candidate_counts = {
        target: len({
            target_id
            for _, target_id in source_target_pairs(full_graph, target)
        })
        for target in TARGETS
    }
    overlaps = {}
    for left, right in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        overlaps[f"{left}_{right}"] = sum(
            len(split_pairs[left][target] & split_pairs[right][target])
            for target in TARGETS
        )

    errors = []
    if split_counts != expected["split_counts"]:
        errors.append(
            f"split_counts={split_counts}, expected={expected['split_counts']}"
        )
    if candidate_counts != expected["candidate_counts"]:
        errors.append(
            "candidate_counts="
            f"{candidate_counts}, expected={expected['candidate_counts']}"
        )
    if any(overlaps.values()):
        errors.append(f"split overlaps are nonzero: {overlaps}")
    if errors:
        raise ValueError("; ".join(errors))
    payload = {
        "status": "pass",
        "dataset": expected["dataset"],
        "split_counts": split_counts,
        "candidate_counts": candidate_counts,
        "split_overlaps": overlaps,
        "targets": list(TARGETS),
    }
    if feature_cache is not None:
        payload["feature_cache"] = _verify_feature_cache(
            feature_cache,
            full_graph,
            expected_feature_dim,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--expected-feature-dim", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = verify(
        args.graph_cache,
        args.baseline_summary,
        args.feature_cache,
        args.expected_feature_dim,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
