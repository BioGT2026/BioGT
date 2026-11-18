#!/usr/bin/env python
"""Audit train/validation/test triple splits for direct edge leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SPLITS = ("train", "valid", "test")


def read_triples(path: Path) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            head, relation, tail = line.rstrip("\n").split("\t")
            triples.add((head, relation, tail))
    return triples


def canonical_group(
    triple: tuple[str, str, str],
    reverse_suffix: str,
) -> tuple[str, str, str]:
    head, relation, tail = triple
    if relation.endswith(reverse_suffix):
        return tail, relation[: -len(reverse_suffix)], head
    return head, relation, tail


def undirected_relation_group(
    triple: tuple[str, str, str],
    reverse_suffix: str,
) -> tuple[str, str, str]:
    head, relation, tail = triple
    base_relation = (
        relation[: -len(reverse_suffix)]
        if relation.endswith(reverse_suffix)
        else relation
    )
    endpoint_a, endpoint_b = sorted((head, tail))
    return endpoint_a, base_relation, endpoint_b


def undirected_pair(triple: tuple[str, str, str]) -> tuple[str, str]:
    head, _, tail = triple
    return tuple(sorted((head, tail)))


def overlap_counts(
    left: set[tuple[str, str, str]],
    right: set[tuple[str, str, str]],
    reverse_suffix: str,
) -> dict[str, int]:
    left_canonical = {canonical_group(item, reverse_suffix) for item in left}
    right_canonical = {canonical_group(item, reverse_suffix) for item in right}
    left_undirected_relation = {
        undirected_relation_group(item, reverse_suffix) for item in left
    }
    right_undirected_relation = {
        undirected_relation_group(item, reverse_suffix) for item in right
    }
    left_pairs = {undirected_pair(item) for item in left}
    right_pairs = {undirected_pair(item) for item in right}
    return {
        "exact_triples": len(left & right),
        "canonical_forward_reverse_groups": len(left_canonical & right_canonical),
        "undirected_relation_groups": len(
            left_undirected_relation & right_undirected_relation
        ),
        "undirected_entity_pairs_any_relation": len(left_pairs & right_pairs),
    }


def referenced_entities(
    triples: dict[str, set[tuple[str, str, str]]],
    candidate_path: Path,
) -> set[str]:
    entities = {
        entity
        for split_triples in triples.values()
        for head, _, tail in split_triples
        for entity in (head, tail)
    }
    if candidate_path.exists():
        entities.update(
            line.strip()
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return entities


def audit_original_id_coverage(
    dataset_path: Path,
    entities: set[str],
    max_error_examples: int,
) -> dict[str, object]:
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        return {
            "metadata_path": str(metadata_path),
            "metadata_present": False,
            "referenced_entities": len(entities),
            "missing_or_blank_original_id_count": len(entities),
            "missing_or_blank_original_id_examples": sorted(entities)[:max_error_examples],
            "duplicate_original_id_count": 0,
            "duplicate_original_id_examples": {},
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mappings = metadata.get("dgl_to_original_id", {})
    missing = []
    original_to_entities: dict[str, list[str]] = {}
    for entity in sorted(entities):
        node_type, separator, dgl_id = entity.partition(":")
        original_id = mappings.get(node_type, {}).get(dgl_id)
        normalized = str(original_id).strip() if original_id is not None else ""
        if not separator or not normalized:
            missing.append(entity)
            continue
        original_to_entities.setdefault(f"{node_type}:{normalized}", []).append(entity)

    duplicates = {
        original_id: mapped_entities
        for original_id, mapped_entities in original_to_entities.items()
        if len(mapped_entities) > 1
    }
    return {
        "metadata_path": str(metadata_path),
        "metadata_present": True,
        "referenced_entities": len(entities),
        "missing_or_blank_original_id_count": len(missing),
        "missing_or_blank_original_id_examples": missing[:max_error_examples],
        "duplicate_original_id_count": len(duplicates),
        "duplicate_original_id_examples": dict(
            list(duplicates.items())[:max_error_examples]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--reverse-relation-suffix", default="_reverse")
    parser.add_argument("--fail-on-overlap", action="store_true")
    parser.add_argument("--require-original-id-coverage", action="store_true")
    parser.add_argument("--max-error-examples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    triples = {
        split: read_triples(args.dataset_path / f"{split}.txt")
        for split in SPLITS
    }
    comparisons = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        comparisons[f"{left}_vs_{right}"] = overlap_counts(
            triples[left],
            triples[right],
            args.reverse_relation_suffix,
        )

    provenance = audit_original_id_coverage(
        args.dataset_path,
        referenced_entities(
            triples,
            args.dataset_path / "candidate_entities.txt",
        ),
        max(0, args.max_error_examples),
    )
    payload = {
        "dataset_path": str(args.dataset_path),
        "split_counts": {split: len(values) for split, values in triples.items()},
        "comparisons": comparisons,
        "node_mapping_provenance": provenance,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    has_overlap = any(
        count
        for comparison in comparisons.values()
        for count in comparison.values()
    )
    has_mapping_errors = bool(
        provenance["missing_or_blank_original_id_count"]
        or provenance["duplicate_original_id_count"]
    )
    if (args.fail_on_overlap and has_overlap) or (
        args.require_original_id_coverage and has_mapping_errors
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
