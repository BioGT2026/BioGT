#!/usr/bin/env python
"""Prepare BioGT triples for the vendored baseline implementations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sniff_dialect(path: Path, delimiter: str | None) -> csv.Dialect:
    if delimiter:
        class Dialect(csv.excel):
            pass

        Dialect.delimiter = delimiter
        return Dialect
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    return csv.Sniffer().sniff(sample, delimiters="\t,")


def typed_entity(node_type: str, identifier: str, enabled: bool) -> str:
    return f"{node_type}::{identifier}" if enabled else identifier


def read_triples(args: argparse.Namespace) -> list[tuple[str, str, str, str | None]]:
    dialect = sniff_dialect(args.edges, args.delimiter)
    include_node_types = {
        value.strip().lower() for value in (args.include_node_types or "").split(",") if value.strip()
    }
    include_relations = {
        value.strip().lower() for value in (args.include_relations or "").split(",") if value.strip()
    }
    with args.edges.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        missing = {
            args.source_col,
            args.relation_col,
            args.target_col,
        } - set(reader.fieldnames or [])
        if include_node_types:
            for type_col in [args.source_type_col, args.target_type_col]:
                if type_col not in (reader.fieldnames or []):
                    missing.add(type_col)
        if args.split_col:
            missing -= {None}
            if args.split_col not in (reader.fieldnames or []):
                missing.add(args.split_col)
        if args.direction_col and args.direction_col not in (reader.fieldnames or []):
            missing.add(args.direction_col)
        if missing:
            raise SystemExit(f"Missing required columns: {sorted(missing)}")

        rows: list[tuple[str, str, str, str | None]] = []
        seen = set()
        read_count = 0
        for line_no, row in enumerate(reader, 2):
            if args.max_rows and read_count >= args.max_rows:
                break
            read_count += 1
            source_type = (row.get(args.source_type_col) or "").strip().lower()
            target_type = (row.get(args.target_type_col) or "").strip().lower()
            if include_node_types:
                if source_type not in include_node_types and target_type not in include_node_types:
                    continue
            source_id = (row.get(args.source_col) or "").strip()
            rel = (row.get(args.relation_col) or "").strip()
            target_id = (row.get(args.target_col) or "").strip()
            if include_relations and rel.lower() not in include_relations:
                continue
            split = (row.get(args.split_col) or "").strip().lower() if args.split_col else None
            if not source_id or not rel or not target_id:
                if args.skip_bad_rows:
                    continue
                raise SystemExit(f"Blank triple field at input line {line_no}")

            if args.canonical_source_type and args.canonical_target_type:
                canonical_source = args.canonical_source_type.lower()
                canonical_target = args.canonical_target_type.lower()
                if source_type == canonical_source and target_type == canonical_target:
                    head = typed_entity(source_type, source_id, args.type_prefix_entities)
                    tail = typed_entity(target_type, target_id, args.type_prefix_entities)
                elif source_type == canonical_target and target_type == canonical_source:
                    head = typed_entity(target_type, target_id, args.type_prefix_entities)
                    tail = typed_entity(source_type, source_id, args.type_prefix_entities)
                else:
                    continue
                rel = args.canonical_relation or f"{canonical_source}_{canonical_target}"
            else:
                head = typed_entity(source_type, source_id, args.type_prefix_entities)
                tail = typed_entity(target_type, target_id, args.type_prefix_entities)

            key = (head, rel, tail, split)
            if key in seen:
                pass
            else:
                seen.add(key)
                rows.append(key)
            direction = (row.get(args.direction_col) or "").strip() if args.direction_col else ""
            if (
                args.make_bidirectional
                and not (args.canonical_source_type and args.canonical_target_type)
                and direction.lower() in args.bidirectional_values
            ):
                rev_key = (tail, f"{rel}{args.reverse_relation_suffix}", head, split)
                if rev_key not in seen:
                    seen.add(rev_key)
                    rows.append(rev_key)
    return rows


def assign_splits(
    rows: list[tuple[str, str, str, str | None]],
    args: argparse.Namespace,
) -> dict[str, list[tuple[str, str, str]]]:
    splits = {"train": [], "valid": [], "test": []}
    if args.split_col:
        mapping = {
            args.train_value.lower(): "train",
            args.valid_value.lower(): "valid",
            args.test_value.lower(): "test",
        }
        for head, rel, tail, split in rows:
            out_split = mapping.get((split or "").lower())
            if out_split:
                splits[out_split].append((head, rel, tail))
        return splits

    grouped_rows: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for head, rel, tail, _ in rows:
        if rel.endswith(args.reverse_relation_suffix):
            group_key = (tail, rel[: -len(args.reverse_relation_suffix)], head)
        else:
            group_key = (head, rel, tail)
        grouped_rows.setdefault(group_key, []).append((head, rel, tail))

    triple_groups = list(grouped_rows.values())
    rng = random.Random(args.seed)
    rng.shuffle(triple_groups)
    n_total = len(triple_groups)
    n_test = int(round(n_total * args.test_frac))
    n_valid = int(round(n_total * args.valid_frac))
    splits["test"] = [triple for group in triple_groups[:n_test] for triple in group]
    splits["valid"] = [triple for group in triple_groups[n_test : n_test + n_valid] for triple in group]
    splits["train"] = [triple for group in triple_groups[n_test + n_valid :] for triple in group]
    return splits


def write_kge_dataset(out_dir: Path, splits: dict[str, list[tuple[str, str, str]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    triples = [triple for split in splits.values() for triple in split]
    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    (out_dir / "entities.dict").write_text(
        "".join(f"{idx}\t{name}\n" for idx, name in enumerate(entities)),
        encoding="utf-8",
    )
    (out_dir / "relations.dict").write_text(
        "".join(f"{idx}\t{name}\n" for idx, name in enumerate(relations)),
        encoding="utf-8",
    )
    for split, rows in splits.items():
        (out_dir / f"{split}.txt").write_text(
            "".join(f"{h}\t{r}\t{t}\n" for h, r, t in rows),
            encoding="utf-8",
        )


def write_candidate_entities(
    out_dir: Path,
    splits: dict[str, list[tuple[str, str, str]]],
) -> None:
    candidates = sorted({tail for rows in splits.values() for _, _, tail in rows})
    (out_dir / "candidate_entities.txt").write_text(
        "".join(f"{entity}\n" for entity in candidates),
        encoding="utf-8",
    )


def copy_to_external(dataset: str, kge_dir: Path, compgcn_dir: Path) -> None:
    targets = [
        ROOT / "external" / "KnowledgeGraphEmbedding" / "data" / dataset,
        ROOT / "external" / "CompGCN" / "data" / dataset,
    ]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(kge_dir, targets[0])
    shutil.copytree(compgcn_dir, targets[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True, help="Input BioGT edge TSV/CSV.")
    parser.add_argument("--dataset", required=True, help="Output dataset name.")
    parser.add_argument("--source-col", required=True)
    parser.add_argument("--source-type-col", default="Node 1 Data Type")
    parser.add_argument("--relation-col", required=True)
    parser.add_argument("--include-relations", default=None, help="Comma-separated relation IDs to keep.")
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--target-type-col", default="Node 2 Data Type")
    parser.add_argument("--include-node-types", default=None, help="Comma-separated node types to keep when either endpoint has the type.")
    parser.add_argument("--split-col", default=None)
    parser.add_argument("--direction-col", default=None)
    parser.add_argument("--make-bidirectional", action="store_true")
    parser.add_argument("--bidirectional-values", default="doubledirected")
    parser.add_argument("--reverse-relation-suffix", default="_reverse")
    parser.add_argument("--canonical-source-type", default=None)
    parser.add_argument("--canonical-target-type", default=None)
    parser.add_argument("--canonical-relation", default=None)
    parser.add_argument("--type-prefix-entities", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--train-value", default="train")
    parser.add_argument("--valid-value", default="valid")
    parser.add_argument("--test-value", default="test")
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--out-root", type=Path, default=ROOT / "data")
    parser.add_argument("--install-external", action="store_true")
    parser.add_argument("--skip-bad-rows", action="store_true")
    args = parser.parse_args()
    args.bidirectional_values = {
        value.strip().lower() for value in args.bidirectional_values.split(",") if value.strip()
    }

    rows = read_triples(args)
    if not rows:
        raise SystemExit("No triples were read from the input file.")
    splits = assign_splits(rows, args)
    empty = [split for split, triples in splits.items() if not triples]
    if empty:
        raise SystemExit(f"Empty split(s): {empty}. Check split values or fractions.")

    dataset_root = args.out_root / args.dataset
    kge_dir = dataset_root / "kge"
    compgcn_dir = dataset_root / "compgcn"
    write_kge_dataset(kge_dir, splits)
    write_candidate_entities(kge_dir, splits)
    # CompGCN generates inverse relations internally. Feeding generated *_reverse
    # relations would collide with its own reverse-relation naming.
    compgcn_splits = {
        split: [
            (head, rel, tail)
            for head, rel, tail in triples
            if not rel.endswith(args.reverse_relation_suffix)
        ]
        for split, triples in splits.items()
    }
    write_kge_dataset(compgcn_dir, compgcn_splits)
    write_candidate_entities(compgcn_dir, compgcn_splits)

    summary = {
        "dataset": args.dataset,
        "input": str(args.edges),
        "split_counts": {split: len(triples) for split, triples in splits.items()},
        "split_unit": "forward_reverse_edge_group",
        "relation_counts": Counter(r for triples in splits.values() for _, r, _ in triples),
        "entity_count": len({h for triples in splits.values() for h, _, _ in triples} |
                            {t for triples in splits.values() for _, _, t in triples}),
        "candidate_entity_count": len(
            {tail for triples in splits.values() for _, _, tail in triples}
        ),
        "task_direction": (
            f"{args.canonical_source_type}->{args.canonical_target_type}"
            if args.canonical_source_type and args.canonical_target_type
            else "as_input"
        ),
        "type_prefixed_entities": bool(args.type_prefix_entities),
    }
    (dataset_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.install_external:
        copy_to_external(args.dataset, kge_dir, compgcn_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
