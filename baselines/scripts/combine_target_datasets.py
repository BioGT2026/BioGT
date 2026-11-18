#!/usr/bin/env python
"""Combine four exact BioGT target datasets into one joint multi-relation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("protein", "gene", "rna", "metabolite")


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            triples.append(tuple(line.split("\t")))
    return triples


def write_mapping(path: Path, values: set[str]) -> None:
    path.write_text(
        "".join(f"{index}\t{value}\n" for index, value in enumerate(sorted(values))),
        encoding="utf-8",
    )


def write_dataset(
    out_dir: Path,
    splits: dict[str, list[tuple[str, str, str]]],
    candidates_by_target: dict[str, list[str]],
    manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entities = {
        entity
        for triples in splits.values()
        for head, _, tail in triples
        for entity in (head, tail)
    }
    relations = {
        relation
        for triples in splits.values()
        for _, relation, _ in triples
    }
    write_mapping(out_dir / "entities.dict", entities)
    write_mapping(out_dir / "relations.dict", relations)
    for split, triples in splits.items():
        out_dir.joinpath(f"{split}.txt").write_text(
            "".join(
                f"{head}\t{relation}\t{tail}\n"
                for head, relation, tail in triples
            ),
            encoding="utf-8",
        )

    union_candidates = sorted({
        entity
        for candidates in candidates_by_target.values()
        for entity in candidates
    })
    out_dir.joinpath("candidate_entities.txt").write_text(
        "".join(f"{entity}\n" for entity in union_candidates),
        encoding="utf-8",
    )
    for target, candidates in candidates_by_target.items():
        out_dir.joinpath(f"candidate_entities.{target}.txt").write_text(
            "".join(f"{entity}\n" for entity in candidates),
            encoding="utf-8",
        )
    out_dir.joinpath("target_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--install-external", action="store_true")
    args = parser.parse_args()

    targets = [item.strip().lower() for item in args.targets.split(",") if item.strip()]
    if set(targets) != set(DEFAULT_TARGETS):
        raise SystemExit(
            "Joint baseline dataset must contain exactly protein,gene,rna,metabolite"
        )

    splits = {"train": [], "valid": [], "test": []}
    candidates_by_target: dict[str, list[str]] = {}
    target_manifest = {
        "schema_version": 1,
        "joint_targets": True,
        "dataset": args.dataset,
        "targets": {},
    }

    for target in targets:
        source_dataset = f"{args.source_prefix}_{target}"
        source_dir = args.data_root / source_dataset / "kge"
        metadata_path = source_dir / "metadata.json"
        required = [
            metadata_path,
            source_dir / "candidate_entities.txt",
            *(source_dir / f"{split}.txt" for split in splits),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit(f"Missing source files for {target}: {missing}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        relation = metadata["relation"]
        candidates = sorted({
            item.strip()
            for item in (source_dir / "candidate_entities.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if item.strip()
        })
        candidates_by_target[target] = candidates
        for split in splits:
            splits[split].extend(read_triples(source_dir / f"{split}.txt"))
        target_manifest["targets"][target] = {
            "relation": relation,
            "candidate_file": f"candidate_entities.{target}.txt",
            "candidate_count": len(candidates),
            "source_dataset": source_dataset,
            "split_counts": metadata.get("split_counts", {}),
        }

    for split in splits:
        splits[split] = sorted(set(splits[split]))

    dataset_root = args.data_root / args.dataset
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    write_dataset(
        dataset_root / "kge",
        splits,
        candidates_by_target,
        target_manifest,
    )
    write_dataset(
        dataset_root / "compgcn",
        splits,
        candidates_by_target,
        target_manifest,
    )

    summary = {
        "dataset": args.dataset,
        "joint_targets": True,
        "targets": targets,
        "source_prefix": args.source_prefix,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "candidate_counts": {
            target: len(candidates)
            for target, candidates in candidates_by_target.items()
        },
        "relations": {
            target: target_manifest["targets"][target]["relation"]
            for target in targets
        },
    }
    dataset_root.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if args.install_external:
        destinations = [
            ROOT / "external" / "KnowledgeGraphEmbedding" / "data" / args.dataset,
            ROOT / "external" / "CompGCN" / "data" / args.dataset,
        ]
        sources = [dataset_root / "kge", dataset_root / "compgcn"]
        for source, destination in zip(sources, destinations):
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink():
                    destination.unlink()
                else:
                    shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
