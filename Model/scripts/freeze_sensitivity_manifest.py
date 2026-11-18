#!/usr/bin/env python3
"""Freeze the historical embed=64 BioGT sensitivity configurations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PARAMETERS = (
    "target",
    "lr",
    "embed",
    "transformer_heads",
    "transformer_layers",
    "hgcn_layers",
    "dropout",
    "neg",
)


def _normalized_row(row: dict) -> dict:
    return {
        "target": row["target"].strip().lower(),
        "lr": float(row["lr"]),
        "embed": int(row["embed"]),
        "transformer_heads": int(row["transformer_heads"]),
        "transformer_layers": int(row["transformer_layers"]),
        "hgcn_layers": int(row["hgcn_layers"]),
        "dropout": float(row["dropout"]),
        "neg": int(row["neg"]),
    }


def _config_id(config: dict) -> str:
    return (
        f"target-{config['target']}_lr-{config['lr']:g}_embed-{config['embed']}"
        f"_heads-{config['transformer_heads']}"
        f"_transformer-{config['transformer_layers']}"
        f"_hgcn-{config['hgcn_layers']}"
        f"_dropout-{config['dropout']:g}_neg-{config['neg']}"
    ).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--expected-count", type=int, default=212)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    with args.source.open("r", encoding="utf-8", newline="") as handle:
        rows = [_normalized_row(row) for row in csv.DictReader(handle)]
    selected = [row for row in rows if row["embed"] == args.embed]
    unique = {
        tuple(row[name] for name in PARAMETERS): row for row in selected
    }
    if len(unique) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} unique embed={args.embed} "
            f"configurations, found {len(unique)}"
        )

    configs = []
    for config in sorted(unique.values(), key=lambda row: tuple(row[p] for p in PARAMETERS)):
        configs.append({"config_id": _config_id(config), **config})
    payload = {
        "schema_version": 1,
        "source_path": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection": {"embed": args.embed},
        "configuration_count": len(configs),
        "configurations": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(configs)}))


if __name__ == "__main__":
    main()
