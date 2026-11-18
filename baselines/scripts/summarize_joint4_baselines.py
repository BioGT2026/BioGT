#!/usr/bin/env python
"""Create incremental JSON, CSV, and Markdown reports for joint baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TARGETS = ("protein", "gene", "rna", "metabolite")
MODELS = ("PPR", "TransE", "DistMult", "ComplEx", "RotatE", "R-GCN", "CompGCN", "HGT", "NBFNet")
METRICS = ("MRR", "Hits@10", "Recall@50", "NDCG@50")


def metrics_path(runs_root: Path, dataset: str, model: str) -> Path:
    if model in {"TransE", "DistMult", "ComplEx", "RotatE"}:
        return runs_root / "kge" / dataset / model / "metrics.json"
    slug = {
        "PPR": "ppr",
        "R-GCN": "rgcn",
        "CompGCN": "compgcn",
        "HGT": "hgt",
        "NBFNet": "nbfnet",
    }[model]
    return runs_root / slug / dataset / "metrics.json"


def metric_value(metrics: dict, name: str):
    aliases = {
        "MRR": ("MRR", "mrr"),
        "Hits@10": ("Hits@10", "hits@10", "hits_at_10"),
        "Recall@50": ("Recall@50", "recall@50", "recall_at_50"),
        "NDCG@50": ("NDCG@50", "ndcg@50", "ndcg_at_50"),
    }
    return next(
        (metrics[key] for key in aliases[name] if key in metrics),
        None,
    )


def collect(runs_root: Path, dataset: str):
    rows = []
    completed_models = []
    for model in MODELS:
        path = metrics_path(runs_root, dataset, model)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if payload.get("split_protocol", {}).get("test_filter") != (
            "train_plus_valid_without_other_test_labels"
        ):
            continue
        target_payload = payload.get("metrics_by_target", {})
        if set(target_payload) != set(TARGETS):
            continue
        completed_models.append(model)
        for target in TARGETS:
            row = {
                "target": target,
                "model": model,
                "metrics_path": str(path),
            }
            row.update({
                metric: metric_value(target_payload[target], metric)
                for metric in METRICS
            })
            rows.append(row)
    return completed_models, rows


def render_markdown(dataset: str, completed_models: list[str], rows: list[dict]) -> str:
    lines = [
        "# Joint Four-Target Baseline Report",
        "",
        f"Dataset: `{dataset}`",
        "",
        f"Completed models: {len(completed_models)}/{len(MODELS)} "
        f"({', '.join(completed_models) or 'none'})",
        "",
        "| Target | Model | MRR | Hits@10 | Recall@50 | NDCG@50 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            "NA" if row[metric] is None else f"{row[metric]:.6f}"
            for metric in METRICS
        ]
        lines.append(
            f"| {row['target']} | {row['model']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend([
        "",
        "## Per-Target Gates",
        "",
        "| Target | MRR maximum | Hits@10 maximum | Recall@50 maximum | NDCG@50 maximum |",
        "| --- | --- | --- | --- | --- |",
    ])
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        cells = []
        for metric in METRICS:
            available = [
                row for row in target_rows if isinstance(row[metric], (int, float))
            ]
            best = max(available, key=lambda row: row[metric], default=None)
            cells.append(
                "NA" if best is None else f"{best[metric]:.6f} ({best['model']})"
            )
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("baselines/runs"))
    parser.add_argument(
        "--dataset",
        default="biogt_dgl_transductive_seed42_v2_joint4_strict_v2",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("baselines/reports/joint4_strict_v2"),
    )
    args = parser.parse_args()

    completed_models, rows = collect(args.runs_root, args.dataset)
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "expected_models": list(MODELS),
        "completed_models": completed_models,
        "rows": rows,
    }
    (args.out_root / "report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    with (args.out_root / "report.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("target", "model", *METRICS, "metrics_path"),
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.out_root / "report.md").write_text(
        render_markdown(args.dataset, completed_models, rows),
        encoding="utf-8",
    )
    print(json.dumps({"completed_models": completed_models}, indent=2))


if __name__ == "__main__":
    main()
