#!/usr/bin/env python3
"""Select a joint BioGT checkpoint using validation metrics only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TARGETS = ("protein", "gene", "rna", "metabolite")
METRICS = ("mrr", "hits_at_10", "recall_at_50", "ndcg_at_50")
PURE_BIOGT_SCORERS = {"concat_mlp", "interaction_mlp"}


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_candidates(run_root: Path) -> list[dict]:
    candidates = []
    for metrics_path in sorted(run_root.glob("*/metrics_validation.json")):
        run_dir = metrics_path.parent
        status = read_json(run_dir / "status.json") or {}
        metrics_payload = read_json(metrics_path) or {}
        full_ranking = metrics_payload.get(
            "validation",
            {},
        ).get(
            "full_ranking",
            metrics_payload.get("full_ranking", {}),
        )
        macro = full_ranking.get("metrics_macro", {})
        per_target = full_ranking.get("per_target", {})
        checkpoint = run_dir / "model_final_best.pt"
        populated_targets = all(
            isinstance(per_target.get(target), dict)
            and per_target[target].get("ranking_num_triples", 0) > 0
            and all(
                isinstance(per_target[target].get(metric), (int, float))
                for metric in METRICS
            )
            for target in TARGETS
        )
        macro_is_numeric = all(
            isinstance(macro.get(metric), (int, float))
            for metric in METRICS
        )
        expected_macro = (
            {
                metric: (
                    sum(
                        float(per_target[target][metric])
                        for target in TARGETS
                    )
                    / len(TARGETS)
                )
                for metric in METRICS
            }
            if populated_targets
            else {}
        )
        macro_matches_targets = (
            populated_targets
            and macro_is_numeric
            and all(
                abs(
                    float(macro.get(metric, float("nan")))
                    - expected_macro[metric]
                )
                <= 1e-10
                for metric in METRICS
            )
        )
        if (
            status.get("status") != "completed"
            or status.get("scorer_mode") not in PURE_BIOGT_SCORERS
            or run_dir.name.startswith("hybrid_")
            or not checkpoint.exists()
            or set(per_target) != set(TARGETS)
            or not macro_is_numeric
            or not populated_targets
            or not macro_matches_targets
        ):
            continue
        candidates.append({
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "status": status,
            "metrics_macro": {
                metric: float(macro[metric])
                for metric in METRICS
            },
            "selection_targets": list(TARGETS),
            "per_target": per_target,
        })
    return candidates


def selection_key(candidate: dict) -> tuple:
    macro = candidate["metrics_macro"]
    return (
        macro["mrr"],
        macro["ndcg_at_50"],
        macro["hits_at_10"],
        macro["recall_at_50"],
        candidate["run"],
    )


def render_markdown(candidates: list[dict], selected: dict | None) -> str:
    lines = [
        "# Joint BioGT Validation Selection",
        "",
        "Selection uses validation metrics only. Ordering is four-target "
        "macro MRR, NDCG@50, Hits@10, then "
        "Recall@50. RNA remains visible despite its small validation count.",
        "",
        "| Run | MRR | Hits@10 | Recall@50 | NDCG@50 | Selected |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in sorted(candidates, key=selection_key, reverse=True):
        macro = candidate["metrics_macro"]
        lines.append(
            f"| {candidate['run']} | {macro['mrr']:.6f} | "
            f"{macro['hits_at_10']:.6f} | {macro['recall_at_50']:.6f} | "
            f"{macro['ndcg_at_50']:.6f} | "
            f"{'yes' if selected and candidate['run'] == selected['run'] else 'no'} |"
        )
    if selected:
        lines.extend([
            "",
            f"Selected checkpoint: `{selected['checkpoint']}`",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("Model/experiments_biogt_joint_exact_v3"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    out_root = args.out_root or args.run_root
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args.run_root)
    selected = max(candidates, key=selection_key) if candidates else None
    payload = {
        "selection_split": "validation",
        "selection_order": [
            "metrics_macro.mrr",
            "metrics_macro.ndcg_at_50",
            "metrics_macro.hits_at_10",
            "metrics_macro.recall_at_50",
        ],
        "selected": selected,
        "candidates": sorted(candidates, key=selection_key, reverse=True),
    }
    (out_root / "validation_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_root / "validation_selection.md").write_text(
        render_markdown(candidates, selected),
        encoding="utf-8",
    )
    with (out_root / "validation_selection.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", *METRICS, "checkpoint", "selected"],
        )
        writer.writeheader()
        for candidate in sorted(candidates, key=selection_key, reverse=True):
            writer.writerow({
                "run": candidate["run"],
                **candidate["metrics_macro"],
                "checkpoint": candidate["checkpoint"],
                "selected": bool(
                    selected and candidate["run"] == selected["run"]
                ),
            })
    print(
        selected["checkpoint"]
        if selected
        else "no_complete_validation_candidates"
    )


if __name__ == "__main__":
    main()
