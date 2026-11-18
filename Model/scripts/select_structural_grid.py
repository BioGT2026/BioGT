#!/usr/bin/env python3
"""Select a balanced structural-prior configuration from validation metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = ("mrr", "hits_at_10", "recall_at_50", "ndcg_at_50")


def _metric_values(payload: dict) -> dict[str, float]:
    values = {}
    for metric in METRICS:
        value = payload.get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"missing or non-finite validation metric: {metric}")
        values[metric] = float(value)
    return values


def _label(alpha: float, neural_weight: float) -> str:
    return (
        f"alpha{alpha:g}_weight{neural_weight:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def select_configuration(
    metrics_payload: dict,
    baseline_alpha: float,
    require_neural_residual: bool,
) -> dict:
    full_ranking = metrics_payload.get("full_ranking") or {}
    grid = full_ranking.get("structural_prior_grid") or {}
    if not grid:
        raise ValueError("full_ranking.structural_prior_grid is empty")

    baseline_label = _label(baseline_alpha, 0.0)
    baseline_entry = grid.get(baseline_label)
    if baseline_entry is None:
        raise ValueError(
            f"validation grid does not contain the PPR reference {baseline_label}"
        )
    baseline = _metric_values(baseline_entry.get("metrics") or {})

    candidates = []
    for label, entry in sorted(grid.items()):
        alpha = float(entry["alpha"])
        neural_weight = float(entry["neural_weight"])
        if require_neural_residual and neural_weight <= 0.0:
            continue
        values = _metric_values(entry.get("metrics") or {})
        relative_gains = {
            metric: (values[metric] - baseline[metric])
            / max(abs(baseline[metric]), 1e-12)
            for metric in METRICS
        }
        raw_gains = {
            metric: values[metric] - baseline[metric] for metric in METRICS
        }
        dominates_reference = all(gain >= 0.0 for gain in raw_gains.values()) and any(
            gain > 0.0 for gain in raw_gains.values()
        )
        candidates.append(
            {
                "label": label,
                "alpha": alpha,
                "neural_weight": neural_weight,
                "metrics": values,
                "raw_gains_vs_ppr_reference": raw_gains,
                "relative_gains_vs_ppr_reference": relative_gains,
                "dominates_ppr_reference": dominates_reference,
                "selection_key": [
                    1 if dominates_reference else 0,
                    min(relative_gains.values()),
                    sum(relative_gains.values()) / len(METRICS),
                    values["mrr"],
                    -neural_weight,
                ],
            }
        )

    if not candidates:
        raise ValueError("no eligible structural-prior configurations")

    selected = max(candidates, key=lambda item: tuple(item["selection_key"]))
    return {
        "selection_split": "validation",
        "selection_policy": (
            "prefer configurations that Pareto-dominate alpha=0.85 pure PPR; "
            "then maximize the minimum relative gain across MRR, Hits@10, "
            "Recall@50, and NDCG@50"
        ),
        "ppr_reference": {
            "label": baseline_label,
            "alpha": baseline_alpha,
            "neural_weight": 0.0,
            "metrics": baseline,
        },
        "require_neural_residual": require_neural_residual,
        "selected": selected,
        "eligible_configurations": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-alpha", type=float, default=0.85)
    parser.add_argument("--allow-zero-neural-weight", action="store_true")
    args = parser.parse_args()

    with args.metrics.open("r", encoding="utf-8") as handle:
        metrics_payload = json.load(handle)
    selection = select_configuration(
        metrics_payload,
        baseline_alpha=args.baseline_alpha,
        require_neural_residual=not args.allow_zero_neural_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")
    print(json.dumps(selection["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
