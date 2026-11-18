#!/usr/bin/env python3
"""Summarize BioGT running metrics beside capped baseline metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS = ("protein", "gene", "rna", "metabolite")
BASELINE_MODELS = ("PPR", "TransE", "DistMult", "ComplEx", "RotatE", "R-GCN", "CompGCN", "HGT", "NBFNet")
METRICS = ("mrr", "mr", "hits_at_10", "hits_at_50", "hits_at_100", "recall_at_50", "ndcg_at_50")
GATE_METRICS = ("mrr", "hits_at_10", "recall_at_50", "ndcg_at_50")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def metric_value(metrics: dict[str, Any], name: str) -> Any:
    aliases = {
        "mrr": ("mrr", "MRR"),
        "mr": ("mr", "MR"),
        "hits_at_10": ("hits_at_10", "Hits@10"),
        "hits_at_50": ("hits_at_50", "Hits@50"),
        "hits_at_100": ("hits_at_100", "Hits@100"),
        "recall_at_50": ("recall_at_50", "Recall@50"),
        "ndcg_at_50": ("ndcg_at_50", "NDCG@50"),
    }
    for key in aliases[name]:
        if key in metrics:
            return metrics[key]
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.5g}"
    return str(value)


def baseline_metrics_path(
    runs_dir: Path,
    target: str,
    model: str,
    dataset_prefix: str,
    joint: bool = False,
) -> Path | None:
    dataset = dataset_prefix if joint else f"{dataset_prefix}_{target}"
    if model == "PPR":
        return runs_dir / "ppr" / dataset / "metrics.json"
    if model in {"TransE", "DistMult", "ComplEx", "RotatE"}:
        return runs_dir / "kge" / dataset / model / "metrics.json"
    if model == "R-GCN":
        return runs_dir / "rgcn" / dataset / "metrics.json"
    if model == "CompGCN":
        return runs_dir / "compgcn" / dataset / "metrics.json"
    if model == "HGT":
        return runs_dir / "hgt" / dataset / "metrics.json"
    if model == "NBFNet":
        return runs_dir / "nbfnet" / dataset / "metrics.json"
    return None


def collect_baseline_rows(
    runs_dir: Path,
    dataset_prefix: str,
    joint_dataset: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for model in BASELINE_MODELS:
            path = baseline_metrics_path(
                runs_dir,
                target,
                model,
                joint_dataset or dataset_prefix,
                joint=bool(joint_dataset),
            )
            payload = read_json(path) if path else None
            if joint_dataset:
                metrics = (payload or {}).get("metrics_by_target", {}).get(
                    target,
                    {},
                )
            else:
                metrics = (payload or {}).get("metrics", {})
            row = {"target": target, "model": model, "protocol": "full filtered ranking", "path": str(path) if path else "NA"}
            row.update({metric: metric_value(metrics, metric) for metric in METRICS})
            row["candidate_count"] = (
                (payload or {})
                .get("candidate_entity_count_by_target", {})
                .get(target)
            )
            rows.append(row)
    return rows


def collect_biogt_rows(
    biogt_dir: Path,
    joint_metrics_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if joint_metrics_path is not None:
        payload = read_json(joint_metrics_path)
        full_ranking = (
            (payload or {})
            .get("test", {})
            .get("full_ranking", {})
        )
        per_target = full_ranking.get("per_target", {})
        candidate_counts = full_ranking.get("candidate_counts", {})
        for target in TARGETS:
            target_metrics = per_target.get(target)
            if not isinstance(target_metrics, dict):
                rows.append({
                    "target": target,
                    "model": "BioGT",
                    "protocol": "pending",
                    "path": str(joint_metrics_path),
                })
                continue
            row = {
                "target": target,
                "model": "BioGT",
                "protocol": "joint full filtered disease-target ranking",
                "path": str(joint_metrics_path),
            }
            row.update({
                metric: metric_value(target_metrics, metric)
                for metric in METRICS
            })
            row["ranking_num_triples"] = target_metrics.get(
                "ranking_num_triples"
            )
            row["candidate_count"] = candidate_counts.get(target)
            rows.append(row)
        return rows

    for target in TARGETS:
        path = biogt_dir / target / "metrics.json"
        payload = read_json(path)
        if not payload:
            rows.append({"target": target, "model": "BioGT", "protocol": "pending", "path": str(path)})
            continue
        metrics = payload.get("metrics", {})
        full_metrics = metrics.get("full_ranking", {}).get("metrics")
        if isinstance(full_metrics, dict):
            row = {"target": target, "model": "BioGT", "protocol": "full filtered disease-target ranking", "path": str(path)}
            row.update({metric: metric_value(full_metrics, metric) for metric in METRICS})
        else:
            row = {"target": target, "model": "BioGT", "protocol": "sampled ranking + binary metrics", "path": str(path)}
            row.update({metric: metric_value(metrics, metric) for metric in METRICS})
        for extra in ("auc_prc", "auc_roc", "f1", "ranking_num_triples"):
            row[extra] = metrics.get(extra)
        rows.append(row)
    return rows


def baseline_metric_gates(
    baseline_rows: list[dict[str, Any]],
    target: str,
) -> dict[str, dict[str, Any]]:
    gates = {}
    for metric in GATE_METRICS:
        candidates = [
            row
            for row in baseline_rows
            if row["target"] == target and isinstance(row.get(metric), (int, float))
        ]
        best = max(candidates, key=lambda row: row[metric], default=None)
        gates[metric] = {
            "value": best.get(metric) if best else None,
            "model": best.get("model") if best else None,
        }
    return gates


def render_markdown(baseline_rows: list[dict[str, Any]], biogt_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# BioGT Running Comparison",
        "",
        "BioGT rows use full filtered disease-target ranking when `full_ranking` is present in the metrics JSON; otherwise they are marked as sampled ranking and are not directly comparable with capped baselines.",
        "",
        "## Per-Metric Baseline Gates",
        "",
        "| Target | MRR gate | Hits@10 gate | Recall@50 gate | NDCG@50 gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for target in TARGETS:
        gates = baseline_metric_gates(baseline_rows, target)
        gate_cells = [
            (
                f"{fmt(gates[metric]['value'])} "
                f"({gates[metric]['model'] or 'NA'})"
            )
            for metric in GATE_METRICS
        ]
        lines.append(f"| {target} | " + " | ".join(gate_cells) + " |")
    lines.extend(
        [
            "",
            "## BioGT Strict Gate",
            "",
            "PASS requires BioGT to be strictly greater than the maximum value from all baselines for every ranking metric.",
            "",
            "| Target | Protocol | MRR | Hits@10 | Recall@50 | NDCG@50 | Strict gate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in biogt_rows:
        gates = baseline_metric_gates(baseline_rows, row["target"])
        checks = {
            metric: (
                isinstance(row.get(metric), (int, float))
                and isinstance(gates[metric]["value"], (int, float))
                and row[metric] > gates[metric]["value"]
            )
            for metric in GATE_METRICS
        }
        ceiling_metrics = {
            metric
            for metric, gate in gates.items()
            if isinstance(gate["value"], (int, float))
            and gate["value"] >= 1.0
        }
        non_ceiling_checks = [
            passed
            for metric, passed in checks.items()
            if metric not in ceiling_metrics
        ]
        ceiling_label = ", ".join(
            metric.replace("_at_", "@").replace("_", " ")
            for metric in GATE_METRICS
            if metric in ceiling_metrics
        )
        if ceiling_metrics and all(non_ceiling_checks):
            gate_status = f"INFEASIBLE ({ceiling_label} ceiling)"
        elif ceiling_metrics:
            gate_status = f"FAIL + {ceiling_label} ceiling"
        else:
            gate_status = (
                "PASS"
                if checks and all(checks.values())
                else "FAIL/PENDING"
            )
        lines.append(
            f"| {row['target']} | {row['protocol']} | {fmt(row.get('mrr'))} | "
            f"{fmt(row.get('hits_at_10'))} | {fmt(row.get('recall_at_50'))} | "
            f"{fmt(row.get('ndcg_at_50'))} | {gate_status} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-runs-dir", type=Path, default=Path("baselines/remote_runs/seetacloud_5090/fair_v1/runs"))
    parser.add_argument("--dataset-prefix", default="biogt_disease_to_target_fair_seed42_v1")
    parser.add_argument("--joint-dataset", default=None)
    parser.add_argument("--biogt-dir", type=Path, default=Path("Model/remote_runs/seetacloud_5090/experiments_biogt_eval_best_v4_epoch32_fair_protocol"))
    parser.add_argument("--biogt-metrics", type=Path, default=None, help="Joint four-target BioGT metrics JSON; takes precedence over per-target directories.")
    parser.add_argument("--out", type=Path, default=Path("Model/remote_runs/seetacloud_5090/biogt_vs_baselines_fair_v1_running.md"))
    args = parser.parse_args()

    baseline_rows = collect_baseline_rows(
        args.baseline_runs_dir,
        args.dataset_prefix,
        args.joint_dataset,
    )
    biogt_rows = collect_biogt_rows(
        args.biogt_dir,
        joint_metrics_path=args.biogt_metrics,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(baseline_rows, biogt_rows), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
