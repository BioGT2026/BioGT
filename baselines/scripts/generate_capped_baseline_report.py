#!/usr/bin/env python
"""Generate a Markdown report for BioGT capped baseline runs.

The script is intentionally read-only: it consumes baselines/runs/status.json,
per-model metrics.json files, and available logs, then writes a report under
baselines/reports.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("protein", "gene", "rna", "metabolite")
DEFAULT_MODELS = ("PPR", "TransE", "DistMult", "ComplEx", "RotatE", "R-GCN", "CompGCN", "HGT", "NBFNet")
METRIC_COLUMNS = (
    "MRR",
    "MR",
    "Hits@10",
    "Hits@50",
    "Hits@100",
    "Recall@10",
    "Recall@50",
    "Recall@100",
    "NDCG@10",
    "NDCG@50",
    "NDCG@100",
)
KGE_LOG_RE = re.compile(
    r"\bTest\s+(?P<metric>[A-Za-z]+@?\d*|MRR|MR)\s+at step\s+\d+:\s+(?P<value>[-+0-9.eE]+)"
)
COMPGCN_MRR_RE = re.compile(r"\[Epoch\s+\d+\s+test\]:\s+MRR:.*?Avg\s+:\s+(?P<value>[-+0-9.eE]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def normalize_metric_name(name: str) -> str:
    upper = name.upper()
    if upper in {"MRR", "MR"}:
        return upper
    if upper.startswith("HITS@"):
        return "Hits@" + upper.split("@", 1)[1]
    if upper.startswith("RECALL@"):
        return "Recall@" + upper.split("@", 1)[1]
    if upper.startswith("NDCG@"):
        return "NDCG@" + upper.split("@", 1)[1]
    return name


def flatten_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload:
        return {}
    source = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    metrics: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, (int, float)):
            metrics[normalize_metric_name(str(key))] = float(value)
    return metrics


def parse_kge_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    metrics: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = KGE_LOG_RE.search(line)
        if match:
            metrics[normalize_metric_name(match.group("metric"))] = float(match.group("value"))
    return metrics


def parse_compgcn_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    metrics: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = COMPGCN_MRR_RE.search(line)
        if match:
            metrics["MRR"] = float(match.group("value"))
    return metrics


def infer_dataset(status: dict[str, Any], target: str, prefix: str) -> str:
    target_status = status.get("targets", {}).get(target, {})
    return str(target_status.get("dataset") or f"{prefix}_{target}")


def infer_metrics_path(runs_dir: Path, dataset: str, model: str, model_status: dict[str, Any]) -> Path | None:
    explicit = model_status.get("metrics_path")
    if explicit:
        normalized = str(explicit).replace("\\", "/")
        marker = "/baselines/runs/"
        if marker in normalized:
            return runs_dir / normalized.split(marker, 1)[1]
        explicit_path = Path(explicit)
        try:
            if explicit_path.exists():
                return explicit_path
        except OSError:
            pass
    if model == "PPR":
        return runs_dir / "ppr" / dataset / "metrics.json"
    if model == "R-GCN":
        return runs_dir / "rgcn" / dataset / "metrics.json"
    if model == "CompGCN":
        return runs_dir / "compgcn" / dataset / "metrics.json"
    if model == "HGT":
        return runs_dir / "hgt" / dataset / "metrics.json"
    if model == "NBFNet":
        return runs_dir / "nbfnet" / dataset / "metrics.json"
    return None


def infer_log_path(runs_dir: Path, dataset: str, model: str) -> Path:
    names = {
        "PPR": "ppr.log",
        "R-GCN": "rgcn.log",
        "CompGCN": "compgcn.log",
        "HGT": "hgt.log",
        "NBFNet": "nbfnet.log",
    }
    if model in {"TransE", "DistMult", "ComplEx", "RotatE"}:
        name = f"kge_{model}.log"
    else:
        name = names.get(model, f"{model.lower()}.log")
    return runs_dir / "logs" / dataset / name


def last_log_excerpt(log_path: Path, max_lines: int = 8) -> str:
    if not log_path.exists():
        return "log not found"
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return "log is empty"
    traceback_idx = next((idx for idx, line in enumerate(lines) if "Traceback" in line), None)
    if traceback_idx is not None:
        return " | ".join(lines[traceback_idx : traceback_idx + max_lines])
    return " | ".join(lines[-max_lines:])


def format_number(value: float | None) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def format_seconds(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "NA"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


def table_row(values: list[str]) -> str:
    escaped = [value.replace("\n", " ").replace("|", "\\|") for value in values]
    return "| " + " | ".join(escaped) + " |"


def collect_rows(args: argparse.Namespace, status: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs_dir = Path(args.runs_dir)
    for target in args.targets:
        dataset = infer_dataset(status, target, args.dataset_prefix)
        model_statuses = status.get("targets", {}).get(target, {}).get("models", {})
        for model in args.models:
            model_status = model_statuses.get(model, {})
            metrics_path = infer_metrics_path(runs_dir, dataset, model, model_status)
            log_path = infer_log_path(runs_dir, dataset, model)

            metrics = flatten_metrics(load_json(metrics_path)) if metrics_path else {}
            if not metrics and model in {"TransE", "DistMult", "ComplEx", "RotatE"}:
                metrics = parse_kge_log(log_path)
            if not metrics and model == "CompGCN":
                metrics = parse_compgcn_log(log_path)

            status_text = str(model_status.get("status") or ("complete" if metrics else "pending"))
            failure = ""
            if status_text == "failed":
                failure = f"exit_code={model_status.get('exit_code', 'NA')}; {last_log_excerpt(log_path)}"
            elif not metrics and status_text == "complete":
                failure = "metrics not found or not exported"

            rows.append(
                {
                    "target": target,
                    "dataset": dataset,
                    "model": model,
                    "status": status_text,
                    "seconds": model_status.get("seconds"),
                    "metrics": metrics,
                    "metrics_path": str(metrics_path) if metrics_path else "NA",
                    "log_path": str(log_path),
                    "failure": failure,
                }
            )
    return rows


def render_report(args: argparse.Namespace, status: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# BioGT Capped Baseline Report",
        "",
        f"- Generated at: `{utc_now()}`",
        f"- Runs directory: `{Path(args.runs_dir)}`",
        f"- Status file: `{Path(args.status)}`",
        f"- Targets: `{', '.join(args.targets)}`",
        f"- Models: `{', '.join(args.models)}`",
        f"- Experiment scope: single split, single seed, capped baselines",
        "",
        "## Status Summary",
        "",
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    for key in sorted(status_counts):
        lines.append(f"- {key}: {status_counts[key]}")

    lines.extend(
        [
            "",
            "## Target x Model Metrics",
            "",
            table_row(["Target", "Model", "Status", "Runtime"] + list(METRIC_COLUMNS) + ["Metrics", "Log"]),
            table_row(["---"] * (4 + len(METRIC_COLUMNS) + 2)),
        ]
    )
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            table_row(
                [
                    row["target"],
                    row["model"],
                    row["status"],
                    format_seconds(row["seconds"]),
                ]
                + [format_number(metrics.get(metric)) for metric in METRIC_COLUMNS]
                + [row["metrics_path"], row["log_path"]]
            )
        )

    failed_or_missing = [row for row in rows if row["failure"]]
    lines.extend(["", "## Failures And Missing Metrics", ""])
    if failed_or_missing:
        lines.append(table_row(["Target", "Model", "Status", "Reason", "Log"]))
        lines.append(table_row(["---", "---", "---", "---", "---"]))
        for row in failed_or_missing:
            lines.append(table_row([row["target"], row["model"], row["status"], row["failure"], row["log_path"]]))
    else:
        lines.append("No failures or missing metric exports detected.")

    lines.extend(["", "## Dataset Summaries", ""])
    for target in args.targets:
        target_status = status.get("targets", {}).get(target, {})
        summary = target_status.get("summary")
        lines.append(f"### {target}")
        if isinstance(summary, dict):
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append("")
            lines.append("No dataset summary found in status.json.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_csv_arg(value: str, defaults: tuple[str, ...]) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or list(defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--status", type=Path, default=ROOT / "runs" / "status.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "capped_baseline_report.md")
    parser.add_argument("--dataset-prefix", default="biogt_full_seed20260728")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()
    args.targets = parse_csv_arg(args.targets, DEFAULT_TARGETS)
    args.models = parse_csv_arg(args.models, DEFAULT_MODELS)

    status = load_json(args.status) or {"targets": {}, "complete": False}
    rows = collect_rows(args, status)
    report = render_report(args, status, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
