#!/usr/bin/env python3
"""Aggregate BioGT paper-run JSON artifacts without parsing log text."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RANKING_KS = (1, 5, 10, 20, 50, 100)


FIELDS = (
    "loss",
    "auc_prc",
    "auc_roc",
    "accuracy",
    "f1",
    "accuracy_at_0_5",
    "f1_at_0_5",
    "selected_threshold",
    "accuracy_at_selected_threshold",
    "f1_at_selected_threshold",
    "sampled_mrr",
    *(f"sampled_hits_at_{k}" for k in RANKING_KS),
    *(f"sampled_recall_at_{k}" for k in RANKING_KS),
    *(f"sampled_ndcg_at_{k}" for k in RANKING_KS),
    "full_mrr",
    *(f"full_hits_at_{k}" for k in RANKING_KS),
    *(f"full_recall_at_{k}" for k in RANKING_KS),
    *(f"full_ndcg_at_{k}" for k in RANKING_KS),
)


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _infer_phase_label(run_root: Path, task_dir: Path) -> tuple[str, str]:
    relative = task_dir.relative_to(run_root)
    parts = relative.parts
    if len(parts) >= 2:
        return parts[0], "/".join(parts[1:])
    if parts:
        return parts[0], parts[0]
    return "", ""


def collect_rows(run_root: Path) -> list[dict]:
    task_dirs = {
        path.parent for path in run_root.rglob("status.json") if path.parent != run_root
    }
    task_dirs.update(path.parent for path in run_root.rglob("metrics.json"))
    rows = []
    for task_dir in sorted(task_dirs):
        status = _read_json(task_dir / "status.json")
        payload = _read_json(task_dir / "metrics.json")
        phase, label = _infer_phase_label(run_root, task_dir)
        metrics = payload.get("metrics") or {}
        full = (metrics.get("full_ranking") or {}).get("metrics") or {}
        split_ref = payload.get("split_manifest") or {}
        row = {
            "phase": status.get("phase", phase),
            "label": status.get("label", label),
            "target": status.get("target")
            or ",".join(payload.get("target_types") or []),
            "status": status.get(
                "status",
                "completed" if payload else "unknown",
            ),
            "model_type": payload.get("model_type"),
            "split_strategy": payload.get("split_strategy"),
            "evaluation_split": payload.get("evaluation_split"),
            "seed": payload.get("seed"),
            "metrics_path": str(task_dir / "metrics.json"),
            "split_manifest_path": split_ref.get("path"),
            "split_manifest_sha256": split_ref.get("sha256"),
            "reused_from": None,
            "loss": metrics.get("loss"),
            "auc_prc": metrics.get("auc_prc"),
            "auc_roc": metrics.get("auc_roc"),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "accuracy_at_0_5": metrics.get("accuracy_at_0_5", metrics.get("accuracy")),
            "f1_at_0_5": metrics.get("f1_at_0_5", metrics.get("f1")),
            "selected_threshold": metrics.get("selected_threshold"),
            "accuracy_at_selected_threshold": metrics.get("accuracy_at_selected_threshold"),
            "f1_at_selected_threshold": metrics.get("f1_at_selected_threshold"),
            "sampled_mrr": metrics.get("mrr"),
            "full_mrr": full.get("mrr"),
        }
        for k_value in RANKING_KS:
            row[f"sampled_hits_at_{k_value}"] = metrics.get(f"hits_at_{k_value}")
            row[f"sampled_recall_at_{k_value}"] = metrics.get(f"recall_at_{k_value}")
            row[f"sampled_ndcg_at_{k_value}"] = metrics.get(f"ndcg_at_{k_value}")
            row[f"full_hits_at_{k_value}"] = full.get(f"hits_at_{k_value}")
            row[f"full_recall_at_{k_value}"] = full.get(f"recall_at_{k_value}")
            row[f"full_ndcg_at_{k_value}"] = full.get(f"ndcg_at_{k_value}")
        rows.append(row)
    manifest = _read_json(run_root / "run_manifest.json")
    rows_by_key = {
        f"{row['phase']}/{row['label']}": row for row in rows
    }
    for alias, source in sorted((manifest.get("aliases") or {}).items()):
        source_row = rows_by_key.get(source)
        if source_row is None:
            continue
        phase, label = alias.split("/", 1)
        alias_row = dict(source_row)
        alias_row.update(
            {
                "phase": phase,
                "label": label,
                "reused_from": source,
            }
        )
        rows.append(alias_row)
    rows.sort(key=lambda row: (row["phase"], row["label"]))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "phase",
        "label",
        "target",
        "status",
        "model_type",
        "split_strategy",
        "evaluation_split",
        "seed",
        *FIELDS,
        "metrics_path",
        "split_manifest_path",
        "split_manifest_sha256",
        "reused_from",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict], run_root: Path) -> None:
    complete = sum(row["status"] == "completed" for row in rows)
    failed = [row for row in rows if row["status"] == "failed"]
    lines = [
        "# BioGT Paper Experiment Results",
        "",
        f"- Run root: `{run_root}`",
        f"- Discovered tasks: {len(rows)}",
        f"- Completed: {complete}",
        f"- Failed: {len(failed)}",
        "",
        "| Phase | Label | Target | Status | AUC-PRC | AUC-ROC | Acc@0.5 | F1@0.5 | Selected threshold | Acc@selected | F1@selected | MRR | Hits@10 | Recall@50 | NDCG@50 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {phase} | {label} | {target} | {status} | {auc_prc} | {auc_roc} | "
            "{accuracy_at_0_5} | {f1_at_0_5} | {selected_threshold} | "
            "{accuracy_at_selected_threshold} | {f1_at_selected_threshold} | "
            "{full_mrr} | {full_hits_at_10} | {full_recall_at_50} | "
            "{full_ndcg_at_50} |".format(
                **{
                    key: _display(row.get(key))
                    for key in (
                        "phase",
                        "label",
                        "target",
                        "status",
                        "auc_prc",
                        "auc_roc",
                        "accuracy_at_0_5",
                        "f1_at_0_5",
                        "selected_threshold",
                        "accuracy_at_selected_threshold",
                        "f1_at_selected_threshold",
                        "full_mrr",
                        "full_hits_at_10",
                        "full_recall_at_50",
                        "full_ndcg_at_50",
                    )
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows = collect_rows(args.run_root)
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "task_count": len(rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows, args.run_root)
    print(
        json.dumps(
            {
                "task_count": len(rows),
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
