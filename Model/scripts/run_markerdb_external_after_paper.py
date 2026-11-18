#!/usr/bin/env python3
"""Run MarkerDB external validation only after the original paper suite ends."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASELINES = (
    "PPR", "TransE", "DistMult", "ComplEx", "RotatE",
    "R-GCN", "CompGCN", "HGT", "NBFNet",
)
TARGETS = ("protein", "gene", "rna", "metabolite")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _require_original_suite_complete(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Original paper suite status is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts") or {}
    if payload.get("status") != "completed" or int(counts.get("completed", 0)) != 242:
        raise RuntimeError(
            "MarkerDB external validation is deferred until all 242 original "
            f"paper tasks complete; observed status={payload.get('status')!r}, counts={counts}"
        )
    return payload


def _validate_biogt_command(
    command: list[str],
    external_edges: Path,
    export_prefix: str,
) -> None:
    required_tokens = {
        "--split-strategy",
        "external_edges",
        "--split-external-edges-path",
        str(external_edges),
        "--final-eval-split",
        "test",
        "--export-ranking-dataset-path",
        "--export-ranking-dataset-prefix",
        export_prefix,
    }
    missing = required_tokens - set(command)
    if missing:
        raise ValueError(f"BioGT external command misses required tokens: {sorted(missing)}")
    lowered = " ".join(command).lower()
    forbidden = ("load-kge", "structural-prior-mode ppr", "nbfnet", "distill")
    if any(token in lowered for token in forbidden):
        raise ValueError("Pure BioGT external command references a forbidden baseline artifact")


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_now()}] command={json.dumps(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{_now()}] exit_code={result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--paper-suite-status", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--biogt-command-json",
        type=Path,
        required=True,
        help="JSON string array for the frozen pure BioGT external training command.",
    )
    parser.add_argument("--dataset", default="markerdb_external_joint4_v1")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    external_edges = protocol_dir / "markerdb_external_edges.csv"
    audit_path = protocol_dir / "markerdb_external_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("training_graph_excludes_all_evaluable_markerdb_pairs"):
        raise RuntimeError("MarkerDB zero-overlap audit did not pass")
    _require_original_suite_complete(args.paper_suite_status.resolve())

    command = json.loads(args.biogt_command_json.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("--biogt-command-json must contain one JSON string array")
    source_prefix = f"{args.dataset}_target"
    _validate_biogt_command(command, external_edges, source_prefix)

    status_path = run_root / "status.json"
    status = {
        "schema_version": 1,
        "status": "running",
        "started_at": _now(),
        "protocol": "markerdb_external_edge_holdout",
        "original_paper_suite_completed_first": True,
        "joint_targets": list(TARGETS),
        "models": ["BioGT", *BASELINES],
        "external_audit": audit,
        "phases": {},
    }
    _write_json(status_path, status)
    try:
        _run(command, repo_root, run_root / "logs" / "biogt.log")
        status["phases"]["biogt"] = "completed"
        _write_json(status_path, status)

        _run(
            [
                sys.executable,
                str(repo_root / "baselines/scripts/combine_target_datasets.py"),
                "--source-prefix", source_prefix,
                "--dataset", args.dataset,
                "--data-root", str(repo_root / "baselines/data"),
                "--install-external",
            ],
            repo_root,
            run_root / "logs" / "combine_baseline_data.log",
        )
        status["phases"]["baseline_data"] = "completed"
        _write_json(status_path, status)

        _run(
            [
                sys.executable,
                str(repo_root / "baselines/scripts/run_all_capped_baselines.py"),
                "--source-prefix", source_prefix,
                "--dataset", args.dataset,
                "--gpu", args.gpu,
                "--models", ",".join(BASELINES),
                "--keep-going",
            ],
            repo_root,
            run_root / "logs" / "baselines.log",
        )
        status["phases"]["baselines"] = "completed"
        status["status"] = "completed"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        raise
    finally:
        status["finished_at"] = _now()
        _write_json(status_path, status)


if __name__ == "__main__":
    main()
