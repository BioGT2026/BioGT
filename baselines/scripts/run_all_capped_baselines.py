#!/usr/bin/env python
"""Run each capped baseline once and report all four biomarker targets."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("protein", "gene", "rna", "metabolite")
KGE_MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")
MODELS = ("PPR", *KGE_MODELS, "R-GCN", "CompGCN", "HGT", "NBFNet")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n[{utc_now()}] CMD: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[{utc_now()}] EXIT: {proc.returncode}\n")
        return proc.returncode


def prepare_joint_dataset(args: argparse.Namespace) -> dict:
    dataset_root = ROOT / "data" / args.dataset
    summary_path = dataset_root / "summary.json"
    if args.force_prepare or not summary_path.exists():
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "combine_target_datasets.py"),
            "--source-prefix",
            args.source_prefix,
            "--dataset",
            args.dataset,
        ]
        rc = run(cmd, ROOT / "runs" / "logs" / args.dataset / "prepare.log")
        if rc != 0:
            raise SystemExit(f"Joint dataset preparation failed: {args.dataset}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def metrics_path(dataset: str, model: str) -> Path:
    if model in KGE_MODELS:
        return ROOT / "runs" / "kge" / dataset / model / "metrics.json"
    slug = {
        "PPR": "ppr",
        "R-GCN": "rgcn",
        "CompGCN": "compgcn",
        "HGT": "hgt",
        "NBFNet": "nbfnet",
    }[model]
    return ROOT / "runs" / slug / dataset / "metrics.json"


def model_command(
    args: argparse.Namespace,
    model: str,
    kge_max_steps: int,
    kge_valid_steps: int,
) -> list[str]:
    prefix = [sys.executable]
    scripts = ROOT / "scripts"
    if model == "PPR":
        return [*prefix, str(scripts / "run_ppr_baseline.py"), "--dataset", args.dataset]
    if model in KGE_MODELS:
        return [
            *prefix,
            str(scripts / "run_kge_baseline.py"),
            "--dataset",
            args.dataset,
            "--model",
            model,
            "--data-path",
            str(ROOT / "data" / args.dataset / "kge"),
            "--python",
            sys.executable,
            "--cuda",
            "--gpu",
            args.gpu,
            "--seed",
            str(args.seed),
            "--max-steps",
            str(kge_max_steps),
            "--batch-size",
            str(args.kge_batch_size),
            "--valid-steps",
            str(kge_valid_steps),
            "--log-steps",
            str(args.kge_log_steps),
        ]
    if model == "R-GCN":
        return [
            *prefix,
            str(scripts / "run_rgcn_baseline.py"),
            "--dataset",
            args.dataset,
            "--cuda",
            "--gpu",
            args.gpu,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.rgcn_epochs),
            "--validation-interval",
            str(args.rgcn_validation_interval),
        ]
    if model == "CompGCN":
        return [
            *prefix,
            str(scripts / "run_compgcn_baseline.py"),
            "--dataset",
            args.dataset,
            "--data-path",
            str(ROOT / "data" / args.dataset / "compgcn"),
            "--python",
            sys.executable,
            "--gpu",
            args.gpu,
            "--seed",
            str(args.seed),
            "--max-epochs",
            str(args.compgcn_epochs),
        ]
    if model == "HGT":
        return [
            *prefix,
            str(scripts / "run_hgt_baseline.py"),
            "--dataset",
            args.dataset,
            "--cuda",
            "--gpu",
            args.gpu,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.hgt_epochs),
            "--validation-interval",
            str(args.hgt_validation_interval),
        ]
    if model == "NBFNet":
        return [
            *prefix,
            str(scripts / "run_nbfnet_baseline.py"),
            "--dataset",
            args.dataset,
            "--cuda",
            "--gpu",
            args.gpu,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.nbfnet_epochs),
            "--batch-size",
            str(args.nbfnet_batch_size),
            "--max-steps-per-epoch",
            str(args.nbfnet_max_steps_per_epoch),
            "--validation-interval",
            str(args.nbfnet_validation_interval),
        ]
    raise ValueError(f"Unsupported baseline: {model}")


def validate_four_target_metrics(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_test_filter = "train_plus_valid_without_other_test_labels"
    actual_test_filter = payload.get("split_protocol", {}).get("test_filter")
    if actual_test_filter != expected_test_filter:
        raise RuntimeError(
            f"{path} has test_filter={actual_test_filter!r}; "
            f"expected {expected_test_filter!r}"
        )
    actual_targets = set(payload.get("metrics_by_target", {}))
    if actual_targets != set(TARGETS):
        raise RuntimeError(
            f"{path} does not contain exactly four targets: {sorted(actual_targets)}"
        )
    required = {"mrr", "hits@10", "recall@50", "ndcg@50"}
    for target, target_metrics in payload["metrics_by_target"].items():
        missing = required - {key.lower() for key in target_metrics}
        if missing:
            raise RuntimeError(f"{path}: {target} missing metrics {sorted(missing)}")
    return payload


def refresh_report(dataset: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_joint4_baselines.py"),
            "--dataset",
            dataset,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-prefix",
        default="biogt_dgl_transductive_seed42_v2",
        help="Prefix of the four exact target datasets",
    )
    parser.add_argument(
        "--dataset",
        default="biogt_dgl_transductive_seed42_v2_joint4_strict_v2",
        help="Joint four-target dataset name",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--kge-max-steps", type=int, default=100000)
    parser.add_argument("--kge-epochs", type=int, default=100)
    parser.add_argument("--kge-batch-size", type=int, default=1024)
    parser.add_argument("--kge-valid-steps", type=int, default=10000)
    parser.add_argument("--kge-log-steps", type=int, default=100)
    parser.add_argument("--rgcn-epochs", type=int, default=100)
    parser.add_argument("--rgcn-validation-interval", type=int, default=10)
    parser.add_argument("--compgcn-epochs", type=int, default=100)
    parser.add_argument("--hgt-epochs", type=int, default=50)
    parser.add_argument("--hgt-validation-interval", type=int, default=10)
    parser.add_argument("--nbfnet-epochs", type=int, default=20)
    parser.add_argument("--nbfnet-validation-interval", type=int, default=5)
    parser.add_argument("--nbfnet-batch-size", type=int, default=8)
    parser.add_argument(
        "--nbfnet-max-steps-per-epoch",
        type=int,
        default=112,
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated subset used for smoke or recovery runs",
    )
    args = parser.parse_args()

    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    invalid = [model for model in selected_models if model not in MODELS]
    if invalid:
        raise SystemExit(f"Unknown models: {invalid}. Valid: {list(MODELS)}")

    summary = prepare_joint_dataset(args)
    train_count = int(summary["split_counts"]["train"])
    epoch_steps = max(1, math.ceil(train_count / args.kge_batch_size)) * args.kge_epochs
    kge_max_steps = min(args.kge_max_steps, epoch_steps)
    kge_valid_steps = min(args.kge_valid_steps, max(100, kge_max_steps))

    status_path = ROOT / "runs" / f"status_{args.dataset}.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {"started_at": utc_now(), "models": {}}
    status.update(
        {
            "updated_at": utc_now(),
            "complete": False,
            "dataset": args.dataset,
            "summary": summary,
            "config": {
                "seed": args.seed,
                "targets": list(TARGETS),
                "joint_four_output": True,
                "single_split": True,
                "single_seed": True,
                "split_protocol": {
                    "fit": "train_only",
                    "checkpoint_selection": "valid_only_when_supported",
                    "test_queries": "test_only_after_configuration_freeze",
                    "test_filter": (
                        "train_plus_valid_without_other_test_labels"
                    ),
                    "transductive_vocabulary": True,
                },
                "embedding_dim_cap": 128,
                "negative_sample_cap": 5,
                "epoch_cap": 100,
                "kge_max_steps": kge_max_steps,
                "nbfnet_batch_size": args.nbfnet_batch_size,
                "nbfnet_max_steps_per_epoch": (
                    args.nbfnet_max_steps_per_epoch
                ),
                "validation_intervals": {
                    "R-GCN": args.rgcn_validation_interval,
                    "HGT": args.hgt_validation_interval,
                    "NBFNet": args.nbfnet_validation_interval,
                    "KGE_steps": kge_valid_steps,
                    "CompGCN_epochs": 1,
                },
            },
        }
    )
    write_json(status_path, status)

    for model in selected_models:
        path = metrics_path(args.dataset, model)
        prior = status["models"].get(model, {})
        if prior.get("status") == "complete" and path.exists():
            try:
                validate_four_target_metrics(path)
                continue
            except RuntimeError:
                pass

        started = time.time()
        rc = run(
            model_command(args, model, kge_max_steps, kge_valid_steps),
            ROOT / "runs" / "logs" / args.dataset / f"{model.lower()}.log",
        )
        model_status = {
            "status": "failed",
            "exit_code": rc,
            "seconds": round(time.time() - started, 2),
            "metrics_path": str(path),
            "updated_at": utc_now(),
        }
        if rc == 0:
            try:
                payload = validate_four_target_metrics(path)
                model_status["status"] = "complete"
                model_status["metrics_by_target"] = payload["metrics_by_target"]
            except RuntimeError as error:
                model_status["validation_error"] = str(error)
        status["models"][model] = model_status
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        refresh_report(args.dataset)
        if model_status["status"] != "complete" and not args.keep_going:
            raise SystemExit(f"{model} failed four-target validation")

    status["complete"] = all(
        status["models"].get(model, {}).get("status") == "complete"
        for model in MODELS
    )
    status["updated_at"] = utc_now()
    write_json(status_path, status)
    refresh_report(args.dataset)


if __name__ == "__main__":
    main()
