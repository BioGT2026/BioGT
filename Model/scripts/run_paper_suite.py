#!/usr/bin/env python3
"""Run the frozen BioGT paper experiment matrix with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TARGETS = ("protein", "gene", "rna", "metabolite")
JOINT_TARGETS = ",".join(TARGETS)
RANKING_KS = "1,5,10,20,50,100"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


@dataclass(frozen=True)
class Task:
    phase: str
    label: str
    target: str
    cache_key: str
    extra_args: tuple[str, ...]
    final_eval_split: str = "validation"

    @property
    def key(self) -> str:
        return f"{self.phase}/{self.label}"


def _node_random_args(model_type: str = "original") -> tuple[str, ...]:
    return (
        "--model-type",
        model_type,
        "--split-strategy",
        "node_random",
    )


def build_tasks(sensitivity_manifest: dict, category_splits: Path) -> list[Task]:
    tasks = [
        Task(
            "main",
            "model-pure-biogt_joint4",
            JOINT_TARGETS,
            "node_random",
            _node_random_args("original") + ("--fusion-mode", "weighted"),
        )
    ]

    for fold in range(10):
        tasks.append(
            Task(
                "kfold",
                f"joint4_fold-{fold}",
                JOINT_TARGETS,
                f"node_kfold_{fold}",
                (
                    "--model-type",
                    "original",
                    "--split-strategy",
                    "node_kfold",
                    "--split-kfolds",
                    "10",
                    "--split-kfold-index",
                    str(fold),
                    "--split-kfold-val-fold",
                    str((fold + 1) % 10),
                    "--fusion-mode",
                    "weighted",
                ),
                final_eval_split="test",
            )
        )

    for fusion in ("sequential", "hierarchical", "concat", "routing"):
        tasks.append(
            Task(
                "fusion",
                f"joint4_fusion-{fusion}",
                JOINT_TARGETS,
                "node_random",
                _node_random_args() + ("--fusion-mode", fusion),
            )
        )

    ablations = (
        ("no-hgcn", "--disable-hgcn"),
        ("no-transformer", "--disable-transformer"),
        ("no-moe", "--disable-moe"),
    )
    for label, flag in ablations:
        tasks.append(
            Task(
                "ablation",
                f"joint4_abl-{label}",
                JOINT_TARGETS,
                "node_random",
                _node_random_args()
                + ("--fusion-mode", "weighted", flag),
            )
        )

    tasks.append(
        Task(
            "zeroshot",
            "joint4_direct-disease-biomarker-holdout",
            JOINT_TARGETS,
            "target_exclude_joint4",
            (
                "--model-type",
                "original",
                "--split-strategy",
                "target_exclude",
                "--eval_input_type",
                "disease",
                "--eval_output_types",
                JOINT_TARGETS,
                "--fusion-mode",
                "weighted",
            ),
            final_eval_split="test",
        )
    )

    split_files = sorted(category_splits.glob("*.txt"))
    if len(split_files) != 11:
        raise ValueError(
            f"expected 11 category split files in {category_splits}, "
            f"found {len(split_files)}"
        )
    for split_file in split_files:
        category = split_file.stem
        tasks.append(
            Task(
                "category",
                f"joint4_cat-{category}",
                JOINT_TARGETS,
                f"category_{category}",
                (
                    "--model-type",
                    "original",
                    "--split-strategy",
                    "specific_ids",
                    "--split-specific-ids-path",
                    str(split_file),
                    "--fusion-mode",
                    "weighted",
                ),
                final_eval_split="test",
            )
        )

    configurations = sensitivity_manifest.get("configurations") or []
    if len(configurations) != 212:
        raise ValueError(
            f"expected 212 sensitivity configurations, found {len(configurations)}"
        )
    for config in configurations:
        tasks.append(
            Task(
                "sensitivity",
                config["config_id"],
                JOINT_TARGETS,
                "node_random",
                (
                    "--model-type",
                    "original",
                    "--split-strategy",
                    "node_random",
                    "--fusion-mode",
                    "weighted",
                    "--lr",
                    str(config["lr"]),
                    "--default_embed_size",
                    str(config["embed"]),
                    "--hgcn_hidden",
                    str(config["embed"]),
                    "--hgcn_out",
                    str(config["embed"]),
                    "--transformer_heads",
                    str(config["transformer_heads"]),
                    "--num_transformer_layers",
                    str(config["transformer_layers"]),
                    "--num_gnn_layers",
                    str(config["hgcn_layers"]),
                    "--dropout",
                    str(config["dropout"]),
                    "--k_neg_samples",
                    str(config["neg"]),
                ),
            )
        )

    if len(tasks) != 242:
        raise ValueError(f"expected 242 unique paper tasks, built {len(tasks)}")
    return tasks


def select_tasks(tasks: list[Task], only_task_keys: list[str] | None) -> list[Task]:
    if not only_task_keys:
        return tasks
    requested = set(only_task_keys)
    if len(requested) != len(only_task_keys):
        raise ValueError("--only-task values must be unique")
    selected = [task for task in tasks if task.key in requested]
    missing = requested.difference(task.key for task in selected)
    if missing:
        raise ValueError(f"unknown --only-task key(s): {', '.join(sorted(missing))}")
    return selected


def _common_command(
    args,
    task: Task,
    task_dir: Path,
    cache_dir: Path,
    final_eval_split: str = "validation",
) -> list[str]:
    return [
        str(args.python),
        "-m",
        "Model.train_test_single",
        "--edges_csv",
        str(args.repo_root / "Data/merged_edge.csv"),
        "--nodes_csv",
        str(args.repo_root / "Data/merged_node.csv"),
        "--train_target_types",
        JOINT_TARGETS,
        "--epochs",
        str(args.epochs),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--local_model_path",
        str(args.repo_root / "models_cache"),
        "--val_split",
        "0.1",
        "--test_split",
        "0.1",
        "--default_embed_size",
        "64",
        "--hgcn_hidden",
        "64",
        "--hgcn_out",
        "64",
        "--transformer_heads",
        "16",
        "--num_transformer_layers",
        "3",
        "--transformer_ff",
        "512",
        "--num_latents",
        "32",
        "--num_gnn_layers",
        "3",
        "--dropout",
        "0.1",
        "--lr",
        "0.0001",
        "--weight_decay",
        "0.00001",
        "--k_neg_samples",
        "32",
        "--training-engine",
        "unified",
        "--unified-graph-backend",
        "torch",
        "--batch-size",
        "48",
        "--neighbor-fanouts",
        "3,2,1",
        "--steps-per-epoch",
        "224",
        "--validation-batch-size",
        "2",
        "--fixed-validation-seed",
        "1042",
        "--eval-every",
        "5",
        "--scorer-mode",
        "interaction_mlp",
        "--link-loss",
        "bce",
        "--early_stopping_patience",
        str(args.patience),
        "--early_stopping_metrics",
        "val_loss",
        "--checkpoint-metric",
        "full_ranking_macro_mrr",
        "--disable-sequence-features",
        "--disable-isolated-node-filter",
        "--sync-model-target-etypes",
        "--regenerate-features-after-filter",
        "--skip-final-prediction",
        "--enable-full-ranking-eval",
        "--enable-full-ranking-validation",
        "--full-ranking-validation-interval",
        "5",
        "--report-optimal-threshold-metrics",
        "--ranking-k-values",
        RANKING_KS,
        "--final-eval-split",
        final_eval_split,
        "--full-ranking-source-type",
        "disease",
        "--restrict-ranking-etypes-to-source-type",
        "--full-ranking-candidate-batch-size",
        "4096",
        "--feature-cache-path",
        str(args.feature_cache),
        "--split-manifest-output",
        str(cache_dir / "split_manifest.json"),
        "--metrics-output",
        str(task_dir / "metrics.json"),
        "--save_model_path",
        str(task_dir / "model_final.pt"),
        "--graph_cache_path",
        str(cache_dir),
        *task.extra_args,
    ]


def _path_command(
    args,
    task: Task,
    task_dir: Path,
    cache_dir: Path,
    parent_checkpoint: Path,
    *,
    relbf: bool,
    final_eval_split: str,
) -> list[str]:
    command = _common_command(
        args,
        task,
        task_dir,
        cache_dir,
        final_eval_split=final_eval_split,
    )
    command.extend(
        [
            "--load_model_path",
            str(parent_checkpoint),
            "--biogt-path-propagation-hops",
            "5",
            "--biogt-path-pretrain-epochs",
            "1",
            "--biogt-path-pretrain-batch-size",
            "4",
            "--biogt-path-pretrain-lr",
            "0.0003",
            "--biogt-path-pretrain-temperature",
            "0.5",
            "--freeze-biogt-base-for-path-finetune",
            "--biogt-path-pretrain-only",
        ]
    )
    if relbf:
        command.extend(
            [
                "--biogt-relbf-dim",
                "32",
                "--initialize-biogt-relbf-from-checkpoint",
            ]
        )
    else:
        command.append("--initialize-biogt-path-from-checkpoint")
    return command


def _gene9_stage_command(
    args,
    task: Task,
    task_dir: Path,
    cache_dir: Path,
    parent_checkpoint: Path,
    *,
    stage: str,
    final_eval_split: str,
) -> list[str]:
    command = _common_command(
        args,
        task,
        task_dir,
        cache_dir,
        final_eval_split=final_eval_split,
    )
    command.extend(
        [
            "--load_model_path",
            str(parent_checkpoint),
            "--biogt-path-propagation-hops",
            "5",
            "--biogt-relbf-dim",
            "32",
            "--biogt-path-pretrain-epochs",
            "1",
            "--biogt-path-pretrain-batch-size",
            "4",
            "--biogt-path-pretrain-lr",
            "0.0003",
            "--biogt-path-pretrain-temperature",
            "0.5",
            "--biogt-path-pretrain-hard-negatives",
            "64",
            "--biogt-path-pretrain-rank-weight",
            "0.25",
            "--biogt-path-pretrain-lambdarank-weight",
            "1",
            "--biogt-path-pretrain-lambdarank-k",
            "10",
            "--biogt-path-pretrain-lambdarank-temperature",
            "1",
            "--biogt-path-pretrain-types",
            "gene",
            "--freeze-biogt-base-for-path-finetune",
            "--biogt-path-pretrain-only",
        ]
    )
    if stage == "gene_lambdarank":
        command.append("--biogt-relbf-target-only-finetune")
    elif stage == "indicator":
        command.extend(
            [
                "--biogt-relbf-indicator-residual",
                "--initialize-biogt-relbf-indicator-from-checkpoint",
                "--biogt-relbf-indicator-only-finetune",
            ]
        )
    elif stage == "comparator":
        command.extend(
            [
                "--biogt-relbf-indicator-residual",
                "--biogt-relbf-parent-top10-boundary-comparator",
                "--initialize-biogt-relbf-parent-top10-boundary-comparator-from-checkpoint",
                "--biogt-relbf-parent-top10-boundary-comparator-only-finetune",
            ]
        )
    elif stage == "window":
        command.extend(
            [
                "--biogt-relbf-indicator-residual",
                "--biogt-relbf-parent-top10-boundary-comparator",
                "--biogt-relbf-boundary-window-rank-slot-expert",
                "--initialize-biogt-relbf-boundary-window-rank-slot-expert-from-checkpoint",
                "--biogt-relbf-boundary-window-rank-slot-expert-only-finetune",
            ]
        )
    else:
        raise ValueError(f"unknown frozen Gene9 stage: {stage}")
    return command


class SuiteRunner:
    def __init__(self, args, tasks: list[Task]) -> None:
        self.args = args
        self.tasks = tasks
        self.lock = threading.Lock()
        self.active_cache_keys: set[str] = set()
        self.completed_cache_keys: set[str] = set()
        self.status_by_key = {}
        self.status_path = args.run_root / "status.json"
        self.collected_task_keys = self._load_collected_task_keys(
            args.completed_tasks_ledger
        )

    @staticmethod
    def _load_collected_task_keys(ledger_path: Path | None) -> set[str]:
        if ledger_path is None:
            return set()
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        completed = payload.get("completed_tasks", {})
        if isinstance(completed, dict):
            return {str(key) for key in completed}
        if isinstance(completed, list):
            return {str(key) for key in completed}
        raise ValueError("completed_tasks ledger entry must be a mapping or list")

    def _task_dir(self, task: Task) -> Path:
        return self.args.run_root / task.phase / task.label

    def _cache_dir(self, task: Task) -> Path:
        return self.args.run_root / "split_caches" / task.cache_key

    def _task_complete(self, task: Task) -> bool:
        if task.key in self.collected_task_keys:
            return not self.args.rerun
        task_dir = self._task_dir(task)
        status = {}
        try:
            status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return (
            not self.args.rerun
            and status.get("status") == "completed"
            and (task_dir / "metrics.json").is_file()
        )

    def _write_suite_status(self) -> None:
        counts = {}
        for status in self.status_by_key.values():
            state = status.get("status", "pending")
            counts[state] = counts.get(state, 0) + 1
        _write_json(
            self.status_path,
            {
                "status": (
                    "completed"
                    if counts.get("completed", 0) == len(self.tasks)
                    else "running"
                ),
                "updated_at": _now(),
                "task_count": len(self.tasks),
                "counts": counts,
                "tasks": self.status_by_key,
            },
        )

    def _run_task(self, task: Task) -> tuple[Task, int]:
        task_dir = self._task_dir(task)
        cache_dir = self._cache_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_dir = task_dir / "stages" / "base"
        path_dir = task_dir / "stages" / "path5"
        relbf_dir = task_dir / "stages" / "relbf"
        gene_dir = task_dir / "stages" / "gene_lambdarank"
        indicator_dir = task_dir / "stages" / "indicator"
        comparator_dir = task_dir / "stages" / "comparator"
        for stage_dir in (
            base_dir,
            path_dir,
            relbf_dir,
            gene_dir,
            indicator_dir,
            comparator_dir,
        ):
            stage_dir.mkdir(parents=True, exist_ok=True)
        commands = [
            (
                "base",
                _common_command(
                    self.args,
                    task,
                    base_dir,
                    cache_dir,
                    final_eval_split="validation",
                ),
                base_dir,
            ),
            (
                "path5",
                _path_command(
                    self.args,
                    task,
                    path_dir,
                    cache_dir,
                    base_dir / "model_final_best.pt",
                    relbf=False,
                    final_eval_split="validation",
                ),
                path_dir,
            ),
            (
                "relbf",
                _path_command(
                    self.args,
                    task,
                    relbf_dir,
                    cache_dir,
                    path_dir / "model_final_best.pt",
                    relbf=True,
                    final_eval_split="validation",
                ),
                relbf_dir,
            ),
            (
                "gene_lambdarank",
                _gene9_stage_command(
                    self.args,
                    task,
                    gene_dir,
                    cache_dir,
                    relbf_dir / "model_final_best.pt",
                    stage="gene_lambdarank",
                    final_eval_split="validation",
                ),
                gene_dir,
            ),
            (
                "indicator",
                _gene9_stage_command(
                    self.args,
                    task,
                    indicator_dir,
                    cache_dir,
                    gene_dir / "model_final_best.pt",
                    stage="indicator",
                    final_eval_split="validation",
                ),
                indicator_dir,
            ),
            (
                "comparator",
                _gene9_stage_command(
                    self.args,
                    task,
                    comparator_dir,
                    cache_dir,
                    indicator_dir / "model_final_best.pt",
                    stage="comparator",
                    final_eval_split="validation",
                ),
                comparator_dir,
            ),
            (
                "window",
                _gene9_stage_command(
                    self.args,
                    task,
                    task_dir,
                    cache_dir,
                    comparator_dir / "model_final_best.pt",
                    stage="window",
                    final_eval_split=task.final_eval_split,
                ),
                task_dir,
            ),
        ]
        status = {
            "phase": task.phase,
            "label": task.label,
            "target": task.target,
            "cache_key": task.cache_key,
            "status": "running",
            "started_at": _now(),
            "staged_training": True,
            "stages": {},
        }
        _write_json(task_dir / "status.json", status)
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(self.args.repo_root),
                "OMP_NUM_THREADS": str(self.args.threads_per_job),
                "MKL_NUM_THREADS": str(self.args.threads_per_job),
                "OPENBLAS_NUM_THREADS": str(self.args.threads_per_job),
            }
        )
        return_code = 0
        for stage_name, command, stage_dir in commands:
            stage_status = {
                "status": "running",
                "started_at": _now(),
                "command": command,
            }
            status["stages"][stage_name] = stage_status
            _write_json(task_dir / "status.json", status)
            with (stage_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
                with (stage_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                    result = subprocess.run(
                        command,
                        cwd=self.args.repo_root,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
            return_code = result.returncode
            stage_complete = (
                return_code == 0
                and (stage_dir / "metrics.json").is_file()
                and (stage_dir / "model_final_best.pt").is_file()
            )
            stage_status.update(
                {
                    "status": "completed" if stage_complete else "failed",
                    "exit_code": return_code,
                    "finished_at": _now(),
                }
            )
            _write_json(task_dir / "status.json", status)
            if not stage_complete:
                break
        status.update(
            {
                "status": (
                    "completed"
                    if return_code == 0
                    and (task_dir / "metrics.json").is_file()
                    else "failed"
                ),
                "exit_code": return_code,
                "finished_at": _now(),
            }
        )
        status["checkpoint_retained"] = status["status"] == "completed"
        status["all_stage_checkpoints_retained"] = status["status"] == "completed"
        _write_json(task_dir / "status.json", status)
        effective_return_code = (
            return_code
            if status["status"] == "completed"
            else (return_code or -2)
        )
        return task, effective_return_code

    def run(self) -> int:
        pending = []
        for task in self.tasks:
            cache_dir = self._cache_dir(task)
            if (
                (cache_dir / "graphs.bin").is_file()
                and (cache_dir / "metadata" / "num_nodes_dict.pt").is_file()
            ):
                self.completed_cache_keys.add(task.cache_key)
        for task in self.tasks:
            if self._task_complete(task):
                self.status_by_key[task.key] = {
                    "status": "completed",
                    "target": task.target,
                    "cache_key": task.cache_key,
                    "resumed": True,
                    "collected": task.key in self.collected_task_keys,
                }
                self.completed_cache_keys.add(task.cache_key)
            else:
                self.status_by_key[task.key] = {
                    "status": "pending",
                    "target": task.target,
                    "cache_key": task.cache_key,
                }
                pending.append(task)
        self._write_suite_status()

        running = {}
        completed_in_invocation = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.workers
        ) as executor:
            while pending or running:
                if (
                    not running
                    and self.args.max_tasks_per_invocation > 0
                    and completed_in_invocation
                    >= self.args.max_tasks_per_invocation
                ):
                    payload = json.loads(
                        self.status_path.read_text(encoding="utf-8")
                    )
                    payload["status"] = "awaiting_collection"
                    payload["completed_in_invocation"] = completed_in_invocation
                    payload["paused_at"] = _now()
                    _write_json(self.status_path, payload)
                    return 0
                submitted = True
                while len(running) < self.args.workers and submitted:
                    submitted = False
                    for index, task in enumerate(pending):
                        cache_ready = task.cache_key in self.completed_cache_keys
                        if (
                            not cache_ready
                            and task.cache_key in self.active_cache_keys
                        ):
                            continue
                        pending.pop(index)
                        if not cache_ready:
                            self.active_cache_keys.add(task.cache_key)
                        self.status_by_key[task.key]["status"] = "running"
                        future = executor.submit(self._run_task, task)
                        running[future] = task
                        submitted = True
                        break

                if not running:
                    raise RuntimeError("scheduler deadlock: pending tasks but no runnable task")
                done, _ = concurrent.futures.wait(
                    running,
                    timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    self._write_suite_status()
                    continue
                for future in done:
                    task = running.pop(future)
                    self.active_cache_keys.discard(task.cache_key)
                    try:
                        _, return_code = future.result()
                    except Exception as exc:
                        return_code = -1
                        state = "failed"
                        self.status_by_key[task.key] = {
                            "status": "failed",
                            "target": task.target,
                            "cache_key": task.cache_key,
                            "error": str(exc),
                        }
                    else:
                        state = "completed" if return_code == 0 else "failed"
                        self.status_by_key[task.key]["status"] = state
                        self.status_by_key[task.key]["exit_code"] = return_code
                        if state == "completed":
                            self.completed_cache_keys.add(task.cache_key)
                            completed_in_invocation += 1
                    self._write_suite_status()
                    if state == "failed":
                        return return_code or 1

        failures = [
            key
            for key, status in self.status_by_key.items()
            if status.get("status") != "completed"
        ]
        final_payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        final_payload["status"] = "completed" if not failures else "failed"
        final_payload["finished_at"] = _now()
        final_payload["failed_tasks"] = failures
        _write_json(self.status_path, final_payload)
        return 0 if not failures else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--sensitivity-manifest", type=Path, required=True)
    parser.add_argument("--category-splits", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, choices=[1])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads-per-job", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-tasks-per-invocation", type=int, default=1)
    parser.add_argument("--completed-tasks-ledger", type=Path)
    parser.add_argument(
        "--only-task",
        action="append",
        dest="only_tasks",
        help="Run only the exact phase/label task key; repeat for multiple tasks.",
    )
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.run_root = args.run_root.resolve()
    args.feature_cache = args.feature_cache.resolve()
    args.category_splits = args.category_splits.resolve()
    if args.completed_tasks_ledger is not None:
        args.completed_tasks_ledger = args.completed_tasks_ledger.resolve()
    manifest = json.loads(
        args.sensitivity_manifest.read_text(encoding="utf-8")
    )
    tasks = select_tasks(
        build_tasks(manifest, args.category_splits),
        args.only_tasks,
    )
    aliases = {
        "fusion/joint4_fusion-weighted": "main/model-pure-biogt_joint4",
        "ablation/joint4_abl-none": "main/model-pure-biogt_joint4",
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.run_root / "run_manifest.json",
        {
            "schema_version": 1,
            "protocol": "pure_biogt_joint4",
            "training_protocol": {
                "staged": True,
                "stages": [
                    "base_sampled_training",
                    "scalar_path5_full_candidate_finetune",
                    "relbf_full_candidate_finetune",
                    "gene_relbf_lambdarank10_finetune",
                    "gene_indicator_finetune",
                    "gene_parent_top10_11_comparator_finetune",
                    "gene_rank5_20_window_finetune",
                ],
                "heldout_evaluation_stage": "gene_rank5_20_window_only",
                "checkpoint_policy": "retain_all_until_local_sha_collection",
            },
            "joint_target_types": list(TARGETS),
            "ranking_ks": [1, 5, 10, 20, 50, 100],
            "unique_task_count": len(tasks),
            "reused_result_count": len(aliases),
            "aliases": aliases,
            "tasks": [
                {
                    "phase": task.phase,
                    "label": task.label,
                    "target": task.target,
                    "cache_key": task.cache_key,
                    "extra_args": list(task.extra_args),
                    "final_eval_split": task.final_eval_split,
                }
                for task in tasks
            ],
        },
    )
    raise SystemExit(SuiteRunner(args, tasks).run())


if __name__ == "__main__":
    main()
