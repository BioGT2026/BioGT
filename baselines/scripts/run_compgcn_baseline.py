#!/usr/bin/env python
"""Run CompGCN through the patched vendored implementation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPGCN_ROOT = ROOT / "external" / "CompGCN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    parser.add_argument("--score-func", default="distmult", choices=["transe", "distmult", "conve"])
    parser.add_argument("--opn", default="corr", choices=["sub", "mult", "corr"])
    parser.add_argument("--gcn-layer", type=int, default=2)
    parser.add_argument("--gcn-dim", type=int, default=128)
    parser.add_argument("--init-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        probe = subprocess.run(
            [args.python, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode != 0:
            raise SystemExit(
                "PyTorch is required for CompGCN. Install the baseline environment "
                "or pass --python pointing to the BioGT training Python."
            )

    dataset_dir = (args.data_path or COMPGCN_ROOT / "data" / args.dataset).resolve()
    required = [
        "entities.dict",
        "relations.dict",
        "train.txt",
        "valid.txt",
        "test.txt",
    ]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing CompGCN dataset files under {dataset_dir}: {missing}")

    compatibility_dir = COMPGCN_ROOT / "data" / args.dataset
    compatibility_dir.parent.mkdir(parents=True, exist_ok=True)
    if compatibility_dir.is_symlink():
        if compatibility_dir.resolve() != dataset_dir:
            compatibility_dir.unlink()
    elif compatibility_dir.exists() and compatibility_dir.resolve() != dataset_dir:
        raise SystemExit(
            f"CompGCN compatibility path already exists and does not match "
            f"{dataset_dir}: {compatibility_dir}"
        )
    if not compatibility_dir.exists():
        compatibility_dir.symlink_to(dataset_dir, target_is_directory=True)

    for dirname in ["log", "checkpoints"]:
        (COMPGCN_ROOT / dirname).mkdir(exist_ok=True)

    name = args.name or f"biogt_{args.dataset}_{args.score_func}_{args.opn}"
    cmd = [
        args.python,
        "run.py",
        "-name",
        name,
        "-data",
        args.dataset,
        "-candidate_file",
        str(dataset_dir / "candidate_entities.txt"),
        "-score_func",
        args.score_func,
        "-opn",
        args.opn,
        "-gcn_layer",
        str(args.gcn_layer),
        "-gcn_dim",
        str(args.gcn_dim),
        "-init_dim",
        str(args.init_dim),
        "-embed_dim",
        str(args.embedding_dim),
        "-batch",
        str(args.batch_size),
        "-epoch",
        str(args.max_epochs),
        "-max_steps_per_epoch",
        str(args.max_steps_per_epoch),
        "-max_eval_triples",
        str(args.max_eval_triples),
        "-gpu",
        args.gpu,
        "-seed",
        str(args.seed),
    ]
    candidate_manifest = dataset_dir / "target_manifest.json"
    if candidate_manifest.exists():
        cmd.extend(["-candidate_manifest", str(candidate_manifest)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=COMPGCN_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
