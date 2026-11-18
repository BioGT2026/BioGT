#!/usr/bin/env python
"""Run TransE, DistMult, ComplEx, or RotatE through the vendored KGE code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from baseline_io import (
    average_tie_rank,
    evaluation_known_tail_filter,
    load_dataset,
    load_target_specs,
    macro_average_metrics,
    metrics_payload_by_target,
    target_lookups,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
KGE_ROOT = ROOT / "external" / "KnowledgeGraphEmbedding"

MODEL_DEFAULTS = {
    "TransE": {"gamma": 12.0, "lr": 0.0001, "flags": []},
    "DistMult": {"gamma": 200.0, "lr": 0.001, "flags": ["-r", "0.00001"]},
    "ComplEx": {"gamma": 200.0, "lr": 0.001, "flags": ["-de", "-dr", "-r", "0.00001"]},
    "RotatE": {"gamma": 12.0, "lr": 0.0001, "flags": ["-de", "-adv"]},
}


def evaluate_checkpoint(
    args: argparse.Namespace,
    data_path: Path,
    save_path: Path,
) -> dict:
    import torch

    codes_dir = KGE_ROOT / "codes"
    sys.path.insert(0, str(codes_dir))
    try:
        from model import KGEModel
    finally:
        sys.path.pop(0)

    ent2id, _, rel2id, _, splits = load_dataset(data_path)
    target_specs = load_target_specs(data_path, ent2id, rel2id)
    relation_to_target, candidates_by_relation = target_lookups(target_specs)
    known = evaluation_known_tail_filter(splits)
    device = torch.device(
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "cpu"
    )
    defaults = MODEL_DEFAULTS[args.model]
    model = KGEModel(
        model_name=args.model,
        nentity=len(ent2id),
        nrelation=len(rel2id),
        hidden_dim=args.hidden_dim,
        gamma=defaults["gamma"],
        double_entity_embedding=args.model in {"ComplEx", "RotatE"},
        double_relation_embedding=args.model == "ComplEx",
    ).to(device)
    checkpoint_path = save_path / "checkpoint_best"
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Validation-selected checkpoint is missing: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ranks_by_target: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        eval_triples = splits["test"][: args.max_eval_triples or None]
        for query_index, (head, relation, tail) in enumerate(eval_triples, 1):
            target = relation_to_target[relation]
            filtered = known[(head, relation)] - {tail}
            candidates = [
                entity
                for entity in candidates_by_relation[relation]
                if entity not in filtered
            ]
            positive = torch.tensor(
                [[head, relation, tail]],
                dtype=torch.long,
                device=device,
            )
            target_score = float(model(positive, mode="single").reshape(-1)[0].item())
            num_greater = 0
            num_equal = 0
            for start in range(0, len(candidates), args.score_batch_size):
                candidate_batch = torch.tensor(
                    [candidates[start : start + args.score_batch_size]],
                    dtype=torch.long,
                    device=device,
                )
                scores = model((positive, candidate_batch), mode="tail-batch").reshape(-1)
                num_greater += int((scores > target_score).sum().item())
                num_equal += int(
                    torch.isclose(
                        scores,
                        torch.tensor(target_score, device=device),
                    ).sum().item()
                )
            ranks_by_target[target].append(
                average_tie_rank(num_greater, num_equal)
            )
            if query_index % 500 == 0:
                print(
                    f"evaluated_kge_queries={query_index}/{len(eval_triples)}",
                    flush=True,
                )

    metrics, metrics_by_target = metrics_payload_by_target(ranks_by_target)
    payload = {
        "baseline": args.model,
        "dataset": args.dataset,
        "data_path": str(data_path),
        "device": str(device),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "validation_mrr": float(checkpoint["validation_mrr"]),
        "candidate_entity_count_by_target": {
            target: len(spec["candidate_ids"])
            for target, spec in target_specs.items()
        },
        "tie_policy": "average",
        "split_protocol": {
            "fit": "train_only",
            "checkpoint_selection": "valid_only",
            "test_queries": "test_only_after_checkpoint_freeze",
            "test_filter": "train_plus_valid_without_other_test_labels",
        },
        "metrics": metrics,
        "metrics_macro": macro_average_metrics(metrics_by_target),
        "metrics_by_target": metrics_by_target,
    }
    write_json(save_path / "metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--save-root", type=Path, default=ROOT / "runs" / "kge")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--negative-sample-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--cpu-num", type=int, default=8)
    parser.add_argument("--valid-steps", type=int, default=10000)
    parser.add_argument("--log-steps", type=int, default=100)
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
                "PyTorch is required for KGE baselines. Install the baseline "
                "environment or pass --python pointing to the BioGT training Python."
            )

    defaults = MODEL_DEFAULTS[args.model]
    data_path = (args.data_path or KGE_ROOT / "data" / args.dataset).resolve()
    required = ["entities.dict", "relations.dict", "train.txt", "valid.txt", "test.txt"]
    missing = [name for name in required if not (data_path / name).exists()]
    if missing:
        raise SystemExit(f"Missing KGE dataset files under {data_path}: {missing}")

    save_path = (args.save_root / args.dataset / args.model).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "-u",
        str(KGE_ROOT / "codes" / "run.py"),
        "--do_train",
        "--do_valid",
        "--data_path",
        str(data_path),
        "--candidate_entities",
        str(data_path / "candidate_entities.txt"),
        "--tail_only",
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "-n",
        str(args.negative_sample_size),
        "-b",
        str(args.batch_size),
        "-d",
        str(args.hidden_dim),
        "-g",
        str(defaults["gamma"]),
        "-a",
        "1.0",
        "-lr",
        str(defaults["lr"]),
        "--max_steps",
        str(args.max_steps),
        "--valid_steps",
        str(args.valid_steps),
        "--log_steps",
        str(args.log_steps),
        "-save",
        str(save_path),
        "--test_batch_size",
        str(args.test_batch_size),
        "-cpu",
        str(args.cpu_num),
    ]
    if args.cuda:
        cmd.insert(3, "--cuda")
    candidate_manifest = data_path / "target_manifest.json"
    if candidate_manifest.exists():
        tail_only_index = cmd.index("--tail_only")
        cmd[tail_only_index:tail_only_index] = [
            "--candidate_manifest",
            str(candidate_manifest),
        ]
    cmd.extend(defaults["flags"])

    env = os.environ.copy()
    if args.cuda:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=KGE_ROOT, env=env, check=True)
        payload = evaluate_checkpoint(args, data_path, save_path)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
