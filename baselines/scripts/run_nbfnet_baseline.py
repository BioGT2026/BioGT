#!/usr/bin/env python
"""Official PyG NBFNet adapted to joint BioGT tail ranking."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from torch.nn import functional as F
    from torch_geometric.data import Data
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Official NBFNet requires torch, torch-geometric, and torch-scatter."
    ) from exc

from baseline_io import (
    average_tie_rank,
    evaluation_known_tail_filter,
    load_dataset,
    load_split,
    load_target_specs,
    macro_average_metrics,
    metrics_payload_by_target,
    relation_balanced_batches,
    sample_filtered_negative_tails,
    target_lookups,
    training_known_tail_filter,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
NBFNET_ROOT = ROOT / "external" / "NBFNet-PyG"


def load_official_model():
    if not (NBFNET_ROOT / "nbfnet" / "models.py").exists():
        raise SystemExit(
            "Missing official NBFNet-PyG source under "
            f"{NBFNET_ROOT}. Run baselines/scripts/download_baselines.py first."
        )
    os.environ.setdefault("NBFNET_DISABLE_FUSED", "1")
    sys.path.insert(0, str(NBFNET_ROOT))
    try:
        from nbfnet.models import NBFNet
    finally:
        sys.path.pop(0)
    return NBFNet


def source_commit() -> str:
    proc = subprocess.run(
        ["git", "-C", str(NBFNET_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def make_graph(train, num_nodes: int, num_relations: int, device):
    forward = torch.tensor(train, dtype=torch.long, device=device)
    inverse = torch.stack(
        [
            forward[:, 2],
            forward[:, 1] + num_relations,
            forward[:, 0],
        ],
        dim=1,
    )
    edges = torch.cat([forward, inverse], dim=0)
    return Data(
        edge_index=edges[:, [0, 2]].t().contiguous(),
        edge_type=edges[:, 1].contiguous(),
        num_nodes=num_nodes,
    )


@torch.no_grad()
def evaluate(
    model,
    graph,
    triples,
    splits,
    candidates_by_relation,
    relation_to_target,
    device,
    eval_batch_size: int,
    score_batch_size: int,
    max_eval_triples: int,
):
    model.eval()
    known = evaluation_known_tail_filter(splits)
    triples_by_relation = defaultdict(list)
    for triple in triples[: max_eval_triples or None]:
        triples_by_relation[triple[1]].append(triple)

    ranks_by_target = defaultdict(list)
    for relation, relation_triples in sorted(triples_by_relation.items()):
        target = relation_to_target[relation]
        candidates = candidates_by_relation[relation]
        candidate_positions = {
            entity: position for position, entity in enumerate(candidates)
        }
        for start in range(0, len(relation_triples), eval_batch_size):
            batch = relation_triples[start : start + eval_batch_size]
            heads = torch.tensor(
                [head for head, _, _ in batch],
                dtype=torch.long,
                device=device,
            )
            relations = torch.full(
                (len(batch),),
                relation,
                dtype=torch.long,
                device=device,
            )
            tails = torch.tensor(
                [tail for _, _, tail in batch],
                dtype=torch.long,
                device=device,
            )
            output = model.bellmanford(graph, heads, relations)["node_feature"]
            target_features = output[
                torch.arange(len(batch), device=device),
                tails,
            ]
            target_scores = model.mlp(target_features).squeeze(-1)
            greater = torch.zeros(len(batch), dtype=torch.long, device=device)
            equal = torch.zeros(len(batch), dtype=torch.long, device=device)

            for candidate_start in range(0, len(candidates), score_batch_size):
                candidate_slice = candidates[
                    candidate_start : candidate_start + score_batch_size
                ]
                candidate_tensor = torch.tensor(
                    candidate_slice,
                    dtype=torch.long,
                    device=device,
                )
                scores = model.mlp(output[:, candidate_tensor, :]).squeeze(-1)
                valid = torch.ones_like(scores, dtype=torch.bool)
                for row, (head, _, tail) in enumerate(batch):
                    for known_tail in known[(head, relation)] - {tail}:
                        position = candidate_positions.get(known_tail)
                        if (
                            position is not None
                            and candidate_start <= position < candidate_start + len(candidate_slice)
                        ):
                            valid[row, position - candidate_start] = False
                greater += ((scores > target_scores.unsqueeze(1)) & valid).sum(dim=1)
                equal += (
                    torch.isclose(scores, target_scores.unsqueeze(1)) & valid
                ).sum(dim=1)

            ranks_by_target[target].extend(
                average_tie_rank(int(num_greater), int(num_equal))
                for num_greater, num_equal in zip(
                    greater.cpu().tolist(),
                    equal.cpu().tolist(),
                )
            )
            print(
                f"nbfnet_eval target={target} "
                f"queries={min(start + len(batch), len(relation_triples))}/"
                f"{len(relation_triples)}",
                flush=True,
            )
    return metrics_payload_by_target(ranks_by_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--adversarial-temperature", type=float, default=0.5)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.cuda and device.type != "cuda":
        raise SystemExit("CUDA requested for official NBFNet but is unavailable")

    data_path = args.data_path or ROOT / "data" / args.dataset / "kge"
    ent2id, _, rel2id, _, splits = load_dataset(
        data_path,
        ("train", "valid"),
    )
    target_specs = load_target_specs(data_path, ent2id, rel2id)
    relation_to_target, candidates_by_relation = target_lookups(target_specs)
    training_known = training_known_tail_filter(splits)
    train = list(splits["train"])
    graph = make_graph(train, len(ent2id), len(rel2id), device)

    NBFNet = load_official_model()
    model = NBFNet(
        input_dim=args.input_dim,
        hidden_dims=[args.hidden_dim] * args.layers,
        num_relation=len(rel2id) * 2,
        message_func="distmult",
        aggregate_func="pna",
        short_cut=True,
        layer_norm=True,
        dependent=True,
        remove_one_hop=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_epoch = 0
    best_validation_mrr = -math.inf
    best_validation_metrics = {}
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch * 100003)
        total_loss = 0.0
        step_count = 0
        for step, positives in enumerate(
            relation_balanced_batches(
                train,
                args.batch_size,
                args.seed,
                epoch,
            ),
            1,
        ):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break
            negatives = sample_filtered_negative_tails(
                positives,
                candidates_by_relation,
                training_known,
                args.negative_ratio,
                rng,
            )
            positive_tensor = torch.tensor(
                positives,
                dtype=torch.long,
                device=device,
            )[:, [0, 2, 1]].unsqueeze(1)
            negative_tensor = torch.tensor(
                negatives,
                dtype=torch.long,
                device=device,
            )[:, [0, 2, 1]].reshape(len(positives), args.negative_ratio, 3)
            training_batch = torch.cat(
                [positive_tensor, negative_tensor],
                dim=1,
            )
            scores = model(graph, training_batch)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1
            loss_items = F.binary_cross_entropy_with_logits(
                scores,
                labels,
                reduction="none",
            )
            weights = torch.ones_like(scores)
            with torch.no_grad():
                weights[:, 1:] = F.softmax(
                    scores[:, 1:] / args.adversarial_temperature,
                    dim=-1,
                )
            loss = (
                (loss_items * weights).sum(dim=-1) / weights.sum(dim=-1)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            step_count += 1
        print(
            f"epoch={epoch} loss={total_loss / max(step_count, 1):.6f} "
            f"steps={step_count}",
            flush=True,
        )

        if (
            epoch % max(1, args.validation_interval) == 0
            or epoch == args.epochs
        ):
            _, validation_by_target = evaluate(
                model,
                graph,
                splits["valid"],
                splits,
                candidates_by_relation,
                relation_to_target,
                device,
                args.eval_batch_size,
                args.score_batch_size,
                args.max_eval_triples,
            )
            validation_macro = macro_average_metrics(validation_by_target)
            validation_mrr = float(validation_macro.get("MRR", -math.inf))
            print(
                f"epoch={epoch} validation_macro_mrr={validation_mrr:.6f}",
                flush=True,
            )
            if validation_mrr > best_validation_mrr:
                best_epoch = epoch
                best_validation_mrr = validation_mrr
                best_validation_metrics = validation_macro
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    splits["test"] = load_split(data_path, "test", ent2id, rel2id)
    metrics, metrics_by_target = evaluate(
        model,
        graph,
        splits["test"],
        splits,
        candidates_by_relation,
        relation_to_target,
        device,
        args.eval_batch_size,
        args.score_batch_size,
        args.max_eval_triples,
    )
    out = args.out or ROOT / "runs" / "nbfnet" / args.dataset / "metrics.json"
    checkpoint_path = out.with_name("checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
        },
        checkpoint_path,
    )
    payload = {
        "baseline": "NBFNet",
        "implementation": "official NBFNet-PyG adapted for relation-scoped tails",
        "source": "https://github.com/KiddoZhu/NBFNet-PyG",
        "source_commit": source_commit(),
        "dataset": args.dataset,
        "data_path": str(data_path),
        "device": str(device),
        "candidate_entity_count_by_target": {
            target: len(spec["candidate_ids"])
            for target, spec in target_specs.items()
        },
        "tie_policy": "average",
        "best_epoch": best_epoch,
        "validation_metrics_macro": best_validation_metrics,
        "split_protocol": {
            "fit": "train_only",
            "checkpoint_selection": "valid_only",
            "test_queries": "test_only_after_checkpoint_freeze",
            "test_filter": "train_plus_valid_without_other_test_labels",
        },
        "metrics": metrics,
        "metrics_macro": macro_average_metrics(metrics_by_target),
        "metrics_by_target": metrics_by_target,
        "checkpoint": str(checkpoint_path),
    }
    write_json(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
