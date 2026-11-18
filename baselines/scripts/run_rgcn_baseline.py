#!/usr/bin/env python
"""PyTorch R-GCN link-prediction baseline for BioGT triples."""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for the R-GCN baseline. Install the baseline "
        "environment first, for example: pip install -r baselines/requirements.txt"
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


class RelGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, x, edge_src, edge_rel, edge_dst, num_nodes: int):
        out = x.new_zeros((num_nodes, self.rel_weight.shape[-1]))
        deg = x.new_zeros((num_nodes,))
        for rel in range(self.rel_weight.shape[0]):
            mask = edge_rel == rel
            if not bool(mask.any()):
                continue
            src = edge_src[mask]
            dst = edge_dst[mask]
            msg = x[src] @ self.rel_weight[rel]
            out.index_add_(0, dst, msg)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        out = out / deg.clamp_min(1.0).unsqueeze(1)
        return out + self.self_loop(x) + self.bias


class RGCNLinkPredictor(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        convs = []
        in_dim = emb_dim
        for _ in range(layers):
            convs.append(RelGraphConv(in_dim, hidden_dim, num_relations * 2))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def encode(self, edge_src, edge_rel, edge_dst):
        x = self.entity_emb.weight
        for conv in self.convs:
            x = F.relu(conv(x, edge_src, edge_rel, edge_dst, x.shape[0]))
        return x

    def score(self, z, heads, rels, tails):
        return (z[heads] * self.rel_emb(rels) * z[tails]).sum(dim=-1)


def make_train_edges(train, num_relations: int, device):
    src = []
    rel = []
    dst = []
    for head, relation, tail in train:
        src.append(head)
        rel.append(relation)
        dst.append(tail)
        src.append(tail)
        rel.append(relation + num_relations)
        dst.append(head)
    return (
        torch.tensor(src, dtype=torch.long, device=device),
        torch.tensor(rel, dtype=torch.long, device=device),
        torch.tensor(dst, dtype=torch.long, device=device),
    )


def batches(items, batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


@torch.no_grad()
def evaluate(
    model,
    triples,
    splits,
    edge_tensors,
    candidates_by_relation,
    relation_to_target,
    device,
    max_eval_triples: int = 0,
):
    model.eval()
    known = evaluation_known_tail_filter(splits)
    z = model.encode(*edge_tensors)
    ranks_by_target = {}
    eval_triples = triples[: max_eval_triples or None]
    candidate_tensors = {
        relation: torch.tensor(candidate_ids, dtype=torch.long, device=device)
        for relation, candidate_ids in candidates_by_relation.items()
    }
    for head, rel, tail in eval_triples:
        target = relation_to_target[rel]
        ranks_by_target.setdefault(target, [])
        candidate_tensor = candidate_tensors[rel]
        num_candidates = candidate_tensor.numel()
        heads = torch.full((num_candidates,), head, dtype=torch.long, device=device)
        rels = torch.full((num_candidates,), rel, dtype=torch.long, device=device)
        scores = model.score(z, heads, rels, candidate_tensor)
        filter_tails = known[(head, rel)] - {tail}
        if filter_tails:
            filter_mask = torch.isin(
                candidate_tensor,
                torch.tensor(list(filter_tails), dtype=torch.long, device=device),
            )
            scores[filter_mask] = -math.inf
        target_position = (candidate_tensor == tail).nonzero(as_tuple=False).item()
        target_score = scores[target_position]
        ranks_by_target[target].append(
            average_tie_rank(
                int((scores > target_score).sum().item()),
                int(torch.isclose(scores, target_score).sum().item()),
            )
        )
    return metrics_payload_by_target(ranks_by_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    data_path = args.data_path or ROOT / "data" / args.dataset / "kge"
    ent2id, _, rel2id, _, splits = load_dataset(
        data_path,
        ("train", "valid"),
    )
    target_specs = load_target_specs(data_path, ent2id, rel2id)
    relation_to_target, candidates_by_relation = target_lookups(target_specs)
    candidate_tensors = {
        relation: torch.tensor(candidate_ids, dtype=torch.long, device=device)
        for relation, candidate_ids in candidates_by_relation.items()
    }
    num_entities = len(ent2id)
    num_relations = len(rel2id)
    train = list(splits["train"])
    training_known = training_known_tail_filter(splits)
    edge_tensors = make_train_edges(train, num_relations, device)

    model = RGCNLinkPredictor(
        num_entities,
        num_relations,
        args.embedding_dim,
        args.hidden_dim,
        args.layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_epoch = 0
    best_validation_mrr = -math.inf
    best_validation_metrics = {}
    best_state = None

    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch * 100003)
        model.train()
        total_loss = 0.0
        total_items = 0
        for step, batch in enumerate(
            relation_balanced_batches(
                train, args.batch_size, args.seed, epoch
            ),
            1,
        ):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break
            pos = torch.tensor(batch, dtype=torch.long, device=device)
            heads, rels, tails = pos[:, 0], pos[:, 1], pos[:, 2]
            neg = torch.tensor(
                sample_filtered_negative_tails(
                    batch,
                    candidates_by_relation,
                    training_known,
                    args.negative_ratio,
                    rng,
                ),
                dtype=torch.long,
                device=device,
            )
            neg_heads, neg_rels, neg_tails = neg[:, 0], neg[:, 1], neg[:, 2]
            z = model.encode(*edge_tensors)
            pos_score = model.score(z, heads, rels, tails)
            neg_score = model.score(z, neg_heads, neg_rels, neg_tails)
            logits = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_items += len(batch)
        print(f"epoch={epoch} loss={total_loss / max(total_items, 1):.6f}")
        if (
            epoch % max(1, args.validation_interval) == 0
            or epoch == args.epochs
        ):
            _, validation_by_target = evaluate(
                model,
                splits["valid"],
                splits,
                edge_tensors,
                candidates_by_relation,
                relation_to_target,
                device,
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
        splits["test"],
        splits,
        edge_tensors,
        candidates_by_relation,
        relation_to_target,
        device,
        args.max_eval_triples,
    )
    payload = {
        "baseline": "R-GCN-DistMult",
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
    }
    out = args.out or ROOT / "runs" / "rgcn" / args.dataset / "metrics.json"
    write_json(out, payload)
    print(payload)


if __name__ == "__main__":
    main()
