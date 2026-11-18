#!/usr/bin/env python
"""PyG HGTConv link-prediction baseline for joint BioGT triples."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import HGTConv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "HGT requires torch and torch-geometric. Install baselines/requirements.txt."
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
KNOWN_TYPES = {"disease", "gene", "rna", "protein", "metabolite"}


def infer_node_type(identifier: str, fallback: str) -> str:
    value = identifier.lower()
    if "::" in value:
        return value.split("::", 1)[0]
    typed_prefix = value.split(":", 1)[0]
    if typed_prefix in KNOWN_TYPES:
        return typed_prefix
    if value.startswith(("mondo:", "doid:", "omim:")):
        return "disease"
    if value.startswith(("ncbigene:", "gene:")):
        return "gene"
    if value.startswith(("uniprot:", "uniprotkb:")):
        return "protein"
    if value.startswith(("hmdb", "pubchem.compound:")):
        return "metabolite"
    if "rna" in value or value.startswith(("rnacentral:", "mirbase:")):
        return "rna"
    return fallback


def build_heterogeneous_graph(
    train,
    id2ent: dict[int, str],
    id2rel: dict[int, str],
    fallback_type: str,
    device,
):
    entities_by_type = defaultdict(list)
    for entity_id in range(len(id2ent)):
        entities_by_type[infer_node_type(id2ent[entity_id], fallback_type)].append(
            entity_id
        )
    node_types = sorted(entities_by_type)
    type_to_id = {node_type: index for index, node_type in enumerate(node_types)}
    global_type = torch.empty(len(id2ent), dtype=torch.long, device=device)
    global_local = torch.empty(len(id2ent), dtype=torch.long, device=device)
    type_global_ids = {}
    for node_type, entity_ids in entities_by_type.items():
        ids = torch.tensor(entity_ids, dtype=torch.long, device=device)
        type_global_ids[node_type] = ids
        global_type[ids] = type_to_id[node_type]
        global_local[ids] = torch.arange(len(entity_ids), device=device)

    edge_lists = defaultdict(lambda: [[], []])
    for head, relation, tail in train:
        source_type = node_types[int(global_type[head].item())]
        target_type = node_types[int(global_type[tail].item())]
        relation_name = id2rel[relation]
        canonical = (source_type, relation_name, target_type)
        inverse = (target_type, f"{relation_name}_reverse", source_type)
        edge_lists[canonical][0].append(int(global_local[head].item()))
        edge_lists[canonical][1].append(int(global_local[tail].item()))
        edge_lists[inverse][0].append(int(global_local[tail].item()))
        edge_lists[inverse][1].append(int(global_local[head].item()))
    edge_index_dict = {
        edge_type: torch.tensor(indices, dtype=torch.long, device=device)
        for edge_type, indices in edge_lists.items()
    }
    metadata = (node_types, sorted(edge_index_dict))
    return {
        "metadata": metadata,
        "edge_index_dict": edge_index_dict,
        "type_global_ids": type_global_ids,
        "global_type": global_type,
        "global_local": global_local,
        "type_to_id": type_to_id,
    }


class HGTLinkPredictor(nn.Module):
    def __init__(
        self,
        type_sizes: dict[str, int],
        metadata,
        num_relations: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        type_global_ids,
        num_entities: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.type_global_ids = type_global_ids
        self.entity_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(size, hidden_dim)
            for node_type, size in type_sizes.items()
        })
        self.convs = nn.ModuleList(
            HGTConv(
                hidden_dim,
                hidden_dim,
                metadata,
                heads=heads,
            )
            for _ in range(layers)
        )
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        for embedding in self.entity_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def encode(self, edge_index_dict):
        x_dict = {
            node_type: embedding.weight
            for node_type, embedding in self.entity_embeddings.items()
        }
        for conv in self.convs:
            updated = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.gelu(updated.get(node_type, x_dict[node_type]))
                for node_type in x_dict
            }
        z = next(iter(x_dict.values())).new_zeros(
            (self.num_entities, self.hidden_dim)
        )
        for node_type, values in x_dict.items():
            z[self.type_global_ids[node_type]] = values
        return z

    def score(self, z, heads, relations, tails):
        return (
            z[heads] * self.rel_emb(relations) * z[tails]
        ).sum(dim=-1)


@torch.no_grad()
def evaluate(
    model,
    triples,
    splits,
    edge_index_dict,
    candidates_by_relation,
    relation_to_target,
    device,
    max_eval_triples: int,
):
    model.eval()
    known = evaluation_known_tail_filter(splits)
    z = model.encode(edge_index_dict)
    candidate_tensors = {
        relation: torch.tensor(candidate_ids, dtype=torch.long, device=device)
        for relation, candidate_ids in candidates_by_relation.items()
    }
    ranks_by_target = defaultdict(list)
    eval_triples = triples[: max_eval_triples or None]
    for query_index, (head, relation, tail) in enumerate(eval_triples, 1):
        target = relation_to_target[relation]
        candidates = candidate_tensors[relation]
        scores = model.score(
            z,
            torch.full_like(candidates, head),
            torch.full_like(candidates, relation),
            candidates,
        )
        filter_tails = known[(head, relation)] - {tail}
        if filter_tails:
            scores[torch.isin(
                candidates,
                torch.tensor(list(filter_tails), dtype=torch.long, device=device),
            )] = -math.inf
        target_position = (candidates == tail).nonzero(as_tuple=False).item()
        target_score = scores[target_position]
        ranks_by_target[target].append(
            average_tie_rank(
                int((scores > target_score).sum().item()),
                int(torch.isclose(scores, target_score).sum().item()),
            )
        )
        if query_index % 500 == 0:
            print(f"hgt_eval queries={query_index}/{len(eval_triples)}", flush=True)
    return metrics_payload_by_target(ranks_by_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fallback-node-type", default="target")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-triples", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.cuda and device.type != "cuda":
        raise SystemExit("CUDA requested for HGT but is unavailable")

    data_path = args.data_path or ROOT / "data" / args.dataset / "kge"
    ent2id, id2ent, rel2id, id2rel, splits = load_dataset(
        data_path,
        ("train", "valid"),
    )
    target_specs = load_target_specs(data_path, ent2id, rel2id)
    relation_to_target, candidates_by_relation = target_lookups(target_specs)
    training_known = training_known_tail_filter(splits)
    train = list(splits["train"])
    graph = build_heterogeneous_graph(
        train,
        id2ent,
        id2rel,
        args.fallback_node_type,
        device,
    )
    type_sizes = {
        node_type: len(global_ids)
        for node_type, global_ids in graph["type_global_ids"].items()
    }
    model = HGTLinkPredictor(
        type_sizes,
        graph["metadata"],
        len(rel2id),
        args.hidden_dim,
        args.layers,
        args.heads,
        graph["type_global_ids"],
        len(ent2id),
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
        for step, batch in enumerate(
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
            pos = torch.tensor(batch, dtype=torch.long, device=device)
            negatives = torch.tensor(
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
            z = model.encode(graph["edge_index_dict"])
            positive_scores = model.score(z, pos[:, 0], pos[:, 1], pos[:, 2])
            negative_scores = model.score(
                z,
                negatives[:, 0],
                negatives[:, 1],
                negatives[:, 2],
            )
            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)
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
                splits["valid"],
                splits,
                graph["edge_index_dict"],
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
        graph["edge_index_dict"],
        candidates_by_relation,
        relation_to_target,
        device,
        args.max_eval_triples,
    )
    payload = {
        "baseline": "HGT-DistMult",
        "implementation": "torch_geometric.nn.HGTConv",
        "source": "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HGTConv.html",
        "dataset": args.dataset,
        "data_path": str(data_path),
        "device": str(device),
        "node_types": graph["type_to_id"],
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
    out = args.out or ROOT / "runs" / "hgt" / args.dataset / "metrics.json"
    write_json(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
