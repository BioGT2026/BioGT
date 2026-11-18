"""Shared helpers for BioGT baseline scripts."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    forward: dict[str, int] = {}
    reverse: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            idx, name = line.rstrip("\n").split("\t", 1)
            idx_int = int(idx)
            forward[name] = idx_int
            reverse[idx_int] = name
    return forward, reverse


def load_split(
    data_path: Path,
    split: str,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
) -> list[tuple[int, int, int]]:
    triples = []
    with (data_path / f"{split}.txt").open("r", encoding="utf-8") as fh:
        for line in fh:
            head, rel, tail = line.rstrip("\n").split("\t")
            triples.append((ent2id[head], rel2id[rel], ent2id[tail]))
    return triples


def load_dataset(
    data_path: Path,
    split_names: tuple[str, ...] = ("train", "valid", "test"),
):
    ent2id, id2ent = read_mapping(data_path / "entities.dict")
    rel2id, id2rel = read_mapping(data_path / "relations.dict")
    splits = {
        split: load_split(data_path, split, ent2id, rel2id)
        for split in split_names
    }
    return ent2id, id2ent, rel2id, id2rel, splits


def load_candidate_entity_ids(
    data_path: Path,
    ent2id: dict[str, int],
) -> list[int]:
    candidate_path = data_path / "candidate_entities.txt"
    if not candidate_path.exists():
        return list(range(len(ent2id)))
    candidates = []
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entity = line.strip()
            if entity:
                candidates.append(ent2id[entity])
    if not candidates:
        raise ValueError(f"No candidate entities found in {candidate_path}")
    return sorted(set(candidates))


def load_target_specs(
    data_path: Path,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Load relation-scoped candidate sets for singleton or joint datasets."""
    manifest_path = data_path / "target_manifest.json"
    metadata_path = data_path / "metadata.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_targets = payload.get("targets", {})
    elif metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target = metadata.get("target_type")
        relation = metadata.get("relation")
        raw_targets = {
            target: {
                "relation": relation,
                "candidate_file": "candidate_entities.txt",
            }
        } if target and relation else {}
    else:
        raw_targets = {}

    specs: dict[str, dict[str, Any]] = {}
    for target, raw_spec in raw_targets.items():
        relation_names = raw_spec.get("relations") or [raw_spec.get("relation")]
        relation_names = [name for name in relation_names if name]
        relation_ids = [rel2id[name] for name in relation_names if name in rel2id]
        candidate_file = raw_spec.get("candidate_file") or f"candidate_entities.{target}.txt"
        candidate_path = data_path / candidate_file
        if not candidate_path.exists():
            raise ValueError(f"Missing candidate file for target '{target}': {candidate_path}")
        candidate_ids = []
        for entity in candidate_path.read_text(encoding="utf-8").splitlines():
            entity = entity.strip()
            if entity:
                candidate_ids.append(ent2id[entity])
        if not candidate_ids:
            raise ValueError(f"No candidate entities found for target '{target}'")
        specs[str(target)] = {
            "relations": relation_names,
            "relation_ids": sorted(set(relation_ids)),
            "candidate_ids": sorted(set(candidate_ids)),
            "candidate_file": candidate_file,
        }

    if specs:
        return specs

    return {
        "all": {
            "relations": list(rel2id),
            "relation_ids": sorted(rel2id.values()),
            "candidate_ids": load_candidate_entity_ids(data_path, ent2id),
            "candidate_file": "candidate_entities.txt",
        }
    }


def target_lookups(
    target_specs: dict[str, dict[str, Any]],
) -> tuple[dict[int, str], dict[int, list[int]]]:
    relation_to_target: dict[int, str] = {}
    candidates_by_relation: dict[int, list[int]] = {}
    for target, spec in target_specs.items():
        for relation_id in spec["relation_ids"]:
            if relation_id in relation_to_target:
                raise ValueError(
                    f"Relation ID {relation_id} is assigned to multiple targets: "
                    f"{relation_to_target[relation_id]} and {target}"
                )
            relation_to_target[relation_id] = target
            candidates_by_relation[relation_id] = spec["candidate_ids"]
    return relation_to_target, candidates_by_relation


def metrics_payload_by_target(
    ranks_by_target: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    all_ranks = [
        rank
        for target in sorted(ranks_by_target)
        for rank in ranks_by_target[target]
    ]
    return (
        rank_metrics(all_ranks),
        {
            target: rank_metrics(ranks)
            for target, ranks in sorted(ranks_by_target.items())
        },
    )


def macro_average_metrics(
    metrics_by_target: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average each ranking metric equally over target modalities."""
    metric_names = sorted({
        metric
        for target_metrics in metrics_by_target.values()
        for metric in target_metrics
        if metric != "num_triples"
    })
    return {
        metric: sum(
            float(target_metrics[metric])
            for target_metrics in metrics_by_target.values()
            if metric in target_metrics
        ) / sum(
            metric in target_metrics
            for target_metrics in metrics_by_target.values()
        )
        for metric in metric_names
    }


def relation_balanced_batches(
    triples: list[tuple[int, int, int]],
    batch_size: int,
    seed: int,
    epoch: int,
):
    """Match BioGT's fixed-step, round-robin target-modality batching."""
    triples_by_relation: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for triple in triples:
        triples_by_relation[triple[1]].append(triple)
    relations = sorted(triples_by_relation)
    if not relations:
        return
    num_steps = max(len(relations), math.ceil(len(triples) / batch_size))
    rng = random.Random(int(seed) + int(epoch) * 100003)
    offset = int(epoch) % len(relations)
    for step in range(num_steps):
        relation = relations[(step + offset) % len(relations)]
        relation_triples = triples_by_relation[relation]
        yield [
            relation_triples[rng.randrange(len(relation_triples))]
            for _ in range(batch_size)
        ]


def sample_filtered_negative_tails(
    positives: list[tuple[int, int, int]],
    candidates_by_relation: dict[int, list[int]],
    known_tails,
    negative_ratio: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    """Sample relation-scoped tails while excluding every known positive."""
    negatives = []
    for head, relation, _ in positives:
        candidates = candidates_by_relation[relation]
        excluded = known_tails[(head, relation)]
        if len(excluded) >= len(candidates):
            raise RuntimeError(
                f"No filtered negative tail exists for query {(head, relation)}"
            )
        for _ in range(max(1, int(negative_ratio))):
            for _ in range(32):
                tail = candidates[rng.randrange(len(candidates))]
                if tail not in excluded:
                    break
            else:
                tail = next(candidate for candidate in candidates if candidate not in excluded)
            negatives.append((head, relation, tail))
    return negatives


def average_tie_rank(num_greater: int, num_equal: int) -> float:
    """Return the average 1-based rank for a score tied with its peers."""
    return 1.0 + float(num_greater) + 0.5 * float(max(0, num_equal - 1))


def known_tail_filter(splits: dict[str, list[tuple[int, int, int]]]):
    known = defaultdict(set)
    for triples in splits.values():
        for head, rel, tail in triples:
            known[(head, rel)].add(tail)
    return known


def training_known_tail_filter(
    splits: dict[str, list[tuple[int, int, int]]],
):
    """Build a negative-sampling filter without validation/test exposure."""
    return known_tail_filter({"train": splits["train"]})


def evaluation_known_tail_filter(
    splits: dict[str, list[tuple[int, int, int]]],
):
    """Filter test rankings with train/validation truths, never other test labels."""
    return known_tail_filter({
        "train": splits["train"],
        "valid": splits["valid"],
    })


def rank_metrics(ranks: list[float], k_values: tuple[int, ...] = (1, 3, 10, 50, 100)) -> dict[str, float]:
    if not ranks:
        return {}
    n = len(ranks)
    metrics = {
        "MRR": sum(1.0 / rank for rank in ranks) / n,
        "MR": sum(ranks) / n,
        "num_triples": n,
    }
    for k in sorted(set(k_values)):
        hits = [rank <= k for rank in ranks]
        metrics[f"Hits@{k}"] = sum(hits) / n
        # Each query has one relevant answer, so Recall@K is equivalent to Hits@K.
        metrics[f"Recall@{k}"] = metrics[f"Hits@{k}"]
        metrics[f"NDCG@{k}"] = sum((1.0 / math.log2(rank + 1)) if hit else 0.0 for rank, hit in zip(ranks, hits)) / n
    return metrics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
