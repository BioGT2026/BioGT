import json
import logging
import math
import os
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from Model.sampling import (
    BalancedTargetEdgeBatcher,
    FullCoverageTargetEdgeBatcher,
    HardNegativePool,
    KnownPositiveIndex,
    QueryContextSampler,
    RelationCoverageEdgeBatcher,
    build_fixed_edge_batches,
    relation_candidate_nodes,
    sample_filtered_negatives,
    target_modality_for_etype,
)
from Model.unified_model import UnifiedDiseaseAssociationModel


_STRUCTURAL_PPR_CACHE = {}


def parse_int_list(value):
    parsed = [int(item.strip()) for item in str(value or "").split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("Expected a comma-separated list of positive integers")
    return parsed


def parse_target_sampling_weights(value):
    weights = {}
    for entry in str(value or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "Target sampling weights must use target:weight entries"
            )
        target_type, weight_text = entry.split(":", 1)
        target_type = target_type.strip()
        weight = float(weight_text)
        if not target_type or not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                "Target sampling weights require non-empty targets and "
                "finite positive values"
            )
        weights[target_type] = weight
    return weights


def parse_target_negative_counts(value):
    counts = {}
    for entry in str(value or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "Target negative counts must use target:count entries"
            )
        target_type, count_text = entry.split(":", 1)
        target_type = target_type.strip()
        count = int(count_text)
        if not target_type or count <= 0:
            raise ValueError(
                "Target negative counts require non-empty targets and "
                "positive integer counts"
            )
        counts[target_type] = count
    return counts


def parse_target_type_list(value):
    return sorted({
        item.strip()
        for item in str(value or "").split(",")
        if item.strip()
    })


def resolve_unified_graph_backend(requested_backend, device):
    requested_backend = str(requested_backend or "auto").lower()
    if requested_backend == "auto":
        return "torch" if device.type == "cuda" else "dgl"
    if requested_backend not in {"dgl", "torch"}:
        raise ValueError(
            "--unified-graph-backend must be auto, dgl, or torch"
        )
    return requested_backend


def should_run_full_ranking_validation(
    epoch_number,
    total_epochs,
    interval,
    enabled=True,
):
    if not enabled:
        return False
    epoch_number = int(epoch_number)
    return (
        epoch_number == 1
        or epoch_number % max(1, int(interval)) == 0
        or epoch_number == int(total_epochs)
    )


def _graph_for_model(graph, model, device):
    return (
        graph
        if getattr(model, "graph_backend", "dgl") == "torch"
        else graph.to(device)
    )


def _known_graph_for_evaluation(
    train_graph_cpu,
    validation_graph_cpu,
    evaluation_split,
):
    if evaluation_split == "validation":
        return train_graph_cpu
    if evaluation_split == "test":
        return dgl.merge([train_graph_cpu, validation_graph_cpu])
    raise ValueError("evaluation_split must be validation or test")


def select_target_etypes(graph, target_types, source_node_type="disease"):
    target_types = set(target_types)
    forward_etypes = [
        etype
        for etype in graph.canonical_etypes
        if graph.num_edges(etype) > 0
        and etype[0] == source_node_type
        and etype[2] in target_types
    ]
    if forward_etypes:
        return forward_etypes
    reverse_etypes = [
        etype
        for etype in graph.canonical_etypes
        if graph.num_edges(etype) > 0
        and etype[2] == source_node_type
        and etype[0] in target_types
    ]
    if reverse_etypes:
        logging.warning(
            "No forward %s-to-target relations were found; using reverse "
            "target-to-%s relations.",
            source_node_type,
            source_node_type,
        )
        return reverse_etypes
    logging.warning(
        "No %s-to-target relations were found; falling back to all relations "
        "touching the requested target modalities.",
        source_node_type,
    )
    return [
        etype
        for etype in graph.canonical_etypes
        if graph.num_edges(etype) > 0
        and (etype[0] in target_types or etype[2] in target_types)
    ]


def compute_joint_link_loss(
    pos_scores,
    neg_scores,
    negatives_per_positive,
    ranking_loss_weight,
    ranking_temperature,
    moe_aux_loss=None,
    moe_aux_loss_weight=0.01,
):
    losses = []
    bce_values = []
    ranking_values = []
    all_etypes = sorted(set(pos_scores) | set(neg_scores))
    for etype in all_etypes:
        positive = pos_scores.get(etype)
        negative = neg_scores.get(etype)
        if positive is None or negative is None:
            continue
        if positive.numel() == 0 or negative.numel() == 0:
            continue

        positive = positive.reshape(-1)
        negative = negative.reshape(-1)
        bce = 0.5 * (
            F.binary_cross_entropy_with_logits(
                positive,
                torch.ones_like(positive),
            )
            + F.binary_cross_entropy_with_logits(
                negative,
                torch.zeros_like(negative),
            )
        )
        bce_values.append(bce)

        expected_negative_count = positive.numel() * int(negatives_per_positive)
        if negative.numel() != expected_negative_count:
            raise ValueError(
                f"Negative grouping mismatch for {etype}: got {negative.numel()}, "
                f"expected {expected_negative_count}"
            )
        negative_groups = negative.reshape(
            positive.numel(),
            int(negatives_per_positive),
        )
        pairwise_difference = (
            positive.unsqueeze(1) - negative_groups
        ) / max(float(ranking_temperature), 1e-6)
        ranking = -F.logsigmoid(pairwise_difference).mean()
        ranking_values.append(ranking)
        losses.append(bce + float(ranking_loss_weight) * ranking)

    if not losses:
        reference = None
        for score_dict in (pos_scores, neg_scores):
            if score_dict:
                reference = next(iter(score_dict.values()))
                break
        device = reference.device if reference is not None else torch.device("cpu")
        total = torch.zeros((), device=device, requires_grad=True)
        bce_mean = total
        ranking_mean = total
    else:
        total = torch.stack(losses).mean()
        bce_mean = torch.stack(bce_values).mean()
        ranking_mean = torch.stack(ranking_values).mean()

    if moe_aux_loss is not None and float(moe_aux_loss_weight) > 0:
        total = total + float(moe_aux_loss_weight) * moe_aux_loss
    return total, {
        "total": float(total.detach().cpu()),
        "bce": float(bce_mean.detach().cpu()),
        "ranking": float(ranking_mean.detach().cpu()),
        "moe_aux": (
            float(moe_aux_loss.detach().cpu())
            if moe_aux_loss is not None
            else 0.0
        ),
    }


def _append_score(score_store, score_dict):
    for etype, values in score_dict.items():
        score_store.setdefault(etype, []).append(values.detach().cpu())


def _flatten_score_store(score_store):
    return {
        etype: torch.cat(parts)
        for etype, parts in score_store.items()
        if parts
    }


def compute_sampled_metrics(
    pos_scores,
    neg_scores,
    negatives_per_positive,
):
    all_positive = []
    all_negative = []
    reciprocal_ranks = []
    hits_at_10 = []
    recall_at_50 = []
    ndcg_at_50 = []
    per_etype = {}
    ranks_by_target = {}

    for etype in sorted(set(pos_scores) & set(neg_scores)):
        positive = pos_scores[etype].reshape(-1)
        negative = neg_scores[etype].reshape(
            positive.numel(),
            int(negatives_per_positive),
        )
        greater = (negative > positive.unsqueeze(1)).sum(dim=1).float()
        equal = torch.isclose(
            negative,
            positive.unsqueeze(1),
        ).sum(dim=1).float()
        ranks = 1.0 + greater + 0.5 * equal
        reciprocal = 1.0 / ranks
        reciprocal_ranks.append(reciprocal)
        hits_at_10.append((ranks <= 10).float())
        recall_at_50.append((ranks <= 50).float())
        ndcg = torch.where(
            ranks <= 50,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        )
        ndcg_at_50.append(ndcg)
        all_positive.append(positive)
        all_negative.append(negative.reshape(-1))
        target = next(
            (
                node_type
                for node_type in (etype[2], etype[0])
                if node_type in {"protein", "gene", "rna", "metabolite"}
            ),
            str(etype),
        )
        ranks_by_target.setdefault(target, []).append(ranks)
        per_etype[str(etype)] = {
            "mrr": float(reciprocal.mean()),
            "hits_at_10": float((ranks <= 10).float().mean()),
            "recall_at_50": float((ranks <= 50).float().mean()),
            "ndcg_at_50": float(ndcg.mean()),
        }

    if not all_positive:
        return {
            "auc_prc": 0.0,
            "auc_roc": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "mrr": 0.0,
            "hits_at_10": 0.0,
            "recall_at_50": 0.0,
            "ndcg_at_50": 0.0,
            "metrics_macro": {
                "mrr": 0.0,
                "hits_at_10": 0.0,
                "recall_at_50": 0.0,
                "ndcg_at_50": 0.0,
            },
            "per_target": {},
            "per_etype": per_etype,
        }

    positive = torch.cat(all_positive).sigmoid().numpy()
    negative = torch.cat(all_negative).sigmoid().numpy()
    labels = np.concatenate(
        [np.ones(positive.shape[0]), np.zeros(negative.shape[0])]
    )
    probabilities = np.concatenate([positive, negative])
    predictions = (probabilities >= 0.5).astype(np.int64)
    per_target = {}
    for target, rank_parts in sorted(ranks_by_target.items()):
        target_ranks = torch.cat(rank_parts)
        target_hits_10 = (target_ranks <= 10).float()
        target_recall_50 = (target_ranks <= 50).float()
        target_ndcg_50 = torch.where(
            target_ranks <= 50,
            1.0 / torch.log2(target_ranks + 1.0),
            torch.zeros_like(target_ranks),
        )
        per_target[target] = {
            "mrr": float((1.0 / target_ranks).mean()),
            "hits_at_10": float(target_hits_10.mean()),
            "recall_at_50": float(target_recall_50.mean()),
            "ndcg_at_50": float(target_ndcg_50.mean()),
            "ranking_num_triples": int(target_ranks.numel()),
        }
    macro_metric_names = ("mrr", "hits_at_10", "recall_at_50", "ndcg_at_50")
    metrics_macro = {
        metric: float(np.mean([
            target_metrics[metric]
            for target_metrics in per_target.values()
        ]))
        for metric in macro_metric_names
    }
    return {
        "auc_prc": float(average_precision_score(labels, probabilities)),
        "auc_roc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mrr": float(torch.cat(reciprocal_ranks).mean()),
        "hits_at_10": float(torch.cat(hits_at_10).mean()),
        "recall_at_50": float(torch.cat(recall_at_50).mean()),
        "ndcg_at_50": float(torch.cat(ndcg_at_50).mean()),
        "metrics_macro": metrics_macro,
        "per_target": per_target,
        "per_etype": per_etype,
    }


def _source_target_pairs(graph, source_type, target_type):
    pairs = set()
    for etype in graph.canonical_etypes:
        src_type, _, dst_type = etype
        if graph.num_edges(etype) == 0:
            continue
        src_nodes, dst_nodes = graph.edges(etype=etype)
        if src_type == source_type and dst_type == target_type:
            pairs.update(
                (int(src), int(dst))
                for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist())
            )
        elif src_type == target_type and dst_type == source_type:
            pairs.update(
                (int(dst), int(src))
                for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist())
            )
    return pairs


def _ranking_scoring_etype(model, graph, source_type, target_type):
    def matches(etype):
        return (
            (etype[0] == source_type and etype[2] == target_type)
            or (etype[0] == target_type and etype[2] == source_type)
        )

    active_etypes = sorted(
        etype
        for etype in getattr(model, "target_etypes", ())
        if matches(etype)
    )
    if active_etypes:
        return active_etypes[0]

    graph_etypes = sorted(
        etype
        for etype in graph.canonical_etypes
        if matches(etype) and graph.num_edges(etype) > 0
    )
    if not graph_etypes:
        raise ValueError(
            f"No scoring relation connects {source_type} and {target_type}"
        )
    return graph_etypes[0]


def _parse_structural_prior_grid(value):
    specs = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        alpha_text, weight_text = item.split(":", 1)
        alpha = float(alpha_text)
        neural_weight = float(weight_text)
        label = (
            f"a{alpha:g}_w{neural_weight:g}"
            .replace(".", "p")
            .replace("-", "m")
        )
        specs.append((label, alpha, neural_weight))
    return specs


def _relation_ppr_scores(
    graph,
    source_type,
    target_type,
    source_ids,
    alpha,
    max_iter,
    tol,
):
    source_ids = sorted(set(int(value) for value in source_ids))
    cache_key = (
        id(graph),
        source_type,
        target_type,
        tuple(source_ids),
        float(alpha),
        int(max_iter),
        float(tol),
    )
    cached = _STRUCTURAL_PPR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for --structural-prior-mode ppr"
        ) from exc

    source_count = int(graph.num_nodes(source_type))
    target_count = int(graph.num_nodes(target_type))
    relation_pairs = _source_target_pairs(
        graph,
        source_type,
        target_type,
    )
    if not relation_pairs:
        result = {
            source_id: torch.zeros(target_count, dtype=torch.float32)
            for source_id in source_ids
        }
        _STRUCTURAL_PPR_CACHE[cache_key] = result
        return result

    rows = []
    columns = []
    target_offset = source_count
    for source_id, target_id in sorted(relation_pairs):
        shifted_target = target_offset + target_id
        rows.extend((source_id, shifted_target))
        columns.extend((shifted_target, source_id))

    rows_array = np.asarray(rows, dtype=np.int64)
    columns_array = np.asarray(columns, dtype=np.int64)
    node_count = source_count + target_count
    degrees = np.bincount(rows_array, minlength=node_count).astype(np.float32)
    values = (
        1.0 / np.maximum(degrees[rows_array], 1.0)
    ).astype(np.float32)
    transition = csr_matrix(
        (values, (rows_array, columns_array)),
        shape=(node_count, node_count),
        dtype=np.float32,
    )

    result = {}
    batch_size = 512
    teleport = np.float32(1.0 - alpha)
    alpha_value = np.float32(alpha)
    for start in range(0, len(source_ids), batch_size):
        batch_sources = source_ids[start:start + batch_size]
        ranks = np.zeros(
            (len(batch_sources), node_count),
            dtype=np.float32,
        )
        row_ids = np.arange(len(batch_sources))
        source_array = np.asarray(batch_sources, dtype=np.int64)
        ranks[row_ids, source_array] = 1.0
        for _ in range(int(max_iter)):
            updated = transition.T.dot(ranks.T).T.astype(
                np.float32,
                copy=False,
            )
            updated *= alpha_value
            updated[row_ids, source_array] += teleport
            delta = float(np.abs(updated - ranks).sum(axis=1).max())
            ranks = updated
            if delta < float(tol):
                break
        target_scores = ranks[:, target_offset:]
        for row_index, source_id in enumerate(batch_sources):
            result[source_id] = torch.from_numpy(
                target_scores[row_index].copy()
            )

    _STRUCTURAL_PPR_CACHE[cache_key] = result
    return result


def _joint_relation_ppr_scores(
    graph,
    source_type,
    target_types,
    source_ids,
    alpha,
    max_iter,
    tol,
):
    source_ids = sorted(set(int(value) for value in source_ids))
    target_types = tuple(
        target_type
        for target_type in target_types
        if target_type in graph.ntypes and target_type != source_type
    )
    cache_key = (
        "joint",
        id(graph),
        source_type,
        target_types,
        tuple(source_ids),
        float(alpha),
        int(max_iter),
        float(tol),
    )
    cached = _STRUCTURAL_PPR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for --structural-prior-mode joint_ppr"
        ) from exc

    source_count = int(graph.num_nodes(source_type))
    offsets = {}
    node_count = source_count
    for target_type in target_types:
        target_count = int(graph.num_nodes(target_type))
        offsets[target_type] = (node_count, node_count + target_count)
        node_count += target_count

    rows = []
    columns = []
    for target_type in target_types:
        target_offset, _ = offsets[target_type]
        for source_id, target_id in sorted(_source_target_pairs(
            graph,
            source_type,
            target_type,
        )):
            shifted_target = target_offset + target_id
            rows.extend((source_id, shifted_target))
            columns.extend((shifted_target, source_id))

    if not rows:
        result = {
            source_id: {
                target_type: torch.zeros(
                    int(graph.num_nodes(target_type)),
                    dtype=torch.float32,
                )
                for target_type in target_types
            }
            for source_id in source_ids
        }
        _STRUCTURAL_PPR_CACHE[cache_key] = result
        return result

    rows_array = np.asarray(rows, dtype=np.int64)
    columns_array = np.asarray(columns, dtype=np.int64)
    degrees = np.bincount(rows_array, minlength=node_count).astype(np.float32)
    values = (
        1.0 / np.maximum(degrees[rows_array], 1.0)
    ).astype(np.float32)
    transition = csr_matrix(
        (values, (rows_array, columns_array)),
        shape=(node_count, node_count),
        dtype=np.float32,
    )

    result = {}
    batch_size = 512
    teleport = np.float32(1.0 - alpha)
    alpha_value = np.float32(alpha)
    for start in range(0, len(source_ids), batch_size):
        batch_sources = source_ids[start:start + batch_size]
        ranks = np.zeros(
            (len(batch_sources), node_count),
            dtype=np.float32,
        )
        row_ids = np.arange(len(batch_sources))
        source_array = np.asarray(batch_sources, dtype=np.int64)
        ranks[row_ids, source_array] = 1.0
        for _ in range(int(max_iter)):
            updated = transition.T.dot(ranks.T).T.astype(
                np.float32,
                copy=False,
            )
            updated *= alpha_value
            updated[row_ids, source_array] += teleport
            delta = float(np.abs(updated - ranks).sum(axis=1).max())
            ranks = updated
            if delta < float(tol):
                break

        for row_index, source_id in enumerate(batch_sources):
            result[source_id] = {
                target_type: torch.from_numpy(
                    ranks[row_index, start_index:end_index].copy()
                )
                for target_type, (
                    start_index,
                    end_index,
                ) in offsets.items()
            }

    _STRUCTURAL_PPR_CACHE[cache_key] = result
    return result


def _fuse_structural_scores(neural_scores, prior_scores, neural_weight):
    def standardize(scores):
        return (
            scores - scores.mean()
        ) / scores.std(unbiased=False).clamp_min(1e-8)

    return standardize(prior_scores) + float(neural_weight) * standardize(
        neural_scores
    )


def _filtered_ranks_for_source(
    candidate_scores,
    true_target_ids,
    known_target_ids,
    candidate_position,
):
    filtered_scores = candidate_scores.clone()
    for known_target_id in known_target_ids:
        position = candidate_position.get(known_target_id)
        if position is not None:
            filtered_scores[position] = float("-inf")

    ranks = []
    for true_target_id in true_target_ids:
        true_position = candidate_position[true_target_id]
        true_score = candidate_scores[true_position]
        greater = int((filtered_scores > true_score).sum().item())
        equal = int(
            torch.isclose(filtered_scores, true_score).sum().item()
        ) + int(true_target_id in known_target_ids)
        ranks.append(1.0 + greater + 0.5 * max(0, equal - 1))
    return ranks


def _ranking_metrics_from_ranks(ranks):
    ranks = torch.as_tensor(ranks, dtype=torch.float64)
    if ranks.numel() == 0:
        return {
            "mrr": 0.0,
            "hits_at_10": 0.0,
            "recall_at_50": 0.0,
            "ndcg_at_50": 0.0,
            "ranking_num_triples": 0,
        }
    ndcg = torch.where(
        ranks <= 50,
        1.0 / torch.log2(ranks + 1.0),
        torch.zeros_like(ranks),
    )
    return {
        "mrr": float((1.0 / ranks).mean()),
        "hits_at_10": float((ranks <= 10).double().mean()),
        "recall_at_50": float((ranks <= 50).double().mean()),
        "ndcg_at_50": float(ndcg.mean()),
        "ranking_num_triples": int(ranks.numel()),
    }


def evaluate_full_filtered_ranking(
    model,
    candidate_graph_cpu,
    message_graph_cpu,
    eval_graph_cpu,
    known_graph_cpu,
    target_types,
    source_type,
    device,
    candidate_batch_size=4096,
    max_edges=0,
    structural_prior_mode="none",
    structural_prior_alpha=0.85,
    structural_prior_max_iter=50,
    structural_prior_tol=1e-8,
    structural_prior_neural_weight=0.0,
    structural_prior_grid="",
):
    """Rank every held-out disease-target edge against its relation candidate set."""

    if source_type not in candidate_graph_cpu.ntypes:
        raise ValueError(
            f"Full-ranking source node type '{source_type}' is absent"
        )
    batch_size = max(1, int(candidate_batch_size))
    edge_cap = max(0, int(max_edges or 0))
    was_training = model.training
    model.eval()
    graph_for_model = _graph_for_model(
        message_graph_cpu,
        model,
        device,
    )

    with torch.no_grad():
        branches = model.encode_branches(graph_for_model)
        per_target = {}
        all_ranks = []
        candidate_counts = {}
        scoring_etypes = {}
        structural_prior_modes = {"ppr", "joint_ppr"}
        if structural_prior_mode not in {"none", *structural_prior_modes}:
            raise ValueError(
                f"Unsupported structural prior mode: {structural_prior_mode}"
            )
        grid_specs = (
            _parse_structural_prior_grid(structural_prior_grid)
            if structural_prior_mode in structural_prior_modes
            else []
        )
        grid_all_ranks = {label: [] for label, _, _ in grid_specs}
        grid_per_target = {label: {} for label, _, _ in grid_specs}
        joint_structural_scores_by_alpha = {}
        if structural_prior_mode == "joint_ppr":
            required_alphas = {
                float(structural_prior_alpha),
                *(float(alpha) for _, alpha, _ in grid_specs),
            }
            joint_source_ids = set()
            for target_type in target_types:
                target_pairs = _source_target_pairs(
                    eval_graph_cpu,
                    source_type,
                    target_type,
                )
                if max_edges and max_edges > 0:
                    target_pairs = target_pairs[:max_edges]
                joint_source_ids.update(
                    source_id
                    for source_id, _ in target_pairs
                )
            for alpha in sorted(required_alphas):
                joint_structural_scores_by_alpha[
                    alpha
                ] = _joint_relation_ppr_scores(
                    message_graph_cpu,
                    source_type,
                    target_types,
                    joint_source_ids,
                    alpha,
                    structural_prior_max_iter,
                    structural_prior_tol,
                )
        for target_type in target_types:
            if target_type not in candidate_graph_cpu.ntypes:
                per_target[target_type] = _ranking_metrics_from_ranks([])
                candidate_counts[target_type] = 0
                continue

            candidate_pairs = _source_target_pairs(
                candidate_graph_cpu,
                source_type,
                target_type,
            )
            candidate_ids = sorted(
                {target_id for _, target_id in candidate_pairs}
            )
            candidate_counts[target_type] = len(candidate_ids)
            eval_pairs = sorted(
                _source_target_pairs(
                    eval_graph_cpu,
                    source_type,
                    target_type,
                )
            )
            if edge_cap > 0:
                eval_pairs = eval_pairs[:edge_cap]
            known_pairs = _source_target_pairs(
                known_graph_cpu,
                source_type,
                target_type,
            )
            known_by_source = {}
            for source_id, target_id in known_pairs:
                known_by_source.setdefault(source_id, set()).add(target_id)

            if not candidate_ids or not eval_pairs:
                per_target[target_type] = _ranking_metrics_from_ranks([])
                continue

            embeddings = model.modality_embeddings(
                branches,
                target_type,
                node_types={source_type, target_type},
            )
            candidate_tensor = torch.tensor(
                candidate_ids,
                dtype=torch.long,
                device=device,
            )
            candidate_position = {
                target_id: position
                for position, target_id in enumerate(candidate_ids)
            }
            target_embeddings = embeddings[target_type]
            source_embeddings = embeddings[source_type]
            independent_mode = getattr(model, "scorer_mode", None)
            use_global_bilinear = (
                independent_mode == "hybrid_global_bilinear"
            )
            use_kge = independent_mode == "hybrid_kge"
            use_biogt_path = (
                getattr(
                    model,
                    "biogt_path_propagation_hops",
                    0,
                )
                > 0
            )
            if use_global_bilinear:
                global_target_embeddings = branches["input"][target_type]
                global_source_embeddings = branches["input"][source_type]
            scoring_etype = _ranking_scoring_etype(
                model,
                candidate_graph_cpu,
                source_type,
                target_type,
            )
            scoring_etypes[target_type] = list(scoring_etype)
            ranks = []

            pairs_by_source = {}
            for source_id, target_id in eval_pairs:
                if (
                    source_id < source_embeddings.shape[0]
                    and target_id in candidate_position
                ):
                    pairs_by_source.setdefault(source_id, []).append(target_id)

            structural_scores_by_alpha = {}
            if structural_prior_mode == "ppr":
                required_alphas = {
                    float(structural_prior_alpha),
                    *(float(alpha) for _, alpha, _ in grid_specs),
                }
                for alpha in sorted(required_alphas):
                    structural_scores_by_alpha[alpha] = _relation_ppr_scores(
                        message_graph_cpu,
                        source_type,
                        target_type,
                        pairs_by_source,
                        alpha,
                        structural_prior_max_iter,
                        structural_prior_tol,
                    )

            grid_target_ranks = {
                label: []
                for label, _, _ in grid_specs
            }
            for source_id, true_target_ids in sorted(
                pairs_by_source.items()
            ):
                source_embedding = source_embeddings[source_id].unsqueeze(0)
                score_chunks = []
                kge_candidate_scores = None
                if use_kge:
                    kge_candidate_scores = (
                        model.score_kge_candidate_matrix(
                            scoring_etype,
                            torch.tensor([source_id]),
                            candidate_tensor.detach().cpu(),
                        )[0]
                    )
                biogt_path_candidate_scores = None
                if use_biogt_path:
                    biogt_path_candidate_scores = (
                        model.score_biogt_path_candidate_matrix(
                            scoring_etype,
                            torch.tensor([source_id]),
                            candidate_tensor.detach().cpu(),
                            relbf_query_features=source_embedding,
                            relbf_candidate_features=(
                                target_embeddings.index_select(
                                    0,
                                    candidate_tensor,
                                )
                            ),
                        )[0]
                    )
                for start in range(0, candidate_tensor.numel(), batch_size):
                    target_ids = candidate_tensor[start:start + batch_size]
                    target_batch = target_embeddings.index_select(
                        0,
                        target_ids,
                    )
                    source_batch = source_embedding.expand(
                        target_batch.shape[0],
                        -1,
                    )
                    global_pair_features = None
                    if scoring_etype[0] == source_type:
                        pair_features = torch.cat(
                            [source_batch, target_batch],
                            dim=-1,
                        )
                        if use_global_bilinear:
                            global_pair_features = torch.cat(
                                [
                                    global_source_embeddings[
                                        source_id
                                    ].unsqueeze(0).expand(
                                        target_ids.shape[0],
                                        -1,
                                    ),
                                    global_target_embeddings.index_select(
                                        0,
                                        target_ids,
                                    ),
                                ],
                                dim=-1,
                            )
                    else:
                        pair_features = torch.cat(
                            [target_batch, source_batch],
                            dim=-1,
                        )
                        if use_global_bilinear:
                            global_pair_features = torch.cat(
                                [
                                    global_target_embeddings.index_select(
                                        0,
                                        target_ids,
                                    ),
                                    global_source_embeddings[
                                        source_id
                                    ].unsqueeze(0).expand(
                                        target_ids.shape[0],
                                        -1,
                                    ),
                                ],
                                dim=-1,
                            )
                    independent_scores = None
                    if use_kge:
                        independent_scores = kge_candidate_scores[
                            start:start + target_ids.shape[0]
                        ]
                    if use_global_bilinear or use_kge:
                        scores, _ = model._predict_pairs(
                            target_type,
                            pair_features,
                            global_pair_features=global_pair_features,
                            independent_scores=independent_scores,
                        )
                    else:
                        scores, _ = model._predict_pairs(
                            target_type,
                            pair_features,
                        )
                    if use_biogt_path:
                        scores = (
                            scores
                            + biogt_path_candidate_scores[
                                start:start + target_ids.shape[0]
                            ]
                        )
                    score_chunks.append(scores.detach().cpu())
                neural_scores = torch.cat(score_chunks)
                scores = neural_scores
                grid_scores = {}
                if (
                    structural_scores_by_alpha
                    or joint_structural_scores_by_alpha
                ):
                    candidate_ids_cpu = candidate_tensor.detach().cpu()
                    if joint_structural_scores_by_alpha:
                        selected_prior = joint_structural_scores_by_alpha[
                            float(structural_prior_alpha)
                        ][source_id][target_type].index_select(
                            0,
                            candidate_ids_cpu,
                        )
                    else:
                        selected_prior = structural_scores_by_alpha[
                            float(structural_prior_alpha)
                        ][source_id].index_select(0, candidate_ids_cpu)
                    scores = _fuse_structural_scores(
                        neural_scores,
                        selected_prior,
                        structural_prior_neural_weight,
                    )
                    for label, alpha, neural_weight in grid_specs:
                        if joint_structural_scores_by_alpha:
                            prior_scores = joint_structural_scores_by_alpha[
                                float(alpha)
                            ][source_id][target_type].index_select(
                                0,
                                candidate_ids_cpu,
                            )
                        else:
                            prior_scores = structural_scores_by_alpha[
                                float(alpha)
                            ][source_id].index_select(0, candidate_ids_cpu)
                        grid_scores[label] = _fuse_structural_scores(
                            neural_scores,
                            prior_scores,
                            neural_weight,
                        )

                known_targets = known_by_source.get(source_id, set())
                ranks.extend(_filtered_ranks_for_source(
                    scores,
                    true_target_ids,
                    known_targets,
                    candidate_position,
                ))
                for label, candidate_scores in grid_scores.items():
                    grid_target_ranks[label].extend(
                        _filtered_ranks_for_source(
                            candidate_scores,
                            true_target_ids,
                            known_targets,
                            candidate_position,
                        )
                    )

            per_target[target_type] = _ranking_metrics_from_ranks(ranks)
            all_ranks.extend(ranks)
            for label, _, _ in grid_specs:
                target_grid_metrics = _ranking_metrics_from_ranks(
                    grid_target_ranks[label]
                )
                grid_per_target[label][target_type] = target_grid_metrics
                grid_all_ranks[label].extend(grid_target_ranks[label])

    if was_training:
        model.train()

    metrics = _ranking_metrics_from_ranks(all_ranks)
    macro_names = (
        "mrr",
        "hits_at_10",
        "recall_at_50",
        "ndcg_at_50",
    )
    populated_targets = [
        target_metrics
        for target_metrics in per_target.values()
        if target_metrics["ranking_num_triples"] > 0
    ]
    metrics_macro = {
        metric: (
            float(np.mean([
                target_metrics[metric]
                for target_metrics in populated_targets
            ]))
            if populated_targets
            else 0.0
        )
        for metric in macro_names
    }
    stable_targets = {
        target: target_metrics
        for target, target_metrics in per_target.items()
        if target_metrics["ranking_num_triples"] >= 10
    }
    metrics_macro_stable = {
        metric: (
            float(np.mean([
                target_metrics[metric]
                for target_metrics in stable_targets.values()
            ]))
            if stable_targets
            else 0.0
        )
        for metric in macro_names
    }
    structural_grid_payload = {}
    for label, alpha, neural_weight in grid_specs:
        target_metrics = grid_per_target[label]
        populated = [
            values
            for values in target_metrics.values()
            if values["ranking_num_triples"] > 0
        ]
        stable = [
            values
            for values in target_metrics.values()
            if values["ranking_num_triples"] >= 10
        ]
        structural_grid_payload[label] = {
            "alpha": alpha,
            "neural_weight": neural_weight,
            "metrics": _ranking_metrics_from_ranks(
                grid_all_ranks[label]
            ),
            "metrics_macro": {
                name: (
                    float(np.mean([values[name] for values in populated]))
                    if populated
                    else 0.0
                )
                for name in macro_names
            },
            "metrics_macro_stable": {
                name: (
                    float(np.mean([values[name] for values in stable]))
                    if stable
                    else 0.0
                )
                for name in macro_names
            },
            "per_target": target_metrics,
        }
    return {
        "protocol": "full_filtered_relation_candidate_ranking",
        "source_type": source_type,
        "tie_policy": "average",
        "message_graph": "train",
        "candidate_scope": "relation_specific_target_type",
        "candidate_counts": candidate_counts,
        "scoring_etypes": scoring_etypes,
        "structural_prior": {
            "mode": structural_prior_mode,
            "alpha": float(structural_prior_alpha),
            "neural_weight": float(structural_prior_neural_weight),
        },
        "structural_prior_grid": structural_grid_payload,
        "metrics": metrics,
        "metrics_macro": metrics_macro,
        "metrics_macro_stable": metrics_macro_stable,
        "stable_macro_targets": sorted(stable_targets),
        "per_target": per_target,
    }


def evaluate_fixed_batches(
    model,
    message_graph_cpu,
    fixed_batches,
    context_sampler,
    device,
    ranking_loss_weight,
    ranking_temperature,
    context_seed,
):
    was_training = model.training
    model.eval()
    positive_store = {}
    negative_store = {}
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(fixed_batches):
            context_graph, pos_graph, neg_graph = context_sampler.build_batch_graphs(
                message_graph_cpu,
                batch.etype,
                batch.pos_src,
                batch.pos_dst,
                batch.neg_src,
                batch.neg_dst,
                random_seed=int(context_seed) + batch_index,
            )
            context_graph = _graph_for_model(
                context_graph,
                model,
                device,
            )
            pos_graph = _graph_for_model(pos_graph, model, device)
            neg_graph = _graph_for_model(neg_graph, model, device)
            pos_scores, neg_scores = model(
                context_graph,
                None,
                pos_graph,
                neg_graph,
                target_etypes_override={batch.etype},
            )
            loss, _ = compute_joint_link_loss(
                pos_scores,
                neg_scores,
                batch.negatives_per_positive,
                ranking_loss_weight,
                ranking_temperature,
                moe_aux_loss=None,
                moe_aux_loss_weight=0.0,
            )
            losses.append(float(loss.detach().cpu()))
            _append_score(positive_store, pos_scores)
            _append_score(negative_store, neg_scores)

    if was_training:
        model.train()
    positive_scores = _flatten_score_store(positive_store)
    negative_scores = _flatten_score_store(negative_store)
    negatives_per_positive = (
        fixed_batches[0].negatives_per_positive if fixed_batches else 1
    )
    metrics = compute_sampled_metrics(
        positive_scores,
        negative_scores,
        negatives_per_positive,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else math.inf
    return metrics


def _graph_creator_from_args(args, device):
    from Model.modelData import DGLGraphCreator

    category_split_map = {}
    if args.split_category_map:
        for entry in args.split_category_map.split(","):
            if ":" in entry:
                key, value = entry.split(":", 1)
                category_split_map[key.strip()] = value.strip()
    exclude_types = [
        item.strip()
        for item in str(args.split_exclude_types or "").split(",")
        if item.strip()
    ]
    val_fold_index = (
        args.split_kfold_val_fold
        if args.split_kfold_val_fold is not None
        and args.split_kfold_val_fold >= 0
        else None
    )
    return DGLGraphCreator(
        node_csv_path=os.path.abspath(args.nodes_csv),
        edge_csv_path=os.path.abspath(args.edges_csv),
        embed_size=args.default_embed_size,
        val_size=args.val_split,
        test_size=args.test_split,
        random_state=args.seed,
        freeze_pretrained=args.freeze_pretrained,
        graph_cache_path=args.graph_cache_path,
        feature_cache_path=args.feature_cache_path,
        device=device,
        local_model_base_path=args.local_model_path,
        generate_embeddings=not args.disable_pretrained_feature_generation,
        use_sequence_features=not args.disable_sequence_features,
        merged_sequence_features=args.merged_sequence_features,
        split_strategy=args.split_strategy,
        split_category_map=category_split_map,
        split_num_folds=args.split_kfolds,
        split_fold_index=args.split_kfold_index,
        split_val_fold_index=val_fold_index,
        split_disease_node_type=args.split_disease_node_type,
        split_exclude_types=exclude_types,
        split_exclude_action=args.split_exclude_action,
        split_specific_ids_path=args.split_specific_ids_path,
        split_external_edges_path=args.split_external_edges_path,
        split_target_input=args.eval_input_type,
        split_target_output=args.eval_output_types,
        filter_isolated_nodes=not args.disable_isolated_node_filter,
        regenerate_features_after_filter=args.regenerate_features_after_filter,
    )


def _node_feature_tensors(node_features, graph):
    result = {}
    for node_type in graph.ntypes:
        feature = (
            node_features[node_type]
            if node_features is not None and node_type in node_features
            else None
        )
        if isinstance(feature, torch.nn.Embedding):
            result[node_type] = feature.weight.detach().cpu().clone()
        elif isinstance(feature, torch.Tensor):
            result[node_type] = feature.detach().cpu().clone()
    return result


def _checkpoint_payload(model, epoch, metrics, args):
    serializable_args = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    return {
        "architecture": "unified_sparse_v1",
        "epoch": int(epoch),
        "metrics": metrics,
        "model_config": model.export_config(),
        "training_args": serializable_args,
        "model_state_dict": model.state_dict(),
    }


def _save_checkpoint(path, model, epoch, metrics, args):
    if not path:
        return
    save_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        _checkpoint_payload(model, epoch, metrics, args),
        path,
    )


def _load_checkpoint(
    path,
    model,
    device,
    initialize_target_disease_from_shared=False,
    initialize_candidate_bias_from_checkpoint=False,
    initialize_target_residual_from_checkpoint=False,
    initialize_path_propagation_from_checkpoint=False,
    initialize_biogt_path_from_checkpoint=False,
    initialize_biogt_neural_path_from_checkpoint=False,
    initialize_biogt_relbf_from_checkpoint=False,
    initialize_biogt_relbf_jk_from_checkpoint=False,
    initialize_biogt_relbf_edge_attention_from_checkpoint=False,
    initialize_biogt_relbf_deep_from_checkpoint=False,
    initialize_biogt_relbf_indicator_from_checkpoint=False,
    initialize_biogt_relbf_indicator_support_gating_from_checkpoint=False,
    initialize_biogt_relbf_indicator_semantic_from_checkpoint=False,
    initialize_biogt_relbf_query_relation_adapter_from_checkpoint=False,
    initialize_biogt_relbf_indicator_query_boundary_from_checkpoint=False,
    initialize_biogt_relbf_indicator_multirule_from_checkpoint=False,
    initialize_biogt_relbf_indicator_relation_transition_from_checkpoint=False,
    initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint=False,
    initialize_biogt_relbf_path_layer_attention_from_checkpoint=False,
    initialize_biogt_relbf_path_token_transformer_from_checkpoint=False,
    initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint=False,
    initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint=False,
    initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint=False,
    initialize_biogt_relbf_boundary_exchange_from_checkpoint=False,
    initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint=False,
    initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint=False,
    initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint=False,
    initialize_biogt_relbf_multi_depth_score_router_from_checkpoint=False,
    initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint=False,
    initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint=False,
    initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint=False,
):
    relbf_migrations = sum(bool(value) for value in (
        initialize_biogt_relbf_from_checkpoint,
        initialize_biogt_relbf_jk_from_checkpoint,
        initialize_biogt_relbf_edge_attention_from_checkpoint,
        initialize_biogt_relbf_deep_from_checkpoint,
        initialize_biogt_relbf_indicator_from_checkpoint,
        initialize_biogt_relbf_indicator_support_gating_from_checkpoint,
        initialize_biogt_relbf_indicator_semantic_from_checkpoint,
        initialize_biogt_relbf_query_relation_adapter_from_checkpoint,
        initialize_biogt_relbf_indicator_query_boundary_from_checkpoint,
        initialize_biogt_relbf_indicator_multirule_from_checkpoint,
        initialize_biogt_relbf_indicator_relation_transition_from_checkpoint,
        initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint,
        initialize_biogt_relbf_path_layer_attention_from_checkpoint,
        initialize_biogt_relbf_path_token_transformer_from_checkpoint,
        initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint,
        initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint,
        initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint,
        initialize_biogt_relbf_boundary_exchange_from_checkpoint,
        initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint,
        initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint,
        initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint,
        initialize_biogt_relbf_multi_depth_score_router_from_checkpoint,
        initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint,
        initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint,
        initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint,
    ))
    if relbf_migrations > 1:
        raise ValueError("BioGT RelBF migrations must be separate")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        architecture = payload.get("architecture")
        if architecture not in {None, "unified_sparse_v1"}:
            raise ValueError(
                f"Checkpoint architecture '{architecture}' is not supported "
                "by the unified training engine"
            )
        state_dict = payload["model_state_dict"]
        if initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint:
            parent_config = payload.get("model_config", {})
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get(
                    "biogt_relbf_parent_top10_boundary_comparator", False
                )
                or not parent_config.get(
                    "biogt_relbf_boundary_window_rank_slot_expert", False
                )
                or parent_config.get("biogt_relbf_boundary_exchange", False)
                or parent_config.get(
                    "biogt_relbf_challenger_boundary_promotion", False
                )
                or parent_config.get("scorer_mode") not in {
                    "concat_mlp", "interaction_mlp"
                }
            ):
                raise ValueError(
                    "BioGT challenger-promotion migration requires an enabled "
                    "gene9 parent without gene10 or an existing gene11 branch"
                )
        if initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint:
            parent_config = payload.get("model_config", {})
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get(
                    "biogt_relbf_parent_top10_boundary_comparator", False
                )
                or not parent_config.get(
                    "biogt_relbf_boundary_window_rank_slot_expert", False
                )
                or parent_config.get("biogt_relbf_boundary_exchange", False)
                or parent_config.get(
                    "biogt_relbf_challenger_boundary_promotion", False
                )
                or parent_config.get(
                    "biogt_relbf_gatekeeper_challenger_paired_margin", False
                )
                or parent_config.get("scorer_mode") not in {
                    "concat_mlp", "interaction_mlp"
                }
            ):
                raise ValueError(
                    "BioGT paired-margin migration requires an enabled gene9 "
                    "parent without gene10, gene11, or an existing gene12 branch"
                )
        if initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint:
            parent_config = payload.get("model_config", {})
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
                or not parent_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
                or parent_config.get("biogt_relbf_boundary_exchange", False)
                or parent_config.get("biogt_relbf_challenger_boundary_promotion", False)
                or parent_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
                or parent_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
                or parent_config.get("scorer_mode") not in {"concat_mlp", "interaction_mlp"}
            ):
                raise ValueError(
                    "BioGT ordinal-adapter migration requires an enabled gene9 parent only"
                )
        if initialize_biogt_relbf_multi_depth_score_router_from_checkpoint:
            parent_config = payload.get("model_config", {})
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
                or not parent_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
                or parent_config.get("biogt_relbf_boundary_exchange", False)
                or parent_config.get("biogt_relbf_challenger_boundary_promotion", False)
                or parent_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
                or parent_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
                or parent_config.get("biogt_relbf_multi_depth_score_router", False)
                or parent_config.get("scorer_mode") not in {"concat_mlp", "interaction_mlp"}
            ):
                raise ValueError(
                    "BioGT multi-depth score-router migration requires gene9"
                )
        if initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint:
            parent_config = payload.get("model_config", {})
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
                or not parent_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
                or parent_config.get("biogt_relbf_boundary_exchange", False)
                or parent_config.get("biogt_relbf_challenger_boundary_promotion", False)
                or parent_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
                or parent_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
                or parent_config.get("biogt_relbf_multi_depth_score_router", False)
                or parent_config.get(
                    "biogt_relbf_dual_stream_trajectory_agreement_calibrator", False
                )
                or parent_config.get("scorer_mode") not in {"concat_mlp", "interaction_mlp"}
            ):
                raise ValueError(
                    "BioGT dual-stream trajectory-agreement migration requires gene9"
                )
        if initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint:
            parent_config = payload.get("model_config", {})
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
                or not parent_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
                or parent_config.get("biogt_relbf_boundary_exchange", False)
                or parent_config.get("biogt_relbf_challenger_boundary_promotion", False)
                or parent_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
                or parent_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
                or parent_config.get("biogt_relbf_multi_depth_score_router", False)
                or parent_config.get("biogt_relbf_dual_stream_trajectory_agreement_calibrator", False)
                or parent_config.get("biogt_relbf_multi_anchor_rank_landmark_calibrator", False)
                or parent_config.get("scorer_mode") not in {"concat_mlp", "interaction_mlp"}
            ):
                raise ValueError(
                    "BioGT multi-anchor rank-landmark migration requires gene9"
                )
        if initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint:
            parent_config = payload.get("model_config", {})
            forbidden = (
                "biogt_relbf_boundary_exchange",
                "biogt_relbf_challenger_boundary_promotion",
                "biogt_relbf_gatekeeper_challenger_paired_margin",
                "biogt_relbf_boundary_prototype_ordinal_adapter",
                "biogt_relbf_multi_depth_score_router",
                "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
                "biogt_relbf_multi_anchor_rank_landmark_calibrator",
            )
            if (
                parent_config.get("biogt_relbf_dim", 0) <= 0
                or not parent_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
                or not parent_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
                or any(parent_config.get(name, False) for name in forbidden)
                or parent_config.get("biogt_relbf_query_token_top64_candidate_set_transformer", False)
                or parent_config.get("scorer_mode") not in {"concat_mlp", "interaction_mlp"}
            ):
                raise ValueError(
                    "BioGT query-token top64 candidate-set migration requires gene9"
                )
        if (
            initialize_target_disease_from_shared
            or initialize_candidate_bias_from_checkpoint
            or initialize_target_residual_from_checkpoint
            or initialize_path_propagation_from_checkpoint
            or initialize_biogt_path_from_checkpoint
            or initialize_biogt_neural_path_from_checkpoint
            or initialize_biogt_relbf_from_checkpoint
            or initialize_biogt_relbf_jk_from_checkpoint
            or initialize_biogt_relbf_edge_attention_from_checkpoint
            or initialize_biogt_relbf_deep_from_checkpoint
            or initialize_biogt_relbf_indicator_from_checkpoint
            or initialize_biogt_relbf_indicator_support_gating_from_checkpoint
            or initialize_biogt_relbf_indicator_semantic_from_checkpoint
            or initialize_biogt_relbf_query_relation_adapter_from_checkpoint
            or initialize_biogt_relbf_indicator_query_boundary_from_checkpoint
            or initialize_biogt_relbf_indicator_multirule_from_checkpoint
            or initialize_biogt_relbf_indicator_relation_transition_from_checkpoint
            or initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint
            or initialize_biogt_relbf_path_layer_attention_from_checkpoint
            or initialize_biogt_relbf_path_token_transformer_from_checkpoint
            or initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint
            or initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint
            or initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint
            or initialize_biogt_relbf_boundary_exchange_from_checkpoint
            or initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint
            or initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint
            or initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint
            or initialize_biogt_relbf_multi_depth_score_router_from_checkpoint
            or initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint
            or initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint
            or initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint
        ):
            incompatible = model.load_state_dict(
                state_dict,
                strict=False,
            )
            expected_missing = set()
            if initialize_target_disease_from_shared:
                expected_missing.update({
                    f"kge_target_disease_embeddings.{target_type}.weight"
                    for target_type in model.kge_target_disease_embeddings
                })
            if initialize_candidate_bias_from_checkpoint:
                expected_missing.update({
                    f"kge_candidate_biases.{target_type}.weight"
                    for target_type in model.kge_candidate_biases
                })
            if initialize_target_residual_from_checkpoint:
                expected_missing.update({
                    f"kge_target_residual_left_factors.{target_type}"
                    for target_type in (
                        model.kge_target_residual_left_factors
                    )
                })
                expected_missing.update({
                    f"kge_target_residual_right_factors.{target_type}"
                    for target_type in (
                        model.kge_target_residual_right_factors
                    )
                })
            if initialize_path_propagation_from_checkpoint:
                expected_missing.update({
                    f"kge_path_query_gates.{target_type}.weight"
                    for target_type in model.kge_path_query_gates
                })
                expected_missing.update({
                    f"kge_path_query_gates.{target_type}.bias"
                    for target_type in model.kge_path_query_gates
                })
            if initialize_biogt_path_from_checkpoint:
                expected_missing.update({
                    f"biogt_path_query_gates.{target_type}.weight"
                    for target_type in model.biogt_path_query_gates
                })
                expected_missing.update({
                    f"biogt_path_query_gates.{target_type}.bias"
                    for target_type in model.biogt_path_query_gates
                })
            if initialize_biogt_neural_path_from_checkpoint:
                expected_missing.update({
                    f"biogt_path_query_gates.{target_type}.weight"
                    for target_type in model.biogt_path_query_gates
                })
                expected_missing.update({
                    f"biogt_path_query_gates.{target_type}.bias"
                    for target_type in model.biogt_path_query_gates
                })
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith("biogt_neural_path_")
                })
            if initialize_biogt_relbf_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith("biogt_relbf_")
                })
            if initialize_biogt_relbf_jk_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith("biogt_relbf_jk_readouts.")
                })
            if initialize_biogt_relbf_edge_attention_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_edge_attention_queries."
                    )
                })
            if initialize_biogt_relbf_deep_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith("biogt_relbf_deep_")
                })
            if initialize_biogt_relbf_indicator_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith("biogt_relbf_indicator_")
                })
            if initialize_biogt_relbf_indicator_support_gating_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_indicator_support_gates."
                    )
                })
            if initialize_biogt_relbf_indicator_semantic_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_indicator_semantic_readouts."
                    )
                })
            if initialize_biogt_relbf_query_relation_adapter_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_query_relation_adapters."
                    )
                })
            if initialize_biogt_relbf_indicator_query_boundary_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_indicator_query_boundary_adapters."
                    )
                })
            if initialize_biogt_relbf_indicator_multirule_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_indicator_multirule_residuals."
                    )
                })
            if initialize_biogt_relbf_indicator_relation_transition_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_indicator_relation_transition_residuals."
                    )
                })
            if initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_indicator_query_local_relation_transition_residuals."
                    )
                })
            if initialize_biogt_relbf_path_layer_attention_from_checkpoint:
                expected_missing.update({
                    key
                    for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_path_layer_attention_residuals."
                    )
                })
            if initialize_biogt_relbf_path_token_transformer_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith("biogt_relbf_path_token_transformer_residuals.")
                })
            if initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_candidate_competition_boundary_residuals."
                    )
                })
            if initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_parent_top10_boundary_comparator_residuals."
                    )
                })
            if initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_boundary_window_rank_slot_expert_residuals."
                    )
                })
            if initialize_biogt_relbf_boundary_exchange_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith("biogt_relbf_boundary_exchange_residuals.")
                })
            if initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_challenger_boundary_promotion_residuals."
                    )
                })
            if initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_gatekeeper_challenger_paired_margin_residuals."
                    )
                })
            if initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_boundary_prototype_ordinal_adapter_residuals."
                    )
                })
            if initialize_biogt_relbf_multi_depth_score_router_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_multi_depth_score_router_residuals."
                    )
                })
            if initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_dual_stream_trajectory_agreement_calibrator_residuals."
                    )
                })
            if initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_multi_anchor_rank_landmark_calibrator_residuals."
                    )
                })
            if initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint:
                expected_missing.update({
                    key for key in model.state_dict()
                    if key.startswith(
                        "biogt_relbf_query_token_top64_candidate_set_transformer_residuals."
                    )
                })
            if (
                set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys
            ):
                raise ValueError(
                    "Unsafe KGE checkpoint migration or unified migration: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            if initialize_target_disease_from_shared:
                initialized_count = (
                    model.initialize_target_disease_embeddings_from_shared()
                )
                logging.info(
                    "Initialized %d target-specific disease KGE tables from "
                    "the loaded shared disease checkpoint table.",
                    initialized_count,
                )
            if initialize_candidate_bias_from_checkpoint:
                initialized_count = (
                    model.initialize_candidate_biases_from_checkpoint()
                )
                logging.info(
                    "Initialized %d target candidate-bias tables to zero.",
                    initialized_count,
                )
            if initialize_target_residual_from_checkpoint:
                initialized_count = (
                    model.initialize_target_residual_from_checkpoint()
                )
                logging.info(
                    "Initialized %d target low-rank residuals with "
                    "zero-valued scores.",
                    initialized_count,
                )
            if initialize_path_propagation_from_checkpoint:
                initialized_count = (
                    model.initialize_path_propagation_from_checkpoint()
                )
                logging.info(
                    "Initialized %d target path-propagation gates to zero.",
                    initialized_count,
                )
            if (
                initialize_biogt_path_from_checkpoint
                or initialize_biogt_neural_path_from_checkpoint
            ):
                initialized_count = (
                    model.initialize_biogt_path_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT path gates to zero.",
                    initialized_count,
                )
            if initialize_biogt_neural_path_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_neural_path_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT neural diffusion candidate "
                    "projections to zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT RelBF readouts to zero "
                    "contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_jk_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_jk_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT RelBF JK residual readouts "
                    "to zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_edge_attention_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_edge_attention_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT RelBF query-conditioned "
                    "relation-aware edge-attention projections to exact "
                    "identity contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_deep_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_deep_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT RelBF deep-refinement "
                    "residual readouts to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT relation-only indicator "
                    "residual readouts to zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_support_gating_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_support_gating_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT indicator per-support layer "
                    "gates to exact identity contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_semantic_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_semantic_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT indicator semantic residual "
                    "readouts to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_query_relation_adapter_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_query_relation_adapter_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT RelBF query-conditioned "
                    "low-rank relation adapters to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_query_boundary_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_query_boundary_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT indicator query-boundary "
                    "adapters to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_multirule_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_multirule_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT indicator multi-rule "
                    "readouts to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_relation_transition_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_relation_transition_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT indicator relation-transition "
                    "modules to exact identity contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT query-local relation-transition "
                    "modules to exact identity contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_path_layer_attention_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_path_layer_attention_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT path-layer attention modules "
                    "to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_path_token_transformer_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_path_token_transformer_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT path-token Transformer modules "
                    "to exact zero contribution.", initialized_count,
                )
            if initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT candidate-competition "
                    "modules to exact zero contribution.", initialized_count,
                )
            if initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT parent top10 boundary "
                    "comparator modules to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT boundary-window rank-slot "
                    "experts to exact zero contribution.", initialized_count,
                )
            if initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT gatekeeper/challenger paired-margin "
                    "experts to exact zero contribution.", initialized_count,
                )
            if initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT boundary-prototype ordinal adapters to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_multi_depth_score_router_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_multi_depth_score_router_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT multi-depth score routers to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT dual-stream trajectory-agreement calibrators to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT multi-anchor rank-landmark calibrators to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT query-token top64 candidate-set Transformers to exact zero contribution.",
                    initialized_count,
                )
            if initialize_biogt_relbf_boundary_exchange_from_checkpoint:
                initialized_count = (
                    model.initialize_biogt_relbf_boundary_exchange_from_checkpoint()
                )
                logging.info(
                    "Initialized %d pure BioGT boundary-exchange experts to "
                    "exact zero contribution.", initialized_count,
                )
            if initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint:
                initialized_count = model.initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint()
                logging.info(
                    "Initialized %d pure BioGT challenger-only promotion "
                    "experts to exact zero contribution.", initialized_count,
                )
        else:
            model.load_state_dict(state_dict)
            model.target_disease_initialized_from_shared_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "target_disease_initialized_from_shared_checkpoint",
                    False,
                )
            )
            model.candidate_bias_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "candidate_bias_initialized_from_checkpoint",
                    False,
                )
            )
            model.target_residual_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "target_residual_initialized_from_checkpoint",
                    False,
                )
            )
            model.path_propagation_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "path_propagation_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_path_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_path_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_neural_path_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_neural_path_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_jk_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_jk_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_edge_attention_initialized_from_checkpoint = (
                bool(payload.get("model_config", {}).get(
                    "biogt_relbf_edge_attention_initialized_from_checkpoint",
                    False,
                ))
            )
            model.biogt_relbf_deep_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_deep_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_support_gating_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_support_gating_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_semantic_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_semantic_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_query_relation_adapter_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_query_relation_adapter_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_query_boundary_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_query_boundary_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_multirule_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_multirule_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_relation_transition_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_relation_transition_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_indicator_query_local_relation_transition_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_query_local_relation_transition_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_path_layer_attention_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_path_layer_attention_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_path_token_transformer_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_path_token_transformer_initialized_from_checkpoint", False,
                )
            )
            model.biogt_relbf_candidate_competition_boundary_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_candidate_competition_boundary_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_parent_top10_boundary_comparator_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_parent_top10_boundary_comparator_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_boundary_window_rank_slot_expert_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_boundary_window_rank_slot_expert_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_boundary_exchange_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_boundary_exchange_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_challenger_boundary_promotion_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_challenger_boundary_promotion_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_gatekeeper_challenger_paired_margin_initialized_from_checkpoint = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_gatekeeper_challenger_paired_margin_initialized_from_checkpoint",
                    False,
                )
            )
            model.biogt_relbf_deep_migration_provenance = str(
                payload.get("model_config", {}).get(
                    "biogt_relbf_deep_migration_provenance",
                    "none",
                )
            )
            model.biogt_relbf_jk_migration_provenance = str(
                payload.get("model_config", {}).get(
                    "biogt_relbf_jk_migration_provenance",
                    "none",
                )
            )
            model.biogt_relbf_jk_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_jk_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_edge_attention_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_edge_attention_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_deep_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_deep_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_support_gate_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_support_gate_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_semantic_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_semantic_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_query_relation_adapter_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_query_relation_adapter_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_query_boundary_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_query_boundary_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_multirule_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_multirule_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_relation_transition_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_relation_transition_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_indicator_query_local_relation_transition_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_indicator_query_local_relation_transition_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_path_layer_attention_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_path_layer_attention_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_path_token_transformer_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_path_token_transformer_only_finetune", False,
                )
            )
            model.biogt_relbf_candidate_competition_boundary_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_candidate_competition_boundary_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_parent_top10_boundary_comparator_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_parent_top10_boundary_comparator_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_boundary_window_rank_slot_expert_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_boundary_window_rank_slot_expert_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_boundary_exchange_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_boundary_exchange_only_finetune", False
                )
            )
            model.biogt_relbf_challenger_boundary_promotion_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_challenger_boundary_promotion_only_finetune",
                    False,
                )
            )
            model.biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune",
                    False,
                )
            )
            model.biogt_base_frozen_for_path_finetune = bool(
                payload.get("model_config", {}).get(
                    "biogt_base_frozen_for_path_finetune",
                    False,
                )
            )
            model.biogt_path_finetune_types = list(
                payload.get("model_config", {}).get(
                    "biogt_path_finetune_types",
                    [],
                )
            )
        return payload
    if (
        initialize_target_disease_from_shared
        or initialize_candidate_bias_from_checkpoint
        or initialize_target_residual_from_checkpoint
        or initialize_path_propagation_from_checkpoint
        or initialize_biogt_path_from_checkpoint
        or initialize_biogt_neural_path_from_checkpoint
        or initialize_biogt_relbf_from_checkpoint
        or initialize_biogt_relbf_jk_from_checkpoint
        or initialize_biogt_relbf_edge_attention_from_checkpoint
        or initialize_biogt_relbf_deep_from_checkpoint
        or initialize_biogt_relbf_indicator_from_checkpoint
        or initialize_biogt_relbf_indicator_support_gating_from_checkpoint
        or initialize_biogt_relbf_indicator_semantic_from_checkpoint
        or initialize_biogt_relbf_query_relation_adapter_from_checkpoint
        or initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint
        or initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint
        or initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint
        or initialize_biogt_relbf_boundary_exchange_from_checkpoint
        or initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint
        or initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint
    ):
        raise ValueError(
            "Checkpoint migration requires a unified checkpoint payload"
        )
    model.load_state_dict(payload)
    return {"model_state_dict": payload}


def _apply_checkpoint_model_args(args):
    if not args.load_model_path:
        return
    payload = torch.load(args.load_model_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get(
        "architecture"
    ) != "unified_sparse_v1":
        return

    training_args = payload.get("training_args", {})
    model_config = payload.get("model_config", {})
    architecture_keys = (
        "default_embed_size",
        "hgcn_hidden",
        "hgcn_out",
        "num_transformer_layers",
        "transformer_heads",
        "transformer_ff",
        "num_latents",
        "dropout",
        "num_gnn_layers",
        "disable_hgcn",
        "disable_transformer",
        "disable_moe",
        "fusion_mode",
        "modality_fusion_modes",
        "scorer_mode",
        "global_bilinear_pretrain_scope",
        "kge_relation_model",
        "kge_relation_rank",
        "kge_context_input_weight",
        "kge_target_specific_disease_embeddings",
        "kge_target_disease_shared_weight",
        "kge_candidate_bias",
        "kge_target_residual_rank",
        "kge_path_propagation_hops",
        "kge_path_cross_modal",
        "kge_path_cross_modal_types",
        "kge_path_cross_modal_support_types",
        "biogt_path_propagation_hops",
        "biogt_path_support_types",
        "biogt_neural_path_dim",
        "biogt_relbf_dim",
        "biogt_relbf_jk_readout",
        "biogt_relbf_edge_attention",
        "biogt_relbf_deep_refinement",
        "biogt_relbf_indicator_residual",
        "biogt_relbf_indicator_support_gating",
        "biogt_relbf_indicator_semantic_readout",
        "biogt_relbf_query_relation_adapter_rank",
        "biogt_relbf_indicator_query_boundary_rank",
        "biogt_relbf_indicator_multirule_count",
        "biogt_relbf_indicator_multirule_readout_rank",
        "biogt_relbf_indicator_relation_transition_layers",
        "biogt_relbf_indicator_relation_transition_rank",
        "biogt_relbf_indicator_query_local_relation_transition_layers",
        "biogt_relbf_indicator_query_local_relation_transition_rank",
        "biogt_relbf_path_layer_attention_rank",
        "biogt_relbf_path_token_transformer",
        "biogt_relbf_candidate_competition_boundary",
        "biogt_relbf_parent_top10_boundary_comparator",
        "biogt_relbf_boundary_window_rank_slot_expert",
        "biogt_relbf_boundary_exchange",
        "biogt_relbf_challenger_boundary_promotion",
        "biogt_relbf_gatekeeper_challenger_paired_margin",
        "biogt_relbf_boundary_prototype_ordinal_adapter",
        "biogt_relbf_multi_depth_score_router",
        "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
        "biogt_relbf_multi_anchor_rank_landmark_calibrator",
        "biogt_relbf_query_token_top64_candidate_set_transformer",
        "sparse_transformer_topk",
        "sparse_transformer_max_hops",
        "sparse_transformer_hop_penalty",
        "sparse_transformer_evidence_weight",
        "moe_num_experts",
        "moe_top_k",
        "moe_hidden_dim",
        "embedding_cache_size",
    )
    restored = {}
    for key in architecture_keys:
        if (
            key == "kge_context_input_weight"
            and getattr(
                args,
                "override_checkpoint_kge_context_input_weight",
                False,
            )
        ):
            continue
        if (
            key in {
                "biogt_path_propagation_hops",
                "biogt_path_support_types",
            }
            and (
                getattr(
                    args,
                    "initialize_biogt_path_from_checkpoint",
                    False,
                )
                or getattr(
                    args,
                    "initialize_biogt_neural_path_from_checkpoint",
                    False,
                )
            )
        ):
            continue
        if (
            key in {
                "biogt_relbf_indicator_query_local_relation_transition_layers",
                "biogt_relbf_indicator_query_local_relation_transition_rank",
            }
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_path_layer_attention_rank"
            and getattr(
                args,
                "initialize_biogt_relbf_path_layer_attention_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_path_token_transformer"
            and getattr(args, "initialize_biogt_relbf_path_token_transformer_from_checkpoint", False)
        ):
            continue
        if (
            key == "biogt_relbf_candidate_competition_boundary"
            and getattr(
                args,
                "initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_parent_top10_boundary_comparator"
            and getattr(
                args,
                "initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_boundary_window_rank_slot_expert"
            and getattr(
                args,
                "initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_boundary_exchange"
            and getattr(
                args,
                "initialize_biogt_relbf_boundary_exchange_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_challenger_boundary_promotion"
            and getattr(
                args,
                "initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_gatekeeper_challenger_paired_margin"
            and getattr(
                args,
                "initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_boundary_prototype_ordinal_adapter"
            and getattr(
                args,
                "initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_multi_depth_score_router"
            and getattr(
                args,
                "initialize_biogt_relbf_multi_depth_score_router_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_dual_stream_trajectory_agreement_calibrator"
            and getattr(
                args,
                "initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_multi_anchor_rank_landmark_calibrator"
            and getattr(
                args,
                "initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_query_token_top64_candidate_set_transformer"
            and getattr(
                args,
                "initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key in {
                "biogt_relbf_indicator_relation_transition_layers",
                "biogt_relbf_indicator_relation_transition_rank",
            }
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_relation_transition_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_neural_path_dim"
            and getattr(
                args,
                "initialize_biogt_neural_path_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_dim"
            and getattr(
                args,
                "initialize_biogt_relbf_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_jk_readout"
            and getattr(
                args,
                "initialize_biogt_relbf_jk_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_edge_attention"
            and getattr(
                args,
                "initialize_biogt_relbf_edge_attention_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_deep_refinement"
            and getattr(
                args,
                "initialize_biogt_relbf_deep_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_indicator_residual"
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_indicator_support_gating"
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_support_gating_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_indicator_semantic_readout"
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_semantic_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_query_relation_adapter_rank"
            and getattr(
                args,
                "initialize_biogt_relbf_query_relation_adapter_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "biogt_relbf_indicator_query_boundary_rank"
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_query_boundary_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key in {
                "biogt_relbf_indicator_multirule_count",
                "biogt_relbf_indicator_multirule_readout_rank",
            }
            and getattr(
                args,
                "initialize_biogt_relbf_indicator_multirule_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            getattr(
                args,
                "initialize_target_disease_from_shared_checkpoint",
                False,
            )
            and key in {
                "kge_target_specific_disease_embeddings",
                "kge_target_disease_shared_weight",
            }
        ):
            continue
        if (
            key == "kge_candidate_bias"
            and getattr(
                args,
                "initialize_kge_candidate_bias_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key == "kge_target_residual_rank"
            and getattr(
                args,
                "initialize_kge_target_residual_from_checkpoint",
                False,
            )
        ):
            continue
        if (
            key in {
                "kge_path_propagation_hops",
                "kge_path_cross_modal",
                "kge_path_cross_modal_types",
                "kge_path_cross_modal_support_types",
            }
            and getattr(
                args,
                "initialize_kge_path_propagation_from_checkpoint",
                False,
            )
        ):
            continue
        if key in training_args:
            setattr(args, key, training_args[key])
            restored[key] = training_args[key]
    if "graph_backend" in model_config:
        args.unified_graph_backend = model_config["graph_backend"]
        restored["unified_graph_backend"] = model_config["graph_backend"]
    if getattr(
        args,
        "initialize_target_disease_from_shared_checkpoint",
        False,
    ):
        args.kge_target_specific_disease_embeddings = True
        if args.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Target-disease checkpoint migration requires "
                "scorer_mode=hybrid_kge"
            )
    if getattr(
        args,
        "initialize_kge_candidate_bias_from_checkpoint",
        False,
    ):
        args.kge_candidate_bias = True
        if args.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Candidate-bias checkpoint migration requires "
                "scorer_mode=hybrid_kge"
            )
    if getattr(
        args,
        "initialize_kge_target_residual_from_checkpoint",
        False,
    ):
        if args.kge_target_residual_rank <= 0:
            raise ValueError(
                "Target-residual checkpoint migration requires a positive "
                "--kge-target-residual-rank"
            )
        if args.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Target-residual checkpoint migration requires "
                "scorer_mode=hybrid_kge"
            )
    if getattr(
        args,
        "initialize_kge_path_propagation_from_checkpoint",
        False,
    ):
        if args.kge_path_propagation_hops not in {3, 5}:
            raise ValueError(
                "Path-propagation checkpoint migration requires "
                "--kge-path-propagation-hops 3 or 5"
            )
        if args.scorer_mode != "hybrid_kge":
            raise ValueError(
                "Path-propagation checkpoint migration requires "
                "scorer_mode=hybrid_kge"
            )
    if getattr(
        args,
        "initialize_biogt_path_from_checkpoint",
        False,
    ):
        if args.biogt_path_propagation_hops not in {3, 5}:
            raise ValueError(
                "BioGT path checkpoint migration requires "
                "--biogt-path-propagation-hops 3 or 5"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT path checkpoint migration requires a pure "
                "concat_mlp or interaction_mlp checkpoint"
            )
        if args.scorer_mode not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT path checkpoint migration requires a pure BioGT "
                "scorer"
            )
    if getattr(
        args,
        "initialize_biogt_neural_path_from_checkpoint",
        False,
    ):
        if args.biogt_neural_path_dim <= 0:
            raise ValueError(
                "BioGT neural diffusion checkpoint migration requires "
                "--biogt-neural-path-dim > 0"
            )
        if args.biogt_path_propagation_hops != 5:
            raise ValueError(
                "BioGT neural diffusion uses the frozen five-layer path "
                "architecture"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT neural diffusion migration requires a pure BioGT "
                "checkpoint"
            )
        if args.scorer_mode not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT neural diffusion requires a pure BioGT scorer"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_from_checkpoint",
        False,
    ):
        if args.biogt_relbf_dim <= 0:
            raise ValueError(
                "BioGT RelBF checkpoint migration requires "
                "--biogt-relbf-dim > 0"
            )
        if args.biogt_path_propagation_hops != 5:
            raise ValueError(
                "BioGT RelBF requires the frozen five-hop scalar parent"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF migration requires a pure BioGT checkpoint"
            )
        if model_config.get("biogt_path_propagation_hops") != 5:
            raise ValueError(
                "BioGT RelBF migration requires a scalar Path5 checkpoint"
            )
        if args.scorer_mode not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF migration requires a pure BioGT scorer"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_jk_from_checkpoint",
        False,
    ):
        if getattr(args, "initialize_biogt_relbf_from_checkpoint", False):
            raise ValueError(
                "BioGT RelBF and JK migrations must be separate"
            )
        if args.biogt_relbf_dim <= 0 or not args.biogt_relbf_jk_readout:
            raise ValueError(
                "BioGT RelBF JK checkpoint migration requires enabled "
                "RelBF and --biogt-relbf-jk-readout"
            )
        if model_config.get("biogt_relbf_dim", 0) <= 0:
            raise ValueError(
                "BioGT RelBF JK migration requires a RelBF checkpoint"
            )
        if model_config.get("biogt_relbf_jk_readout", False):
            raise ValueError(
                "BioGT RelBF JK migration requires a checkpoint without JK"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF JK migration requires a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_edge_attention_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_edge_attention", False)
        ):
            raise ValueError(
                "BioGT RelBF edge-attention migration requires enabled "
                "RelBF and --biogt-relbf-edge-attention"
            )
        if model_config.get("biogt_relbf_dim", 0) <= 0:
            raise ValueError(
                "BioGT RelBF edge-attention migration requires a RelBF "
                "checkpoint"
            )
        if model_config.get("biogt_relbf_edge_attention", False):
            raise ValueError(
                "BioGT RelBF edge-attention migration requires a checkpoint "
                "without edge attention"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF edge-attention migration requires a pure "
                "BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_deep_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_deep_refinement", False)
        ):
            raise ValueError(
                "BioGT RelBF deep-refinement migration requires enabled "
                "RelBF and --biogt-relbf-deep-refinement"
            )
        if model_config.get("biogt_relbf_dim", 0) <= 0:
            raise ValueError(
                "BioGT RelBF deep-refinement migration requires a RelBF "
                "checkpoint"
            )
        if model_config.get("biogt_relbf_deep_refinement", False):
            raise ValueError(
                "BioGT RelBF deep-refinement migration requires a "
                "checkpoint without deep refinement"
            )
        if (
            model_config.get("biogt_relbf_jk_readout", False)
            or model_config.get("biogt_relbf_edge_attention", False)
        ):
            raise ValueError(
                "BioGT RelBF deep-refinement migration rejects checkpoints "
                "from the frozen JK or edge-attention axes"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF deep-refinement migration requires a pure "
                "BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
        ):
            raise ValueError(
                "BioGT RelBF indicator migration requires enabled RelBF "
                "and --biogt-relbf-indicator-residual"
            )
        if model_config.get("biogt_relbf_dim", 0) <= 0:
            raise ValueError(
                "BioGT RelBF indicator migration requires a RelBF checkpoint"
            )
        if model_config.get("biogt_relbf_indicator_residual", False):
            raise ValueError(
                "BioGT RelBF indicator migration requires a checkpoint "
                "without an indicator residual"
            )
        if any(model_config.get(key, False) for key in (
            "biogt_relbf_jk_readout",
            "biogt_relbf_edge_attention",
            "biogt_relbf_deep_refinement",
        )):
            raise ValueError(
                "BioGT RelBF indicator migration rejects checkpoints from "
                "the frozen JK, edge-attention, or deep axes"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF indicator migration requires a pure BioGT "
                "checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_support_gating_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or not getattr(
                args,
                "biogt_relbf_indicator_support_gating",
                False,
            )
        ):
            raise ValueError(
                "BioGT RelBF indicator support-gating migration requires "
                "enabled RelBF, indicator residual, and support gating"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_indicator_residual", False)
        ):
            raise ValueError(
                "BioGT RelBF indicator support-gating migration requires "
                "an indicator-residual checkpoint"
            )
        if model_config.get(
            "biogt_relbf_indicator_support_gating",
            False,
        ):
            raise ValueError(
                "BioGT RelBF indicator support-gating migration requires a "
                "checkpoint without support gates"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF indicator support-gating migration requires a "
                "pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_semantic_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or not getattr(
                args,
                "biogt_relbf_indicator_semantic_readout",
                False,
            )
            or getattr(
                args,
                "biogt_relbf_indicator_support_gating",
                False,
            )
        ):
            raise ValueError(
                "BioGT RelBF indicator semantic migration requires enabled "
                "RelBF, indicator residual, semantic readout, and no support "
                "gating"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_indicator_residual", False)
            or model_config.get(
                "biogt_relbf_indicator_support_gating",
                False,
            )
            or model_config.get(
                "biogt_relbf_indicator_semantic_readout",
                False,
            )
        ):
            raise ValueError(
                "BioGT RelBF indicator semantic migration requires a plain "
                "indicator-residual checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF indicator semantic migration requires a pure "
                "BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_query_relation_adapter_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or getattr(
                args,
                "biogt_relbf_query_relation_adapter_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "BioGT RelBF query-relation adapter migration requires an "
                "enabled RelBF and a positive adapter rank"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or model_config.get(
                "biogt_relbf_query_relation_adapter_rank",
                0,
            ) > 0
        ):
            raise ValueError(
                "BioGT RelBF query-relation adapter migration requires an "
                "adapter-free pure RelBF checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF query-relation adapter migration requires a "
                "pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_query_boundary_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(
                args,
                "biogt_relbf_indicator_query_boundary_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "BioGT RelBF indicator query-boundary migration requires "
                "an enabled indicator residual and a positive adapter rank"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_indicator_residual", False)
            or model_config.get(
                "biogt_relbf_indicator_query_boundary_rank",
                0,
            ) > 0
        ):
            raise ValueError(
                "BioGT RelBF indicator query-boundary migration requires an "
                "adapter-free indicator checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT RelBF indicator query-boundary migration requires a "
                "pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_multirule_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(args, "biogt_relbf_indicator_multirule_count", 0) <= 0
            or getattr(
                args,
                "biogt_relbf_indicator_multirule_readout_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "BioGT indicator multi-rule migration requires an enabled "
                "indicator residual, positive rule count, and positive rank"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_indicator_residual", False)
            or model_config.get("biogt_relbf_indicator_multirule_count", 0) > 0
        ):
            raise ValueError(
                "BioGT indicator multi-rule migration requires a multi-rule-"
                "free indicator checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT indicator multi-rule migration requires a pure "
                "BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_relation_transition_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(
                args,
                "biogt_relbf_indicator_relation_transition_layers",
                0,
            ) <= 0
            or getattr(
                args,
                "biogt_relbf_indicator_relation_transition_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "BioGT indicator relation-transition migration requires an "
                "enabled indicator residual, propagation layers, and rank"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_indicator_residual", False)
            or model_config.get(
                "biogt_relbf_indicator_relation_transition_layers",
                0,
            ) > 0
            or model_config.get(
                "biogt_relbf_indicator_query_local_relation_transition_layers",
                0,
            ) > 0
        ):
            raise ValueError(
                "BioGT indicator relation-transition migration requires a "
                "relation-transition-free indicator checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT indicator relation-transition migration requires a "
                "pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(
                args,
                "biogt_relbf_indicator_query_local_relation_transition_layers",
                0,
            ) <= 0
            or getattr(
                args,
                "biogt_relbf_indicator_query_local_relation_transition_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "BioGT query-local relation-transition migration requires an "
                "enabled indicator residual, propagation layers, and rank"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_indicator_residual", False)
            or model_config.get(
                "biogt_relbf_indicator_query_local_relation_transition_layers",
                0,
            ) > 0
            or model_config.get(
                "biogt_relbf_indicator_relation_transition_layers",
                0,
            ) > 0
        ):
            raise ValueError(
                "BioGT query-local relation-transition migration requires a "
                "query-local-free indicator checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT query-local relation-transition migration requires a "
                "pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_path_layer_attention_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or getattr(args, "biogt_relbf_path_layer_attention_rank", 0) <= 0
        ):
            raise ValueError(
                "BioGT path-layer attention migration requires an enabled "
                "RelBF branch and positive rank"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or model_config.get("biogt_relbf_path_layer_attention_rank", 0) > 0
        ):
            raise ValueError(
                "BioGT path-layer attention migration requires an "
                "attention-free RelBF checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp",
            "interaction_mlp",
        }:
            raise ValueError(
                "BioGT path-layer attention migration requires a pure "
                "BioGT checkpoint"
            )
    if getattr(args, "initialize_biogt_relbf_path_token_transformer_from_checkpoint", False):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_path_token_transformer", False)
        ):
            raise ValueError(
                "BioGT path-token Transformer migration requires an enabled RelBF branch"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or model_config.get("biogt_relbf_path_token_transformer", False)
        ):
            raise ValueError(
                "BioGT path-token Transformer migration requires a branch-free RelBF checkpoint"
            )
        if model_config.get("scorer_mode") not in {"concat_mlp", "interaction_mlp"}:
            raise ValueError(
                "BioGT path-token Transformer migration requires a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_candidate_competition_boundary", False
            )
        ):
            raise ValueError(
                "BioGT candidate-competition migration requires an enabled "
                "RelBF branch"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or model_config.get(
                "biogt_relbf_candidate_competition_boundary", False
            )
        ):
            raise ValueError(
                "BioGT candidate-competition migration requires a "
                "competition-free RelBF checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp", "interaction_mlp"
        }:
            raise ValueError(
                "BioGT candidate-competition migration requires a pure "
                "BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
        ):
            raise ValueError(
                "BioGT parent top10 boundary-comparator migration requires "
                "an enabled RelBF branch"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or model_config.get(
                "biogt_relbf_parent_top10_boundary_comparator", False
            )
        ):
            raise ValueError(
                "BioGT parent top10 boundary-comparator migration requires "
                "a comparator-free RelBF checkpoint"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp", "interaction_mlp"
        }:
            raise ValueError(
                "BioGT parent top10 boundary-comparator migration requires "
                "a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
        ):
            raise ValueError(
                "BioGT boundary-window migration requires the enabled gene8 "
                "comparator and window expert"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get(
                "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or model_config.get(
                "biogt_relbf_boundary_window_rank_slot_expert", False
            )
        ):
            raise ValueError(
                "BioGT boundary-window migration requires a comparator-enabled "
                "gene8 parent without an existing window expert"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp", "interaction_mlp"
        }:
            raise ValueError(
                "BioGT boundary-window migration requires a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_boundary_exchange_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or not getattr(args, "biogt_relbf_boundary_exchange", False)
        ):
            raise ValueError(
                "BioGT boundary-exchange migration requires the enabled "
                "gene9 parent and exchange expert"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get(
                "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not model_config.get(
                "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or model_config.get("biogt_relbf_boundary_exchange", False)
        ):
            raise ValueError(
                "BioGT boundary-exchange migration requires an enabled "
                "gene9 parent without an existing exchange expert"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp", "interaction_mlp"
        }:
            raise ValueError(
                "BioGT boundary-exchange migration requires a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or not getattr(
                args, "biogt_relbf_challenger_boundary_promotion", False
            )
        ):
            raise ValueError(
                "BioGT challenger-promotion migration requires the enabled "
                "gene9 parent and challenger-only expert without gene10"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get(
                "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not model_config.get(
                "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or model_config.get("biogt_relbf_boundary_exchange", False)
            or model_config.get(
                "biogt_relbf_challenger_boundary_promotion", False
            )
        ):
            raise ValueError(
                "BioGT challenger-promotion migration requires an enabled "
                "gene9 parent without gene10 or an existing gene11 branch"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp", "interaction_mlp"
        }:
            raise ValueError(
                "BioGT challenger-promotion migration requires a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or not getattr(
                args, "biogt_relbf_gatekeeper_challenger_paired_margin", False
            )
        ):
            raise ValueError(
                "BioGT paired-margin migration requires the enabled gene9 "
                "parent and Gene12 expert without gene10 or gene11"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get(
                "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not model_config.get(
                "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or model_config.get("biogt_relbf_boundary_exchange", False)
            or model_config.get(
                "biogt_relbf_challenger_boundary_promotion", False
            )
            or model_config.get(
                "biogt_relbf_gatekeeper_challenger_paired_margin", False
            )
        ):
            raise ValueError(
                "BioGT paired-margin migration requires an enabled gene9 parent"
            )
        if model_config.get("scorer_mode") not in {
            "concat_mlp", "interaction_mlp"
        }:
            raise ValueError(
                "BioGT paired-margin migration requires a pure BioGT checkpoint"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or getattr(args, "biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or not getattr(args, "biogt_relbf_boundary_prototype_ordinal_adapter", False)
        ):
            raise ValueError(
                "BioGT ordinal migration requires the enabled gene9 parent and Gene13"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
            or not model_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
            or model_config.get("biogt_relbf_boundary_exchange", False)
            or model_config.get("biogt_relbf_challenger_boundary_promotion", False)
            or model_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or model_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
        ):
            raise ValueError("BioGT ordinal migration requires an enabled gene9 parent")
    if getattr(
        args,
        "initialize_biogt_relbf_multi_depth_score_router_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or getattr(args, "biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or getattr(args, "biogt_relbf_boundary_prototype_ordinal_adapter", False)
            or not getattr(args, "biogt_relbf_multi_depth_score_router", False)
        ):
            raise ValueError(
                "BioGT multi-depth score-router migration requires gene9 and Gene14"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
            or not model_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
            or model_config.get("biogt_relbf_boundary_exchange", False)
            or model_config.get("biogt_relbf_challenger_boundary_promotion", False)
            or model_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or model_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
            or model_config.get("biogt_relbf_multi_depth_score_router", False)
        ):
            raise ValueError(
                "BioGT multi-depth score-router migration requires gene9"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint",
        False,
    ):
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or getattr(args, "biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or getattr(args, "biogt_relbf_boundary_prototype_ordinal_adapter", False)
            or getattr(args, "biogt_relbf_multi_depth_score_router", False)
            or not getattr(
                args,
                "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
                False,
            )
        ):
            raise ValueError(
                "BioGT trajectory-agreement migration requires gene9 and Gene15"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
            or not model_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
            or model_config.get("biogt_relbf_boundary_exchange", False)
            or model_config.get("biogt_relbf_challenger_boundary_promotion", False)
            or model_config.get("biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or model_config.get("biogt_relbf_boundary_prototype_ordinal_adapter", False)
            or model_config.get("biogt_relbf_multi_depth_score_router", False)
            or model_config.get(
                "biogt_relbf_dual_stream_trajectory_agreement_calibrator", False
            )
        ):
            raise ValueError(
                "BioGT trajectory-agreement migration requires an enabled gene9 parent"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint",
        False,
    ):
        forbidden_args = (
            "biogt_relbf_boundary_exchange",
            "biogt_relbf_challenger_boundary_promotion",
            "biogt_relbf_gatekeeper_challenger_paired_margin",
            "biogt_relbf_boundary_prototype_ordinal_adapter",
            "biogt_relbf_multi_depth_score_router",
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
        )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or any(getattr(args, name, False) for name in forbidden_args)
            or not getattr(
                args, "biogt_relbf_multi_anchor_rank_landmark_calibrator", False
            )
        ):
            raise ValueError(
                "BioGT multi-anchor rank-landmark migration requires gene9 and Gene16"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
            or not model_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
            or any(model_config.get(name, False) for name in forbidden_args)
            or model_config.get("biogt_relbf_multi_anchor_rank_landmark_calibrator", False)
        ):
            raise ValueError(
                "BioGT multi-anchor rank-landmark migration requires an enabled gene9 parent"
            )
    if getattr(
        args,
        "initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint",
        False,
    ):
        forbidden_args = (
            "biogt_relbf_boundary_exchange",
            "biogt_relbf_challenger_boundary_promotion",
            "biogt_relbf_gatekeeper_challenger_paired_margin",
            "biogt_relbf_boundary_prototype_ordinal_adapter",
            "biogt_relbf_multi_depth_score_router",
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
            "biogt_relbf_multi_anchor_rank_landmark_calibrator",
        )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or any(getattr(args, name, False) for name in forbidden_args)
            or not getattr(
                args,
                "biogt_relbf_query_token_top64_candidate_set_transformer",
                False,
            )
        ):
            raise ValueError(
                "BioGT query-token top64 candidate-set migration requires gene9 and Gene17"
            )
        if (
            model_config.get("biogt_relbf_dim", 0) <= 0
            or not model_config.get("biogt_relbf_parent_top10_boundary_comparator", False)
            or not model_config.get("biogt_relbf_boundary_window_rank_slot_expert", False)
            or any(model_config.get(name, False) for name in forbidden_args)
            or model_config.get("biogt_relbf_query_token_top64_candidate_set_transformer", False)
        ):
            raise ValueError(
                "BioGT query-token top64 candidate-set migration requires an enabled gene9 parent"
            )
    logging.info(
        "Restored unified architecture arguments from checkpoint: %s",
        restored,
    )


def _best_checkpoint_path(save_model_path):
    if not save_model_path:
        return None
    base, extension = os.path.splitext(save_model_path)
    return f"{base}_best{extension or '.pt'}"


def _metric_improved(metric_name, current, best, min_delta):
    if metric_name in {"loss", "val_loss", "train_loss"}:
        return current < best - min_delta
    return current > best + min_delta


def _build_full_candidate_softmax_state(
    graph,
    target_etypes,
    target_types,
    candidate_nodes_by_etype,
):
    target_types = set(target_types)
    state = {}
    for etype in target_etypes:
        modality = target_modality_for_etype(etype, target_types)
        candidate_nodes = (candidate_nodes_by_etype or {}).get(etype)
        if modality is None or candidate_nodes is None:
            raise ValueError(
                f"Full-candidate softmax requires relation candidates for {etype}"
            )
        candidates = torch.unique(
            candidate_nodes.cpu().long(),
            sorted=True,
        )
        if candidates.numel() == 0:
            raise ValueError(f"No ranking candidates are available for {etype}")
        candidate_is_source = etype[0] == modality
        query_to_positions = {}
        for known_etype in graph.canonical_etypes:
            if (
                known_etype[0] != etype[0]
                or known_etype[2] != etype[2]
            ):
                continue
            known_src, known_dst = graph.edges(etype=known_etype)
            if candidate_is_source:
                query_ids = known_dst.cpu().long()
                positive_ids = known_src.cpu().long()
            else:
                query_ids = known_src.cpu().long()
                positive_ids = known_dst.cpu().long()
            positions = torch.searchsorted(candidates, positive_ids)
            valid = positions < candidates.numel()
            if not bool(valid.all()):
                raise ValueError(
                    f"Train positives fall outside relation candidates for {etype}"
                )
            if not torch.equal(
                candidates.index_select(0, positions),
                positive_ids,
            ):
                raise ValueError(
                    f"Train positives fall outside relation candidates for {etype}"
                )
            for query_id, position in zip(
                query_ids.tolist(),
                positions.tolist(),
            ):
                query_to_positions.setdefault(int(query_id), []).append(
                    int(position)
                )
        state[etype] = {
            "candidates": candidates,
            "positive_positions_by_query": {
                query_id: torch.tensor(
                    sorted(set(positions)),
                    dtype=torch.long,
                )
                for query_id, positions in query_to_positions.items()
            },
        }
    return state


def _full_candidate_softmax_loss(
    model,
    etype,
    pos_src,
    pos_dst,
    target_types,
    relation_state,
    ranking_temperature,
    score_candidate_matrix=None,
):
    modality = target_modality_for_etype(etype, target_types)
    candidate_is_source = etype[0] == modality
    query_ids = pos_dst.cpu().long() if candidate_is_source else pos_src.cpu().long()
    positive_ids = (
        pos_src.cpu().long()
        if candidate_is_source
        else pos_dst.cpu().long()
    )
    candidates = relation_state["candidates"]
    positive_positions = torch.searchsorted(candidates, positive_ids)
    valid = positive_positions < candidates.numel()
    if not bool(valid.all()):
        raise ValueError(f"Positive targets fall outside candidates for {etype}")
    if not torch.equal(
        candidates.index_select(0, positive_positions),
        positive_ids,
    ):
        raise ValueError(f"Positive targets fall outside candidates for {etype}")
    scorer = (
        score_candidate_matrix
        if score_candidate_matrix is not None
        else model.score_kge_candidate_matrix
    )
    logits = scorer(
        etype,
        query_ids,
        candidates,
    )
    known_mask = torch.zeros(
        logits.shape,
        dtype=torch.bool,
    )
    positives_by_query = relation_state["positive_positions_by_query"]
    for row, query_id in enumerate(query_ids.tolist()):
        known_positions = positives_by_query.get(int(query_id))
        if known_positions is not None:
            known_mask[row, known_positions] = True
    known_mask[
        torch.arange(query_ids.numel()),
        positive_positions,
    ] = False
    logits = logits.masked_fill(
        known_mask.to(logits.device),
        torch.finfo(logits.dtype).min,
    )
    return F.cross_entropy(
        logits / max(float(ranking_temperature), 1e-6),
        positive_positions.to(logits.device),
    )


def pretrain_global_bilinear_branch(
    model,
    graph,
    target_etypes,
    target_types,
    known_positive_index,
    candidate_nodes_by_etype,
    dense_optimizer,
    sparse_optimizer,
    epochs,
    batch_size,
    negatives_per_positive,
    ranking_loss_weight,
    ranking_temperature,
    gradient_clip_norm,
    seed,
    pretrain_scope="target_relations",
    max_edges_per_relation=0,
    pretrain_objective="bce_pairwise",
    negatives_by_target=None,
    hard_negatives=128,
    pretrain_target_types=None,
):
    pretrain_scope = str(pretrain_scope or "target_relations").lower()
    if pretrain_scope not in {
        "target_relations",
        "all_train_relations",
    }:
        raise ValueError(
            "pretrain_scope must be target_relations or all_train_relations"
        )
    pretrain_objective = str(
        pretrain_objective or "bce_pairwise"
    ).lower()
    if pretrain_objective not in {
        "bce_pairwise",
        "sampled_softmax",
        "hard_sampled_softmax",
        "full_candidate_softmax",
    }:
        raise ValueError(
            "pretrain_objective must be bce_pairwise, sampled_softmax, "
            "hard_sampled_softmax, or full_candidate_softmax"
        )
    if (
        pretrain_objective == "full_candidate_softmax"
        and pretrain_scope != "target_relations"
    ):
        raise ValueError(
            "full_candidate_softmax requires pretrain_scope=target_relations"
        )
    hard_negatives = max(1, int(hard_negatives))
    negatives_by_target = {
        str(target_type): max(1, int(count))
        for target_type, count in (negatives_by_target or {}).items()
    }
    pretrain_target_types = sorted({
        str(target_type).strip()
        for target_type in (pretrain_target_types or [])
        if str(target_type).strip()
    })
    unknown_target_types = (
        set(pretrain_target_types) - set(target_types)
    )
    if unknown_target_types:
        raise ValueError(
            "Unknown pretraining target types: "
            f"{sorted(unknown_target_types)}"
        )
    if (
        pretrain_target_types
        and pretrain_scope != "target_relations"
    ):
        raise ValueError(
            "Target-type-restricted pretraining requires "
            "pretrain_scope=target_relations"
        )
    epochs = max(0, int(epochs))
    if epochs == 0:
        return {
            "epochs": 0,
            "steps_per_epoch": 0,
            "losses": [],
            "scope": pretrain_scope,
            "relation_count": 0,
            "available_edges": 0,
            "covered_edges_per_epoch": 0,
            "max_edges_per_relation": max(
                0,
                int(max_edges_per_relation),
            ),
            "objective": pretrain_objective,
            "negatives_by_target": negatives_by_target,
            "hard_negatives": hard_negatives,
            "target_types": pretrain_target_types,
        }
    if model.scorer_mode not in {
        "hybrid_global_bilinear",
        "hybrid_kge",
    }:
        raise ValueError(
            "Independent bilinear pretraining requires "
            "scorer_mode=hybrid_global_bilinear or hybrid_kge"
        )
    if (
        pretrain_objective == "full_candidate_softmax"
        and model.scorer_mode != "hybrid_kge"
    ):
        raise ValueError(
            "full_candidate_softmax requires scorer_mode=hybrid_kge"
        )
    if pretrain_scope == "all_train_relations":
        if model.scorer_mode != "hybrid_kge":
            raise ValueError(
                "all_train_relations pretraining requires "
                "scorer_mode=hybrid_kge"
            )
        pretrain_etypes = [
            etype
            for etype in graph.canonical_etypes
            if graph.num_edges(etype) > 0
        ]
        batcher = RelationCoverageEdgeBatcher(
            graph,
            pretrain_etypes,
            batch_size=batch_size,
            max_edges_per_relation=max_edges_per_relation,
            seed=seed,
        )
        pretrain_known_positive_index = KnownPositiveIndex(
            graph,
            {
                (etype[0], etype[2])
                for etype in pretrain_etypes
            },
        )
        score_pairs = model.score_kge_pretraining_pairs
        relation_count = len(batcher.etypes)
        available_edges = batcher.available_edges
        covered_edges_per_epoch = batcher.covered_edges_per_epoch
    else:
        pretrain_etypes = [
            etype
            for etype in target_etypes
            if (
                not pretrain_target_types
                or target_modality_for_etype(
                    etype,
                    target_types,
                ) in pretrain_target_types
            )
        ]
        if not pretrain_etypes:
            raise ValueError(
                "No target relations remain after applying the "
                "pretraining target-type restriction"
            )
        batcher = FullCoverageTargetEdgeBatcher(
            graph,
            pretrain_etypes,
            target_types,
            batch_size=batch_size,
            seed=seed,
        )
        pretrain_known_positive_index = known_positive_index
        score_pairs = model.score_global_pairs
        relation_count = len(batcher.target_etypes)
        available_edges = sum(
            graph.num_edges(etype)
            for etype in batcher.target_etypes
        )
        covered_edges_per_epoch = available_edges
    full_candidate_state = (
        _build_full_candidate_softmax_state(
            graph,
            pretrain_etypes,
            target_types,
            candidate_nodes_by_etype,
        )
        if pretrain_objective == "full_candidate_softmax"
        else {}
    )
    generator = torch.Generator()
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        generator.manual_seed(seed + (epoch + 1) * 104729)
        losses = []
        for etype, _, pos_src, pos_dst in batcher.iter_epoch(epoch):
            modality = target_modality_for_etype(
                etype,
                target_types,
            )
            batch_negatives = negatives_by_target.get(
                modality,
                negatives_per_positive,
            )
            if pretrain_objective == "full_candidate_softmax":
                neg_src = None
                neg_dst = None
            else:
                neg_src, neg_dst = sample_filtered_negatives(
                    graph,
                    etype,
                    pos_src,
                    pos_dst,
                    batch_negatives,
                    target_types,
                    pretrain_known_positive_index,
                    generator,
                    candidate_nodes_by_etype,
                )
            dense_optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad(set_to_none=True)
            if pretrain_objective == "full_candidate_softmax":
                loss = _full_candidate_softmax_loss(
                    model,
                    etype,
                    pos_src,
                    pos_dst,
                    target_types,
                    full_candidate_state[etype],
                    ranking_temperature,
                )
            elif pretrain_objective == "hard_sampled_softmax":
                selected_count = min(hard_negatives, batch_negatives)
                with torch.no_grad():
                    pool_scores = score_pairs(
                        etype,
                        neg_src,
                        neg_dst,
                    ).reshape(pos_src.shape[0], batch_negatives)
                    selected = torch.topk(
                        pool_scores,
                        k=selected_count,
                        dim=1,
                    ).indices
                    offsets = (
                        torch.arange(
                            pos_src.shape[0],
                            device=selected.device,
                        )
                        * batch_negatives
                    ).unsqueeze(1)
                    selected_flat = (
                        selected + offsets
                    ).reshape(-1).detach().cpu()
                neg_src = neg_src.index_select(0, selected_flat)
                neg_dst = neg_dst.index_select(0, selected_flat)
            else:
                selected_count = batch_negatives
            if pretrain_objective != "full_candidate_softmax":
                pos_scores = {
                    etype: score_pairs(
                        etype,
                        pos_src,
                        pos_dst,
                    )
                }
                neg_scores = {
                    etype: score_pairs(
                        etype,
                        neg_src,
                        neg_dst,
                    )
                }
                if pretrain_objective in {
                    "sampled_softmax",
                    "hard_sampled_softmax",
                }:
                    positive = pos_scores[etype]
                    negative = neg_scores[etype].reshape(
                        positive.shape[0],
                        selected_count,
                    )
                    logits = torch.cat(
                        [positive.unsqueeze(1), negative],
                        dim=1,
                    ) / max(float(ranking_temperature), 1e-6)
                    loss = F.cross_entropy(
                        logits,
                        torch.zeros(
                            positive.shape[0],
                            dtype=torch.long,
                            device=positive.device,
                        ),
                    )
                else:
                    loss, _ = compute_joint_link_loss(
                        pos_scores,
                        neg_scores,
                        batch_negatives,
                        ranking_loss_weight,
                        ranking_temperature,
                        moe_aux_loss=None,
                        moe_aux_loss_weight=0.0,
                    )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Global bilinear pretraining produced a non-finite loss"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.dense_parameters(),
                max_norm=gradient_clip_norm,
            )
            dense_optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)
        logging.info(
            "Global bilinear pretrain epoch %d/%d loss=%.5f "
            "steps=%d relations=%d covered_edges=%d available_edges=%d "
            "scope=%s objective=%s.",
            epoch + 1,
            epochs,
            epoch_loss,
            len(batcher),
            relation_count,
            covered_edges_per_epoch,
            available_edges,
            pretrain_scope,
            pretrain_objective,
        )
    model.clear_embedding_cache()
    return {
        "epochs": epochs,
        "steps_per_epoch": len(batcher),
        "losses": epoch_losses,
        "scope": pretrain_scope,
        "relation_count": relation_count,
        "available_edges": available_edges,
        "covered_edges_per_epoch": covered_edges_per_epoch,
        "max_edges_per_relation": max(
            0,
            int(max_edges_per_relation),
        ),
        "objective": pretrain_objective,
        "negatives_by_target": negatives_by_target,
        "hard_negatives": hard_negatives,
        "target_types": (
            pretrain_target_types
            or sorted({
                target_modality_for_etype(etype, target_types)
                for etype in pretrain_etypes
                if target_modality_for_etype(etype, target_types)
            })
        ),
    }


def _lambdarank_delta_ndcg_weights(
    positive_score,
    negative_scores,
    k,
):
    """Return detached binary-relevance delta-NDCG@K swap weights."""
    k = int(k)
    if k <= 0:
        raise ValueError("BioGT path LambdaRank K must be positive")
    if negative_scores.numel() == 0:
        return negative_scores.detach().clone()
    with torch.no_grad():
        scores = torch.cat((positive_score.reshape(1), negative_scores))
        order = torch.argsort(scores, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(
            1,
            scores.numel() + 1,
            dtype=order.dtype,
            device=order.device,
        )
        discounts = torch.where(
            ranks <= k,
            1.0 / torch.log2(ranks.to(scores.dtype) + 1.0),
            torch.zeros_like(ranks, dtype=scores.dtype),
        )
        return (discounts[0] - discounts[1:]).abs()


def _biogt_path_full_candidate_loss(
    scaled_logits,
    known_positive_mask,
    row_indices,
    positive_positions,
    hard_negative_count=0,
    rank_weight=0.0,
    topk_boundary_weight=0.0,
    topk_boundary_k=10,
    topk_boundary_margin=0.0,
    lambdarank_weight=0.0,
    lambdarank_k=10,
    lambdarank_temperature=1.0,
    soft_hits_weight=0.0,
    soft_hits_k=10,
    soft_hits_temperature=1.0,
    normalized_soft_hits_weight=0.0,
    normalized_soft_hits_k=10,
    normalized_soft_hits_temperature=1.0,
    return_topk_details=False,
    return_lambdarank_details=False,
    return_soft_hits_details=False,
    return_normalized_soft_hits_details=False,
):
    hard_negative_count = int(hard_negative_count)
    rank_weight = float(rank_weight)
    topk_boundary_weight = float(topk_boundary_weight)
    topk_boundary_k = int(topk_boundary_k)
    topk_boundary_margin = float(topk_boundary_margin)
    lambdarank_weight = float(lambdarank_weight)
    lambdarank_k = int(lambdarank_k)
    lambdarank_temperature = float(lambdarank_temperature)
    soft_hits_weight = float(soft_hits_weight)
    soft_hits_k = int(soft_hits_k)
    soft_hits_temperature = float(soft_hits_temperature)
    normalized_soft_hits_weight = float(normalized_soft_hits_weight)
    normalized_soft_hits_k = int(normalized_soft_hits_k)
    normalized_soft_hits_temperature = float(
        normalized_soft_hits_temperature
    )
    if hard_negative_count < 0:
        raise ValueError("BioGT path hard-negative count must be non-negative")
    if not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError("BioGT path ranking weight must be finite and non-negative")
    if not math.isfinite(topk_boundary_weight) or topk_boundary_weight < 0:
        raise ValueError(
            "BioGT path Top-K boundary weight must be finite and non-negative"
        )
    if topk_boundary_k <= 0:
        raise ValueError("BioGT path Top-K boundary K must be positive")
    if not math.isfinite(topk_boundary_margin) or topk_boundary_margin < 0:
        raise ValueError(
            "BioGT path Top-K boundary margin must be finite and non-negative"
        )
    if not math.isfinite(lambdarank_weight) or lambdarank_weight < 0:
        raise ValueError(
            "BioGT path LambdaRank weight must be finite and non-negative"
        )
    if lambdarank_k <= 0:
        raise ValueError("BioGT path LambdaRank K must be positive")
    if (
        not math.isfinite(lambdarank_temperature)
        or lambdarank_temperature <= 0
    ):
        raise ValueError(
            "BioGT path LambdaRank temperature must be finite and positive"
        )
    if not math.isfinite(soft_hits_weight) or soft_hits_weight < 0:
        raise ValueError(
            "BioGT path Soft-Hits weight must be finite and non-negative"
        )
    if soft_hits_k <= 0:
        raise ValueError("BioGT path Soft-Hits K must be positive")
    if (
        not math.isfinite(soft_hits_temperature)
        or soft_hits_temperature <= 0
    ):
        raise ValueError(
            "BioGT path Soft-Hits temperature must be finite and positive"
        )
    if (
        not math.isfinite(normalized_soft_hits_weight)
        or normalized_soft_hits_weight < 0
    ):
        raise ValueError(
            "BioGT path normalized Soft-Hits weight must be finite and "
            "non-negative"
        )
    if normalized_soft_hits_k <= 0:
        raise ValueError("BioGT path normalized Soft-Hits K must be positive")
    if (
        not math.isfinite(normalized_soft_hits_temperature)
        or normalized_soft_hits_temperature <= 0
    ):
        raise ValueError(
            "BioGT path normalized Soft-Hits temperature must be finite "
            "and positive"
        )
    if scaled_logits.ndim != 2 or not scaled_logits.is_floating_point():
        raise ValueError(
            "BioGT path full-candidate logits must be a floating 2D tensor"
        )
    if (
        known_positive_mask.shape != scaled_logits.shape
        or known_positive_mask.dtype != torch.bool
    ):
        raise ValueError(
            "BioGT path known-positive mask must be boolean and match logits"
        )
    if (
        row_indices.ndim != 1
        or positive_positions.ndim != 1
        or row_indices.numel() != positive_positions.numel()
        or row_indices.numel() == 0
    ):
        raise ValueError(
            "BioGT path positive row and candidate indices must be non-empty "
            "aligned vectors"
        )
    if (
        row_indices.device != scaled_logits.device
        or positive_positions.device != scaled_logits.device
        or known_positive_mask.device != scaled_logits.device
    ):
        raise ValueError(
            "BioGT path logits, mask, and positive indices must share a device"
        )
    if (
        bool((row_indices < 0).any())
        or bool((row_indices >= scaled_logits.shape[0]).any())
        or bool((positive_positions < 0).any())
        or bool((positive_positions >= scaled_logits.shape[1]).any())
    ):
        raise ValueError("BioGT path positive indices are out of bounds")
    if not bool(known_positive_mask[row_indices, positive_positions].all()):
        raise ValueError(
            "BioGT path positive indices must be present in the known mask"
        )
    if not bool(torch.isfinite(scaled_logits).all()):
        raise ValueError("BioGT path full-candidate logits must be finite")

    negative_logsumexp = torch.logsumexp(
        scaled_logits.masked_fill(
            known_positive_mask,
            torch.finfo(scaled_logits.dtype).min,
        ),
        dim=1,
    )
    positive_scores = scaled_logits[
        row_indices,
        positive_positions,
    ]
    listwise_loss = (
        torch.logaddexp(
            positive_scores,
            negative_logsumexp.index_select(0, row_indices),
        )
        - positive_scores
    ).mean()

    ranking_loss = scaled_logits.new_zeros(())
    selected_pair_count = 0
    if hard_negative_count > 0 and rank_weight > 0:
        pairwise_losses = []
        for row in torch.unique(row_indices).tolist():
            negative_scores = scaled_logits[row].masked_select(
                ~known_positive_mask[row]
            )
            selected_count = min(
                hard_negative_count,
                int(negative_scores.numel()),
            )
            if selected_count == 0:
                continue
            hard_negative_scores = torch.topk(
                negative_scores,
                k=selected_count,
                largest=True,
                sorted=False,
            ).values
            row_positive_scores = positive_scores[
                row_indices == row
            ]
            row_pairwise_losses = F.softplus(
                hard_negative_scores.unsqueeze(0)
                - row_positive_scores.unsqueeze(1)
            ).reshape(-1)
            pairwise_losses.append(row_pairwise_losses)
            selected_pair_count += int(row_pairwise_losses.numel())
        if pairwise_losses:
            ranking_loss = torch.cat(pairwise_losses).mean()

    topk_boundary_loss = scaled_logits.new_zeros(())
    selected_boundary_count = 0
    if topk_boundary_weight > 0:
        boundary_losses = []
        for row in torch.unique(row_indices).tolist():
            negative_scores = scaled_logits[row].masked_select(
                ~known_positive_mask[row]
            )
            if negative_scores.numel() == 0:
                continue
            effective_k = min(topk_boundary_k, int(negative_scores.numel()))
            kth_negative = torch.topk(
                negative_scores,
                k=effective_k,
                largest=True,
                sorted=True,
            ).values[-1]
            row_positive_scores = positive_scores[row_indices == row]
            row_boundary_losses = F.softplus(
                kth_negative
                + topk_boundary_margin
                - row_positive_scores
            )
            boundary_losses.append(row_boundary_losses)
            selected_boundary_count += int(row_boundary_losses.numel())
        if boundary_losses:
            topk_boundary_loss = torch.cat(boundary_losses).mean()

    loss = listwise_loss
    if hard_negative_count > 0 and rank_weight > 0:
        loss = loss + rank_weight * ranking_loss
    if topk_boundary_weight > 0:
        loss = loss + topk_boundary_weight * topk_boundary_loss
    lambdarank_loss = scaled_logits.new_zeros(())
    selected_lambdarank_pair_count = 0
    if lambdarank_weight > 0:
        lambdarank_losses = []
        for row in torch.unique(row_indices).tolist():
            negative_scores = scaled_logits[row].masked_select(
                ~known_positive_mask[row]
            )
            selected_count = min(
                max(lambdarank_k, hard_negative_count),
                int(negative_scores.numel()),
            )
            if selected_count == 0:
                continue
            hard_negative_scores = torch.topk(
                negative_scores,
                k=selected_count,
                largest=True,
                sorted=True,
            ).values
            for positive_score in positive_scores[row_indices == row]:
                delta_ndcg = _lambdarank_delta_ndcg_weights(
                    positive_score.detach(),
                    hard_negative_scores.detach(),
                    lambdarank_k,
                )
                relevant = delta_ndcg > 0
                if not bool(relevant.any()):
                    continue
                pair_losses = F.softplus(
                    (
                        hard_negative_scores[relevant] - positive_score
                    ) / lambdarank_temperature
                ) * delta_ndcg[relevant]
                lambdarank_losses.append(pair_losses)
                selected_lambdarank_pair_count += int(pair_losses.numel())
        if lambdarank_losses:
            lambdarank_loss = torch.cat(lambdarank_losses).mean()
            loss = loss + lambdarank_weight * lambdarank_loss
    soft_hits_loss = scaled_logits.new_zeros(())
    soft_hits_positive_count = 0
    if soft_hits_weight > 0:
        soft_hits_losses = []
        cutoff = float(soft_hits_k) + 0.5
        for row in torch.unique(row_indices).tolist():
            negative_scores = scaled_logits[row].masked_select(
                ~known_positive_mask[row]
            )
            for positive_score in positive_scores[row_indices == row]:
                soft_rank = positive_score.new_ones(())
                if negative_scores.numel() > 0:
                    soft_rank = soft_rank + torch.sigmoid(
                        (negative_scores - positive_score)
                        / soft_hits_temperature
                    ).sum()
                soft_hits_losses.append(
                    F.softplus(
                        (soft_rank - cutoff) / float(soft_hits_k)
                    ).reshape(1)
                )
                soft_hits_positive_count += 1
        if soft_hits_losses:
            soft_hits_loss = torch.cat(soft_hits_losses).mean()
            loss = loss + soft_hits_weight * soft_hits_loss
    normalized_soft_hits_loss = scaled_logits.new_zeros(())
    normalized_soft_hits_positive_count = 0
    normalized_soft_hits_negative_count = 0
    if normalized_soft_hits_weight > 0:
        normalized_soft_hits_losses = []
        for row in torch.unique(row_indices).tolist():
            negative_scores = scaled_logits[row].masked_select(
                ~known_positive_mask[row]
            )
            negative_count = int(negative_scores.numel())
            if negative_count == 0:
                continue
            effective_k = min(
                normalized_soft_hits_k,
                negative_count + 1,
            )
            cutoff_fraction = float(effective_k) / float(negative_count + 1)
            for positive_score in positive_scores[row_indices == row]:
                exceedance_fraction = torch.sigmoid(
                    (negative_scores - positive_score)
                    / normalized_soft_hits_temperature
                ).mean()
                normalized_soft_hits_losses.append(
                    F.softplus(
                        (exceedance_fraction - cutoff_fraction)
                        / normalized_soft_hits_temperature
                    ).reshape(1)
                )
                normalized_soft_hits_positive_count += 1
                normalized_soft_hits_negative_count += negative_count
        if normalized_soft_hits_losses:
            normalized_soft_hits_loss = torch.cat(
                normalized_soft_hits_losses
            ).mean()
            loss = (
                loss
                + normalized_soft_hits_weight * normalized_soft_hits_loss
            )
    result = (
        loss,
        listwise_loss,
        ranking_loss,
        selected_pair_count,
    )
    if return_topk_details:
        result = result + (topk_boundary_loss, selected_boundary_count)
    if return_lambdarank_details:
        result = result + (
            lambdarank_loss,
            selected_lambdarank_pair_count,
        )
    if return_soft_hits_details:
        result = result + (
            soft_hits_loss,
            soft_hits_positive_count,
        )
    if return_normalized_soft_hits_details:
        result = result + (
            normalized_soft_hits_loss,
            normalized_soft_hits_positive_count,
            normalized_soft_hits_negative_count,
        )
    return result


def pretrain_biogt_path_branch(
    model,
    graph,
    target_etypes,
    target_types,
    candidate_nodes_by_etype,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    ranking_temperature,
    gradient_clip_norm,
    seed,
    pretrain_target_types=None,
    hard_negative_count=0,
    rank_weight=0.0,
    topk_boundary_weight=0.0,
    topk_boundary_k=10,
    topk_boundary_margin=0.0,
    lambdarank_weight=0.0,
    lambdarank_k=10,
    lambdarank_temperature=1.0,
    soft_hits_weight=0.0,
    soft_hits_k=10,
    soft_hits_temperature=1.0,
    normalized_soft_hits_weight=0.0,
    normalized_soft_hits_k=10,
    normalized_soft_hits_temperature=1.0,
):
    epochs = max(0, int(epochs))
    hard_negative_count = int(hard_negative_count)
    rank_weight = float(rank_weight)
    topk_boundary_weight = float(topk_boundary_weight)
    topk_boundary_k = int(topk_boundary_k)
    topk_boundary_margin = float(topk_boundary_margin)
    lambdarank_weight = float(lambdarank_weight)
    lambdarank_k = int(lambdarank_k)
    lambdarank_temperature = float(lambdarank_temperature)
    soft_hits_weight = float(soft_hits_weight)
    soft_hits_k = int(soft_hits_k)
    soft_hits_temperature = float(soft_hits_temperature)
    normalized_soft_hits_weight = float(normalized_soft_hits_weight)
    normalized_soft_hits_k = int(normalized_soft_hits_k)
    normalized_soft_hits_temperature = float(
        normalized_soft_hits_temperature
    )
    if hard_negative_count < 0:
        raise ValueError(
            "--biogt-path-pretrain-hard-negatives must be non-negative"
        )
    if not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError(
            "--biogt-path-pretrain-rank-weight must be finite and "
            "non-negative"
        )
    if not math.isfinite(topk_boundary_weight) or topk_boundary_weight < 0:
        raise ValueError(
            "--biogt-path-pretrain-topk-boundary-weight must be finite and "
            "non-negative"
        )
    if topk_boundary_k <= 0:
        raise ValueError(
            "--biogt-path-pretrain-topk-boundary-k must be positive"
        )
    if not math.isfinite(topk_boundary_margin) or topk_boundary_margin < 0:
        raise ValueError(
            "--biogt-path-pretrain-topk-boundary-margin must be finite and "
            "non-negative"
        )
    if not math.isfinite(lambdarank_weight) or lambdarank_weight < 0:
        raise ValueError(
            "--biogt-path-pretrain-lambdarank-weight must be finite and "
            "non-negative"
        )
    if lambdarank_k <= 0:
        raise ValueError(
            "--biogt-path-pretrain-lambdarank-k must be positive"
        )
    if (
        not math.isfinite(lambdarank_temperature)
        or lambdarank_temperature <= 0
    ):
        raise ValueError(
            "--biogt-path-pretrain-lambdarank-temperature must be finite "
            "and positive"
        )
    if not math.isfinite(soft_hits_weight) or soft_hits_weight < 0:
        raise ValueError(
            "--biogt-path-pretrain-soft-hits-weight must be finite and "
            "non-negative"
        )
    if soft_hits_k <= 0:
        raise ValueError(
            "--biogt-path-pretrain-soft-hits-k must be positive"
        )
    if (
        not math.isfinite(soft_hits_temperature)
        or soft_hits_temperature <= 0
    ):
        raise ValueError(
            "--biogt-path-pretrain-soft-hits-temperature must be finite "
            "and positive"
        )
    if (
        not math.isfinite(normalized_soft_hits_weight)
        or normalized_soft_hits_weight < 0
    ):
        raise ValueError(
            "--biogt-path-pretrain-normalized-soft-hits-weight must be "
            "finite and non-negative"
        )
    if normalized_soft_hits_k <= 0:
        raise ValueError(
            "--biogt-path-pretrain-normalized-soft-hits-k must be positive"
        )
    if (
        not math.isfinite(normalized_soft_hits_temperature)
        or normalized_soft_hits_temperature <= 0
    ):
        raise ValueError(
            "--biogt-path-pretrain-normalized-soft-hits-temperature must "
            "be finite and positive"
        )
    requested_types = sorted({
        str(target_type).strip()
        for target_type in (pretrain_target_types or [])
        if str(target_type).strip()
    })
    if epochs == 0:
        return {
            "epochs": 0,
            "steps_per_epoch": 0,
            "losses": [],
            "listwise_losses": [],
            "ranking_losses": [],
            "objective": "full_candidate_softmax",
            "hard_negatives": hard_negative_count,
            "rank_weight": rank_weight,
            "ranking_auxiliary_enabled": bool(
                hard_negative_count > 0 and rank_weight > 0
            ),
            "topk_boundary_weight": topk_boundary_weight,
            "topk_boundary_k": topk_boundary_k,
            "topk_boundary_margin": topk_boundary_margin,
            "topk_boundary_enabled": bool(topk_boundary_weight > 0),
            "topk_boundary_losses": [],
            "selected_boundaries_per_epoch": [],
            "lambdarank_weight": lambdarank_weight,
            "lambdarank_k": lambdarank_k,
            "lambdarank_temperature": lambdarank_temperature,
            "lambdarank_enabled": bool(lambdarank_weight > 0),
            "lambdarank_losses": [],
            "selected_lambdarank_pairs_per_epoch": [],
            "soft_hits_weight": soft_hits_weight,
            "soft_hits_k": soft_hits_k,
            "soft_hits_temperature": soft_hits_temperature,
            "soft_hits_enabled": bool(soft_hits_weight > 0),
            "soft_hits_losses": [],
            "soft_hits_positive_counts_per_epoch": [],
            "normalized_soft_hits_weight": normalized_soft_hits_weight,
            "normalized_soft_hits_k": normalized_soft_hits_k,
            "normalized_soft_hits_temperature": (
                normalized_soft_hits_temperature
            ),
            "normalized_soft_hits_enabled": bool(
                normalized_soft_hits_weight > 0
            ),
            "normalized_soft_hits_losses": [],
            "normalized_soft_hits_positive_counts_per_epoch": [],
            "normalized_soft_hits_negative_counts_per_epoch": [],
            "target_types": requested_types,
            "available_edges": 0,
            "covered_edges_per_epoch": 0,
            "batch_unit": "unique_queries",
            "unique_queries_per_epoch": 0,
        }
    if (
        model.biogt_path_propagation_hops <= 0
        or not model.biogt_path_query_gates
        or model.scorer_mode not in {
            "concat_mlp",
            "interaction_mlp",
        }
    ):
        raise ValueError(
            "BioGT path pretraining requires an enabled pure BioGT path "
            "branch"
        )
    unknown_types = set(requested_types) - set(target_types)
    if unknown_types:
        raise ValueError(
            "Unknown BioGT path pretraining target types: "
            f"{sorted(unknown_types)}"
        )
    selected_etypes = [
        etype
        for etype in target_etypes
        if (
            not requested_types
            or target_modality_for_etype(
                etype,
                target_types,
            ) in requested_types
        )
    ]
    if not selected_etypes:
        raise ValueError(
            "No target relations remain for BioGT path pretraining"
        )
    relation_state = _build_full_candidate_softmax_state(
        graph,
        selected_etypes,
        target_types,
        candidate_nodes_by_etype,
    )
    path_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if (
            name.startswith("biogt_path_")
            or name.startswith("biogt_neural_path_")
            or name.startswith("biogt_relbf_")
        )
        and parameter.requires_grad
    ]
    if not path_parameters:
        raise ValueError(
            "BioGT path pretraining has no trainable path parameters"
        )
    optimizer = torch.optim.AdamW(
        path_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    cached_base_embeddings = model.cache_embeddings(
        graph,
        cache_key=("biogt-path-pretrain-base", tuple(selected_etypes)),
        required_etypes=selected_etypes,
        force=True,
    )

    def score_path_candidates(scoring_etype, query_ids, candidate_ids):
        modality = target_modality_for_etype(
            scoring_etype,
            target_types,
        )
        candidate_is_source = scoring_etype[0] == modality
        query_ids_cpu = torch.as_tensor(
            query_ids,
            dtype=torch.long,
        ).cpu()
        candidate_ids_cpu = torch.as_tensor(
            candidate_ids,
            dtype=torch.long,
        ).cpu()
        query_count = query_ids_cpu.numel()
        candidate_count = candidate_ids_cpu.numel()
        if candidate_is_source:
            src_ids = candidate_ids_cpu.repeat(query_count)
            dst_ids = query_ids_cpu.repeat_interleave(candidate_count)
        else:
            src_ids = query_ids_cpu.repeat_interleave(candidate_count)
            dst_ids = candidate_ids_cpu.repeat(query_count)
        base_scores = model.score_cached_pairs(
            cached_base_embeddings,
            scoring_etype,
            src_ids,
            dst_ids,
            include_biogt_path=False,
        ).reshape(query_count, candidate_count)
        path_scores = model.score_biogt_path_candidate_matrix(
            scoring_etype,
            query_ids,
            candidate_ids,
            detach_query_features=True,
            relbf_query_features=(
                cached_base_embeddings["modalities"][modality][
                    "disease"
                ].index_select(0, query_ids_cpu)
                if model.biogt_relbf_dim > 0
                else None
            ),
            relbf_candidate_features=(
                cached_base_embeddings["modalities"][modality][
                    modality
                ].index_select(0, candidate_ids_cpu)
                if model.biogt_relbf_dim > 0
                else None
            ),
        )
        return base_scores.to(path_scores.device) + path_scores

    training_positions_by_etype = {}
    for etype in selected_etypes:
        modality = target_modality_for_etype(etype, target_types)
        candidate_is_source = etype[0] == modality
        pos_src, pos_dst = graph.edges(etype=etype)
        query_ids = (
            pos_dst.cpu().long()
            if candidate_is_source
            else pos_src.cpu().long()
        )
        positive_ids = (
            pos_src.cpu().long()
            if candidate_is_source
            else pos_dst.cpu().long()
        )
        candidates = relation_state[etype]["candidates"]
        positions = torch.searchsorted(candidates, positive_ids)
        if (
            not bool((positions < candidates.numel()).all())
            or not torch.equal(
                candidates.index_select(0, positions),
                positive_ids,
            )
        ):
            raise ValueError(
                f"BioGT path positives fall outside candidates for {etype}"
            )
        grouped_positions = {}
        for query_id, position in zip(
            query_ids.tolist(),
            positions.tolist(),
        ):
            grouped_positions.setdefault(int(query_id), []).append(
                int(position)
            )
        training_positions_by_etype[etype] = {
            query_id: torch.tensor(values, dtype=torch.long)
            for query_id, values in grouped_positions.items()
        }

    available_edges = sum(
        graph.num_edges(etype)
        for etype in selected_etypes
    )
    grouped_edges = sum(
        positions.numel()
        for grouped in training_positions_by_etype.values()
        for positions in grouped.values()
    )
    if grouped_edges != available_edges:
        raise RuntimeError(
            "Grouped BioGT path coverage does not match selected train edges: "
            f"grouped={grouped_edges}, available={available_edges}"
        )
    batch_counts_by_etype = {
        etype: math.ceil(len(grouped) / max(1, int(batch_size)))
        for etype, grouped in training_positions_by_etype.items()
    }
    steps_per_epoch = (
        max(batch_counts_by_etype.values())
        if (
            model.biogt_neural_path_dim > 0
            or model.biogt_relbf_dim > 0
        )
        else sum(batch_counts_by_etype.values())
    )
    unique_queries_per_epoch = sum(
        len(grouped)
        for grouped in training_positions_by_etype.values()
    )
    def compute_batch_loss(etype, batch_queries):
        query_ids = torch.tensor(batch_queries, dtype=torch.long)
        logits = score_path_candidates(
            etype,
            query_ids,
            relation_state[etype]["candidates"],
        )
        scaled_logits = logits / max(
            float(ranking_temperature),
            1e-6,
        )
        known_mask = torch.zeros(
            scaled_logits.shape,
            dtype=torch.bool,
            device=scaled_logits.device,
        )
        row_indices = []
        positive_positions = []
        for row, query_id in enumerate(batch_queries):
            known_positions = relation_state[etype][
                "positive_positions_by_query"
            ].get(int(query_id))
            if known_positions is not None:
                known_mask[row, known_positions.to(logits.device)] = True
            train_positions = training_positions_by_etype[etype][
                int(query_id)
            ]
            row_indices.extend([row] * train_positions.numel())
            positive_positions.extend(train_positions.tolist())
        row_indices = torch.tensor(
            row_indices,
            dtype=torch.long,
            device=logits.device,
        )
        positive_positions = torch.tensor(
            positive_positions,
            dtype=torch.long,
            device=logits.device,
        )
        (
            loss,
            listwise_loss,
            ranking_loss,
            selected_pair_count,
            topk_boundary_loss,
            selected_boundary_count,
            lambdarank_loss,
            selected_lambdarank_pair_count,
            soft_hits_loss,
            soft_hits_positive_count,
            normalized_soft_hits_loss,
            normalized_soft_hits_positive_count,
            normalized_soft_hits_negative_count,
        ) = _biogt_path_full_candidate_loss(
            scaled_logits,
            known_mask,
            row_indices,
            positive_positions,
            hard_negative_count=hard_negative_count,
            rank_weight=rank_weight,
            topk_boundary_weight=topk_boundary_weight,
            topk_boundary_k=topk_boundary_k,
            topk_boundary_margin=topk_boundary_margin,
            lambdarank_weight=lambdarank_weight,
            lambdarank_k=lambdarank_k,
            lambdarank_temperature=lambdarank_temperature,
            soft_hits_weight=soft_hits_weight,
            soft_hits_k=soft_hits_k,
            soft_hits_temperature=soft_hits_temperature,
            normalized_soft_hits_weight=normalized_soft_hits_weight,
            normalized_soft_hits_k=normalized_soft_hits_k,
            normalized_soft_hits_temperature=(
                normalized_soft_hits_temperature
            ),
            return_topk_details=True,
            return_lambdarank_details=True,
            return_soft_hits_details=True,
            return_normalized_soft_hits_details=True,
        )
        if not all(
            torch.isfinite(value)
            for value in (
                loss,
                listwise_loss,
                ranking_loss,
                topk_boundary_loss,
                lambdarank_loss,
                soft_hits_loss,
                normalized_soft_hits_loss,
            )
        ):
            raise RuntimeError(
                "BioGT path pretraining produced a non-finite loss"
            )
        return (
            loss,
            listwise_loss,
            ranking_loss,
            topk_boundary_loss,
            lambdarank_loss,
            soft_hits_loss,
            normalized_soft_hits_loss,
            int(positive_positions.numel()),
            selected_pair_count,
            selected_boundary_count,
            selected_lambdarank_pair_count,
            soft_hits_positive_count,
            normalized_soft_hits_positive_count,
            normalized_soft_hits_negative_count,
        )

    epoch_losses = []
    epoch_listwise_losses = []
    epoch_ranking_losses = []
    epoch_topk_boundary_losses = []
    epoch_lambdarank_losses = []
    epoch_soft_hits_losses = []
    epoch_normalized_soft_hits_losses = []
    selected_pairs_per_epoch = []
    selected_boundaries_per_epoch = []
    selected_lambdarank_pairs_per_epoch = []
    soft_hits_positive_counts_per_epoch = []
    normalized_soft_hits_positive_counts_per_epoch = []
    normalized_soft_hits_negative_counts_per_epoch = []
    for epoch in range(epochs):
        model.train()
        losses = []
        listwise_losses = []
        ranking_losses = []
        topk_boundary_losses = []
        lambdarank_losses = []
        soft_hits_losses = []
        normalized_soft_hits_losses = []
        selected_pair_count = 0
        selected_boundary_count = 0
        selected_lambdarank_pair_count = 0
        soft_hits_positive_count = 0
        normalized_soft_hits_positive_count = 0
        normalized_soft_hits_negative_count = 0
        generator = torch.Generator().manual_seed(
            int(seed) + int(epoch) * 100003
        )
        batches_by_etype = {}
        for etype in sorted(selected_etypes):
            query_ids = sorted(training_positions_by_etype[etype])
            permutation = torch.randperm(
                len(query_ids),
                generator=generator,
            ).tolist()
            shuffled_queries = [query_ids[index] for index in permutation]
            batches_by_etype[etype] = [
                shuffled_queries[
                    start:start + max(1, int(batch_size))
                ]
                for start in range(
                    0,
                    len(shuffled_queries),
                    max(1, int(batch_size)),
                )
            ]

        covered_edges = 0
        if (
            model.biogt_neural_path_dim > 0
            or model.biogt_relbf_dim > 0
        ):
            ordered_etypes = sorted(selected_etypes)
            for step in range(steps_per_epoch):
                model.zero_grad(set_to_none=True)
                step_losses = []
                for etype in ordered_etypes:
                    relation_batches = batches_by_etype[etype]
                    (
                        loss,
                        listwise_loss,
                        ranking_loss,
                        topk_boundary_loss,
                        lambdarank_loss,
                        soft_hits_loss,
                        normalized_soft_hits_loss,
                        positive_count,
                        batch_selected_pair_count,
                        batch_selected_boundary_count,
                        batch_selected_lambdarank_pair_count,
                        batch_soft_hits_positive_count,
                        batch_normalized_soft_hits_positive_count,
                        batch_normalized_soft_hits_negative_count,
                    ) = compute_batch_loss(
                        etype,
                        relation_batches[step % len(relation_batches)],
                    )
                    (loss / len(ordered_etypes)).backward()
                    step_losses.append(float(loss.detach().cpu()))
                    listwise_losses.append(
                        float(listwise_loss.detach().cpu())
                    )
                    ranking_losses.append(
                        float(ranking_loss.detach().cpu())
                    )
                    topk_boundary_losses.append(
                        float(topk_boundary_loss.detach().cpu())
                    )
                    lambdarank_losses.append(
                        float(lambdarank_loss.detach().cpu())
                    )
                    soft_hits_losses.append(
                        float(soft_hits_loss.detach().cpu())
                    )
                    normalized_soft_hits_losses.append(
                        float(normalized_soft_hits_loss.detach().cpu())
                    )
                    selected_pair_count += batch_selected_pair_count
                    selected_boundary_count += (
                        batch_selected_boundary_count
                    )
                    selected_lambdarank_pair_count += (
                        batch_selected_lambdarank_pair_count
                    )
                    soft_hits_positive_count += (
                        batch_soft_hits_positive_count
                    )
                    normalized_soft_hits_positive_count += (
                        batch_normalized_soft_hits_positive_count
                    )
                    normalized_soft_hits_negative_count += (
                        batch_normalized_soft_hits_negative_count
                    )
                    if step < len(relation_batches):
                        covered_edges += positive_count
                torch.nn.utils.clip_grad_norm_(
                    path_parameters,
                    max_norm=gradient_clip_norm,
                )
                optimizer.step()
                losses.append(float(np.mean(step_losses)))
        else:
            batches = [
                (etype, batch_queries)
                for etype, relation_batches in batches_by_etype.items()
                for batch_queries in relation_batches
            ]
            batch_order = torch.randperm(
                len(batches),
                generator=generator,
            ).tolist()
            for batch_index in batch_order:
                etype, batch_queries = batches[batch_index]
                model.zero_grad(set_to_none=True)
                (
                    loss,
                    listwise_loss,
                    ranking_loss,
                    topk_boundary_loss,
                    lambdarank_loss,
                    soft_hits_loss,
                    normalized_soft_hits_loss,
                    positive_count,
                    batch_selected_pair_count,
                    batch_selected_boundary_count,
                    batch_selected_lambdarank_pair_count,
                    batch_soft_hits_positive_count,
                    batch_normalized_soft_hits_positive_count,
                    batch_normalized_soft_hits_negative_count,
                ) = compute_batch_loss(
                    etype,
                    batch_queries,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    path_parameters,
                    max_norm=gradient_clip_norm,
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                listwise_losses.append(
                    float(listwise_loss.detach().cpu())
                )
                ranking_losses.append(
                    float(ranking_loss.detach().cpu())
                )
                topk_boundary_losses.append(
                    float(topk_boundary_loss.detach().cpu())
                )
                lambdarank_losses.append(
                    float(lambdarank_loss.detach().cpu())
                )
                soft_hits_losses.append(
                    float(soft_hits_loss.detach().cpu())
                )
                normalized_soft_hits_losses.append(
                    float(normalized_soft_hits_loss.detach().cpu())
                )
                selected_pair_count += batch_selected_pair_count
                selected_boundary_count += batch_selected_boundary_count
                selected_lambdarank_pair_count += (
                    batch_selected_lambdarank_pair_count
                )
                soft_hits_positive_count += (
                    batch_soft_hits_positive_count
                )
                normalized_soft_hits_positive_count += (
                    batch_normalized_soft_hits_positive_count
                )
                normalized_soft_hits_negative_count += (
                    batch_normalized_soft_hits_negative_count
                )
                covered_edges += positive_count
        if covered_edges != available_edges:
            raise RuntimeError(
                "BioGT path query-group coverage mismatch: "
                f"covered={covered_edges}, available={available_edges}"
            )
        epoch_loss = float(np.mean(losses))
        epoch_listwise_loss = float(np.mean(listwise_losses))
        epoch_ranking_loss = float(np.mean(ranking_losses))
        epoch_topk_boundary_loss = float(
            np.mean(topk_boundary_losses)
        )
        epoch_lambdarank_loss = float(np.mean(lambdarank_losses))
        epoch_soft_hits_loss = float(np.mean(soft_hits_losses))
        epoch_normalized_soft_hits_loss = float(
            np.mean(normalized_soft_hits_losses)
        )
        epoch_losses.append(epoch_loss)
        epoch_listwise_losses.append(epoch_listwise_loss)
        epoch_ranking_losses.append(epoch_ranking_loss)
        epoch_topk_boundary_losses.append(epoch_topk_boundary_loss)
        epoch_lambdarank_losses.append(epoch_lambdarank_loss)
        epoch_soft_hits_losses.append(epoch_soft_hits_loss)
        epoch_normalized_soft_hits_losses.append(
            epoch_normalized_soft_hits_loss
        )
        selected_pairs_per_epoch.append(selected_pair_count)
        selected_boundaries_per_epoch.append(selected_boundary_count)
        selected_lambdarank_pairs_per_epoch.append(
            selected_lambdarank_pair_count
        )
        soft_hits_positive_counts_per_epoch.append(
            soft_hits_positive_count
        )
        normalized_soft_hits_positive_counts_per_epoch.append(
            normalized_soft_hits_positive_count
        )
        normalized_soft_hits_negative_counts_per_epoch.append(
            normalized_soft_hits_negative_count
        )
        logging.info(
            "BioGT path pretrain epoch %d/%d loss=%.5f "
            "listwise_loss=%.5f ranking_loss=%.5f "
            "topk_boundary_loss=%.5f steps=%d "
            "lambdarank_loss=%.5f "
            "soft_hits_loss=%.5f "
            "normalized_soft_hits_loss=%.5f "
            "covered_edges=%d available_edges=%d objective=%s "
            "hard_negatives=%d rank_weight=%.5f selected_pairs=%d "
            "topk_boundary_weight=%.5f topk_boundary_k=%d "
            "topk_boundary_margin=%.5f selected_boundaries=%d "
            "lambdarank_weight=%.5f lambdarank_k=%d "
            "lambdarank_temperature=%.5f "
            "selected_lambdarank_pairs=%d "
            "soft_hits_weight=%.5f soft_hits_k=%d "
            "soft_hits_temperature=%.5f soft_hits_positive_count=%d "
            "normalized_soft_hits_weight=%.5f "
            "normalized_soft_hits_k=%d "
            "normalized_soft_hits_temperature=%.5f "
            "normalized_soft_hits_positive_count=%d "
            "normalized_soft_hits_negative_count=%d.",
            epoch + 1,
            epochs,
            epoch_loss,
            epoch_listwise_loss,
            epoch_ranking_loss,
            epoch_topk_boundary_loss,
            steps_per_epoch,
            epoch_lambdarank_loss,
            epoch_soft_hits_loss,
            epoch_normalized_soft_hits_loss,
            available_edges,
            available_edges,
            "full_candidate_softmax",
            hard_negative_count,
            rank_weight,
            selected_pair_count,
            topk_boundary_weight,
            topk_boundary_k,
            topk_boundary_margin,
            selected_boundary_count,
            lambdarank_weight,
            lambdarank_k,
            lambdarank_temperature,
            selected_lambdarank_pair_count,
            soft_hits_weight,
            soft_hits_k,
            soft_hits_temperature,
            soft_hits_positive_count,
            normalized_soft_hits_weight,
            normalized_soft_hits_k,
            normalized_soft_hits_temperature,
            normalized_soft_hits_positive_count,
            normalized_soft_hits_negative_count,
        )
    model.clear_embedding_cache()
    return {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "losses": epoch_losses,
        "listwise_losses": epoch_listwise_losses,
        "ranking_losses": epoch_ranking_losses,
        "topk_boundary_losses": epoch_topk_boundary_losses,
        "lambdarank_losses": epoch_lambdarank_losses,
        "soft_hits_losses": epoch_soft_hits_losses,
        "normalized_soft_hits_losses": epoch_normalized_soft_hits_losses,
        "objective": "full_candidate_softmax",
        "hard_negatives": hard_negative_count,
        "rank_weight": rank_weight,
        "ranking_auxiliary_enabled": bool(
            hard_negative_count > 0 and rank_weight > 0
        ),
        "selected_pairs_per_epoch": selected_pairs_per_epoch,
        "topk_boundary_weight": topk_boundary_weight,
        "topk_boundary_k": topk_boundary_k,
        "topk_boundary_margin": topk_boundary_margin,
        "topk_boundary_enabled": bool(topk_boundary_weight > 0),
        "selected_boundaries_per_epoch": selected_boundaries_per_epoch,
        "lambdarank_weight": lambdarank_weight,
        "lambdarank_k": lambdarank_k,
        "lambdarank_temperature": lambdarank_temperature,
        "lambdarank_enabled": bool(lambdarank_weight > 0),
        "selected_lambdarank_pairs_per_epoch": (
            selected_lambdarank_pairs_per_epoch
        ),
        "soft_hits_weight": soft_hits_weight,
        "soft_hits_k": soft_hits_k,
        "soft_hits_temperature": soft_hits_temperature,
        "soft_hits_enabled": bool(soft_hits_weight > 0),
        "soft_hits_positive_counts_per_epoch": (
            soft_hits_positive_counts_per_epoch
        ),
        "normalized_soft_hits_weight": normalized_soft_hits_weight,
        "normalized_soft_hits_k": normalized_soft_hits_k,
        "normalized_soft_hits_temperature": normalized_soft_hits_temperature,
        "normalized_soft_hits_enabled": bool(
            normalized_soft_hits_weight > 0
        ),
        "normalized_soft_hits_positive_counts_per_epoch": (
            normalized_soft_hits_positive_counts_per_epoch
        ),
        "normalized_soft_hits_negative_counts_per_epoch": (
            normalized_soft_hits_negative_counts_per_epoch
        ),
        "target_types": (
            requested_types
            or sorted({
                target_modality_for_etype(etype, target_types)
                for etype in selected_etypes
            })
        ),
        "available_edges": available_edges,
        "covered_edges_per_epoch": available_edges,
        "batch_unit": "unique_queries",
        "unique_queries_per_epoch": unique_queries_per_epoch,
        "target_weighting": (
            "equal_modality_gradient_accumulation"
            if (
                model.biogt_neural_path_dim > 0
                or model.biogt_relbf_dim > 0
            )
            else "edge_proportional"
        ),
        "readout_warmup_steps": (
            1
            if (
                model.biogt_neural_path_dim > 0
                or model.biogt_relbf_dim > 0
            )
            else 0
        ),
        "batch_counts_by_target": {
            target_modality_for_etype(
                etype,
                target_types,
            ): count
            for etype, count in batch_counts_by_etype.items()
        },
    }


def run_unified_training(args, device):
    if args.model_type.lower() != "original":
        raise ValueError(
            "The unified engine implements the shared BioGT model only. "
            "Use --training-engine legacy for GCN/GAT/Transformer baselines."
        )
    for path_name in ("nodes_csv", "edges_csv"):
        path = os.path.abspath(getattr(args, path_name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} was not found: {path}")
    if (
        getattr(
            args,
            "initialize_biogt_relbf_jk_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-jk-from-checkpoint requires "
            "--load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_edge_attention_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-edge-attention-from-checkpoint "
            "requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_deep_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-deep-from-checkpoint requires "
            "--load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_indicator_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-indicator-from-checkpoint requires "
            "--load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_indicator_support_gating_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-indicator-support-gating-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-indicator-query-local-relation-"
            "transition-from-checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_path_layer_attention_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-path-layer-attention-from-checkpoint "
            "requires --load_model_path"
        )
    if (
        getattr(args, "initialize_biogt_relbf_path_token_transformer_from_checkpoint", False)
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-path-token-transformer-from-checkpoint "
            "requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-candidate-competition-boundary-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-parent-top10-boundary-comparator-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-boundary-window-rank-slot-expert-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_boundary_exchange_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-boundary-exchange-from-checkpoint "
            "requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_query_relation_adapter_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-query-relation-adapter-from-checkpoint "
            "requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_indicator_query_boundary_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-indicator-query-boundary-from-checkpoint "
            "requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_indicator_relation_transition_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-indicator-relation-transition-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-challenger-boundary-promotion-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-gatekeeper-challenger-paired-margin-from-"
            "checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-boundary-prototype-ordinal-adapter-from-checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_multi_depth_score_router_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-multi-depth-score-router-from-checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-multi-anchor-rank-landmark-"
            "calibrator-from-checkpoint requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-query-token-top64-candidate-set-"
            "transformer-from-checkpoint requires --load_model_path"
        )
    _apply_checkpoint_model_args(args)

    start_time = time.time()
    logging.info("Initializing graph data for the unified sampled engine.")
    graph_creator = _graph_creator_from_args(args, device)
    g_full_cpu, g_train_cpu, g_val_cpu, g_test_cpu = graph_creator.get_graphs()
    node_features = _node_feature_tensors(
        graph_creator.get_node_features(),
        g_full_cpu,
    )
    target_types = [
        item.strip()
        for item in args.train_target_types.split(",")
        if item.strip() and item.strip() in g_full_cpu.ntypes
    ]
    if not target_types:
        raise ValueError("None of --train_target_types exist in the graph")
    target_disease_finetune_types = parse_target_type_list(
        getattr(args, "kge_target_disease_finetune_types", "")
    )
    candidate_bias_finetune_types = parse_target_type_list(
        getattr(args, "kge_candidate_bias_finetune_types", "")
    )
    target_residual_finetune_types = parse_target_type_list(
        getattr(args, "kge_target_residual_finetune_types", "")
    )
    path_propagation_finetune_types = parse_target_type_list(
        getattr(args, "kge_path_propagation_finetune_types", "")
    )
    biogt_path_pretrain_types = parse_target_type_list(
        getattr(args, "biogt_path_pretrain_types", "")
    )
    unknown_target_disease_finetune_types = (
        set(target_disease_finetune_types) - set(target_types)
    )
    if unknown_target_disease_finetune_types:
        raise ValueError(
            "Unknown --kge-target-disease-finetune-types values: "
            f"{sorted(unknown_target_disease_finetune_types)}"
        )
    unknown_candidate_bias_finetune_types = (
        set(candidate_bias_finetune_types) - set(target_types)
    )
    if unknown_candidate_bias_finetune_types:
        raise ValueError(
            "Unknown --kge-candidate-bias-finetune-types values: "
            f"{sorted(unknown_candidate_bias_finetune_types)}"
        )
    unknown_target_residual_finetune_types = (
        set(target_residual_finetune_types) - set(target_types)
    )
    if unknown_target_residual_finetune_types:
        raise ValueError(
            "Unknown --kge-target-residual-finetune-types values: "
            f"{sorted(unknown_target_residual_finetune_types)}"
        )
    unknown_path_propagation_finetune_types = (
        set(path_propagation_finetune_types) - set(target_types)
    )
    if unknown_path_propagation_finetune_types:
        raise ValueError(
            "Unknown --kge-path-propagation-finetune-types values: "
            f"{sorted(unknown_path_propagation_finetune_types)}"
        )
    unknown_biogt_path_pretrain_types = (
        set(biogt_path_pretrain_types) - set(target_types)
    )
    if unknown_biogt_path_pretrain_types:
        raise ValueError(
            "Unknown --biogt-path-pretrain-types values: "
            f"{sorted(unknown_biogt_path_pretrain_types)}"
        )
    active_finetune_type_sets = sum(
        bool(values)
        for values in (
            target_disease_finetune_types,
            candidate_bias_finetune_types,
            target_residual_finetune_types,
            path_propagation_finetune_types,
        )
    )
    if active_finetune_type_sets > 1:
        raise ValueError(
            "Target-disease, candidate-bias, target-residual, and "
            "path-propagation "
            "fine-tuning types are mutually exclusive"
        )

    train_target_etypes = select_target_etypes(
        g_train_cpu,
        target_types,
        source_node_type=args.source_node_type,
    )
    val_target_etypes = [
        etype
        for etype in train_target_etypes
        if etype in g_val_cpu.canonical_etypes and g_val_cpu.num_edges(etype) > 0
    ]
    test_target_etypes = [
        etype
        for etype in train_target_etypes
        if etype in g_test_cpu.canonical_etypes and g_test_cpu.num_edges(etype) > 0
    ]
    if not train_target_etypes:
        raise ValueError("No target edge types were found in the training graph")

    graph_backend = resolve_unified_graph_backend(
        args.unified_graph_backend,
        device,
    )
    model = UnifiedDiseaseAssociationModel(
        node_feature_embeddings=node_features,
        hgcn_hidden=args.hgcn_hidden,
        hgcn_out=args.hgcn_out,
        num_transformer_layers=args.num_transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff=args.transformer_ff,
        num_latents=args.num_latents,
        ntypes=g_full_cpu.ntypes,
        etypes=g_full_cpu.canonical_etypes,
        target_node_types=target_types,
        num_nodes_dict={
            node_type: g_full_cpu.num_nodes(node_type)
            for node_type in g_full_cpu.ntypes
        },
        node_info=graph_creator.get_node_info(),
        dropout=args.dropout,
        num_gnn_layers=args.num_gnn_layers,
        freeze_pretrained=args.freeze_pretrained,
        disable_hgcn=args.disable_hgcn,
        disable_transformer=args.disable_transformer,
        disable_moe=args.disable_moe,
        fusion_mode=args.fusion_mode,
        modality_fusion_modes=args.modality_fusion_modes,
        scorer_mode=args.scorer_mode,
        kge_pretrain_scope=args.global_bilinear_pretrain_scope,
        kge_relation_model=args.kge_relation_model,
        kge_relation_rank=args.kge_relation_rank,
        kge_context_input_weight=args.kge_context_input_weight,
        kge_target_specific_disease_embeddings=(
            args.kge_target_specific_disease_embeddings
        ),
        kge_target_disease_shared_weight=(
            args.kge_target_disease_shared_weight
        ),
        kge_candidate_bias=args.kge_candidate_bias,
        kge_target_residual_rank=args.kge_target_residual_rank,
        kge_path_propagation_hops=args.kge_path_propagation_hops,
        kge_path_cross_modal=args.kge_path_cross_modal,
        kge_path_cross_modal_types=args.kge_path_cross_modal_types,
        kge_path_cross_modal_support_types=(
            args.kge_path_cross_modal_support_types
        ),
        biogt_path_propagation_hops=(
            getattr(args, "biogt_path_propagation_hops", 0)
        ),
        biogt_path_support_types=getattr(
            args,
            "biogt_path_support_types",
            "",
        ),
        biogt_neural_path_dim=getattr(
            args,
            "biogt_neural_path_dim",
            0,
        ),
        biogt_relbf_dim=getattr(args, "biogt_relbf_dim", 0),
        biogt_relbf_jk_readout=getattr(
            args,
            "biogt_relbf_jk_readout",
            False,
        ),
        biogt_relbf_edge_attention=getattr(
            args,
            "biogt_relbf_edge_attention",
            False,
        ),
        biogt_relbf_deep_refinement=getattr(
            args,
            "biogt_relbf_deep_refinement",
            False,
        ),
        biogt_relbf_indicator_residual=getattr(
            args,
            "biogt_relbf_indicator_residual",
            False,
        ),
        biogt_relbf_indicator_support_gating=getattr(
            args,
            "biogt_relbf_indicator_support_gating",
            False,
        ),
        biogt_relbf_indicator_semantic_readout=getattr(
            args,
            "biogt_relbf_indicator_semantic_readout",
            False,
        ),
        biogt_relbf_query_relation_adapter_rank=getattr(
            args,
            "biogt_relbf_query_relation_adapter_rank",
            0,
        ),
        biogt_relbf_indicator_query_boundary_rank=getattr(
            args,
            "biogt_relbf_indicator_query_boundary_rank",
            0,
        ),
        biogt_relbf_indicator_multirule_count=getattr(
            args,
            "biogt_relbf_indicator_multirule_count",
            0,
        ),
        biogt_relbf_indicator_multirule_readout_rank=getattr(
            args,
            "biogt_relbf_indicator_multirule_readout_rank",
            0,
        ),
        biogt_relbf_indicator_relation_transition_layers=getattr(
            args,
            "biogt_relbf_indicator_relation_transition_layers",
            0,
        ),
        biogt_relbf_indicator_relation_transition_rank=getattr(
            args,
            "biogt_relbf_indicator_relation_transition_rank",
            0,
        ),
        biogt_relbf_indicator_query_local_relation_transition_layers=getattr(
            args,
            "biogt_relbf_indicator_query_local_relation_transition_layers",
            0,
        ),
        biogt_relbf_indicator_query_local_relation_transition_rank=getattr(
            args,
            "biogt_relbf_indicator_query_local_relation_transition_rank",
            0,
        ),
        biogt_relbf_path_layer_attention_rank=getattr(
            args,
            "biogt_relbf_path_layer_attention_rank",
            0,
        ),
        biogt_relbf_path_token_transformer=getattr(
            args, "biogt_relbf_path_token_transformer", False
        ),
        biogt_relbf_candidate_competition_boundary=getattr(
            args, "biogt_relbf_candidate_competition_boundary", False
        ),
        biogt_relbf_parent_top10_boundary_comparator=getattr(
            args, "biogt_relbf_parent_top10_boundary_comparator", False
        ),
        biogt_relbf_boundary_window_rank_slot_expert=getattr(
            args, "biogt_relbf_boundary_window_rank_slot_expert", False
        ),
        biogt_relbf_boundary_exchange=getattr(
            args, "biogt_relbf_boundary_exchange", False
        ),
        biogt_relbf_challenger_boundary_promotion=getattr(
            args, "biogt_relbf_challenger_boundary_promotion", False
        ),
        biogt_relbf_gatekeeper_challenger_paired_margin=getattr(
            args, "biogt_relbf_gatekeeper_challenger_paired_margin", False
        ),
        biogt_relbf_boundary_prototype_ordinal_adapter=getattr(
            args, "biogt_relbf_boundary_prototype_ordinal_adapter", False
        ),
        biogt_relbf_multi_depth_score_router=getattr(
            args, "biogt_relbf_multi_depth_score_router", False
        ),
        biogt_relbf_dual_stream_trajectory_agreement_calibrator=getattr(
            args,
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
            False,
        ),
        biogt_relbf_multi_anchor_rank_landmark_calibrator=getattr(
            args,
            "biogt_relbf_multi_anchor_rank_landmark_calibrator",
            False,
        ),
        biogt_relbf_query_token_top64_candidate_set_transformer=getattr(
            args,
            "biogt_relbf_query_token_top64_candidate_set_transformer",
            False,
        ),
        global_bilinear_score_weight=(
            args.global_bilinear_score_weight
        ),
        contextual_score_weight=args.contextual_score_weight,
        sparse_transformer_topk=args.sparse_transformer_topk,
        sparse_transformer_max_hops=args.sparse_transformer_max_hops,
        sparse_transformer_hop_penalty=args.sparse_transformer_hop_penalty,
        sparse_transformer_evidence_weight=args.sparse_transformer_evidence_weight,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden_dim=args.moe_hidden_dim,
        embedding_cache_size=args.embedding_cache_size,
        graph_backend=graph_backend,
    ).to(device)
    model.set_target_etypes(train_target_etypes)
    path_graph_metadata = model.configure_train_only_path_graph(
        g_train_cpu,
        train_target_etypes,
    )
    if path_graph_metadata:
        logging.info(
            "Configured train-only path propagation graph: %s",
            path_graph_metadata,
        )
    biogt_path_graph_metadata = (
        model.configure_train_only_biogt_path_graph(
            g_train_cpu,
            train_target_etypes,
        )
    )
    if biogt_path_graph_metadata:
        logging.info(
            "Configured train-only pure BioGT path graph: %s",
            biogt_path_graph_metadata,
        )
    logging.info(
        "Unified model initialized with %d trainable parameters, modalities %s, "
        "and graph backend %s.",
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        target_types,
        graph_backend,
    )
    logging.info("Modality fusion modes: %s", model.modality_fusion_modes)

    if args.load_model_path:
        _load_checkpoint(
            args.load_model_path,
            model,
            device,
            initialize_target_disease_from_shared=getattr(
                args,
                "initialize_target_disease_from_shared_checkpoint",
                False,
            ),
            initialize_candidate_bias_from_checkpoint=getattr(
                args,
                "initialize_kge_candidate_bias_from_checkpoint",
                False,
            ),
            initialize_target_residual_from_checkpoint=getattr(
                args,
                "initialize_kge_target_residual_from_checkpoint",
                False,
            ),
            initialize_path_propagation_from_checkpoint=getattr(
                args,
                "initialize_kge_path_propagation_from_checkpoint",
                False,
            ),
            initialize_biogt_path_from_checkpoint=getattr(
                args,
                "initialize_biogt_path_from_checkpoint",
                False,
            ),
            initialize_biogt_neural_path_from_checkpoint=getattr(
                args,
                "initialize_biogt_neural_path_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_jk_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_jk_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_edge_attention_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_edge_attention_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_deep_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_deep_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_support_gating_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_support_gating_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_semantic_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_semantic_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_query_relation_adapter_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_query_relation_adapter_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_query_boundary_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_query_boundary_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_multirule_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_multirule_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_relation_transition_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_relation_transition_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_indicator_query_local_relation_transition_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_path_layer_attention_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_path_layer_attention_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_path_token_transformer_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_path_token_transformer_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_candidate_competition_boundary_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_parent_top10_boundary_comparator_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_boundary_window_rank_slot_expert_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_boundary_exchange_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_boundary_exchange_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_challenger_boundary_promotion_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_gatekeeper_challenger_paired_margin_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_boundary_prototype_ordinal_adapter_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_multi_depth_score_router_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_multi_depth_score_router_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint",
                False,
            ),
            initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint=getattr(
                args,
                "initialize_biogt_relbf_query_token_top64_candidate_set_transformer_from_checkpoint",
                False,
            ),
        )
        logging.info("Loaded unified checkpoint from %s.", args.load_model_path)

    freeze_shared_kge_for_target_disease = bool(
        getattr(
            args,
            "freeze_shared_kge_for_target_disease_finetune",
            False,
        )
    )
    freeze_kge_for_candidate_bias = bool(
        getattr(
            args,
            "freeze_kge_for_candidate_bias_finetune",
            False,
        )
    )
    freeze_kge_for_target_residual = bool(
        getattr(
            args,
            "freeze_kge_for_target_residual_finetune",
            False,
        )
    )
    freeze_kge_for_path_propagation = bool(
        getattr(
            args,
            "freeze_kge_for_path_propagation_finetune",
            False,
        )
    )
    active_kge_freeze_modes = sum(
        (
            freeze_shared_kge_for_target_disease,
            freeze_kge_for_candidate_bias,
            freeze_kge_for_target_residual,
            freeze_kge_for_path_propagation,
        )
    )
    if active_kge_freeze_modes > 1:
        raise ValueError(
            "Target-disease, candidate-bias, target-residual, and "
            "path-propagation KGE "
            "freezing modes are mutually exclusive"
        )
    shared_kge_frozen_parameter_count = 0
    candidate_bias_frozen_parameter_count = 0
    target_residual_frozen_parameter_count = 0
    path_propagation_frozen_parameter_count = 0
    biogt_path_frozen_parameter_count = 0
    if (
        target_disease_finetune_types
        and not freeze_shared_kge_for_target_disease
    ):
        raise ValueError(
            "--kge-target-disease-finetune-types requires "
            "--freeze-shared-kge-for-target-disease-finetune"
        )
    if freeze_shared_kge_for_target_disease:
        if not args.load_model_path:
            raise ValueError(
                "--freeze-shared-kge-for-target-disease-finetune requires "
                "--load_model_path"
            )
        if not args.kge_target_specific_disease_embeddings:
            raise ValueError(
                "--freeze-shared-kge-for-target-disease-finetune requires "
                "target-specific disease embeddings from either a migrated "
                "shared checkpoint or an already specialized checkpoint"
            )
        if (
            args.global_bilinear_pretrain_epochs <= 0
            and args.global_bilinear_pretrain_target_finetune_epochs <= 0
        ):
            raise ValueError(
                "--freeze-shared-kge-for-target-disease-finetune requires "
                "positive target-relation pretraining epochs"
            )
        if args.global_bilinear_pretrain_scope != "target_relations":
            raise ValueError(
                "--freeze-shared-kge-for-target-disease-finetune requires "
                "--global-bilinear-pretrain-scope target_relations"
            )
        shared_kge_frozen_parameter_count = (
            model.freeze_shared_kge_for_target_disease_finetune(
                target_disease_finetune_types or None
            )
        )
        logging.info(
            "Froze %d shared or excluded target KGE parameters; "
            "target-specific disease fine-tuning types=%s.",
            shared_kge_frozen_parameter_count,
            model.target_disease_finetune_types,
        )
    if (
        candidate_bias_finetune_types
        and not freeze_kge_for_candidate_bias
    ):
        raise ValueError(
            "--kge-candidate-bias-finetune-types requires "
            "--freeze-kge-for-candidate-bias-finetune"
        )
    if freeze_kge_for_candidate_bias:
        if not args.load_model_path:
            raise ValueError(
                "--freeze-kge-for-candidate-bias-finetune requires "
                "--load_model_path"
            )
        if not args.kge_candidate_bias:
            raise ValueError(
                "--freeze-kge-for-candidate-bias-finetune requires "
                "--kge-candidate-bias"
            )
        if (
            args.global_bilinear_pretrain_epochs <= 0
            and args.global_bilinear_pretrain_target_finetune_epochs <= 0
        ):
            raise ValueError(
                "--freeze-kge-for-candidate-bias-finetune requires "
                "positive target-relation pretraining epochs"
            )
        if args.global_bilinear_pretrain_scope != "target_relations":
            raise ValueError(
                "--freeze-kge-for-candidate-bias-finetune requires "
                "--global-bilinear-pretrain-scope target_relations"
            )
        candidate_bias_frozen_parameter_count = (
            model.freeze_kge_for_candidate_bias_finetune(
                candidate_bias_finetune_types or None
            )
        )
        logging.info(
            "Froze %d existing or excluded KGE parameters; "
            "candidate-bias fine-tuning types=%s.",
            candidate_bias_frozen_parameter_count,
            model.candidate_bias_finetune_types,
        )
    if (
        target_residual_finetune_types
        and not freeze_kge_for_target_residual
    ):
        raise ValueError(
            "--kge-target-residual-finetune-types requires "
            "--freeze-kge-for-target-residual-finetune"
        )
    if freeze_kge_for_target_residual:
        if not args.load_model_path:
            raise ValueError(
                "--freeze-kge-for-target-residual-finetune requires "
                "--load_model_path"
            )
        if args.kge_target_residual_rank <= 0:
            raise ValueError(
                "--freeze-kge-for-target-residual-finetune requires a "
                "positive --kge-target-residual-rank"
            )
        if (
            args.global_bilinear_pretrain_epochs <= 0
            and args.global_bilinear_pretrain_target_finetune_epochs <= 0
        ):
            raise ValueError(
                "--freeze-kge-for-target-residual-finetune requires "
                "positive target-relation pretraining epochs"
            )
        if args.global_bilinear_pretrain_scope != "target_relations":
            raise ValueError(
                "--freeze-kge-for-target-residual-finetune requires "
                "--global-bilinear-pretrain-scope target_relations"
            )
        target_residual_frozen_parameter_count = (
            model.freeze_kge_for_target_residual_finetune(
                target_residual_finetune_types or None
            )
        )
        logging.info(
            "Froze %d existing or excluded KGE parameters; "
            "target-residual fine-tuning types=%s.",
            target_residual_frozen_parameter_count,
            model.target_residual_finetune_types,
        )
    if (
        path_propagation_finetune_types
        and not freeze_kge_for_path_propagation
    ):
        raise ValueError(
            "--kge-path-propagation-finetune-types requires "
            "--freeze-kge-for-path-propagation-finetune"
        )
    if freeze_kge_for_path_propagation:
        if not args.load_model_path:
            raise ValueError(
                "--freeze-kge-for-path-propagation-finetune requires "
                "--load_model_path"
            )
        if args.kge_path_propagation_hops not in {3, 5}:
            raise ValueError(
                "--freeze-kge-for-path-propagation-finetune requires "
                "--kge-path-propagation-hops 3 or 5"
            )
        if (
            args.global_bilinear_pretrain_epochs <= 0
            and args.global_bilinear_pretrain_target_finetune_epochs <= 0
        ):
            raise ValueError(
                "--freeze-kge-for-path-propagation-finetune requires "
                "positive target-relation pretraining epochs"
            )
        if args.global_bilinear_pretrain_scope != "target_relations":
            raise ValueError(
                "--freeze-kge-for-path-propagation-finetune requires "
                "--global-bilinear-pretrain-scope target_relations"
            )
        path_propagation_frozen_parameter_count = (
            model.freeze_kge_for_path_propagation_finetune(
                path_propagation_finetune_types or None
            )
        )
        logging.info(
            "Froze %d existing or excluded parameters; "
            "path-propagation fine-tuning types=%s.",
            path_propagation_frozen_parameter_count,
            model.path_propagation_finetune_types,
        )
    freeze_biogt_base_for_path = bool(
        getattr(
            args,
            "freeze_biogt_base_for_path_finetune",
            False,
        )
    )
    biogt_relbf_target_only_finetune = bool(
        getattr(args, "biogt_relbf_target_only_finetune", False)
    )
    biogt_relbf_jk_only_finetune = bool(
        getattr(args, "biogt_relbf_jk_only_finetune", False)
    )
    biogt_relbf_edge_attention_only_finetune = bool(getattr(
        args,
        "biogt_relbf_edge_attention_only_finetune",
        False,
    ))
    biogt_relbf_deep_only_finetune = bool(getattr(
        args,
        "biogt_relbf_deep_only_finetune",
        False,
    ))
    biogt_relbf_indicator_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_only_finetune",
        False,
    ))
    biogt_relbf_indicator_support_gate_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_support_gate_only_finetune",
        False,
    ))
    biogt_relbf_indicator_semantic_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_semantic_only_finetune",
        False,
    ))
    biogt_relbf_query_relation_adapter_only_finetune = bool(getattr(
        args,
        "biogt_relbf_query_relation_adapter_only_finetune",
        False,
    ))
    biogt_relbf_indicator_query_boundary_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_query_boundary_only_finetune",
        False,
    ))
    biogt_relbf_indicator_multirule_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_multirule_only_finetune",
        False,
    ))
    biogt_relbf_indicator_relation_transition_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_relation_transition_only_finetune",
        False,
    ))
    biogt_relbf_indicator_query_local_relation_transition_only_finetune = bool(getattr(
        args,
        "biogt_relbf_indicator_query_local_relation_transition_only_finetune",
        False,
    ))
    biogt_relbf_path_layer_attention_only_finetune = bool(getattr(
        args,
        "biogt_relbf_path_layer_attention_only_finetune",
        False,
    ))
    biogt_relbf_path_token_transformer_only_finetune = bool(getattr(
        args, "biogt_relbf_path_token_transformer_only_finetune", False
    ))
    biogt_relbf_candidate_competition_boundary_only_finetune = bool(getattr(
        args,
        "biogt_relbf_candidate_competition_boundary_only_finetune",
        False,
    ))
    biogt_relbf_parent_top10_boundary_comparator_only_finetune = bool(getattr(
        args,
        "biogt_relbf_parent_top10_boundary_comparator_only_finetune",
        False,
    ))
    biogt_relbf_boundary_window_rank_slot_expert_only_finetune = bool(getattr(
        args,
        "biogt_relbf_boundary_window_rank_slot_expert_only_finetune",
        False,
    ))
    biogt_relbf_boundary_exchange_only_finetune = bool(getattr(
        args, "biogt_relbf_boundary_exchange_only_finetune", False
    ))
    biogt_relbf_challenger_boundary_promotion_only_finetune = bool(getattr(
        args, "biogt_relbf_challenger_boundary_promotion_only_finetune", False
    ))
    biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune = bool(getattr(
        args,
        "biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune",
        False,
    ))
    biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune = bool(getattr(
        args, "biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune", False
    ))
    biogt_relbf_multi_depth_score_router_only_finetune = bool(getattr(
        args, "biogt_relbf_multi_depth_score_router_only_finetune", False
    ))
    biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune = bool(
        getattr(
            args,
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune",
            False,
        )
    )
    biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune = bool(
        getattr(
            args,
            "biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune",
            False,
        )
    )
    biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune = bool(
        getattr(
            args,
            "biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune",
            False,
        )
    )
    relbf_finetune_modes = sum(bool(value) for value in (
        biogt_relbf_target_only_finetune,
        biogt_relbf_jk_only_finetune,
        biogt_relbf_edge_attention_only_finetune,
        biogt_relbf_deep_only_finetune,
        biogt_relbf_indicator_only_finetune,
        biogt_relbf_indicator_support_gate_only_finetune,
        biogt_relbf_indicator_semantic_only_finetune,
        biogt_relbf_query_relation_adapter_only_finetune,
        biogt_relbf_indicator_query_boundary_only_finetune,
        biogt_relbf_indicator_multirule_only_finetune,
        biogt_relbf_indicator_relation_transition_only_finetune,
        biogt_relbf_indicator_query_local_relation_transition_only_finetune,
        biogt_relbf_path_layer_attention_only_finetune,
        biogt_relbf_path_token_transformer_only_finetune,
        biogt_relbf_candidate_competition_boundary_only_finetune,
        biogt_relbf_parent_top10_boundary_comparator_only_finetune,
        biogt_relbf_boundary_window_rank_slot_expert_only_finetune,
        biogt_relbf_boundary_exchange_only_finetune,
        biogt_relbf_challenger_boundary_promotion_only_finetune,
        biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune,
        biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune,
        biogt_relbf_multi_depth_score_router_only_finetune,
        biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune,
        biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune,
        biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune,
    ))
    if relbf_finetune_modes > 1:
        raise ValueError(
            "RelBF specialized fine-tuning modes are mutually exclusive"
        )
    if biogt_relbf_target_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-target-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if getattr(args, "biogt_relbf_dim", 0) <= 0:
            raise ValueError(
                "--biogt-relbf-target-only-finetune requires an enabled "
                "BioGT RelBF branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-target-only-finetune requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_jk_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-jk-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-jk-only-finetune requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_jk_readout", False)
        ):
            raise ValueError(
                "--biogt-relbf-jk-only-finetune requires an enabled JK "
                "readout"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-jk-only-finetune requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_edge_attention_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-edge-attention-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-edge-attention-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_edge_attention", False)
        ):
            raise ValueError(
                "--biogt-relbf-edge-attention-only-finetune requires an "
                "enabled edge-attention branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-edge-attention-only-finetune requires "
                "explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_deep_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-deep-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-deep-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_deep_refinement", False)
        ):
            raise ValueError(
                "--biogt-relbf-deep-only-finetune requires an enabled "
                "deep-refinement branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-deep-only-finetune requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_indicator_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
        ):
            raise ValueError(
                "--biogt-relbf-indicator-only-finetune requires an enabled "
                "relation-only indicator residual"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-indicator-only-finetune requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_indicator_support_gate_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-support-gate-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-support-gate-only-finetune "
            "requires --load_model_path"
        )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_dual_stream_trajectory_agreement_calibrator_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-dual-stream-trajectory-agreement-"
            "calibrator-from-checkpoint requires --load_model_path"
        )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or not getattr(
                args,
                "biogt_relbf_indicator_support_gating",
                False,
            )
        ):
            raise ValueError(
                "--biogt-relbf-indicator-support-gate-only-finetune "
                "requires enabled indicator support gating"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-indicator-support-gate-only-finetune "
                "requires explicit --biogt-path-pretrain-types"
            )
    if (
        getattr(
            args,
            "initialize_biogt_relbf_multi_anchor_rank_landmark_calibrator_from_checkpoint",
            False,
        )
        and not args.load_model_path
    ):
        raise ValueError(
            "--initialize-biogt-relbf-multi-anchor-rank-landmark-"
            "calibrator-from-checkpoint requires --load_model_path"
        )
    if biogt_relbf_indicator_semantic_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-semantic-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-semantic-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or not getattr(
                args,
                "biogt_relbf_indicator_semantic_readout",
                False,
            )
            or getattr(
                args,
                "biogt_relbf_indicator_support_gating",
                False,
            )
        ):
            raise ValueError(
                "--biogt-relbf-indicator-semantic-only-finetune requires "
                "the semantic readout without support gating"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-indicator-semantic-only-finetune requires "
                "explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_query_relation_adapter_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-query-relation-adapter-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-query-relation-adapter-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or getattr(
                args,
                "biogt_relbf_query_relation_adapter_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "--biogt-relbf-query-relation-adapter-only-finetune requires "
                "an enabled positive-rank adapter"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-query-relation-adapter-only-finetune requires "
                "explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_indicator_query_boundary_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-query-boundary-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-query-boundary-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(
                args,
                "biogt_relbf_indicator_query_boundary_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "--biogt-relbf-indicator-query-boundary-only-finetune "
                "requires an enabled positive-rank adapter"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-indicator-query-boundary-only-finetune "
                "requires explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_indicator_multirule_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-multirule-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-multirule-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(args, "biogt_relbf_indicator_multirule_count", 0) <= 0
            or getattr(
                args,
                "biogt_relbf_indicator_multirule_readout_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "--biogt-relbf-indicator-multirule-only-finetune requires "
                "an enabled positive-size multi-rule branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-indicator-multirule-only-finetune requires "
                "explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_indicator_relation_transition_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-relation-transition-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-relation-transition-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(
                args,
                "biogt_relbf_indicator_relation_transition_layers",
                0,
            ) <= 0
            or getattr(
                args,
                "biogt_relbf_indicator_relation_transition_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "--biogt-relbf-indicator-relation-transition-only-finetune "
                "requires an enabled positive-size relation-transition branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "--biogt-relbf-indicator-relation-transition-only-finetune "
                "requires explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_indicator_query_local_relation_transition_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-indicator-query-local-relation-transition-only-"
                "finetune requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-indicator-query-local-relation-transition-only-"
                "finetune requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_indicator_residual", False)
            or getattr(
                args,
                "biogt_relbf_indicator_query_local_relation_transition_layers",
                0,
            ) <= 0
            or getattr(
                args,
                "biogt_relbf_indicator_query_local_relation_transition_rank",
                0,
            ) <= 0
        ):
            raise ValueError(
                "Query-local relation-transition-only fine-tuning requires "
                "an enabled positive-size branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Query-local relation-transition-only fine-tuning requires "
                "explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_path_layer_attention_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-path-layer-attention-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-path-layer-attention-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or getattr(args, "biogt_relbf_path_layer_attention_rank", 0) <= 0
        ):
            raise ValueError(
                "Path-layer-attention-only fine-tuning requires an enabled "
                "positive-rank branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Path-layer-attention-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_path_token_transformer_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-path-token-transformer-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-path-token-transformer-only-finetune requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_path_token_transformer", False)
        ):
            raise ValueError(
                "Path-token-Transformer-only fine-tuning requires an enabled branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Path-token-Transformer-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_candidate_competition_boundary_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-candidate-competition-boundary-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-candidate-competition-boundary-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_candidate_competition_boundary", False
            )
        ):
            raise ValueError(
                "Candidate-competition-only fine-tuning requires an enabled "
                "branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Candidate-competition-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_parent_top10_boundary_comparator_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-parent-top10-boundary-comparator-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-parent-top10-boundary-comparator-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
        ):
            raise ValueError(
                "Parent-top10-boundary-comparator-only fine-tuning requires "
                "an enabled branch"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Parent-top10-boundary-comparator-only fine-tuning requires "
                "explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_boundary_window_rank_slot_expert_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-boundary-window-rank-slot-expert-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-boundary-window-rank-slot-expert-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
        ):
            raise ValueError(
                "Boundary-window-only fine-tuning requires the gene8 "
                "comparator and window expert"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Boundary-window-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_boundary_exchange_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-boundary-exchange-only-finetune requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-boundary-exchange-only-finetune requires "
                "--load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or not getattr(args, "biogt_relbf_boundary_exchange", False)
        ):
            raise ValueError(
                "Boundary-exchange-only fine-tuning requires the complete "
                "gene9 parent and exchange expert"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Boundary-exchange-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_challenger_boundary_promotion_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-challenger-boundary-promotion-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-challenger-boundary-promotion-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or not getattr(
                args, "biogt_relbf_challenger_boundary_promotion", False
            )
        ):
            raise ValueError(
                "Challenger-promotion-only fine-tuning requires the complete "
                "gene9 parent and gene11 expert without gene10"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Challenger-promotion-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-relbf-gatekeeper-challenger-paired-margin-only-finetune "
                "requires --freeze-biogt-base-for-path-finetune"
            )
        if not args.load_model_path:
            raise ValueError(
                "--biogt-relbf-gatekeeper-challenger-paired-margin-only-finetune "
                "requires --load_model_path"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(
                args, "biogt_relbf_parent_top10_boundary_comparator", False
            )
            or not getattr(
                args, "biogt_relbf_boundary_window_rank_slot_expert", False
            )
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or not getattr(
                args, "biogt_relbf_gatekeeper_challenger_paired_margin", False
            )
        ):
            raise ValueError(
                "Paired-margin-only fine-tuning requires the complete gene9 "
                "parent and Gene12 expert without gene10 or gene11"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Paired-margin-only fine-tuning requires explicit "
                "--biogt-path-pretrain-types"
            )
    if biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune:
        if not freeze_biogt_base_for_path or not args.load_model_path:
            raise ValueError(
                "Ordinal-adapter-only fine-tuning requires a frozen loaded parent"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or getattr(args, "biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or not getattr(args, "biogt_relbf_boundary_prototype_ordinal_adapter", False)
        ):
            raise ValueError(
                "Ordinal-adapter-only fine-tuning requires the complete gene9 parent and Gene13"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Ordinal-adapter-only fine-tuning requires explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_multi_depth_score_router_only_finetune:
        if not freeze_biogt_base_for_path or not args.load_model_path:
            raise ValueError(
                "Multi-depth-router-only fine-tuning requires a frozen loaded parent"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or getattr(args, "biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or getattr(args, "biogt_relbf_boundary_prototype_ordinal_adapter", False)
            or not getattr(args, "biogt_relbf_multi_depth_score_router", False)
        ):
            raise ValueError(
                "Multi-depth-router-only fine-tuning requires gene9 and Gene14"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Multi-depth-router-only fine-tuning requires explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune:
        if not freeze_biogt_base_for_path or not args.load_model_path:
            raise ValueError(
                "Trajectory-agreement-only fine-tuning requires a frozen loaded parent"
            )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or getattr(args, "biogt_relbf_boundary_exchange", False)
            or getattr(args, "biogt_relbf_challenger_boundary_promotion", False)
            or getattr(args, "biogt_relbf_gatekeeper_challenger_paired_margin", False)
            or getattr(args, "biogt_relbf_boundary_prototype_ordinal_adapter", False)
            or getattr(args, "biogt_relbf_multi_depth_score_router", False)
            or not getattr(
                args,
                "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
                False,
            )
        ):
            raise ValueError(
                "Trajectory-agreement-only fine-tuning requires gene9 and Gene15"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Trajectory-agreement-only fine-tuning requires explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune:
        if not freeze_biogt_base_for_path or not args.load_model_path:
            raise ValueError(
                "Multi-anchor-landmark-only fine-tuning requires a frozen loaded parent"
            )
        forbidden_args = (
            "biogt_relbf_boundary_exchange",
            "biogt_relbf_challenger_boundary_promotion",
            "biogt_relbf_gatekeeper_challenger_paired_margin",
            "biogt_relbf_boundary_prototype_ordinal_adapter",
            "biogt_relbf_multi_depth_score_router",
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
        )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or any(getattr(args, name, False) for name in forbidden_args)
            or not getattr(
                args, "biogt_relbf_multi_anchor_rank_landmark_calibrator", False
            )
        ):
            raise ValueError(
                "Multi-anchor-landmark-only fine-tuning requires gene9 and Gene16"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Multi-anchor-landmark-only fine-tuning requires explicit --biogt-path-pretrain-types"
            )
    if biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune:
        if not freeze_biogt_base_for_path or not args.load_model_path:
            raise ValueError(
                "Query-token-top64-only fine-tuning requires a frozen loaded parent"
            )
        forbidden_args = (
            "biogt_relbf_boundary_exchange",
            "biogt_relbf_challenger_boundary_promotion",
            "biogt_relbf_gatekeeper_challenger_paired_margin",
            "biogt_relbf_boundary_prototype_ordinal_adapter",
            "biogt_relbf_multi_depth_score_router",
            "biogt_relbf_dual_stream_trajectory_agreement_calibrator",
            "biogt_relbf_multi_anchor_rank_landmark_calibrator",
        )
        if (
            getattr(args, "biogt_relbf_dim", 0) <= 0
            or not getattr(args, "biogt_relbf_parent_top10_boundary_comparator", False)
            or not getattr(args, "biogt_relbf_boundary_window_rank_slot_expert", False)
            or any(getattr(args, name, False) for name in forbidden_args)
            or not getattr(
                args,
                "biogt_relbf_query_token_top64_candidate_set_transformer",
                False,
            )
        ):
            raise ValueError(
                "Query-token-top64-only fine-tuning requires gene9 and Gene17"
            )
        if not biogt_path_pretrain_types:
            raise ValueError(
                "Query-token-top64-only fine-tuning requires explicit --biogt-path-pretrain-types"
            )
    if freeze_biogt_base_for_path:
        if not args.load_model_path:
            raise ValueError(
                "--freeze-biogt-base-for-path-finetune requires "
                "--load_model_path"
            )
        if getattr(args, "biogt_path_propagation_hops", 0) not in {3, 5}:
            raise ValueError(
                "--freeze-biogt-base-for-path-finetune requires "
                "--biogt-path-propagation-hops 3 or 5"
            )
        if getattr(args, "biogt_path_pretrain_epochs", 0) <= 0:
            raise ValueError(
                "--freeze-biogt-base-for-path-finetune requires positive "
                "--biogt-path-pretrain-epochs"
            )
        biogt_path_frozen_parameter_count = (
            model.freeze_biogt_base_for_path_finetune(
                biogt_path_pretrain_types or None,
                relbf_target_only=biogt_relbf_target_only_finetune,
                relbf_jk_only=biogt_relbf_jk_only_finetune,
                relbf_edge_attention_only=(
                    biogt_relbf_edge_attention_only_finetune
                ),
                relbf_deep_only=biogt_relbf_deep_only_finetune,
                relbf_indicator_only=(
                    biogt_relbf_indicator_only_finetune
                ),
                relbf_indicator_support_gate_only=(
                    biogt_relbf_indicator_support_gate_only_finetune
                ),
                relbf_indicator_semantic_only=(
                    biogt_relbf_indicator_semantic_only_finetune
                ),
                relbf_query_relation_adapter_only=(
                    biogt_relbf_query_relation_adapter_only_finetune
                ),
                relbf_indicator_query_boundary_only=(
                    biogt_relbf_indicator_query_boundary_only_finetune
                ),
                relbf_indicator_multirule_only=(
                    biogt_relbf_indicator_multirule_only_finetune
                ),
                relbf_indicator_relation_transition_only=(
                    biogt_relbf_indicator_relation_transition_only_finetune
                ),
                relbf_indicator_query_local_relation_transition_only=(
                    biogt_relbf_indicator_query_local_relation_transition_only_finetune
                ),
                relbf_path_layer_attention_only=(
                    biogt_relbf_path_layer_attention_only_finetune
                ),
                relbf_path_token_transformer_only=(
                    biogt_relbf_path_token_transformer_only_finetune
                ),
                relbf_candidate_competition_boundary_only=(
                    biogt_relbf_candidate_competition_boundary_only_finetune
                ),
                relbf_parent_top10_boundary_comparator_only=(
                    biogt_relbf_parent_top10_boundary_comparator_only_finetune
                ),
                relbf_boundary_window_rank_slot_expert_only=(
                    biogt_relbf_boundary_window_rank_slot_expert_only_finetune
                ),
                relbf_boundary_exchange_only=(
                    biogt_relbf_boundary_exchange_only_finetune
                ),
                relbf_challenger_boundary_promotion_only=(
                    biogt_relbf_challenger_boundary_promotion_only_finetune
                ),
                relbf_gatekeeper_challenger_paired_margin_only=(
                    biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune
                ),
                relbf_boundary_prototype_ordinal_adapter_only=(
                    biogt_relbf_boundary_prototype_ordinal_adapter_only_finetune
                ),
                relbf_multi_depth_score_router_only=(
                    biogt_relbf_multi_depth_score_router_only_finetune
                ),
                relbf_dual_stream_trajectory_agreement_calibrator_only=(
                    biogt_relbf_dual_stream_trajectory_agreement_calibrator_only_finetune
                ),
                relbf_multi_anchor_rank_landmark_calibrator_only=(
                    biogt_relbf_multi_anchor_rank_landmark_calibrator_only_finetune
                ),
                relbf_query_token_top64_candidate_set_transformer_only=(
                    biogt_relbf_query_token_top64_candidate_set_transformer_only_finetune
                ),
            )
        )
        logging.info(
            "Froze %d BioGT base or excluded path parameters; "
            "pure path fine-tuning types=%s.",
            biogt_path_frozen_parameter_count,
            model.biogt_path_finetune_types,
        )
        if biogt_relbf_target_only_finetune:
            logging.info(
                "RelBF target-only fine-tuning keeps shared PNA and "
                "excluded target parameters frozen."
            )
        if biogt_relbf_jk_only_finetune:
            logging.info(
                "RelBF JK-only fine-tuning keeps the base, Path5, and "
                "parent RelBF parameters frozen."
            )
        if biogt_relbf_edge_attention_only_finetune:
            logging.info(
                "RelBF edge-attention-only fine-tuning keeps Transformer, "
                "HGCN, Path5, and parent RelBF parameters frozen."
            )
        if biogt_relbf_deep_only_finetune:
            logging.info(
                "RelBF deep-only fine-tuning keeps Transformer, HGCN, "
                "Path5, and the parent three-layer RelBF frozen."
            )
        if biogt_relbf_indicator_only_finetune:
            logging.info(
                "RelBF indicator-only fine-tuning keeps Transformer, HGCN, "
                "Path5, and disease-conditioned parent RelBF frozen."
            )
        if biogt_relbf_indicator_support_gate_only_finetune:
            logging.info(
                "RelBF indicator support-gate-only fine-tuning keeps "
                "Transformer, HGCN, Path5, parent RelBF, and indicator "
                "propagation/readout parameters frozen."
            )
        if biogt_relbf_indicator_semantic_only_finetune:
            logging.info(
                "RelBF indicator semantic-only fine-tuning keeps "
                "Transformer, HGCN, Path5, parent RelBF, indicator "
                "propagation, and original indicator readout frozen."
            )
        if biogt_relbf_query_relation_adapter_only_finetune:
            logging.info(
                "RelBF query-relation-adapter-only fine-tuning keeps "
                "Transformer, HGCN, Path5, parent RelBF, and indicator "
                "parameters frozen."
            )
        if biogt_relbf_indicator_multirule_only_finetune:
            logging.info(
                "RelBF indicator multi-rule-only fine-tuning keeps HGCN, "
                "Transformer, Path5, parent RelBF, and indicator parent "
                "parameters frozen."
            )
        if biogt_relbf_indicator_relation_transition_only_finetune:
            logging.info(
                "RelBF indicator relation-transition-only fine-tuning keeps "
                "HGCN, Transformer, Path5, parent RelBF, and indicator parent "
                "parameters frozen."
            )
        if biogt_relbf_indicator_query_local_relation_transition_only_finetune:
            logging.info(
                "Query-local relation-transition-only fine-tuning keeps HGCN, "
                "Transformer, Path5, parent RelBF, and indicator parent frozen."
            )
        if biogt_relbf_path_layer_attention_only_finetune:
            logging.info(
                "Path-layer-attention-only fine-tuning keeps HGCN, "
                "Transformer, Path5, and parent RelBF parameters frozen."
            )
        if biogt_relbf_path_token_transformer_only_finetune:
            logging.info(
                "Path-token-Transformer-only fine-tuning keeps HGCN, "
                "Transformer, Path5, and parent RelBF parameters frozen."
            )
        if biogt_relbf_candidate_competition_boundary_only_finetune:
            logging.info(
                "Candidate-competition-only fine-tuning keeps HGCN, "
                "Transformer, Path5, and parent RelBF parameters frozen."
            )
        if biogt_relbf_parent_top10_boundary_comparator_only_finetune:
            logging.info(
                "Parent-top10-boundary-comparator-only fine-tuning keeps "
                "HGCN, Transformer, Path5, and parent RelBF parameters frozen."
            )
        if biogt_relbf_boundary_window_rank_slot_expert_only_finetune:
            logging.info(
                "Boundary-window-only fine-tuning keeps the complete gene8 "
                "parent frozen and trains only the rank-slot expert."
            )
        if biogt_relbf_boundary_exchange_only_finetune:
            logging.info(
                "Boundary-exchange-only fine-tuning keeps the complete gene9 "
                "parent frozen and trains only the centered exchange expert."
            )
        if biogt_relbf_challenger_boundary_promotion_only_finetune:
            logging.info(
                "Challenger-promotion-only fine-tuning keeps the complete "
                "gene9 parent frozen and trains only the Gene11 expert."
            )
        if biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune:
            logging.info(
                "Paired-margin-only fine-tuning keeps the complete gene9 "
                "parent frozen and trains only the Gene12 expert."
            )

    dense_parameters = model.dense_parameters()
    sparse_parameters = model.sparse_embedding_parameters()
    dense_optimizer = torch.optim.AdamW(
        dense_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sparse_optimizer = (
        torch.optim.SparseAdam(sparse_parameters, lr=args.embedding_lr)
        if sparse_parameters
        else None
    )

    fanouts = parse_int_list(args.neighbor_fanouts)
    context_sampler = QueryContextSampler(fanouts, seed=args.seed)
    target_sampling_weights = parse_target_sampling_weights(
        args.target_sampling_weights
    )
    pretrain_negatives_by_target = parse_target_negative_counts(
        args.global_bilinear_pretrain_negatives_by_target
    )
    train_batcher = BalancedTargetEdgeBatcher(
        g_train_cpu,
        train_target_etypes,
        target_types,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        modality_weights=target_sampling_weights,
    )
    logging.info(
        "Target sampling weights: %s",
        target_sampling_weights or "uniform round-robin",
    )
    logging.info(
        "Pretraining negatives by target: %s",
        pretrain_negatives_by_target or "shared default",
    )
    pretrain_finetune_types = (
        target_disease_finetune_types
        or candidate_bias_finetune_types
        or target_residual_finetune_types
        or path_propagation_finetune_types
    )
    known_positive_index = KnownPositiveIndex.for_target_etypes(
        g_train_cpu,
        train_target_etypes,
    )
    candidate_nodes_by_etype = relation_candidate_nodes(
        g_full_cpu,
        train_target_etypes,
        target_types,
    )
    biogt_neural_path_graph_metadata = (
        model.configure_train_only_biogt_neural_path_graph(
            g_train_cpu,
            train_target_etypes,
            candidate_nodes_by_etype,
        )
    )
    if biogt_neural_path_graph_metadata:
        logging.info(
            "Configured compact train-only pure BioGT neural/RelBF "
            "graph: %s",
            biogt_neural_path_graph_metadata,
        )
    logging.info(
        "Relation-scoped training candidates: %s",
        {
            str(etype): int(nodes.numel())
            for etype, nodes in candidate_nodes_by_etype.items()
        },
    )
    freeze_independent = bool(
        getattr(args, "freeze_independent_after_pretrain", False)
    )
    if freeze_independent:
        if (
            args.global_bilinear_pretrain_epochs <= 0
            and args.global_bilinear_pretrain_target_finetune_epochs <= 0
            and not args.load_model_path
        ):
            raise ValueError(
                "--freeze-independent-after-pretrain requires "
                "a positive pretraining or target-finetuning epoch count, "
                "or a loaded checkpoint"
            )
        if model.scorer_mode != "hybrid_kge":
            raise ValueError(
                "--freeze-independent-after-pretrain requires "
                "--scorer-mode hybrid_kge"
            )
    biogt_path_pretraining = pretrain_biogt_path_branch(
        model,
        g_train_cpu,
        train_target_etypes,
        target_types,
        candidate_nodes_by_etype,
        epochs=getattr(args, "biogt_path_pretrain_epochs", 0),
        batch_size=getattr(
            args,
            "biogt_path_pretrain_batch_size",
            128,
        ),
        learning_rate=getattr(
            args,
            "biogt_path_pretrain_lr",
            3e-4,
        ),
        weight_decay=args.weight_decay,
        ranking_temperature=getattr(
            args,
            "biogt_path_pretrain_temperature",
            0.5,
        ),
        hard_negative_count=getattr(
            args,
            "biogt_path_pretrain_hard_negatives",
            0,
        ),
        rank_weight=getattr(
            args,
            "biogt_path_pretrain_rank_weight",
            0.0,
        ),
        topk_boundary_weight=getattr(
            args,
            "biogt_path_pretrain_topk_boundary_weight",
            0.0,
        ),
        topk_boundary_k=getattr(
            args,
            "biogt_path_pretrain_topk_boundary_k",
            10,
        ),
        topk_boundary_margin=getattr(
            args,
            "biogt_path_pretrain_topk_boundary_margin",
            0.0,
        ),
        lambdarank_weight=getattr(
            args,
            "biogt_path_pretrain_lambdarank_weight",
            0.0,
        ),
        lambdarank_k=getattr(
            args,
            "biogt_path_pretrain_lambdarank_k",
            10,
        ),
        lambdarank_temperature=getattr(
            args,
            "biogt_path_pretrain_lambdarank_temperature",
            1.0,
        ),
        soft_hits_weight=getattr(
            args,
            "biogt_path_pretrain_soft_hits_weight",
            0.0,
        ),
        soft_hits_k=getattr(
            args,
            "biogt_path_pretrain_soft_hits_k",
            10,
        ),
        soft_hits_temperature=getattr(
            args,
            "biogt_path_pretrain_soft_hits_temperature",
            1.0,
        ),
        normalized_soft_hits_weight=getattr(
            args,
            "biogt_path_pretrain_normalized_soft_hits_weight",
            0.0,
        ),
        normalized_soft_hits_k=getattr(
            args,
            "biogt_path_pretrain_normalized_soft_hits_k",
            10,
        ),
        normalized_soft_hits_temperature=getattr(
            args,
            "biogt_path_pretrain_normalized_soft_hits_temperature",
            1.0,
        ),
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed + 600011,
        pretrain_target_types=biogt_path_pretrain_types,
    )
    global_pretraining = pretrain_global_bilinear_branch(
        model,
        g_train_cpu,
        train_target_etypes,
        target_types,
        known_positive_index,
        candidate_nodes_by_etype,
        dense_optimizer,
        sparse_optimizer,
        epochs=args.global_bilinear_pretrain_epochs,
        batch_size=args.global_bilinear_pretrain_batch_size,
        negatives_per_positive=(
            args.global_bilinear_pretrain_negatives
        ),
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_temperature=args.ranking_temperature,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed + 700001,
        pretrain_scope=args.global_bilinear_pretrain_scope,
        max_edges_per_relation=(
            args.global_bilinear_pretrain_max_edges_per_relation
        ),
        pretrain_objective=args.global_bilinear_pretrain_objective,
        negatives_by_target=pretrain_negatives_by_target,
        hard_negatives=args.global_bilinear_pretrain_hard_negatives,
        pretrain_target_types=pretrain_finetune_types,
    )
    target_finetuning = pretrain_global_bilinear_branch(
        model,
        g_train_cpu,
        train_target_etypes,
        target_types,
        known_positive_index,
        candidate_nodes_by_etype,
        dense_optimizer,
        sparse_optimizer,
        epochs=args.global_bilinear_pretrain_target_finetune_epochs,
        batch_size=args.global_bilinear_pretrain_batch_size,
        negatives_per_positive=(
            args.global_bilinear_pretrain_negatives
        ),
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_temperature=args.ranking_temperature,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed + 800003,
        pretrain_scope="target_relations",
        max_edges_per_relation=0,
        pretrain_objective=args.global_bilinear_pretrain_objective,
        negatives_by_target=pretrain_negatives_by_target,
        hard_negatives=args.global_bilinear_pretrain_hard_negatives,
        pretrain_target_types=pretrain_finetune_types,
    )
    global_pretraining["target_finetuning"] = target_finetuning
    global_pretraining[
        "shared_kge_frozen_for_target_disease_finetune"
    ] = freeze_shared_kge_for_target_disease
    global_pretraining["shared_kge_frozen_parameter_count"] = int(
        shared_kge_frozen_parameter_count
    )
    global_pretraining["target_disease_finetune_types"] = (
        target_disease_finetune_types
    )
    global_pretraining[
        "kge_frozen_for_candidate_bias_finetune"
    ] = freeze_kge_for_candidate_bias
    global_pretraining["candidate_bias_frozen_parameter_count"] = int(
        candidate_bias_frozen_parameter_count
    )
    global_pretraining["candidate_bias_finetune_types"] = (
        candidate_bias_finetune_types
    )
    global_pretraining[
        "kge_frozen_for_target_residual_finetune"
    ] = freeze_kge_for_target_residual
    global_pretraining["target_residual_frozen_parameter_count"] = int(
        target_residual_frozen_parameter_count
    )
    global_pretraining["target_residual_finetune_types"] = (
        target_residual_finetune_types
    )
    global_pretraining[
        "kge_frozen_for_path_propagation_finetune"
    ] = freeze_kge_for_path_propagation
    global_pretraining[
        "path_propagation_frozen_parameter_count"
    ] = int(path_propagation_frozen_parameter_count)
    global_pretraining["path_propagation_finetune_types"] = (
        path_propagation_finetune_types
    )
    global_pretraining["path_graph"] = path_graph_metadata
    biogt_path_pretraining["path_graph"] = (
        biogt_path_graph_metadata
    )
    biogt_path_pretraining["base_frozen_for_finetune"] = (
        freeze_biogt_base_for_path
    )
    biogt_path_pretraining["relbf_jk_only_finetune"] = (
        biogt_relbf_jk_only_finetune
    )
    biogt_path_pretraining["relbf_edge_attention_only_finetune"] = (
        biogt_relbf_edge_attention_only_finetune
    )
    biogt_path_pretraining["relbf_path_layer_attention_only_finetune"] = (
        biogt_relbf_path_layer_attention_only_finetune
    )
    biogt_path_pretraining["relbf_path_token_transformer_only_finetune"] = (
        biogt_relbf_path_token_transformer_only_finetune
    )
    biogt_path_pretraining[
        "relbf_candidate_competition_boundary_only_finetune"
    ] = biogt_relbf_candidate_competition_boundary_only_finetune
    biogt_path_pretraining[
        "relbf_parent_top10_boundary_comparator_only_finetune"
    ] = biogt_relbf_parent_top10_boundary_comparator_only_finetune
    biogt_path_pretraining[
        "relbf_boundary_window_rank_slot_expert_only_finetune"
    ] = biogt_relbf_boundary_window_rank_slot_expert_only_finetune
    biogt_path_pretraining[
        "relbf_boundary_exchange_only_finetune"
    ] = biogt_relbf_boundary_exchange_only_finetune
    biogt_path_pretraining[
        "relbf_challenger_boundary_promotion_only_finetune"
    ] = biogt_relbf_challenger_boundary_promotion_only_finetune
    biogt_path_pretraining[
        "relbf_gatekeeper_challenger_paired_margin_only_finetune"
    ] = biogt_relbf_gatekeeper_challenger_paired_margin_only_finetune
    biogt_path_pretraining["relbf_deep_only_finetune"] = (
        biogt_relbf_deep_only_finetune
    )
    biogt_path_pretraining["relbf_indicator_only_finetune"] = (
        biogt_relbf_indicator_only_finetune
    )
    biogt_path_pretraining[
        "relbf_indicator_support_gate_only_finetune"
    ] = biogt_relbf_indicator_support_gate_only_finetune
    biogt_path_pretraining[
        "relbf_indicator_semantic_only_finetune"
    ] = biogt_relbf_indicator_semantic_only_finetune
    biogt_path_pretraining[
        "relbf_query_relation_adapter_only_finetune"
    ] = biogt_relbf_query_relation_adapter_only_finetune
    biogt_path_pretraining[
        "relbf_indicator_query_boundary_only_finetune"
    ] = biogt_relbf_indicator_query_boundary_only_finetune
    biogt_path_pretraining[
        "relbf_indicator_multirule_only_finetune"
    ] = biogt_relbf_indicator_multirule_only_finetune
    biogt_path_pretraining[
        "relbf_indicator_relation_transition_only_finetune"
    ] = biogt_relbf_indicator_relation_transition_only_finetune
    biogt_path_pretraining[
        "relbf_indicator_query_local_relation_transition_only_finetune"
    ] = biogt_relbf_indicator_query_local_relation_transition_only_finetune
    biogt_path_pretraining["frozen_parameter_count"] = (
        biogt_path_frozen_parameter_count
    )
    biogt_path_pretrain_only = bool(
        getattr(args, "biogt_path_pretrain_only", False)
    )
    if biogt_path_pretrain_only:
        if not freeze_biogt_base_for_path:
            raise ValueError(
                "--biogt-path-pretrain-only requires "
                "--freeze-biogt-base-for-path-finetune"
            )
        if getattr(args, "biogt_path_pretrain_epochs", 0) <= 0:
            raise ValueError(
                "--biogt-path-pretrain-only requires positive "
                "--biogt-path-pretrain-epochs"
            )
        logging.info(
            "Pure BioGT path pretrain-only mode will skip sampled main "
            "training and proceed directly to validation."
        )
    if freeze_independent:
        frozen_parameter_count = (
            model.freeze_independent_bilinear_branch()
        )
        logging.info(
            "Froze %d independent KGE branch parameters after pretraining.",
            frozen_parameter_count,
        )
        global_pretraining["frozen_after_pretraining"] = True
        global_pretraining["frozen_parameter_count"] = int(
            frozen_parameter_count
        )
    else:
        global_pretraining["frozen_after_pretraining"] = False
    fixed_validation_batches = build_fixed_edge_batches(
        g_val_cpu,
        val_target_etypes,
        target_types,
        known_positive_index,
        batch_size=args.validation_batch_size,
        negatives_per_positive=args.k_neg_samples,
        seed=args.fixed_validation_seed,
        max_edges=args.validation_max_edges,
        candidate_nodes_by_etype=candidate_nodes_by_etype,
    )
    logging.info(
        "Prepared %d fixed validation batches; validation runs every %d epochs.",
        len(fixed_validation_batches),
        args.eval_every,
    )

    hard_negative_pool = HardNegativePool()
    best_checkpoint_path = _best_checkpoint_path(args.save_model_path)
    checkpoint_metric = args.checkpoint_metric
    best_metric = (
        math.inf
        if checkpoint_metric in {"loss", "val_loss", "train_loss"}
        else -math.inf
    )
    validation_events_without_improvement = 0
    last_metrics = {}
    random_generator = torch.Generator()
    epoch_number = 0
    if args.epochs <= 0 and not args.load_model_path:
        raise ValueError(
            "--epochs 0 requires --load_model_path for evaluation-only mode"
        )

    main_training_epochs = (
        0 if biogt_path_pretrain_only else args.epochs
    )
    for epoch in range(main_training_epochs):
        epoch_number = epoch + 1
        should_refresh_hard_negatives = (
            not args.disable_hard_negative_mining
            and epoch_number >= args.hard_negative_warmup
            and (
                (epoch_number - args.hard_negative_warmup)
                % max(1, args.hard_negative_interval)
                == 0
            )
        )
        if should_refresh_hard_negatives:
            refreshed = hard_negative_pool.refresh(
                model,
                g_train_cpu,
                train_target_etypes,
                target_types,
                known_positive_index,
                epoch=epoch_number,
                max_edges=args.hard_negative_max_edges,
                candidates_per_positive=args.hard_negative_candidates,
                negatives_per_positive=args.k_neg_samples,
                margin=args.hard_negative_margin,
                score_chunk_size=args.score_chunk_size,
                seed=args.seed,
                candidate_nodes_by_etype=candidate_nodes_by_etype,
            )
            logging.info(
                "Epoch %d refreshed %d semi-hard negative groups.",
                epoch_number,
                refreshed,
            )

        model.train()
        random_generator.manual_seed(args.seed + epoch_number * 104729)
        epoch_losses = []
        epoch_bce = []
        epoch_ranking = []
        epoch_moe_aux = []
        epoch_global_aux = []

        for batch_index, (etype, eids, pos_src, pos_dst) in enumerate(
            train_batcher.iter_epoch(epoch)
        ):
            neg_src, neg_dst = sample_filtered_negatives(
                g_train_cpu,
                etype,
                pos_src,
                pos_dst,
                args.k_neg_samples,
                target_types,
                known_positive_index,
                random_generator,
                candidate_nodes_by_etype,
            )
            neg_src, neg_dst = hard_negative_pool.mix_with_random(
                etype,
                eids,
                neg_src,
                neg_dst,
                args.k_neg_samples,
                args.hard_negative_ratio,
            )
            context_graph, pos_graph, neg_graph = context_sampler.build_batch_graphs(
                g_train_cpu,
                etype,
                pos_src,
                pos_dst,
                neg_src,
                neg_dst,
                random_seed=(
                    args.seed
                    + epoch_number * 1000003
                    + batch_index
                ),
            )
            context_graph = _graph_for_model(
                context_graph,
                model,
                device,
            )
            pos_graph = _graph_for_model(pos_graph, model, device)
            neg_graph = _graph_for_model(neg_graph, model, device)

            dense_optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad(set_to_none=True)
            pos_scores, neg_scores = model(
                context_graph,
                None,
                pos_graph,
                neg_graph,
                target_etypes_override={etype},
            )
            loss, components = compute_joint_link_loss(
                pos_scores,
                neg_scores,
                args.k_neg_samples,
                args.ranking_loss_weight,
                args.ranking_temperature,
                moe_aux_loss=model.get_moe_aux_loss(),
                moe_aux_loss_weight=args.moe_aux_loss_weight,
            )
            if (
                args.global_bilinear_aux_weight > 0
                and model.scorer_mode
                in {"hybrid_global_bilinear", "hybrid_kge"}
            ):
                global_pos_scores, global_neg_scores = (
                    model.get_global_bilinear_scores()
                )
                global_aux_loss, _ = compute_joint_link_loss(
                    global_pos_scores,
                    global_neg_scores,
                    args.k_neg_samples,
                    args.ranking_loss_weight,
                    args.ranking_temperature,
                    moe_aux_loss=None,
                    moe_aux_loss_weight=0.0,
                )
                loss = (
                    loss
                    + args.global_bilinear_aux_weight * global_aux_loss
                )
                components["global_aux"] = float(
                    global_aux_loss.detach().cpu()
                )
                components["total"] = float(loss.detach().cpu())
            else:
                components["global_aux"] = 0.0
            if not torch.isfinite(loss):
                logging.warning(
                    "Skipping non-finite loss in epoch %d batch %d.",
                    epoch_number,
                    batch_index,
                )
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dense_parameters,
                max_norm=args.gradient_clip_norm,
            )
            dense_optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            model.clear_embedding_cache()

            epoch_losses.append(components["total"])
            epoch_bce.append(components["bce"])
            epoch_ranking.append(components["ranking"])
            epoch_moe_aux.append(components["moe_aux"])
            epoch_global_aux.append(components["global_aux"])

        train_metrics = {
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.inf,
            "train_bce": float(np.mean(epoch_bce)) if epoch_bce else math.inf,
            "train_ranking": float(np.mean(epoch_ranking)) if epoch_ranking else math.inf,
            "train_moe_aux": float(np.mean(epoch_moe_aux)) if epoch_moe_aux else 0.0,
            "train_global_aux": (
                float(np.mean(epoch_global_aux))
                if epoch_global_aux
                else 0.0
            ),
        }
        should_evaluate = (
            epoch_number == 1
            or epoch_number % max(1, args.eval_every) == 0
            or epoch_number == args.epochs
        )
        if not should_evaluate:
            logging.info(
                "Epoch %d/%d train_loss=%.5f bce=%.5f rank=%.5f "
                "(validation deferred)",
                epoch_number,
                args.epochs,
                train_metrics["train_loss"],
                train_metrics["train_bce"],
                train_metrics["train_ranking"],
            )
            continue

        validation_metrics = evaluate_fixed_batches(
            model,
            g_train_cpu,
            fixed_validation_batches,
            context_sampler,
            device,
            args.ranking_loss_weight,
            args.ranking_temperature,
            context_seed=args.fixed_validation_seed,
        )
        full_ranking_due = should_run_full_ranking_validation(
            epoch_number,
            args.epochs,
            args.full_ranking_validation_interval,
            enabled=args.enable_full_ranking_validation,
        )
        if full_ranking_due:
            validation_full_ranking = evaluate_full_filtered_ranking(
                model,
                g_full_cpu,
                g_train_cpu,
                g_val_cpu,
                g_train_cpu,
                target_types,
                args.full_ranking_source_type,
                device,
                candidate_batch_size=args.full_ranking_candidate_batch_size,
                max_edges=args.full_ranking_max_edges,
                structural_prior_mode=args.structural_prior_mode,
                structural_prior_alpha=args.structural_prior_alpha,
                structural_prior_max_iter=args.structural_prior_max_iter,
                structural_prior_tol=args.structural_prior_tol,
                structural_prior_neural_weight=args.structural_prior_neural_weight,
                structural_prior_grid=args.structural_prior_grid,
            )
            validation_metrics["full_ranking"] = validation_full_ranking
            for metric_name, metric_value in validation_full_ranking[
                "metrics"
            ].items():
                validation_metrics[
                    f"full_ranking_{metric_name}"
                ] = metric_value
            for metric_name, metric_value in validation_full_ranking[
                "metrics_macro"
            ].items():
                validation_metrics[
                    f"full_ranking_macro_{metric_name}"
                ] = metric_value
            for metric_name, metric_value in validation_full_ranking[
                "metrics_macro_stable"
            ].items():
                validation_metrics[
                    f"full_ranking_stable_macro_{metric_name}"
                ] = metric_value
            logging.info(
                "Validation full ranking: macro MRR=%.5f Hits@10=%.5f "
                "Recall@50=%.5f NDCG@50=%.5f",
                validation_full_ranking["metrics_macro"]["mrr"],
                validation_full_ranking["metrics_macro"]["hits_at_10"],
                validation_full_ranking["metrics_macro"]["recall_at_50"],
                validation_full_ranking["metrics_macro"]["ndcg_at_50"],
            )
        last_metrics = {**train_metrics, **validation_metrics}
        logging.info(
            "Epoch %d/%d train_loss=%.5f val_loss=%.5f AUC-PRC=%.5f "
            "MRR=%.5f Hits@10=%.5f",
            epoch_number,
            args.epochs,
            train_metrics["train_loss"],
            validation_metrics["loss"],
            validation_metrics["auc_prc"],
            validation_metrics["mrr"],
            validation_metrics["hits_at_10"],
        )

        current_metric = last_metrics.get(checkpoint_metric)
        if current_metric is None:
            if checkpoint_metric.startswith("full_ranking_"):
                logging.info(
                    "Checkpoint metric '%s' is deferred until the next "
                    "full-ranking validation event.",
                    checkpoint_metric,
                )
                continue
            raise ValueError(
                f"Checkpoint metric '{checkpoint_metric}' is unavailable. "
                f"Available metrics: {sorted(last_metrics)}"
            )
        if _metric_improved(
            checkpoint_metric,
            current_metric,
            best_metric,
            args.early_stopping_min_delta,
        ):
            best_metric = current_metric
            validation_events_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_path,
                model,
                epoch_number,
                last_metrics,
                args,
            )
        else:
            validation_events_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and validation_events_without_improvement
            >= args.early_stopping_patience
        ):
            logging.info(
                "Early stopping after %d validation events without improvement.",
                validation_events_without_improvement,
            )
            break

    if main_training_epochs > 0:
        _save_checkpoint(
            args.save_model_path,
            model,
            epoch_number,
            last_metrics,
            args,
        )
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            _load_checkpoint(best_checkpoint_path, model, device)
            model.clear_embedding_cache()
            logging.info(
                "Reloaded best %s checkpoint for final evaluation.",
                checkpoint_metric,
            )

    final_metrics = dict(last_metrics)
    final_metrics["global_bilinear_pretraining"] = global_pretraining
    final_metrics["biogt_path_pretraining"] = (
        biogt_path_pretraining
    )
    evaluation_split = args.final_eval_split
    eval_graph_cpu = (
        g_val_cpu
        if evaluation_split == "validation"
        else g_test_cpu
    )
    eval_target_etypes = (
        val_target_etypes
        if evaluation_split == "validation"
        else test_target_etypes
    )
    known_graph_cpu = _known_graph_for_evaluation(
        g_train_cpu,
        g_val_cpu,
        evaluation_split,
    )
    evaluation_known_positive_index = (
        known_positive_index
        if evaluation_split == "validation"
        else KnownPositiveIndex.for_target_etypes(
            known_graph_cpu,
            train_target_etypes,
        )
    )
    should_run_final_evaluation = (
        not args.skip_final_test_eval
        and (
            evaluation_split == "validation"
            or args.test_split > 0
        )
        and bool(eval_target_etypes)
    )
    if should_run_final_evaluation:
        fixed_test_batches = build_fixed_edge_batches(
            eval_graph_cpu,
            eval_target_etypes,
            target_types,
            evaluation_known_positive_index,
            batch_size=args.validation_batch_size,
            negatives_per_positive=args.k_neg_samples,
            seed=args.fixed_validation_seed + 1,
            max_edges=args.validation_max_edges,
            candidate_nodes_by_etype=candidate_nodes_by_etype,
        )
        test_metrics = evaluate_fixed_batches(
            model,
            g_train_cpu,
            fixed_test_batches,
            context_sampler,
            device,
            args.ranking_loss_weight,
            args.ranking_temperature,
            context_seed=args.fixed_validation_seed + 1,
        )
        test_metrics["split_protocol"] = {
            "training_labels": "train_only",
            "selection_labels": "validation_only",
            "evaluation_split": evaluation_split,
            "test_filter": (
                "train_plus_valid_without_other_test_labels"
                if evaluation_split == "test"
                else "train_without_other_validation_labels"
            ),
        }
        if args.enable_full_ranking_eval:
            test_full_ranking = evaluate_full_filtered_ranking(
                model,
                g_full_cpu,
                g_train_cpu,
                eval_graph_cpu,
                known_graph_cpu,
                target_types,
                args.full_ranking_source_type,
                device,
                candidate_batch_size=args.full_ranking_candidate_batch_size,
                max_edges=args.full_ranking_max_edges,
                structural_prior_mode=args.structural_prior_mode,
                structural_prior_alpha=args.structural_prior_alpha,
                structural_prior_max_iter=args.structural_prior_max_iter,
                structural_prior_tol=args.structural_prior_tol,
                structural_prior_neural_weight=args.structural_prior_neural_weight,
                structural_prior_grid=args.structural_prior_grid,
            )
            test_metrics["full_ranking"] = test_full_ranking
            for metric_name, metric_value in test_full_ranking[
                "metrics"
            ].items():
                test_metrics[
                    f"full_ranking_{metric_name}"
                ] = metric_value
            for metric_name, metric_value in test_full_ranking[
                "metrics_macro"
            ].items():
                test_metrics[
                    f"full_ranking_macro_{metric_name}"
                ] = metric_value
            for metric_name, metric_value in test_full_ranking[
                "metrics_macro_stable"
            ].items():
                test_metrics[
                    f"full_ranking_stable_macro_{metric_name}"
                ] = metric_value
            logging.info(
                "%s full ranking: macro MRR=%.5f Hits@10=%.5f "
                "Recall@50=%.5f NDCG@50=%.5f",
                evaluation_split.capitalize(),
                test_full_ranking["metrics_macro"]["mrr"],
                test_full_ranking["metrics_macro"]["hits_at_10"],
                test_full_ranking["metrics_macro"]["recall_at_50"],
                test_full_ranking["metrics_macro"]["ndcg_at_50"],
            )
        final_metrics[evaluation_split] = test_metrics
        logging.info(
            "Final %s AUC-PRC=%.5f MRR=%.5f Hits@10=%.5f "
            "NDCG@50=%.5f.",
            evaluation_split,
            test_metrics["auc_prc"],
            test_metrics["mrr"],
            test_metrics["hits_at_10"],
            test_metrics["ndcg_at_50"],
        )

    if biogt_path_pretrain_only:
        _save_checkpoint(
            args.save_model_path,
            model,
            epoch_number,
            final_metrics,
            args,
        )
        if best_checkpoint_path:
            _save_checkpoint(
                best_checkpoint_path,
                model,
                epoch_number,
                final_metrics,
                args,
            )
        logging.info(
            "Saved pure BioGT path pretrain-only final and best checkpoints."
        )

    if args.metrics_output:
        metrics_path = os.path.abspath(args.metrics_output)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(final_metrics, metrics_file, indent=2, sort_keys=True)

    logging.info(
        "Unified sampled training completed in %.1f seconds.",
        time.time() - start_time,
    )
    return {
        "model": model,
        "metrics": final_metrics,
        "train_target_etypes": train_target_etypes,
        "fixed_validation_batches": fixed_validation_batches,
    }
