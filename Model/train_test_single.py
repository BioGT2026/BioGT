import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import dgl
import numpy as np
import argparse
import hashlib
import json
import time
import traceback
import signal
import sys

# Import from local modules
from Model.modelData import DGLGraphCreator
from Model.model import DiseaseAssociationModel
from Model.gcn_model import GCNModel
from Model.gat_model import GATModel
from Model.transformer_model import SimpleTransformerModel
from Model.kg_embedding_model import KGEmbeddingModel, ComplExModel, RotatEModel
from Model.rgcn_model import RGCNModel
from Model.network_propagation_model import NetworkPropagationModel
from Model.ppr_model import PPRModel
from Model.modelEvaluator import ModelEvaluator
from Model.common import *

# Global variables for signal handling
_training_interrupted = False
_current_model = None
_current_epoch = 0
_current_save_path = None
_current_metric_name = None
_current_metric_value = None
_structural_ppr_cache = {}


def _parse_category_split_map(mapping_str):
    """Parse a comma-separated list like 'metabolite:val,gene:test' into a dict."""
    if not mapping_str:
        return {}

    assignments = {}
    entries = [entry.strip() for entry in mapping_str.split(',') if entry.strip()]
    for entry in entries:
        if ':' not in entry:
            logging.warning(f"Skipping invalid category split entry '{entry}'. Expected format 'node_type:target_set'.")
            continue
        node_type, target_set = entry.split(':', 1)
        node_type = node_type.strip()
        target_set = target_set.strip().lower()
        if target_set not in {"train", "val", "test"}:
            logging.warning(f"Skipping category split entry '{entry}' with unsupported target set '{target_set}'.")
            continue
        if not node_type:
            logging.warning(f"Skipping category split entry '{entry}' with empty node type.")
            continue
        assignments[node_type] = target_set
    return assignments


def _parse_node_type_list(list_str):
    if not list_str:
        return []
    return [item.strip() for item in list_str.split(',') if item.strip()]

def _parse_ranking_ks(k_values):
    return ModelEvaluator.normalize_ranking_ks(k_values)

def _default_eval_metrics(k_values=None):
    metrics = {'auc_prc': 0.0, 'auc_roc': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'loss': float('inf')}
    metrics.update(ModelEvaluator.empty_ranking_metrics(k_values))
    return metrics


def _full_ranking_metric_values(payload):
    """Expose exact-ranking payload values under CLI checkpoint metric names."""
    if not payload:
        return {}

    values = {}
    scopes = (
        ("full_ranking", "metrics"),
        ("full_ranking_macro", "metrics_macro"),
        ("full_ranking_stable_macro", "metrics_macro_stable"),
    )
    for checkpoint_prefix, payload_key in scopes:
        metrics = payload.get(payload_key, {})
        for metric_name in ("mrr", "hits_at_10", "recall_at_50", "ndcg_at_50"):
            if metric_name in metrics:
                values[f"{checkpoint_prefix}_{metric_name}"] = metrics[metric_name]

    # Preserve legacy early-stopping names used by historical launchers.
    metrics = payload.get("metrics", {})
    if "mrr" in metrics:
        values["full_mrr"] = metrics["mrr"]
    for metric_name, legacy_name in (
        ("hits_at_10", "full_hits_at_10"),
        ("recall_at_50", "full_recall_at_50"),
        ("ndcg_at_50", "full_ndcg_at_50"),
    ):
        if metric_name in metrics:
            values[legacy_name] = metrics[metric_name]
    return values


def _macro_full_ranking_metrics(per_target_metrics, metric_names):
    populated = {
        target: metrics
        for target, metrics in per_target_metrics.items()
        if metrics.get("ranking_num_triples", 0) > 0
    }
    stable = {
        target: metrics
        for target, metrics in populated.items()
        if metrics.get("ranking_num_triples", 0) >= 10
    }

    def aggregate(targets):
        return {
            metric_name: (
                float(np.mean([metrics[metric_name] for metrics in targets.values()]))
                if targets
                else 0.0
            )
            for metric_name in metric_names
        }

    return aggregate(populated), aggregate(stable), sorted(stable)


def _resolve_ranking_predictor(model_module, scoring_etype, target_type):
    """Resolve the predictor used by pair scoring for exact candidate ranking."""
    if getattr(model_module, "disable_moe", False):
        predictor = getattr(model_module, "shared_predictor", None)
        if predictor is not None:
            return predictor

    expert_predictors = getattr(model_module, "expert_predictors", None)
    if expert_predictors is not None and target_type in expert_predictors:
        return expert_predictors[target_type]

    relation_predictors = getattr(model_module, "moe_predictors", None)
    relation_key = str(scoring_etype)
    if relation_predictors is not None and relation_key in relation_predictors:
        return relation_predictors[relation_key]

    raise KeyError(
        "No ranking predictor for target "
        f"{target_type!r} and relation {scoring_etype!r}."
    )

def _ranking_log_fields(metrics, k_values):
    k_values = _parse_ranking_ks(k_values)
    primary_k = k_values[0]
    secondary_k = 50 if 50 in k_values else k_values[-1]
    return (
        f"MRR: {metrics.get('mrr', 0.0):.4f} | "
        f"Hits@{primary_k}: {metrics.get(f'hits_at_{primary_k}', 0.0):.4f} | "
        f"Recall@{secondary_k}: {metrics.get(f'recall_at_{secondary_k}', 0.0):.4f} | "
        f"NDCG@{secondary_k}: {metrics.get(f'ndcg_at_{secondary_k}', 0.0):.4f}"
    )

def _json_safe_metrics(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metrics(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def _write_metrics_output(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_json_safe_metrics(payload), fh, indent=2)


def _canonical_etype_label(etype):
    return "::".join(str(part) for part in etype)


def _sorted_ints(values):
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return sorted(int(value) for value in values)


def _typed_original_ids(dgl_ids, ntype, node_dglid_to_orig):
    mapping = (node_dglid_to_orig or {}).get(ntype, {})
    records = []
    for dgl_id in _sorted_ints(dgl_ids):
        original_id = mapping.get(dgl_id)
        if original_id is None:
            original_id = mapping.get(str(dgl_id))
        records.append(
            {
                "dgl_id": dgl_id,
                "original_id": None if original_id is None else str(original_id),
            }
        )
    return records


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_split_manifest(graph_creator, args):
    metadata = graph_creator.get_split_metadata() or {}
    node_mapping = graph_creator.node_dglid_to_orig
    payload = {
        "schema_version": 1,
        "strategy": args.split_strategy,
        "seed": int(args.seed),
        "val_split": float(args.val_split),
        "test_split": float(args.test_split),
        "kfolds": int(args.split_kfolds or 0),
        "test_fold_index": int(args.split_kfold_index or 0),
        "val_fold_index": int(args.split_kfold_val_fold or -1),
        "specific_ids_path": args.split_specific_ids_path or None,
        "external_edges_path": args.split_external_edges_path or None,
        "external_edges_sha256": (
            _sha256_file(args.split_external_edges_path)
            if args.split_external_edges_path
            and os.path.isfile(args.split_external_edges_path)
            else None
        ),
        "metadata_available": bool(metadata),
        "node_splits": {},
        "node_folds": [],
        "matched_typed_nodes": {},
        "edge_counts": {},
        "excluded_cross_edge_counts": {},
        "excluded_cross_edge_total": int(
            metadata.get("excluded_cross_edge_total", 0)
        ),
        "external_edge_audit": {
            key: int(metadata.get(key, 0))
            for key in (
                "requested_directed_pair_count",
                "matched_directed_pair_count",
                "unmatched_directed_pair_count",
                "test_edge_count",
            )
            if key in metadata
        },
    }

    for split_name, typed_nodes in (metadata.get("node_splits") or {}).items():
        payload["node_splits"][split_name] = {
            ntype: _typed_original_ids(node_ids, ntype, node_mapping)
            for ntype, node_ids in sorted(typed_nodes.items())
        }
    for fold in metadata.get("node_folds") or []:
        payload["node_folds"].append(
            {
                ntype: _typed_original_ids(node_ids, ntype, node_mapping)
                for ntype, node_ids in sorted(fold.items())
            }
        )
    payload["matched_typed_nodes"] = {
        ntype: _typed_original_ids(node_ids, ntype, node_mapping)
        for ntype, node_ids in sorted(
            (metadata.get("matched_typed_nodes") or {}).items()
        )
    }

    excluded_counts = metadata.get("excluded_cross_edge_counts") or {}
    payload["excluded_cross_edge_counts"] = {
        _canonical_etype_label(etype): int(count)
        for etype, count in sorted(
            excluded_counts.items(),
            key=lambda item: _canonical_etype_label(item[0]),
        )
    }
    graphs = {
        "full": graph_creator.g_full,
        "train": graph_creator.g_train,
        "validation": graph_creator.g_val,
        "test": graph_creator.g_test,
    }
    for graph_name, graph in graphs.items():
        payload["edge_counts"][graph_name] = {
            _canonical_etype_label(etype): int(graph.num_edges(etype))
            for etype in sorted(
                graph.canonical_etypes,
                key=_canonical_etype_label,
            )
        }
    return payload


def _write_split_manifest(path, payload):
    if not path:
        return None
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(serialized)
    os.replace(temp_path, output_path)
    return {"path": output_path, "sha256": digest}


def _canonical_source_target_pairs(graph, source_type, target_type):
    pairs = set()
    for etype in graph.canonical_etypes:
        src_type, _, dst_type = etype
        if src_type == source_type and dst_type == target_type:
            src_ids, dst_ids = graph.edges(etype=etype)
            pairs.update(
                (int(src_id), int(dst_id))
                for src_id, dst_id in zip(src_ids.tolist(), dst_ids.tolist())
            )
        elif src_type == target_type and dst_type == source_type:
            src_ids, dst_ids = graph.edges(etype=etype)
            pairs.update(
                (int(dst_id), int(src_id))
                for src_id, dst_id in zip(src_ids.tolist(), dst_ids.tolist())
            )
    return pairs


def export_dgl_ranking_dataset(
    output_root,
    dataset_prefix,
    g_full_cpu,
    g_train_cpu,
    g_val_cpu,
    g_test_cpu,
    node_info,
    source_type,
    target_types,
):
    """Export exact DGL split support in the KGE baseline text format."""
    output_root = os.path.abspath(output_root)
    dgl_to_orig = (node_info or {}).get("dglid_to_orig", {})
    payload = {"source_type": source_type, "targets": {}}
    split_graphs = {
        "train": g_train_cpu,
        "valid": g_val_cpu,
        "test": g_test_cpu,
    }

    for target_type in target_types:
        split_pairs = {
            split_name: _canonical_source_target_pairs(
                split_graph,
                source_type,
                target_type,
            )
            for split_name, split_graph in split_graphs.items()
        }
        candidate_ids = sorted(
            {
                target_id
                for _, target_id in _canonical_source_target_pairs(
                    g_full_cpu,
                    source_type,
                    target_type,
                )
            }
        )
        source_ids = sorted(
            {
                source_id
                for pairs in split_pairs.values()
                for source_id, _ in pairs
            }
        )
        target_ids = sorted(
            {
                target_id
                for pairs in split_pairs.values()
                for _, target_id in pairs
            }
            | set(candidate_ids)
        )

        entity_names = [
            *(f"{source_type}:{node_id}" for node_id in source_ids),
            *(f"{target_type}:{node_id}" for node_id in target_ids),
        ]
        dataset_name = f"{dataset_prefix}_{target_type}"
        dataset_dir = os.path.join(output_root, dataset_name)
        target_dir = os.path.join(dataset_dir, "kge")
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "entities.dict"), "w", encoding="utf-8") as handle:
            for entity_id, entity_name in enumerate(entity_names):
                handle.write(f"{entity_id}\t{entity_name}\n")
        relation_name = f"{source_type}_{target_type}"
        with open(os.path.join(target_dir, "relations.dict"), "w", encoding="utf-8") as handle:
            handle.write(f"0\t{relation_name}\n")
        for split_name, pairs in split_pairs.items():
            with open(os.path.join(target_dir, f"{split_name}.txt"), "w", encoding="utf-8") as handle:
                for source_id, target_id in sorted(pairs):
                    handle.write(
                        f"{source_type}:{source_id}\t{relation_name}\t{target_type}:{target_id}\n"
                    )
        with open(os.path.join(target_dir, "candidate_entities.txt"), "w", encoding="utf-8") as handle:
            for target_id in candidate_ids:
                handle.write(f"{target_type}:{target_id}\n")

        target_metadata = {
            "source_type": source_type,
            "target_type": target_type,
            "relation": relation_name,
            "split_counts": {
                split_name: len(pairs)
                for split_name, pairs in split_pairs.items()
            },
            "candidate_count": len(candidate_ids),
            "source_count": len(source_ids),
            "entity_count": len(entity_names),
            "entity_id_scheme": "node_type:dgl_id",
            "dgl_to_original_id": {
                source_type: {
                    str(node_id): str(dgl_to_orig.get(source_type, {}).get(node_id, ""))
                    for node_id in source_ids
                },
                target_type: {
                    str(node_id): str(dgl_to_orig.get(target_type, {}).get(node_id, ""))
                    for node_id in target_ids
                },
            },
        }
        with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(target_metadata, handle, indent=2)
        with open(os.path.join(dataset_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(target_metadata, handle, indent=2)
        payload["targets"][target_type] = target_metadata

    os.makedirs(output_root, exist_ok=True)
    with open(
        os.path.join(output_root, f"{dataset_prefix}_export_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _relation_ppr_scores(
    g_message_cpu,
    disease_type,
    target_type,
    source_ids,
    alpha,
    max_iter,
    tol,
):
    """Return train-only bipartite PPR vectors keyed by disease DGL ID."""
    source_ids = sorted(set(int(value) for value in source_ids))
    cache_key = (
        id(g_message_cpu),
        disease_type,
        target_type,
        tuple(source_ids),
        float(alpha),
        int(max_iter),
        float(tol),
    )
    cached = _structural_ppr_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for --structural-prior-mode ppr"
        ) from exc

    num_sources = int(g_message_cpu.num_nodes(disease_type))
    num_targets = int(g_message_cpu.num_nodes(target_type))
    relation_pairs = _canonical_source_target_pairs(
        g_message_cpu,
        disease_type,
        target_type,
    )

    if not relation_pairs:
        result = {
            source_id: torch.zeros(num_targets, dtype=torch.float32)
            for source_id in source_ids
        }
        _structural_ppr_cache[cache_key] = result
        return result

    rows = []
    cols = []
    target_offset = num_sources
    for source_id, target_id in sorted(relation_pairs):
        shifted_target = target_offset + target_id
        rows.extend((source_id, shifted_target))
        cols.extend((shifted_target, source_id))

    rows_np = np.asarray(rows, dtype=np.int64)
    cols_np = np.asarray(cols, dtype=np.int64)
    num_nodes = num_sources + num_targets
    degrees = np.bincount(rows_np, minlength=num_nodes).astype(np.float32)
    data = (1.0 / np.maximum(degrees[rows_np], 1.0)).astype(np.float32)
    transition = csr_matrix(
        (data, (rows_np, cols_np)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    batch_size = 512
    teleport = np.float32(1.0 - alpha)
    alpha32 = np.float32(alpha)
    result = {}
    for start in range(0, len(source_ids), batch_size):
        batch_sources = source_ids[start:start + batch_size]
        ranks = np.zeros((len(batch_sources), num_nodes), dtype=np.float32)
        row_ids = np.arange(len(batch_sources))
        source_array = np.asarray(batch_sources, dtype=np.int64)
        ranks[row_ids, source_array] = 1.0
        for _ in range(max_iter):
            updated = transition.T.dot(ranks.T).T.astype(np.float32, copy=False)
            updated *= alpha32
            updated[row_ids, source_array] += teleport
            delta = float(np.abs(updated - ranks).sum(axis=1).max())
            ranks = updated
            if delta < tol:
                break
        target_scores = ranks[:, target_offset:]
        for batch_index, source_id in enumerate(batch_sources):
            result[source_id] = torch.from_numpy(target_scores[batch_index].copy())

    _structural_ppr_cache[cache_key] = result
    return result

def _parse_relation_type_list(list_str):
    if not list_str:
        return set()
    return {item.strip() for item in list_str.split(',') if item.strip()}


def _parse_int_list(list_str, default=None):
    if not list_str:
        return list(default or [])

    parsed = []
    if isinstance(list_str, (list, tuple, set)):
        raw_items = list_str
    else:
        raw_items = str(list_str).split(',')

    for item in raw_items:
        item = str(item).strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            logging.warning(f"Skipping invalid integer value '{item}'.")
            continue
        if value <= 0:
            logging.warning(f"Skipping non-positive ranking K value '{item}'.")
            continue
        if value not in parsed:
            parsed.append(value)

    return parsed or list(default or [])
def select_target_etypes(graph, target_node_types, args, stage_label="training"):
    """Select canonical edge types used for training or evaluation."""
    target_node_type_set = {item.strip() for item in target_node_types if item.strip()}
    relation_filter = _parse_relation_type_list(getattr(args, 'target_relation_types', None))
    target_edge_scope = getattr(args, 'target_edge_scope', 'all_target_incident')

    selected = []
    if getattr(args, 'restrict_ranking_etypes_to_source_type', False):
        source_type = getattr(args, 'full_ranking_source_type', 'disease')
        selected = [
            etype
            for etype in graph.canonical_etypes
            if _matches_source_target_relation(etype, source_type, target_node_type_set)
        ]
    elif target_edge_scope == 'input_output':
        input_type = (getattr(args, 'eval_input_type', None) or '').strip()
        output_types = set(_parse_node_type_list(getattr(args, 'eval_output_types', None)))
        for etype in graph.canonical_etypes:
            src_type, rel_type, dst_type = etype
            is_target = (
                (src_type == input_type and dst_type in output_types)
                or (dst_type == input_type and src_type in output_types)
            )
            if is_target:
                selected.append(etype)
    else:
        for etype in graph.canonical_etypes:
            src_type, _, dst_type = etype
            if dst_type in target_node_type_set or src_type in target_node_type_set:
                selected.append(etype)

    if relation_filter:
        selected = [etype for etype in selected if etype[1] in relation_filter]

    if not selected:
        logging.warning(
            "No %s target edge types found with target_edge_scope='%s', target_node_types=%s, eval_input_type='%s', eval_output_types='%s', target_relation_types='%s'.",
            stage_label,
            target_edge_scope,
            sorted(target_node_type_set),
            getattr(args, 'eval_input_type', None),
            getattr(args, 'eval_output_types', None),
            getattr(args, 'target_relation_types', None),
        )
    return selected

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C, SIGTERM, etc.) during training"""
    global _training_interrupted, _current_model, _current_epoch, _current_save_path
    
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM",
        signal.SIGHUP: "SIGHUP"
    }
    signal_name = signal_names.get(signum, f"Signal {signum}")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Received {signal_name} - Training interruption requested")
    logging.info(f"{'='*60}")
    
    _training_interrupted = True
    
    # Attempt to save the current model state
    if _current_model is not None and _current_save_path is not None:
        try:
            # Generate interrupted model save path
            base_name = _current_save_path.replace(".pt", "")
            interrupted_model_path = f"{base_name}_epoch{_current_epoch}_interrupted.pt"

            # Create directory if it doesn't exist
            save_dir = os.path.dirname(interrupted_model_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # Save model state
            torch.save(_current_model.state_dict(), interrupted_model_path)

            logging.info(f"✓ Emergency model save successful!")
            logging.info(f"  Saved epoch {_current_epoch} model to: {interrupted_model_path}")
            if _current_metric_name and _current_metric_value is not None:
                logging.info(f"  Last {_current_metric_name}: {_current_metric_value:.4f}")
            
        except Exception as e:
            logging.error(f"✗ Failed to save model during interruption: {e}")
            traceback.print_exc()
    else:
        if _current_model is None:
            logging.warning("No model available to save during interruption")
        if _current_save_path is None:
            logging.warning("No save path specified - cannot save model during interruption")
    
    logging.info(f"Training will be stopped gracefully after current operations complete...")
    logging.info(f"{'='*60}\n")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    try:
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)   # Hangup signal
        logging.info("Signal handlers registered for graceful shutdown (Ctrl+C, SIGTERM, SIGHUP)")
    except Exception as e:
        logging.warning(f"Failed to setup some signal handlers: {e}")

def update_training_state(model, epoch, save_path, metric_name=None, metric_value=None):
    """Update global training state for signal handler"""
    global _current_model, _current_epoch, _current_save_path, _current_metric_name, _current_metric_value
    _current_model = model
    _current_epoch = epoch
    _current_save_path = save_path
    _current_metric_name = metric_name
    _current_metric_value = metric_value

def check_training_interrupted():
    """Check if training was interrupted by signal"""
    return _training_interrupted

def compute_loss(
    pos_scores_dict,
    neg_scores_dict,
    loss_weights=None,
    balanced_link_loss=False,
    link_loss="bce",
    ranking_loss_margin=0.0,
    ranking_loss_temperature=1.0,
):
    """ 
    Computes Binary Cross Entropy loss for link prediction with per-etype weighting.
    
    Args:
        pos_scores_dict: Dictionary of positive scores per edge type
        neg_scores_dict: Dictionary of negative scores per edge type
        loss_weights: Dictionary of weights per edge type {etype: weight}. 
                     If None, defaults to equal weights (1.0).
                     Higher weights increase importance of minority classes.
    """
    if not pos_scores_dict and not neg_scores_dict:
        return torch.tensor(0.0, requires_grad=True, device=torch.device('cpu'))
    
    device = None
    total_loss = 0.0
    total_weight = 0.0
    
    # Compute loss per edge type with optional weighting
    all_etypes = set(list(pos_scores_dict.keys()) + list(neg_scores_dict.keys()))
    
    for etype in all_etypes:
        pos_scores = pos_scores_dict.get(etype, None)
        neg_scores = neg_scores_dict.get(etype, None)
        
        # Skip if both are empty
        if (pos_scores is None or pos_scores.numel() == 0) and \
           (neg_scores is None or neg_scores.numel() == 0):
            continue
        
        # Prepare scores for this edge type
        etype_scores = []
        etype_labels = []
        
        if pos_scores is not None and pos_scores.numel() > 0:
            pos_scores_clamped = torch.clamp(pos_scores, min=-10.0, max=10.0)
            etype_scores.append(pos_scores_clamped)
            etype_labels.append(torch.ones(pos_scores_clamped.numel(), device=pos_scores.device))
            if device is None:
                device = pos_scores.device
        
        if neg_scores is not None and neg_scores.numel() > 0:
            neg_scores_clamped = torch.clamp(neg_scores, min=-10.0, max=10.0)
            etype_scores.append(neg_scores_clamped)
            etype_labels.append(torch.zeros(neg_scores_clamped.numel(), device=neg_scores.device))
            if device is None:
                device = neg_scores.device
        
        if not etype_scores:
            continue
        
        use_ranking_loss = (
            link_loss in {"bpr", "sampled_softmax"}
            and pos_scores is not None
            and neg_scores is not None
            and pos_scores.numel() > 0
            and neg_scores.numel() > 0
            and neg_scores.numel() % pos_scores.numel() == 0
        )
        if use_ranking_loss:
            pos_scores_clamped = torch.clamp(pos_scores, min=-10.0, max=10.0)
            neg_scores_grouped = torch.clamp(
                neg_scores,
                min=-10.0,
                max=10.0,
            ).view(pos_scores.numel(), -1)
            if link_loss == "bpr":
                score_difference = (
                    pos_scores_clamped.unsqueeze(1)
                    - neg_scores_grouped
                    - float(ranking_loss_margin)
                )
                etype_loss = F.softplus(-score_difference).mean()
            else:
                temperature = max(float(ranking_loss_temperature), 1e-6)
                logits = torch.cat(
                    [pos_scores_clamped.unsqueeze(1), neg_scores_grouped],
                    dim=1,
                ) / temperature
                labels = torch.zeros(
                    logits.shape[0],
                    dtype=torch.long,
                    device=logits.device,
                )
                etype_loss = F.cross_entropy(logits, labels)
        elif balanced_link_loss and pos_scores is not None and neg_scores is not None and pos_scores.numel() > 0 and neg_scores.numel() > 0:
            pos_scores_clamped = torch.clamp(pos_scores, min=-10.0, max=10.0)
            neg_scores_clamped = torch.clamp(neg_scores, min=-10.0, max=10.0)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores_clamped,
                torch.ones(pos_scores_clamped.numel(), device=pos_scores.device),
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores_clamped,
                torch.zeros(neg_scores_clamped.numel(), device=neg_scores.device),
            )
            etype_loss = 0.5 * (pos_loss + neg_loss)
        else:
            # Concatenate scores and labels for this edge type
            scores_tensor = torch.cat(etype_scores)
            labels_tensor = torch.cat(etype_labels)

            # Compute BCE loss for this edge type
            etype_loss = F.binary_cross_entropy_with_logits(scores_tensor, labels_tensor)
        
        # Apply weight if provided
        weight = 1.0
        if loss_weights is not None and etype in loss_weights:
            weight = loss_weights[etype]
        
        total_loss += etype_loss * weight
        total_weight += weight
    
    # Average by total weight
    if total_weight > 0:
        return total_loss / total_weight
    else:
        dev = device if device else torch.device('cpu')
        return torch.tensor(0.0, requires_grad=True, device=dev)

def evaluate_graph_metrics(
    model_module,
    g_eval_cpu,
    node_features_embed_cpu,
    args,
    device,
    eval_target_etypes=None,
    stage_label="Evaluation",
    g_message_cpu=None,
    g_known_cpu=None,
    decision_threshold=None,
    select_decision_threshold=False,
):
    """Evaluate the model on a given graph (e.g., validation or test) using full-graph processing."""
    ranking_ks = _parse_ranking_ks(getattr(args, 'ranking_ks', '10,50,100'))
    if g_eval_cpu is None or g_eval_cpu.number_of_edges() == 0:
        logging.warning("Evaluation graph is empty or None. Skipping evaluation.")
        return _default_eval_metrics(ranking_ks)

    model_module.eval()
    eval_metrics = _default_eval_metrics(ranking_ks)

    with torch.no_grad():
        g_eval = g_eval_cpu.to(device)
        message_graph_cpu = g_message_cpu if g_message_cpu is not None else g_eval_cpu
        g_message = message_graph_cpu.to(device)
        known_graph_cpu = g_known_cpu if g_known_cpu is not None else g_eval_cpu
        candidate_tensors, known_codes = _build_relation_candidate_state(
            known_graph_cpu,
            args,
            device,
        )
        node_features_embed = {k: v.to(device) for k, v in node_features_embed_cpu.items()}
        
        if DEBUG_LOGGING:
            logging.info(f"DEBUG evaluate_graph_metrics {stage_label}: node_features_embed keys = {list(node_features_embed.keys())}")
            logging.info(f"DEBUG evaluate_graph_metrics {stage_label}: g_eval.ntypes = {g_eval.ntypes}")
            
            # Log node counts for each type in the evaluation graph
            node_counts = {ntype: g_eval.num_nodes(ntype) for ntype in g_eval.ntypes}
            logging.info(f"DEBUG evaluate_graph_metrics {stage_label}: g_eval node counts = {node_counts}")
        
        # Check which node types are missing from node_features_embed
        missing_in_features = set(g_eval.ntypes) - set(node_features_embed.keys())
        if missing_in_features:
            logging.warning(f"Node types in g_eval but missing from node_features_embed: {missing_in_features}")

        # For evaluation, we create a negative graph from the evaluation graph
        # Evaluation negatives are grouped by positive edge for sampled tail ranking:
        # (head, relation, true_tail) is compared against k sampled tails.
        neg_edges_eval = {}
        for etype in g_eval.canonical_etypes:
            # Only sample negatives for the edge types we are evaluating on
            if eval_target_etypes and etype not in eval_target_etypes:
                continue
            if g_eval.num_edges(etype) > 0:
                num_neg_samples = g_eval.num_edges(etype) * args.k_neg_samples
                try:
                    src_type, _, dst_type = etype
                    pos_src, pos_dst = g_eval.edges(etype=etype)
                    if getattr(args, "restrict_ranking_etypes_to_source_type", False):
                        sampled_edges = _sample_filtered_relation_negatives(
                            etype,
                            pos_src,
                            pos_dst,
                            args.k_neg_samples,
                            g_eval,
                            args,
                            candidate_tensors,
                            known_codes,
                        )
                        if sampled_edges is None:
                            logging.warning(
                                "No relation-specific candidate pool for edge type %s.",
                                etype,
                            )
                            continue
                        neg_src, neg_dst = sampled_edges
                    elif _target_endpoint_is_source(etype, args):
                        num_src_nodes = g_eval.num_nodes(src_type)
                        if num_src_nodes <= 1:
                            logging.warning(f"Skipping negative sampling for edge type {etype}: source target type has <=1 node.")
                            continue
                        neg_src = torch.randint(0, num_src_nodes, (num_neg_samples,), device=device)
                        neg_dst = pos_dst.repeat_interleave(args.k_neg_samples)
                        repeated_pos_src = pos_src.repeat_interleave(args.k_neg_samples)
                        for _ in range(10):
                            collision_mask = neg_src == repeated_pos_src
                            if not torch.any(collision_mask):
                                break
                            neg_src[collision_mask] = torch.randint(
                                0,
                                num_src_nodes,
                                (int(collision_mask.sum().item()),),
                                device=device,
                            )
                    else:
                        num_dst_nodes = g_eval.num_nodes(dst_type)
                        if num_dst_nodes <= 1:
                            logging.warning(f"Skipping negative sampling for edge type {etype}: destination type has <=1 node.")
                            continue
                        neg_src = pos_src.repeat_interleave(args.k_neg_samples)
                        neg_dst = torch.randint(0, num_dst_nodes, (num_neg_samples,), device=device)
                        repeated_pos_dst = pos_dst.repeat_interleave(args.k_neg_samples)
                        for _ in range(10):
                            collision_mask = neg_dst == repeated_pos_dst
                            if not torch.any(collision_mask):
                                break
                            neg_dst[collision_mask] = torch.randint(
                                0,
                                num_dst_nodes,
                                (int(collision_mask.sum().item()),),
                                device=device,
                            )
                    neg_edges_eval[etype] = (neg_src, neg_dst)
                except Exception as e:
                    logging.warning(f"Failed to generate grouped negative samples for edge type {etype}: {e}")
        
        neg_g_eval = dgl.heterograph(
            neg_edges_eval,
            num_nodes_dict={ntype: g_eval.num_nodes(ntype) for ntype in g_eval.ntypes}
        )

        # The model's forward pass for a single graph takes the graph for message passing,
        # features, a positive graph, and a negative graph.
        edge_weights_eval = {
            etype: g_message.edges[etype].data['weight']
            for etype in g_message.canonical_etypes
            if 'weight' in g_message.edges[etype].data
        }
        
        if DEBUG_LOGGING:
            logging.info(f"DEBUG evaluate_graph_metrics {stage_label}: g_eval.canonical_etypes = {g_eval.canonical_etypes}")
            logging.info(f"DEBUG evaluate_graph_metrics {stage_label}: calling model.forward with training/message graph as context and g_eval as pos_graph...")
        
        pos_scores, neg_scores = model_module(
            g_message,
            node_features_embed, 
            g_eval, 
            neg_g_eval, 
            edge_weights=edge_weights_eval
        )
        
        if DEBUG_LOGGING:
            logging.info(f"DEBUG evaluate_graph_metrics {stage_label}: model returned pos_scores keys = {list(pos_scores.keys())}, neg_scores keys = {list(neg_scores.keys())}")

        if (
            getattr(model_module, 'enable_explainability', False)
            and model_module.enable_explainability
            and getattr(model_module, 'fusion_mode', None) == 'weighted'
            and hasattr(model_module, 'get_last_explainability')
        ):
            explainability_payload = model_module.get_last_explainability()
            if explainability_payload:
                log_explainability_summary(explainability_payload, f"{stage_label}")

        # Filter scores to only include target edge types for evaluation
        if eval_target_etypes:
            pos_scores = {k: v for k, v in pos_scores.items() if k in eval_target_etypes}
            neg_scores = {k: v for k, v in neg_scores.items() if k in eval_target_etypes}
            logging.debug(f"Evaluation metrics calculated for target etypes: {eval_target_etypes}")

        # DEBUG: Log the scores returned by model
        if DEBUG_LOGGING:
            logging.debug(f"DEBUG {stage_label}: pos_scores keys = {list(pos_scores.keys())}")
            logging.debug(f"DEBUG {stage_label}: neg_scores keys = {list(neg_scores.keys())}")
            logging.debug(f"DEBUG {stage_label}: eval_target_etypes = {eval_target_etypes}")
        
        loss = compute_loss(pos_scores, neg_scores)
        eval_metrics['loss'] = loss.item() if not torch.isnan(loss) else float('inf')

        # Check if any scores were collected before computing metrics
        has_pos_scores = any(t.numel() > 0 for t in pos_scores.values())
        has_neg_scores = any(t.numel() > 0 for t in neg_scores.values())

        if has_pos_scores or has_neg_scores:
            try:
                # Compute metrics per edge type and also aggregated
                eval_metrics['auc_prc_per_etype'] = ModelEvaluator.compute_auc_prc(pos_scores, neg_scores, per_etype=True)
                eval_metrics['auc_roc_per_etype'] = ModelEvaluator.compute_auc_roc(pos_scores, neg_scores, per_etype=True)
                
                # Aggregated metrics (for overall performance view)
                eval_metrics['auc_prc'] = np.mean(list(eval_metrics['auc_prc_per_etype'].values())) if eval_metrics['auc_prc_per_etype'] else 0.0
                eval_metrics['auc_roc'] = np.mean(list(eval_metrics['auc_roc_per_etype'].values())) if eval_metrics['auc_roc_per_etype'] else 0.0
                
                # Accuracy and F1 are usually computed on aggregated scores
                accuracy_at_0_5 = ModelEvaluator.compute_accuracy(pos_scores, neg_scores, threshold=0.5)
                f1_at_0_5 = ModelEvaluator.compute_f1_score(pos_scores, neg_scores, threshold=0.5)
                eval_metrics['accuracy'] = accuracy_at_0_5
                eval_metrics['f1'] = f1_at_0_5
                eval_metrics['accuracy_at_0_5'] = accuracy_at_0_5
                eval_metrics['f1_at_0_5'] = f1_at_0_5

                threshold_selection = None
                if select_decision_threshold:
                    threshold_selection = ModelEvaluator.select_f1_optimal_threshold(
                        pos_scores,
                        neg_scores,
                    )
                    decision_threshold = threshold_selection["threshold"]
                if decision_threshold is not None:
                    selected_threshold = float(decision_threshold)
                    eval_metrics['selected_threshold'] = selected_threshold
                    eval_metrics['threshold_selection_objective'] = 'validation_f1'
                    eval_metrics['accuracy_at_selected_threshold'] = ModelEvaluator.compute_accuracy(
                        pos_scores,
                        neg_scores,
                        threshold=selected_threshold,
                    )
                    eval_metrics['f1_at_selected_threshold'] = ModelEvaluator.compute_f1_score(
                        pos_scores,
                        neg_scores,
                        threshold=selected_threshold,
                    )
                if threshold_selection is not None:
                    eval_metrics['threshold_selection'] = threshold_selection

                # Sampled ranking metrics: each positive edge is ranked against its sampled negatives.
                ranking_metrics = ModelEvaluator.compute_sampled_ranking_metrics(
                    pos_scores,
                    neg_scores,
                    k_values=ranking_ks,
                    negatives_per_positive=args.k_neg_samples,
                    per_etype=False,
                )
                eval_metrics.update(ranking_metrics)
                eval_metrics['sampled_ranking_per_etype'] = ModelEvaluator.compute_sampled_ranking_metrics(
                    pos_scores,
                    neg_scores,
                    k_values=ranking_ks,
                    negatives_per_positive=args.k_neg_samples,
                    per_etype=True,
                )
            except Exception as e_metrics:
                logging.error(f"Error computing final metrics: {e_metrics}")
                traceback.print_exc()
                # Metrics will remain at their default initial values (0.0 or inf for loss)
        else:
            logging.warning("No scores collected during evaluation. Metrics will be default.")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logging.debug("Cleared CUDA cache at end of evaluation_on_test_set")
    model_module.train()
    return eval_metrics


def _parse_structural_prior_grid(grid_spec):
    specs = []
    for item in (grid_spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            alpha_text, weight_text = item.split(":", 1)
            alpha = float(alpha_text)
            neural_weight = float(weight_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid structural prior grid item '{item}'; expected alpha:weight"
            ) from exc
        label = (
            f"alpha{alpha:g}_weight{neural_weight:g}"
            .replace(".", "p")
            .replace("-", "m")
        )
        specs.append((label, alpha, neural_weight))
    return specs


def compute_disease_target_full_ranking(
    model_module,
    g_full_cpu,
    g_message_cpu,
    g_test_cpu,
    node_features_embed_cpu,
    target_types,
    args,
    device,
    g_filter_cpu=None,
):
    """Compute full filtered disease-to-target ranking for baseline comparison."""
    if not getattr(args, "enable_full_ranking_eval", False):
        return None
    if not hasattr(model_module, "get_embeddings"):
        logging.warning("Full ranking skipped: model has no get_embeddings method.")
        return None

    ranking_ks = _parse_ranking_ks(args.ranking_ks)
    disease_type = args.full_ranking_source_type
    if disease_type not in g_full_cpu.ntypes:
        logging.warning("Full ranking skipped: source node type '%s' not in graph.", disease_type)
        return None

    target_types = [target for target in target_types if target in g_full_cpu.ntypes]
    if not target_types:
        logging.warning("Full ranking skipped: no requested target types are present in graph.")
        return None

    g_filter_cpu = g_filter_cpu if g_filter_cpu is not None else g_full_cpu
    known_by_target_type = {target: {} for target in target_types}
    candidate_ids_by_target_type = {target: set() for target in target_types}
    for etype in g_full_cpu.canonical_etypes:
        src_type, _, dst_type = etype
        if src_type == disease_type and dst_type in candidate_ids_by_target_type:
            src_ids, dst_ids = g_full_cpu.edges(etype=etype)
            candidate_ids_by_target_type[dst_type].update(
                int(dst_id) for dst_id in dst_ids.tolist()
            )
        elif dst_type == disease_type and src_type in candidate_ids_by_target_type:
            src_ids, dst_ids = g_full_cpu.edges(etype=etype)
            candidate_ids_by_target_type[src_type].update(
                int(src_id) for src_id in src_ids.tolist()
            )

    for etype in g_filter_cpu.canonical_etypes:
        src_type, _, dst_type = etype
        if src_type == disease_type and dst_type in known_by_target_type:
            src_ids, dst_ids = g_filter_cpu.edges(etype=etype)
            known = known_by_target_type[dst_type]
            for src_id, dst_id in zip(src_ids.tolist(), dst_ids.tolist()):
                known.setdefault(int(src_id), set()).add(int(dst_id))
        elif dst_type == disease_type and src_type in known_by_target_type:
            src_ids, dst_ids = g_filter_cpu.edges(etype=etype)
            known = known_by_target_type[src_type]
            for src_id, dst_id in zip(src_ids.tolist(), dst_ids.tolist()):
                known.setdefault(int(dst_id), set()).add(int(src_id))

    eval_pairs_by_target_type = {target: [] for target in target_types}
    for etype in g_test_cpu.canonical_etypes:
        src_type, _, dst_type = etype
        if src_type == disease_type and dst_type in eval_pairs_by_target_type:
            src_ids, dst_ids = g_test_cpu.edges(etype=etype)
            eval_pairs_by_target_type[dst_type].extend((int(s), int(t)) for s, t in zip(src_ids.tolist(), dst_ids.tolist()))
        elif dst_type == disease_type and src_type in eval_pairs_by_target_type:
            src_ids, dst_ids = g_test_cpu.edges(etype=etype)
            eval_pairs_by_target_type[src_type].extend((int(d), int(s)) for s, d in zip(src_ids.tolist(), dst_ids.tolist()))

    was_training = model_module.training
    model_module.eval()
    with torch.no_grad():
        g_message = g_message_cpu.to(device)
        features = {ntype: tensor.to(device) for ntype, tensor in node_features_embed_cpu.items()}
        embeddings = model_module.get_embeddings(g_message, features)

    if disease_type not in embeddings:
        logging.warning("Full ranking skipped: no embeddings for source node type '%s'.", disease_type)
        return None

    batch_size = max(1, int(args.full_ranking_candidate_batch_size))
    max_edges = int(args.full_ranking_max_edges or 0)
    ranks_by_target_type = {}
    all_ranks = []
    scoring_etypes = {}
    grid_specs = (
        _parse_structural_prior_grid(args.structural_prior_grid)
        if args.structural_prior_mode == "ppr"
        else []
    )
    grid_ranks_by_label = {label: [] for label, _, _ in grid_specs}
    grid_metrics_by_target = {label: {} for label, _, _ in grid_specs}

    for target_type, eval_pairs in eval_pairs_by_target_type.items():
        if not eval_pairs:
            ranks_by_target_type[target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            for label, _, _ in grid_specs:
                grid_metrics_by_target[label][target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            continue
        if target_type not in embeddings:
            logging.warning("Full ranking skipped for target '%s': missing embeddings.", target_type)
            ranks_by_target_type[target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            for label, _, _ in grid_specs:
                grid_metrics_by_target[label][target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            continue
        if target_type not in getattr(model_module, "target_node_types", [target_type]):
            logging.warning(
                "Full ranking skipped for target '%s': model target types are %s.",
                target_type,
                getattr(model_module, "target_node_types", None),
            )
            ranks_by_target_type[target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            for label, _, _ in grid_specs:
                grid_metrics_by_target[label][target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            continue

        if max_edges > 0:
            eval_pairs = eval_pairs[:max_edges]

        source_embeddings = embeddings[disease_type]
        target_embeddings = embeddings[target_type]
        num_targets = int(target_embeddings.shape[0])
        candidate_ids = sorted(idx for idx in candidate_ids_by_target_type.get(target_type, set()) if 0 <= idx < num_targets)
        if not candidate_ids:
            logging.warning("Full ranking skipped for target '%s': no relation-specific candidate nodes.", target_type)
            ranks_by_target_type[target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            for label, _, _ in grid_specs:
                grid_metrics_by_target[label][target_type] = ModelEvaluator._ranking_metrics_from_ranks([], ranking_ks)
            continue
        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)
        candidate_id_set = set(candidate_ids)
        scoring_etype = _ranking_scoring_etype(
            model_module,
            g_full_cpu,
            disease_type,
            target_type,
        )
        scoring_etypes[target_type] = list(scoring_etype)
        predictor = _resolve_ranking_predictor(
            model_module,
            scoring_etype,
            target_type,
        )
        predictor_in_dim = predictor[0].in_features

        pairs_by_source = {}
        for source_id, true_target_id in eval_pairs:
            if 0 <= source_id < source_embeddings.shape[0] and 0 <= true_target_id < num_targets:
                pairs_by_source.setdefault(source_id, []).append(true_target_id)

        structural_scores_by_alpha = {}
        if args.structural_prior_mode == "ppr":
            required_alphas = {
                float(args.structural_prior_alpha),
                *(float(alpha) for _, alpha, _ in grid_specs),
            }
            for alpha in sorted(required_alphas):
                structural_scores_by_alpha[alpha] = _relation_ppr_scores(
                    g_message_cpu,
                    disease_type,
                    target_type,
                    pairs_by_source.keys(),
                    alpha,
                    args.structural_prior_max_iter,
                    args.structural_prior_tol,
                )

        target_ranks = []
        grid_target_ranks = {label: [] for label, _, _ in grid_specs}
        for source_id, true_target_ids in pairs_by_source.items():
            source_emb = source_embeddings[source_id].unsqueeze(0)
            score_chunks = []
            for start in range(0, num_targets, batch_size):
                target_batch = target_embeddings[start:start + batch_size]
                source_batch = source_emb.expand(target_batch.shape[0], -1)
                if scoring_etype[0] == disease_type:
                    pair_features = torch.cat(
                        [source_batch, target_batch],
                        dim=1,
                    )
                else:
                    pair_features = torch.cat(
                        [target_batch, source_batch],
                        dim=1,
                    )
                if pair_features.shape[1] != predictor_in_dim:
                    raise ValueError(
                        f"Full ranking predictor dimension mismatch for {target_type}: "
                        f"expected {predictor_in_dim}, got {pair_features.shape[1]}"
                    )
                score_chunks.append(predictor(pair_features).view(-1).detach().cpu())
            neural_scores = torch.cat(score_chunks)
            scores = neural_scores
            if structural_scores_by_alpha:
                scores = model_module.fuse_structural_scores(
                    neural_scores,
                    structural_scores_by_alpha[float(args.structural_prior_alpha)][source_id],
                    candidate_ids=candidate_tensor,
                )
            grid_scores = {
                label: model_module.fuse_structural_scores(
                    neural_scores,
                    structural_scores_by_alpha[float(alpha)][source_id],
                    candidate_ids=candidate_tensor,
                    neural_weight=neural_weight,
                )
                for label, alpha, neural_weight in grid_specs
            }
            known_targets = known_by_target_type.get(target_type, {}).get(source_id, set())
            for true_target_id in true_target_ids:
                if true_target_id not in candidate_id_set:
                    continue
                filtered_scores = scores.clone()
                other_known = [idx for idx in known_targets if idx != true_target_id and 0 <= idx < num_targets]
                if other_known:
                    filtered_scores[other_known] = float("-inf")
                true_score = filtered_scores[true_target_id]
                candidate_scores = filtered_scores[candidate_tensor]
                num_greater = int(torch.sum(candidate_scores > true_score).item())
                num_equal = int(torch.isclose(candidate_scores, true_score).sum().item())
                rank = 1.0 + num_greater + 0.5 * max(0, num_equal - 1)
                target_ranks.append(rank)
                for label, candidate_score_vector in grid_scores.items():
                    grid_filtered_scores = candidate_score_vector.clone()
                    if other_known:
                        grid_filtered_scores[other_known] = float("-inf")
                    grid_true_score = grid_filtered_scores[true_target_id]
                    grid_candidate_scores = grid_filtered_scores[candidate_tensor]
                    grid_num_greater = int(
                        torch.sum(grid_candidate_scores > grid_true_score).item()
                    )
                    grid_num_equal = int(
                        torch.isclose(grid_candidate_scores, grid_true_score).sum().item()
                    )
                    grid_target_ranks[label].append(
                        1.0
                        + grid_num_greater
                        + 0.5 * max(0, grid_num_equal - 1)
                    )

        ranks_by_target_type[target_type] = ModelEvaluator._ranking_metrics_from_ranks(target_ranks, ranking_ks)
        all_ranks.extend(target_ranks)
        for label, _, _ in grid_specs:
            target_grid_ranks = grid_target_ranks[label]
            grid_metrics_by_target[label][target_type] = ModelEvaluator._ranking_metrics_from_ranks(
                target_grid_ranks,
                ranking_ks,
            )
            grid_ranks_by_label[label].extend(target_grid_ranks)
        logging.info(
            "Full filtered ranking for %s: %s",
            target_type,
            _ranking_log_fields(ranks_by_target_type[target_type], args.ranking_ks),
        )

    aggregate = ModelEvaluator._ranking_metrics_from_ranks(all_ranks, ranking_ks)
    structural_grid_payload = {}
    for label, alpha, neural_weight in grid_specs:
        grid_metrics = ModelEvaluator._ranking_metrics_from_ranks(
            grid_ranks_by_label[label],
            ranking_ks,
        )
        structural_grid_payload[label] = {
            "alpha": alpha,
            "neural_weight": neural_weight,
            "metrics": grid_metrics,
            "per_target_type": grid_metrics_by_target[label],
        }
        logging.info(
            "Structural prior grid %s: %s",
            label,
            _ranking_log_fields(grid_metrics, args.ranking_ks),
        )
    if was_training:
        model_module.train()
    macro_metric_names = [
        metric_name
        for metric_name in aggregate
        if metric_name != "ranking_num_triples"
    ]
    metrics_macro, metrics_macro_stable, stable_macro_targets = (
        _macro_full_ranking_metrics(ranks_by_target_type, macro_metric_names)
    )
    return {
        "source_type": disease_type,
        "message_graph": "train",
        "known_positive_filter_graph": (
            "full" if g_filter_cpu is g_full_cpu else "train+validation"
        ),
        "candidate_scope": "relation_specific_target_type",
        "tie_policy": "average",
        "structural_prior": {
            "mode": args.structural_prior_mode,
            "message_graph": "train",
            "alpha": args.structural_prior_alpha,
            "max_iter": args.structural_prior_max_iter,
            "tol": args.structural_prior_tol,
            "neural_weight": args.structural_prior_neural_weight,
            "normalization": "per_query_candidate_zscore",
        },
        "candidate_batch_size": batch_size,
        "max_edges": max_edges,
        "candidate_counts": {target: len(ids) for target, ids in candidate_ids_by_target_type.items()},
        "scoring_etypes": scoring_etypes,
        "metrics": aggregate,
        "metrics_macro": metrics_macro,
        "metrics_macro_stable": metrics_macro_stable,
        "stable_macro_targets": stable_macro_targets,
        "per_target_type": ranks_by_target_type,
        "structural_prior_grid": structural_grid_payload,
    }


def _matches_source_target_relation(etype, source_type, target_node_types):
    src_type, _, dst_type = etype
    return (
        src_type == source_type and dst_type in target_node_types
    ) or (
        dst_type == source_type and src_type in target_node_types
    )


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


def _target_endpoint_is_source(etype, args):
    src_type, _, dst_type = etype
    if not getattr(args, "restrict_ranking_etypes_to_source_type", False):
        return False
    target_node_types = set(_parse_node_type_list(getattr(args, "train_target_types", "")))
    return src_type in target_node_types and dst_type == getattr(args, "full_ranking_source_type", "disease")


def _build_relation_candidate_state(g_known_cpu, args, device):
    source_type = getattr(args, "full_ranking_source_type", "disease")
    target_types = set(_parse_node_type_list(getattr(args, "train_target_types", "")))
    if source_type not in g_known_cpu.ntypes:
        return {}, {}

    num_source_nodes = g_known_cpu.num_nodes(source_type)
    candidate_sets = {target_type: set() for target_type in target_types}
    known_codes = {}
    for etype in g_known_cpu.canonical_etypes:
        src_type, _, dst_type = etype
        if src_type == source_type and dst_type in target_types:
            source_ids, target_ids = g_known_cpu.edges(etype=etype)
            target_type = dst_type
        elif dst_type == source_type and src_type in target_types:
            target_ids, source_ids = g_known_cpu.edges(etype=etype)
            target_type = src_type
        else:
            continue
        candidate_sets[target_type].update(int(value) for value in target_ids.tolist())
        codes = target_ids.to(torch.long) * num_source_nodes + source_ids.to(torch.long)
        known_codes[etype] = torch.unique(codes).to(device)

    candidate_tensors = {
        target_type: torch.tensor(
            sorted(candidate_ids),
            dtype=torch.long,
            device=device,
        )
        for target_type, candidate_ids in candidate_sets.items()
        if candidate_ids
    }
    return candidate_tensors, known_codes


def _sample_filtered_relation_negatives(
    etype,
    pos_src,
    pos_dst,
    negatives_per_positive,
    graph,
    args,
    candidate_tensors,
    known_codes,
):
    src_type, _, dst_type = etype
    source_type = getattr(args, "full_ranking_source_type", "disease")
    target_is_source = _target_endpoint_is_source(etype, args)
    target_type = src_type if target_is_source else dst_type
    candidate_pool = candidate_tensors.get(target_type)
    if candidate_pool is None or candidate_pool.numel() == 0:
        return None

    num_samples = pos_src.numel() * negatives_per_positive
    pool_indices = torch.randint(
        0,
        candidate_pool.numel(),
        (num_samples,),
        device=pos_src.device,
    )
    negative_targets = candidate_pool[pool_indices]
    if target_is_source:
        source_ids = pos_dst.repeat_interleave(negatives_per_positive)
    else:
        source_ids = pos_src.repeat_interleave(negatives_per_positive)

    known_for_etype = known_codes.get(etype)
    if known_for_etype is not None and known_for_etype.numel() > 0:
        num_source_nodes = graph.num_nodes(source_type)
        for _ in range(20):
            pair_codes = negative_targets * num_source_nodes + source_ids
            collision_mask = torch.isin(pair_codes, known_for_etype)
            if not torch.any(collision_mask):
                break
            replacement_indices = torch.randint(
                0,
                candidate_pool.numel(),
                (int(collision_mask.sum().item()),),
                device=pos_src.device,
            )
            negative_targets[collision_mask] = candidate_pool[replacement_indices]

    if target_is_source:
        return negative_targets, source_ids
    return source_ids, negative_targets


def train_model(args, device):
    """
    Core training logic for a single device (GPU or CPU).
    """
    logging.info(f"Entered train_model. Using device: {device}")
    args.ranking_ks = ','.join(str(k) for k in _parse_ranking_ks(args.ranking_ks))
    logging.info("Ranking metrics will be computed for K=%s.", args.ranking_ks)

    try:
        logging.info(f"Initializing DGLGraphCreator at {time.strftime('%X')}...")
        node_csv_path = os.path.abspath(args.nodes_csv)
        edge_csv_path = os.path.abspath(args.edges_csv)
        if not os.path.exists(node_csv_path):
            logging.error(f"Error: Node CSV not found at: {node_csv_path}")
            return
        if not os.path.exists(edge_csv_path):
            logging.error(f"Error: Edge CSV not found at: {edge_csv_path}")
            return

        category_split_map = _parse_category_split_map(args.split_category_map)
        exclude_types = _parse_node_type_list(args.split_exclude_types)
        exclude_action = (args.split_exclude_action or "train").strip().lower()
        if exclude_action not in {"train", "drop"}:
            logging.warning(
                "Unsupported split exclude action '%s'. Falling back to 'train'.",
                exclude_action,
            )
            exclude_action = "train"

        val_fold_index = args.split_kfold_val_fold if args.split_kfold_val_fold is not None else -1
        if val_fold_index < 0:
            val_fold_index = None

        logging.info(
            "Using split strategy '%s' (folds=%s, fold_index=%s, val_fold_index=%s).",
            args.split_strategy,
            args.split_kfolds,
            args.split_kfold_index,
            val_fold_index,
        )
        if category_split_map:
            logging.info("Category split assignments: %s", category_split_map)
        if exclude_types:
            logging.info(
                "Type exclusion split settings: exclude types=%s, action=%s",
                exclude_types,
                exclude_action,
            )

        if args.split_strategy.lower() in {"disease", "disease-centric", "disease_centered"}:
            logging.info(
                "Disease-centric split settings: node type '%s'",
                args.split_disease_node_type,
            )

        graph_creator = DGLGraphCreator(
            node_csv_path=node_csv_path,
            edge_csv_path=edge_csv_path,
            embed_size=args.default_embed_size,
            val_size=args.val_split,
            test_size=args.test_split,
            random_state=args.seed,
            freeze_pretrained=args.freeze_pretrained,
            graph_cache_path=args.graph_cache_path,
            feature_cache_path=args.feature_cache_path,
            device=device,
            local_model_base_path=args.local_model_path, # Pass the local model path
            generate_embeddings=not args.default_node_features_only,
            use_sequence_features=not args.disable_sequence_features,
            merged_sequence_features=args.merged_sequence_features,
            split_strategy=args.split_strategy,
            split_category_map=category_split_map,
            split_num_folds=args.split_kfolds,
            split_fold_index=args.split_kfold_index,
            split_val_fold_index=val_fold_index,
            split_disease_node_type=args.split_disease_node_type,
            split_exclude_types=exclude_types,
            split_exclude_action=exclude_action,
            split_specific_ids_path=args.split_specific_ids_path,
            split_external_edges_path=args.split_external_edges_path,
            split_cohort_ids_path=args.split_cohort_ids_path,
            split_val_cohort_ids_path=args.split_val_cohort_ids_path,
            split_holdout_scope=args.split_holdout_scope,
            split_target_input=args.eval_input_type,
            split_target_output=args.eval_output_types,
            use_evidence_weights=not args.disable_evidence_weights,
            source_ablation_mode=args.source_ablation_mode,
            source_ablation_sources=_parse_node_type_list(args.source_ablation_sources),
            source_ablation_match=args.source_ablation_match,
            graph_node_types=_parse_node_type_list(args.graph_node_types),
            graph_relation_types=_parse_node_type_list(args.graph_relation_types),
            filter_isolated_nodes=not (
                args.disable_isolated_node_filter
                or args.disable_isolated_node_filtering
            ),
            regenerate_features_after_filter=args.regenerate_features_after_filter,
        )
        logging.info(f"DGLGraphCreator instance created at {time.strftime('%X')}.")
        logging.info("Data loading and graph creation/loading complete.")

        g_full_cpu, g_train_cpu, g_val_cpu, g_test_cpu = graph_creator.get_graphs()
        g_validation_known_cpu = dgl.merge([g_train_cpu, g_val_cpu])
        node_features_embed_cpu = graph_creator.get_node_features()
        node_info = graph_creator.get_node_info()
        split_manifest = _build_split_manifest(graph_creator, args)
        split_manifest_ref = _write_split_manifest(
            args.split_manifest_output,
            split_manifest,
        )

        fold_splits = graph_creator.get_fold_splits()
        if fold_splits:
            logging.info("K-fold split produced %d folds. Held-out test fold index: %d", len(fold_splits), args.split_kfold_index)

        logging.info("Node feature keys from DGLGraphCreator: %s", list(node_features_embed_cpu.keys() if node_features_embed_cpu else []))
        logging.debug("Graphs and node features (full) remain on CPU for sampled training.")
        logging.debug("Sampled subgraphs (blocks) and batch features will be moved to device by DataLoader/training loop.")

        if args.export_ranking_dataset_path:
            export_payload = export_dgl_ranking_dataset(
                args.export_ranking_dataset_path,
                args.export_ranking_dataset_prefix,
                g_full_cpu,
                g_train_cpu,
                g_val_cpu,
                g_test_cpu,
                node_info,
                args.full_ranking_source_type,
                _parse_node_type_list(args.train_target_types),
            )
            logging.info(
                "Exported exact DGL ranking datasets to %s: %s",
                args.export_ranking_dataset_path,
                {
                    target: metadata.get("split_counts", {})
                    for target, metadata in export_payload.get("targets", {}).items()
                },
            )
            if args.export_ranking_data_only:
                return

    except Exception as e:
        logging.error(f"Error during data loading/graph creation: {e}")
        traceback.print_exc()
        return

    ntypes = g_full_cpu.ntypes
    etypes = g_full_cpu.canonical_etypes
    train_target_node_types = args.train_target_types.split(',')

    if args.enable_explainability and args.fusion_mode != 'weighted':
        logging.warning(
            "Explainability output is enabled but fusion mode '%s' is not 'weighted'. No explainability summaries will be generated.",
            args.fusion_mode
        )

    # Create a dictionary with the number of nodes for each node type
    num_nodes_dict = {ntype: g_full_cpu.num_nodes(ntype) for ntype in ntypes}

    # --- Simplified Model Initialization for Single Device ---
    try:
        # The model is initialized on CPU first, then moved to the target device.
        # This assumes the model's internal initialization doesn't require a device_map.
        
        # Create a device map for the model to ensure all parts are on the correct device
        device_str = str(device)
        device_map = {
            'embeddings': device_str, 'hgnn': device_str, 
            'transformer': device_str, 'predictors': device_str, 'default': device_str
        }

        # The model expects a dictionary of tensors. If `node_features_embed_cpu` contains `nn.Embedding`
        # objects, we extract their weight tensors. This fixes the "'Embedding' object has no attribute 'shape'" error.
        # The original code also had an incorrect `.to(device)` call on the dictionary.
        node_feature_tensors = {
            ntype: (feature.weight.data if isinstance(feature, torch.nn.Embedding) else feature)
            for ntype, feature in node_features_embed_cpu.items()
        }
        
        # DETAILED DEBUG: Log feature dimensions and raw contents before model init
        logging.info("=" * 80)
        logging.info("CRITICAL: FEATURE DIMENSION CHECK BEFORE MODEL INIT:")
        logging.info("=" * 80)
        
        for ntype, tensor in node_feature_tensors.items():
            # Skip projection layers and other non-tensor modules
            if isinstance(tensor, torch.nn.Module):
                logging.info(f"\nNode type: '{ntype}'")
                logging.info(f"  Module type: {type(tensor).__name__}")
                logging.info(f"  Skipping detailed inspection (not a tensor)")
                continue
                
            logging.info(f"\nNode type: '{ntype}'")
            logging.info(f"  Shape: {tensor.shape}")
            logging.info(f"  Dtype: {tensor.dtype}")
            logging.info(f"  Device: {tensor.device}")
            logging.info(f"  Min value: {tensor.min().item():.6f}")
            logging.info(f"  Max value: {tensor.max().item():.6f}")
            logging.info(f"  Mean value: {tensor.mean().item():.6f}")
            
            # If dimensions look suspicious (like 4), log a warning with extra info
            if len(tensor.shape) == 2 and tensor.shape[1] in [4, 5, 8]:
                logging.warning(f"  ⚠️  SUSPICIOUS DIMENSION: {tensor.shape[1]} (might indicate fallback or incomplete feature generation)")
                # Log sample values from first few rows
                if tensor.shape[0] > 0:
                    logging.warning(f"     First row values: {tensor[0, :].tolist()}")
                    if tensor.shape[0] > 1:
                        logging.warning(f"     Second row values: {tensor[1, :].tolist()}")
        
        logging.info("=" * 80)
        logging.info("END FEATURE DIMENSION CHECK")
        logging.info("=" * 80)

        # Select model based on model_type argument
        model_type = args.model_type.lower()
        
        if model_type == 'gcn':
            logging.info(f"Initializing GCN Model...")
            model = GCNModel(
                node_feature_embeddings=node_feature_tensors,
                gcn_hidden=args.hgcn_hidden,
                gcn_out=args.hgcn_out,
                num_gcn_layers=args.num_gnn_layers,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                freeze_pretrained=args.freeze_pretrained
            ).to(device)
        
        elif model_type == 'gat':
            logging.info(f"Initializing GAT Model...")
            model = GATModel(
                node_feature_embeddings=node_feature_tensors,
                gat_hidden=args.hgcn_hidden,
                gat_out=args.hgcn_out,
                num_gat_layers=args.num_gnn_layers,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                num_heads=args.transformer_heads,
                freeze_pretrained=args.freeze_pretrained
            ).to(device)
        
        elif model_type == 'transformer':
            logging.info(f"Initializing Transformer-only Model...")
            model = SimpleTransformerModel(
                node_feature_embeddings=node_feature_tensors,
                transformer_hidden=args.hgcn_hidden,
                transformer_out=args.hgcn_out,
                num_transformer_layers=args.num_transformer_layers,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                transformer_heads=args.transformer_heads,
                transformer_ff=args.transformer_ff,
                freeze_pretrained=args.freeze_pretrained
            ).to(device)

        elif model_type == 'kg_embedding':
            logging.info("Initializing KG embedding baseline (DistMult)...")
            model = KGEmbeddingModel(
                node_feature_embeddings=node_feature_tensors,
                embedding_dim=args.hgcn_out,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                freeze_pretrained=args.freeze_pretrained,
            ).to(device)

        elif model_type == 'complex':
            logging.info("Initializing ComplEx KG embedding baseline...")
            model = ComplExModel(
                node_feature_embeddings=node_feature_tensors,
                embedding_dim=args.hgcn_out,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                freeze_pretrained=args.freeze_pretrained,
            ).to(device)

        elif model_type == 'rotate':
            logging.info("Initializing RotatE KG embedding baseline...")
            model = RotatEModel(
                node_feature_embeddings=node_feature_tensors,
                embedding_dim=args.hgcn_out,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                freeze_pretrained=args.freeze_pretrained,
            ).to(device)

        elif model_type == 'rgcn':
            logging.info("Initializing R-GCN heterogeneous baseline...")
            model = RGCNModel(
                node_feature_embeddings=node_feature_tensors,
                rgcn_hidden=args.hgcn_hidden,
                rgcn_out=args.hgcn_out,
                num_rgcn_layers=args.num_gnn_layers,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                freeze_pretrained=args.freeze_pretrained,
            ).to(device)

        elif model_type == 'network_propagation':
            logging.info("Initializing network propagation baseline...")
            model = NetworkPropagationModel(
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                node_info=node_info,
                restart_prob=args.propagation_restart_prob,
                max_steps=args.propagation_steps,
                dropout=args.dropout,
            ).to(device)

        elif model_type == 'ppr':
            logging.info("Initializing PPR baseline...")
            model = PPRModel(
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                node_info=node_info,
                restart_prob=args.propagation_restart_prob,
                max_steps=args.propagation_steps,
                dropout=args.dropout,
            ).to(device)
        
        else:  # 'original' or default
            logging.info(f"Initializing Original DiseaseAssociationModel (HGCN+Transformer)...")
            model = DiseaseAssociationModel(
                node_feature_embeddings=node_feature_tensors,
                hgcn_hidden=args.hgcn_hidden,
                hgcn_out=args.hgcn_out,
                num_transformer_layers=args.num_transformer_layers,
                transformer_heads=args.transformer_heads,
                transformer_ff=args.transformer_ff,
                num_latents=args.num_latents,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=train_target_node_types,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=args.dropout,
                num_gnn_layers=args.num_gnn_layers,
                freeze_pretrained=args.freeze_pretrained,
                disable_hgcn=args.disable_hgcn,
                disable_transformer=args.disable_transformer,
                disable_moe=args.disable_moe,
                disable_mla=args.disable_mla,
                disable_feature_gating=args.disable_feature_gating,
                fusion_mode=args.fusion_mode,
                enable_explainability=args.enable_explainability,
                explainability_top_k=args.explainability_top_k,
                explainability_max_hops=args.explainability_max_hops,
                explainability_hop_penalty=args.explainability_hop_penalty,
                explainability_virtual_edge_penalty=args.explainability_virtual_edge_penalty,
                explainability_virtual_sim_threshold=args.explainability_virtual_sim_threshold,
                explainability_attribution_method=args.explainability_attribution_method,
                scorer_mode=args.scorer_mode,
                structural_prior_mode=args.structural_prior_mode,
                structural_prior_neural_weight=args.structural_prior_neural_weight,
            ).to(device)

    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        traceback.print_exc()
        return

    logging.info(f"Model ({args.model_type}) moved to {device}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params:,} trainable parameters.")

    # --- Create the full initial feature dictionary from the model's embeddings ---
    # This ensures that all node types, including those without pre-trained features,
    # have initial feature vectors for the GNN.
    # These features will be on the correct device since the model is.
    full_node_features = {}
    with torch.no_grad():
        # The model's node_embeddings module is the source of truth for all features.
        for ntype in g_full_cpu.ntypes:
            if ntype in model.node_embeddings:
                num_nodes = g_full_cpu.num_nodes(ntype)
                if num_nodes > 0:
                    # Create a tensor of all node IDs for this type
                    node_ids = torch.arange(num_nodes, device=device)
                    # Get the embeddings for all nodes of this type
                    full_node_features[ntype] = model.node_embeddings[ntype](node_ids)
                else:
                    # Handle node types with no nodes
                    embedding_dim = model.node_embeddings[ntype].embedding_dim
                    full_node_features[ntype] = torch.empty(0, embedding_dim, device=device)
            else:
                logging.warning(f"Node type '{ntype}' from graph is not in model.node_embeddings. It will have no features.")
    
    # We also need a CPU version for functions that expect it (e.g., evaluation on CPU graphs)
    full_node_features_cpu = {k: v.cpu() for k, v in full_node_features.items()}

    skip_training = args.model_type.lower() in {'network_propagation', 'ppr'}
    if skip_training:
        logging.info("%s is a non-parametric baseline. Skipping gradient training and evaluating directly.", args.model_type)
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            try:
                logging.info(f"Loading model state from: {args.load_model_path}...")
                state_dict = torch.load(args.load_model_path, map_location=device)
                
                # Check for dimension mismatches before loading
                dimension_mismatch_errors = []
                for param_name, param in state_dict.items():
                    if param_name in model.state_dict():
                        current_shape = model.state_dict()[param_name].shape
                        checkpoint_shape = param.shape
                        if current_shape != checkpoint_shape:
                            dimension_mismatch_errors.append(
                                f"  - {param_name}: checkpoint shape {checkpoint_shape} vs current model shape {current_shape}"
                            )
                
                if dimension_mismatch_errors:
                    logging.error("CRITICAL: Dimension mismatch detected when loading checkpoint. This usually indicates:")
                    logging.error("  1. The graph structure has changed (nodes were filtered or dataset modified)")
                    logging.error("  2. The model architecture doesn't match the checkpoint")
                    logging.error("  3. Different feature generation parameters were used")
                    logging.error("\nMismatched parameters:")
                    for error in dimension_mismatch_errors:
                        logging.error(error)
                    logging.error("\nCache and checkpoint must be regenerated. Please delete the cache and checkpoint directories and retrain.")
                    raise RuntimeError(f"Dimension mismatch: {len(dimension_mismatch_errors)} parameters have incompatible shapes")
                
                model.load_state_dict(state_dict)
                logging.info("Model state loaded successfully.")
                skip_training = True
                logging.info("Skipping training as model was loaded.")
            except Exception as e:
                logging.error(f"Error loading model state from {args.load_model_path}: {e}")
                logging.info("Proceeding with training from scratch.")
                skip_training = False
        else:
            logging.warning(f"Specified load_model_path '{args.load_model_path}' not found. Training from scratch.")
    
    last_saved_best_model_path = None

    if not skip_training:
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()
        
        try:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        except Exception as e:
            logging.error(f"Error creating optimizer: {e}")
            traceback.print_exc()
            return
        logging.info("Optimizer created.")

        # Initialize training state for signal handler
        update_training_state(model, 0, args.save_model_path)

        logging.info("Starting training with graph sampling...")
        
        # Multi-target early stopping: monitor multiple metrics simultaneously
        # Parse user-specified metrics or use all available
        ranking_ks = _parse_ranking_ks(args.ranking_ks)
        all_available_metrics = [
            'mrr', 'full_mrr', 'auc_roc', 'auc_prc', 'f1', 'val_loss', 'train_loss',
            'full_ranking_mrr', 'full_ranking_hits_at_10',
            'full_ranking_recall_at_50', 'full_ranking_ndcg_at_50',
            'full_ranking_macro_mrr', 'full_ranking_macro_hits_at_10',
            'full_ranking_macro_recall_at_50', 'full_ranking_macro_ndcg_at_50',
            'full_ranking_stable_macro_mrr',
            'full_ranking_stable_macro_hits_at_10',
            'full_ranking_stable_macro_recall_at_50',
            'full_ranking_stable_macro_ndcg_at_50',
        ]
        for k_value in ranking_ks:
            all_available_metrics.extend([f'hits_at_{k_value}', f'recall_at_{k_value}', f'ndcg_at_{k_value}'])
            all_available_metrics.extend([
                f'full_hits_at_{k_value}',
                f'full_recall_at_{k_value}',
                f'full_ndcg_at_{k_value}',
            ])
        if args.early_stopping_metrics:
            monitored_metrics = [m.strip() for m in args.early_stopping_metrics.split(',') if m.strip()]
            # Validate that all specified metrics are available
            invalid_metrics = [m for m in monitored_metrics if m not in all_available_metrics]
            if invalid_metrics:
                logging.warning(f"Invalid metrics in --early_stopping_metrics: {invalid_metrics}. Using all available metrics instead.")
                monitored_metrics = all_available_metrics
        else:
            monitored_metrics = all_available_metrics
        
        logging.info(f"Monitoring metrics for early stopping: {', '.join(monitored_metrics)}")
        checkpoint_metric = (args.checkpoint_metric or "").strip() or None
        if checkpoint_metric and checkpoint_metric not in monitored_metrics:
            logging.error(
                "Checkpoint metric '%s' must also be included in "
                "--early_stopping_metrics (%s).",
                checkpoint_metric,
                ", ".join(monitored_metrics),
            )
            return
        logging.info(
            "Checkpoint selection metric: %s",
            checkpoint_metric or "any monitored metric (legacy behavior)",
        )

        best_metrics = {}  # Track best value for each metric
        
        # Initialize best_metrics with initial values
        for metric in monitored_metrics:
            if metric in ['val_loss', 'train_loss'] or metric.startswith('top_k_enrichment_p_value_at_'):
                best_metrics[metric] = float('inf')
            else:
                best_metrics[metric] = 0.0
        
        epochs_no_improve = 0

        # Use all canonical edge types from the training graph for selecting training edges.
        # The train_target_node_types variable is still passed to the model for its internal logic.
        # Fix: Ensure we correctly identify edge types where the destination node is in our target types
        train_target_etypes = select_target_etypes(
            g_train_cpu,
            train_target_node_types,
            args,
            stage_label="training",
        )

        if not train_target_etypes:
            logging.warning(f"No training edge types found for the specified target node types: {train_target_node_types}.")
            logging.warning("Falling back to all available edge types in the training graph.")
            train_target_etypes = g_train_cpu.canonical_etypes
            if not train_target_etypes:
                logging.error("No edge types found in the training graph at all. Cannot proceed.")
                return

        logging.info(f"Training will focus on {len(train_target_etypes)} edge type(s) matching target node types: {train_target_node_types}")
        logging.debug(f"Target training etypes: {train_target_etypes}")

        # Optional: align model target etypes to training graph etypes.
        # Disabled by default for backward compatibility with previous behavior.
        if args.sync_model_target_etypes and hasattr(model, 'set_target_etypes'):
            model.set_target_etypes(train_target_etypes)
        
        # Log edge type distribution to help diagnose imbalance issues
        if len(train_target_etypes) > 1:
            logging.info("Edge type distribution in training set:")
            total_train_edges = sum(g_train_cpu.num_edges(etype) for etype in train_target_etypes)
            etype_stats = []
            for etype in train_target_etypes:
                edge_count = g_train_cpu.num_edges(etype)
                edge_pct = (edge_count / total_train_edges * 100) if total_train_edges > 0 else 0
                etype_stats.append((etype, edge_count, edge_pct))
            # Sort by edge count descending
            etype_stats.sort(key=lambda x: x[1], reverse=True)
            for etype, count, pct in etype_stats:
                logging.info(f"  {etype[0]}->{etype[2]}: {count} edges ({pct:.1f}%)")
            logging.info(f"  Total: {total_train_edges} edges across {len(train_target_etypes)} types")

        # For evaluation, we also focus on the validation graph edges corresponding to the training targets
        val_target_etypes = select_target_etypes(
            g_val_cpu,
            train_target_node_types,
            args,
            stage_label="validation",
        )

        logging.info(f"Validation will focus on {len(val_target_etypes)} edge type(s) in the validation set.")
        logging.debug(f"Target validation etypes: {val_target_etypes}")
        logging.info(f"DEBUG: train_target_etypes = {train_target_etypes}")
        logging.info(f"DEBUG: val_target_etypes = {val_target_etypes}")
        logging.info(f"DEBUG: Model target etypes after set_target_etypes = {model.target_etypes if hasattr(model, 'target_etypes') else 'N/A'}")

        g_train = g_train_cpu.to(device)
        candidate_tensors, known_codes = _build_relation_candidate_state(
            g_full_cpu,
            args,
            device,
        )
        if args.restrict_ranking_etypes_to_source_type:
            logging.info(
                "Relation-specific negative candidate counts: %s",
                {
                    target_type: int(candidate_ids.numel())
                    for target_type, candidate_ids in candidate_tensors.items()
                },
            )
        # 将边权重从CPU图复制到移动到设备上的图中
        for etype in g_train_cpu.canonical_etypes:
            if 'weight' in g_train_cpu.edges[etype].data:
                g_train.edges[etype].data['weight'] = g_train_cpu.edges[etype].data['weight'].to(device)
        
        logging.info("Data moved to device. Starting full-graph training loop...")

        for epoch in range(args.epochs):
            # Check for training interruption at the start of each epoch
            if check_training_interrupted():
                logging.info(f"Training interrupted by signal at epoch {epoch+1}. Stopping training loop.")
                break
                
            # Update training state for signal handler
            update_training_state(model, epoch + 1, args.save_model_path)
            
            model.train()
            start_time = time.time()
            
            # --- Full-graph training logic ---
            # 1. Negative sampling for the entire training graph
            # Manually iterate over canonical etypes to avoid ambiguity in older DGL versions.
            neg_edges = {}
            for etype in g_train.canonical_etypes:
                if g_train.num_edges(etype) > 0:
                    # Only sample negatives for the edge types we are training on
                    if etype not in train_target_etypes:
                        continue
                    num_neg_samples = g_train.num_edges(etype) * args.k_neg_samples
                    if args.restrict_ranking_etypes_to_source_type:
                        pos_src, pos_dst = g_train.edges(etype=etype)
                        sampled_edges = _sample_filtered_relation_negatives(
                            etype,
                            pos_src,
                            pos_dst,
                            args.k_neg_samples,
                            g_train,
                            args,
                            candidate_tensors,
                            known_codes,
                        )
                        if sampled_edges is None:
                            logging.warning(
                                "No relation-specific candidate pool for training edge type %s.",
                                etype,
                            )
                            continue
                        neg_src, neg_dst = sampled_edges
                    elif _target_endpoint_is_source(etype, args):
                        src_type, _, _ = etype
                        pos_src, pos_dst = g_train.edges(etype=etype)
                        neg_src = torch.randint(
                            0,
                            g_train.num_nodes(src_type),
                            (num_neg_samples,),
                            device=device,
                        )
                        neg_dst = pos_dst.repeat_interleave(args.k_neg_samples)
                    else:
                        neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
                            g_train, num_neg_samples, etype=etype, exclude_self_loops=True
                        )
                    neg_edges[etype] = (neg_src, neg_dst)
            
            neg_graph = dgl.heterograph(
                neg_edges,
                num_nodes_dict={ntype: g_train.num_nodes(ntype) for ntype in g_train.ntypes}
            )

            # 2. Forward pass
            # The model's forward pass needs the graph for structure, node features,
            # the positive graph, and the negative graph.
            # For training, g_train is used for message passing and as the positive graph.
            # 我们传递边权重给模型
            edge_weights = {}
            if not args.disable_hgcn: # Only create edge_weights if HGCN is enabled
                edge_weights = {etype: g_train.edges[etype].data['weight'] for etype in g_train.canonical_etypes if 'weight' in g_train.edges[etype].data}
             
            pos_scores, neg_scores = model(g_train, full_node_features, g_train, neg_graph, edge_weights=edge_weights)

            # 3. Compute loss weights based on edge type distribution (minority class upweighting)
            # Calculate loss weights to address class imbalance: minority edge types get higher weights
            loss_weights = {}
            total_edges = sum(g_train.num_edges(etype) for etype in train_target_etypes if g_train.num_edges(etype) > 0)
            
            if total_edges > 0:
                for etype in train_target_etypes:
                    num_edges = g_train.num_edges(etype)
                    if num_edges > 0:
                        # Inverse frequency weighting: minority classes get higher weights
                        edge_ratio = num_edges / total_edges
                        # Weight = 1 / sqrt(frequency) for balanced emphasis on minority classes
                        loss_weights[etype] = 1.0 / (edge_ratio ** 0.5)
            
            # Normalize weights so average weight is 1.0
            if loss_weights:
                avg_weight = sum(loss_weights.values()) / len(loss_weights)
                loss_weights = {k: v / avg_weight for k, v in loss_weights.items()}
            
            # Log loss weights once at first epoch for transparency
            if epoch == 0 and loss_weights:
                # Display full edge type info with edge counts for clarity
                weight_items = []
                for etype, weight in sorted(loss_weights.items(), key=lambda x: x[1], reverse=True):
                    edge_count = g_train.num_edges(etype)
                    edge_pct = (edge_count / total_edges * 100) if total_edges > 0 else 0
                    # Format: (src->dst: weight, count, %)
                    weight_items.append(f"{etype[0]}->{etype[2]}: {weight:.3f} (n={edge_count}, {edge_pct:.1f}%)")
                
                logging.info(f"Loss weights for {len(loss_weights)} edge types (minority upweighting):")
                for item in weight_items:
                    logging.info(f"  {item}")
            
            # 4. Compute loss with per-etype weighting
            pos_scores_for_loss = {k: v for k, v in pos_scores.items() if k in train_target_etypes}
            neg_scores_for_loss = {k: v for k, v in neg_scores.items() if k in train_target_etypes}
            loss = compute_loss(
                pos_scores_for_loss,
                neg_scores_for_loss,
                loss_weights=loss_weights,
                balanced_link_loss=args.balanced_link_loss,
                link_loss=args.link_loss,
                ranking_loss_margin=args.ranking_loss_margin,
                ranking_loss_temperature=args.ranking_loss_temperature,
            )

            # 4. Backward pass and optimization with gradient clipping
            optimizer.zero_grad()
            if loss.requires_grad and not torch.isnan(loss):
                loss.backward()
                
                # Gradient clipping to prevent instability in RNA training
                # This prevents gradient explosion when RNA has sparse edges
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
                
                optimizer.step()
            elif torch.isnan(loss):
                logging.warning(f"Warning: Loss is NaN in epoch {epoch+1}. Skipping optimizer step.")
            
            train_loss = loss.item() if not torch.isnan(loss) else float('inf')
            
            # Check for interruption before evaluation
            if check_training_interrupted():
                logging.info(f"Training interrupted. Skipping evaluation for epoch {epoch+1}.")
                # Force break from epoch loop
                break
            
            evaluate_start_time = time.time()
            val_metrics_dict = evaluate_graph_metrics(
                model,
                g_val_cpu,
                full_node_features_cpu,
                args,
                device,
                eval_target_etypes=val_target_etypes,
                stage_label="Validation",
                g_message_cpu=g_train_cpu,
                g_known_cpu=g_validation_known_cpu,
            )
            evaluate_end_time = time.time()
            evaluate_time = evaluate_end_time - evaluate_start_time
            
            val_loss = val_metrics_dict['loss'] if val_metrics_dict else float('inf')
            # Use the aggregated metric from the target etypes for logging and early stopping
            val_auc_prc = val_metrics_dict['auc_prc'] if val_metrics_dict else 0.0
            val_auc_roc = val_metrics_dict['auc_roc'] if val_metrics_dict else 0.0
            val_accuracy = val_metrics_dict['accuracy'] if val_metrics_dict else 0.0
            val_f1 = val_metrics_dict['f1'] if val_metrics_dict else 0.0
            val_mrr = val_metrics_dict.get('mrr', 0.0) if val_metrics_dict else 0.0
            val_full_ranking_payload = None
            full_ranking_interval = max(1, int(args.full_ranking_validation_interval))
            if (
                args.enable_full_ranking_validation
                and (epoch == 0 or (epoch + 1) % full_ranking_interval == 0)
            ):
                val_full_ranking_payload = compute_disease_target_full_ranking(
                    model,
                    g_full_cpu,
                    g_train_cpu,
                    g_val_cpu,
                    full_node_features_cpu,
                    train_target_node_types,
                    args,
                    device,
                    g_filter_cpu=g_validation_known_cpu,
                )
                if val_full_ranking_payload:
                    val_metrics_dict["full_ranking"] = val_full_ranking_payload

            end_time = time.time()
            epoch_time = end_time - start_time

            # Multi-target early stopping: collect only the monitored metric values
            all_available_metrics_values = {
                'auc_prc': val_auc_prc,
                'auc_roc': val_auc_roc,
                'f1': val_f1,
                'mrr': val_mrr,
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            if val_full_ranking_payload:
                full_metrics = val_full_ranking_payload.get("metrics", {})
                all_available_metrics_values["full_mrr"] = full_metrics.get("mrr", 0.0)
                for k_value in ranking_ks:
                    all_available_metrics_values[f"full_hits_at_{k_value}"] = full_metrics.get(f"hits_at_{k_value}", 0.0)
                    all_available_metrics_values[f"full_recall_at_{k_value}"] = full_metrics.get(f"recall_at_{k_value}", 0.0)
                    all_available_metrics_values[f"full_ndcg_at_{k_value}"] = full_metrics.get(f"ndcg_at_{k_value}", 0.0)
                all_available_metrics_values.update(
                    _full_ranking_metric_values(val_full_ranking_payload)
                )
            for k_value in ranking_ks:
                all_available_metrics_values[f'hits_at_{k_value}'] = val_metrics_dict.get(f'hits_at_{k_value}', 0.0) if val_metrics_dict else 0.0
                all_available_metrics_values[f'recall_at_{k_value}'] = val_metrics_dict.get(f'recall_at_{k_value}', 0.0) if val_metrics_dict else 0.0
                all_available_metrics_values[f'ndcg_at_{k_value}'] = val_metrics_dict.get(f'ndcg_at_{k_value}', 0.0) if val_metrics_dict else 0.0
            # Only keep metrics that are being monitored
            current_metrics = {k: v for k, v in all_available_metrics_values.items() if k in monitored_metrics}
            early_stopping_observation_ready = len(current_metrics) == len(monitored_metrics)
            
            # Update training state with current metrics
            # Use val_loss as the primary metric for state tracking (for signal handler)
            update_training_state(model, epoch + 1, args.save_model_path, 'val_loss', val_loss)
            
            # Enhanced logging to show per-etype metrics
            log_msg = (
                f"Epoch {epoch+1:03d}/{args.epochs} | train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | val_AUC_PRC: {val_auc_prc:.4f} | val_AUC_ROC: {val_auc_roc:.4f} | "
                f"val_acc: {val_accuracy:.4f} | val_f1: {val_f1:.4f} | "
                f"val_{_ranking_log_fields(val_metrics_dict or {}, ranking_ks)} | "
                f"Patience: {epochs_no_improve}/{args.early_stopping_patience} | Total Time: {epoch_time:.2f}s"
            )
            if val_full_ranking_payload:
                log_msg += (
                    "\n    Full validation ranking: "
                    + _ranking_log_fields(
                        val_full_ranking_payload.get("metrics", {}),
                        args.ranking_ks,
                    )
                )
            
            # Log per-etype metrics for detailed tracking
            if val_metrics_dict:
                if 'auc_roc_per_etype' in val_metrics_dict and val_metrics_dict['auc_roc_per_etype']:
                    # Group by target node type for better readability
                    roc_by_target_type = {}
                    for etype, metric in val_metrics_dict['auc_roc_per_etype'].items():
                        target_type = etype[2]  # (src, rel, dst) -> dst
                        if target_type not in roc_by_target_type:
                            roc_by_target_type[target_type] = []
                        roc_by_target_type[target_type].append((etype, metric))
                    
                    etype_roc_str = " | ".join([
                        f"{ttype}={','.join([f'{e[1]:.3f}' for e in etypes])}"
                        for ttype, etypes in sorted(roc_by_target_type.items())
                    ])
                    log_msg += f"\n    Per-Type AUC-ROC: {etype_roc_str}"
                
                if 'auc_prc_per_etype' in val_metrics_dict and val_metrics_dict['auc_prc_per_etype']:
                    # Group by target node type for AUC-PRC as well
                    prc_by_target_type = {}
                    for etype, metric in val_metrics_dict['auc_prc_per_etype'].items():
                        target_type = etype[2]
                        if target_type not in prc_by_target_type:
                            prc_by_target_type[target_type] = []
                        prc_by_target_type[target_type].append((etype, metric))
                    
                    etype_prc_str = " | ".join([
                        f"{ttype}={','.join([f'{e[1]:.3f}' for e in etypes])}"
                        for ttype, etypes in sorted(prc_by_target_type.items())
                    ])
                    log_msg += f"\n    Per-Type AUC-PRC: {etype_prc_str}"

                if 'sampled_ranking_per_etype' in val_metrics_dict and val_metrics_dict['sampled_ranking_per_etype']:
                    mrr_by_target_type = {}
                    for etype, metric_dict in val_metrics_dict['sampled_ranking_per_etype'].items():
                        target_type = etype[2]
                        if target_type not in mrr_by_target_type:
                            mrr_by_target_type[target_type] = []
                        mrr_by_target_type[target_type].append((etype, metric_dict.get('mrr', 0.0)))

                    etype_mrr_str = " | ".join([
                        f"{ttype}={','.join([f'{e[1]:.3f}' for e in etypes])}"
                        for ttype, etypes in sorted(mrr_by_target_type.items())
                    ])
                    log_msg += f"\n    Per-Type sampled MRR: {etype_mrr_str}"
            
            logging.info(log_msg)
            log_memory_usage(f"After epoch {epoch+1} training", device)
            
            # Multi-target early stopping: check if ANY monitored metric improved
            if not early_stopping_observation_ready:
                logging.info(
                    "Skipping early-stopping update until all monitored metrics are observed: %s",
                    ", ".join(monitored_metrics),
                )
                continue

            any_metric_improved = False
            improved_metrics = []
            
            for metric_name, current_value in current_metrics.items():
                if metric_name not in best_metrics:
                    continue
                    
                best_value = best_metrics[metric_name]
                metric_lower_is_better = metric_name in ['val_loss', 'train_loss'] or metric_name.startswith('top_k_enrichment_p_value_at_')
                
                # Check if this metric improved
                if metric_lower_is_better:
                    if current_value < best_value - args.early_stopping_min_delta:
                        improved_metrics.append((metric_name, best_value, current_value))
                        best_metrics[metric_name] = current_value
                        any_metric_improved = True
                else:  # Higher is better
                    if current_value > best_value + args.early_stopping_min_delta:
                        improved_metrics.append((metric_name, best_value, current_value))
                        best_metrics[metric_name] = current_value
                        any_metric_improved = True
            
            if any_metric_improved:
                epochs_no_improve = 0
                
                # Log which metrics improved
                improved_log = ", ".join([f"{m[0]}: {m[1]:.4f}→{m[2]:.4f}" for m in improved_metrics])
                logging.info(f"  ✓ Metrics improved: {improved_log}")
                improved_metric_names = {metric[0] for metric in improved_metrics}
                checkpoint_improved = (
                    checkpoint_metric is None
                    or checkpoint_metric in improved_metric_names
                )
                if args.save_model_path and checkpoint_improved:
                    try:
                        # Define a consistent path for the best model
                        base_name = os.path.splitext(args.save_model_path)[0]
                        best_model_path = f"{base_name}_best.pt"
                        
                        logging.info(f"  Saving new best model to {best_model_path}")
                        
                        save_dir = os.path.dirname(best_model_path)
                        if save_dir and not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        
                        torch.save(model.state_dict(), best_model_path)
                        last_saved_best_model_path = best_model_path # Keep track of it
                        logging.info(f"  Successfully saved new best model.")

                    except Exception as e_save:
                        logging.error(f"  Error saving best model state: {e_save}")
                        traceback.print_exc()
                elif args.save_model_path and checkpoint_metric:
                    logging.info(
                        "  Best checkpoint unchanged because '%s' did not improve.",
                        checkpoint_metric,
                    )
            else:
                epochs_no_improve += 1

            # Check for training interruption before early stopping check
            if check_training_interrupted():
                logging.info(f"Training interrupted after completing epoch {epoch+1}. Exiting training loop.")
                break
        

            if epochs_no_improve >= args.early_stopping_patience:
                logging.info(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                logging.info(f"No improvement in monitored metrics ({', '.join(monitored_metrics)}) for {args.early_stopping_patience} consecutive epochs.")
                # Display best values for all monitored metrics
                best_values_log = ", ".join([f"{m}: {best_metrics[m]:.4f}" for m in monitored_metrics if m in best_metrics])
                logging.info(f"Best monitored metrics achieved: {best_values_log} (Best model path: {last_saved_best_model_path if last_saved_best_model_path else 'Not saved'})")
                break
        
        if check_training_interrupted():
            logging.info("Training was interrupted by signal. Emergency model save already completed.")
        else:
            logging.info("Training finished normally.")
            # Always save the final model state if training completed without interruption
            if args.save_model_path:
                try:
                    final_model_path = args.save_model_path
                    logging.info(f"Saving final model state of epoch {epoch+1} to: {final_model_path}")
                    save_dir = os.path.dirname(final_model_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), final_model_path)
                    logging.info(f"Successfully saved final model to {final_model_path}.")
                except Exception as e:
                    logging.error(f"Error saving final model state to {final_model_path}: {e}")
                    traceback.print_exc()

    logging.info("\nFinal Evaluation/Prediction phase...")
    model_to_eval = model

    final_model_path_to_load = None

    # Define the expected path for the best model
    if args.save_model_path:
        base_name = os.path.splitext(args.save_model_path)[0]
        best_model_path = f"{base_name}_best.pt"
    else:
        best_model_path = None

    # Priority 1: Use the 'best' model if it was saved.
    if best_model_path and os.path.exists(best_model_path):
        final_model_path_to_load = best_model_path
        logging.info(f"Found a 'best' model file. Prioritizing it for final actions: {best_model_path}")
    # Priority 2: If skipping training, use the explicitly provided model.
    elif skip_training and args.load_model_path and os.path.exists(args.load_model_path):
        final_model_path_to_load = args.load_model_path
        logging.info(f"Training was skipped. Using model from --load_model_path for final actions: {args.load_model_path}")
    # Priority 3: Use the 'final' model from the completed training run.
    elif not skip_training and args.save_model_path and os.path.exists(args.save_model_path):
        final_model_path_to_load = args.save_model_path
        logging.info(f"No 'best' model was found. Using the final model from the completed training run: {args.save_model_path}")


    if final_model_path_to_load:
        logging.info(f"Loading model state from {final_model_path_to_load} for final actions.")
        try:
            state_dict = torch.load(final_model_path_to_load, map_location=device)
            model_to_eval.load_state_dict(state_dict)
        except Exception as e_load_final:
            logging.error(f"Error loading model state from {final_model_path_to_load}: {e_load_final}. Using current model state.")
    else:
        logging.info("No specific model path found for final eval/prediction. Using current model state.")

    model_to_eval.eval()

    selected_validation_threshold = None
    validation_threshold_metrics = None
    if args.report_optimal_threshold_metrics:
        validation_target_etypes = select_target_etypes(
            g_val_cpu,
            train_target_node_types,
            args,
            stage_label="validation-threshold-selection",
        )
        validation_threshold_metrics = evaluate_graph_metrics(
            model_to_eval,
            g_val_cpu,
            full_node_features_cpu,
            args,
            device,
            eval_target_etypes=validation_target_etypes,
            stage_label="Validation threshold selection",
            g_message_cpu=g_train_cpu,
            g_known_cpu=g_validation_known_cpu,
            select_decision_threshold=True,
        )
        selected_validation_threshold = validation_threshold_metrics.get(
            "selected_threshold"
        )

    # Perform final evaluation on the requested held-out split.
    if args.test_split > 0:
        final_eval_split = args.final_eval_split
        eval_graph_cpu = g_val_cpu if final_eval_split == "validation" else g_test_cpu
        stage_label = "Validation" if final_eval_split == "validation" else "Test"
        logging.info("Performing final evaluation on the %s set...", final_eval_split)
        test_target_etypes = select_target_etypes(
            eval_graph_cpu,
            train_target_node_types,
            args,
            stage_label=final_eval_split,
        )
                
        test_metrics = evaluate_graph_metrics(
            model_to_eval,
            eval_graph_cpu,
            full_node_features_cpu,
            args,
            device,
            eval_target_etypes=test_target_etypes,
            stage_label=stage_label,
            g_message_cpu=g_train_cpu,
            g_known_cpu=(
                g_validation_known_cpu
                if final_eval_split == "validation"
                else g_full_cpu
            ),
            decision_threshold=selected_validation_threshold,
            select_decision_threshold=False,
        )
        if test_metrics:
            full_ranking_payload = compute_disease_target_full_ranking(
                model_to_eval,
                g_full_cpu,
                g_train_cpu,
                eval_graph_cpu,
                full_node_features_cpu,
                train_target_node_types,
                args,
                device,
                g_filter_cpu=(
                    g_validation_known_cpu
                    if final_eval_split == "validation"
                    else g_full_cpu
                ),
            )
            if full_ranking_payload:
                test_metrics["full_ranking"] = full_ranking_payload
                for metric_name, metric_value in full_ranking_payload.get("metrics", {}).items():
                    test_metrics[f"full_ranking_{metric_name}"] = metric_value
            logging.info(f"--- Final {stage_label} Set Metrics (for target types: {train_target_node_types}) ---")
            logging.info(f"Loss: {test_metrics['loss']:.4f}, AUC-PRC: {test_metrics['auc_prc']:.4f}, AUC-ROC: {test_metrics['auc_roc']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
            logging.info(f"Sampled ranking metrics: {_ranking_log_fields(test_metrics, args.ranking_ks)}")
            if full_ranking_payload:
                logging.info(f"Full filtered disease-target ranking metrics: {_ranking_log_fields(full_ranking_payload.get('metrics', {}), args.ranking_ks)}")
            if 'auc_roc_per_etype' in test_metrics:
                etype_metrics_str = ", ".join([f"{etype[2]}: {metric:.4f}" for etype, metric in test_metrics['auc_roc_per_etype'].items()])
                if etype_metrics_str:
                    logging.info(f"Test AUC-ROC per type: ({etype_metrics_str})")
            if 'sampled_ranking_per_etype' in test_metrics:
                etype_ranking_str = ", ".join([
                    f"{etype[2]}: MRR={metric.get('mrr', 0.0):.4f}"
                    for etype, metric in test_metrics['sampled_ranking_per_etype'].items()
                ])
                if etype_ranking_str:
                    logging.info(f"Test sampled MRR per type: ({etype_ranking_str})")
            _write_metrics_output(
                args.metrics_output,
                {
                    "model_type": args.model_type,
                    "target_types": train_target_node_types,
                    "split_strategy": args.split_strategy,
                    "evaluation_split": final_eval_split,
                    "seed": args.seed,
                    "ranking_ks": _parse_ranking_ks(args.ranking_ks),
                    "split_manifest": split_manifest_ref,
                    "split_summary": {
                        "metadata_available": split_manifest["metadata_available"],
                        "excluded_cross_edge_total": split_manifest[
                            "excluded_cross_edge_total"
                        ],
                        "edge_counts": split_manifest["edge_counts"],
                    },
                    "metrics": test_metrics,
                    "threshold_selection": (
                        {
                            "source_split": "validation",
                            "threshold": selected_validation_threshold,
                            "objective": "f1",
                            "validation_accuracy": validation_threshold_metrics.get(
                                "accuracy_at_selected_threshold"
                            ),
                            "validation_f1": validation_threshold_metrics.get(
                                "f1_at_selected_threshold"
                            ),
                        }
                        if validation_threshold_metrics is not None
                        else None
                    ),
                },
            )

    if args.skip_final_prediction:
        logging.info("Skipping final prediction task (--skip-final-prediction).")
    elif args.eval_input_name and args.eval_input_type and args.eval_output_types:
        logging.info(f"Running specific prediction for input: '{args.eval_input_name}' ({args.eval_input_type}) -> {args.eval_output_types}")
        try:
            # Correctly extract tensors from embedding objects for the evaluator
            # This is now replaced by using the complete CPU feature set
            evaluator = ModelEvaluator(model_to_eval, node_info, device, full_node_features_cpu)
            eval_output_node_types_list = [s.strip() for s in args.eval_output_types.split(',')]
            
            # Loop through each output type and get top_k for each.
            for output_type in eval_output_node_types_list:
                logging.info(f"--- Predicting Top {args.eval_top_k} for node type: {output_type} ---")
                _ = evaluator.predict_top_k(
                    eval_g_cpu=g_full_cpu,
                    input_node_name=args.eval_input_name,
                    input_node_type=args.eval_input_type,
                    output_node_types=[output_type], # Pass one type at a time
                    top_k=args.eval_top_k
                )

        except Exception as e_pred_final:
            logging.error(f"Error during final prediction: {e_pred_final}")
            traceback.print_exc()
    else:
        logging.info("No specific prediction input provided. Skipping final prediction task.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disease-Biomarker Association Prediction Model Training (Single GPU Optimized)')
    parser.add_argument('--nodes_csv', type=str, required=True, help='Path to node CSV file')
    parser.add_argument('--edges_csv', type=str, required=True, help='Path to edge CSV file')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of edges for validation set. Sum of val_split and test_split must be < 1.0.')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of edges for test set. The rest is for training. If > 0, final evaluation on the test set is performed.')
    parser.add_argument('--graph_cache_path', type=str, default=None, help='Path to directory for caching/loading processed graph data.')
    parser.add_argument('--default_embed_size', type=int, default=64, help='Target embedding size for all node features.')
    parser.add_argument('--freeze_pretrained', action='store_true', help='Freeze pre-trained embeddings during training.')
    parser.add_argument('--hgcn_hidden', type=int, default=64, help='HGCN hidden layer size.')
    parser.add_argument('--hgcn_out', type=int, default=64, help='HGCN output embedding size. Must be divisible by --transformer_heads.')
    parser.add_argument('--num_gnn_layers', type=int, default=3, help='Number of GNN layers in the model. Must match --gnn_sampling_layers.')
    parser.add_argument('--num_transformer_layers', type=int, default=3, help='Number of Transformer layers in the model.')
    parser.add_argument('--transformer_heads', type=int, default=16, help='Number of attention heads in the Transformer. Must divide --hgcn_out.')
    parser.add_argument('--transformer_ff', type=int, default=512, help='Transformer feed-forward layer size.')
    parser.add_argument('--num_latents', type=int, default=32, help='Number of latent vectors for the Perceiver model.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate used in the model.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--k_neg_samples', type=int, default=5, help='Number of negative samples to generate for each positive edge during training.')
    parser.add_argument('--balanced-link-loss', action='store_true', help='Balance positive and negative BCE terms per edge type instead of weighting by raw sample counts.')
    parser.add_argument('--link-loss', type=str, default='bce', choices=['bce', 'bpr', 'sampled_softmax'], help='Link prediction objective. Ranking losses require grouped negatives per positive edge.')
    parser.add_argument('--ranking-loss-margin', type=float, default=0.0, help='Optional margin used by the BPR objective.')
    parser.add_argument('--ranking-loss-temperature', type=float, default=1.0, help='Temperature used by sampled-softmax ranking loss.')
    parser.add_argument('--ranking_ks', '--ranking-k-values', dest='ranking_ks', type=str, default='10,50,100', help='Comma-separated K values used for sampled ranking metrics (Hits@K, Recall@K, NDCG@K).')
    parser.add_argument('--training-engine', type=str, default='unified', choices=['unified', 'legacy'], help='Use the sampled shared-encoder engine or the original full-graph compatibility engine.')
    parser.add_argument('--unified-graph-backend', type=str, default='auto', choices=['auto', 'dgl', 'torch'], help='Graph convolution backend for unified BioGT. Auto uses pure PyTorch graph ops on CUDA and DGL on CPU.')
    parser.add_argument('--batch-size', type=int, default=128, help='Number of positive target edges per sampled training batch.')
    parser.add_argument('--steps-per-epoch', type=int, default=0, help='Balanced target-edge batches per epoch; 0 derives the count from the training edge count.')
    parser.add_argument('--neighbor-fanouts', type=str, default='15,10,5', help='Comma-separated in/out neighbor fanouts used to build query context at each hop.')
    parser.add_argument('--eval-every', type=int, default=5, help='Run validation every N epochs. Epoch 1 and the final epoch are always evaluated.')
    parser.add_argument('--validation-batch-size', type=int, default=256, help='Positive edges per fixed validation/test batch.')
    parser.add_argument('--validation-max-edges', type=int, default=0, help='Optional maximum validation/test positives per edge type; 0 uses all positives.')
    parser.add_argument('--fixed-validation-seed', type=int, default=1042, help='Seed used once to construct reusable validation negatives.')
    parser.add_argument('--ranking-loss-weight', type=float, default=0.3, help='Lambda for pairwise ranking loss added to balanced BCE.')
    parser.add_argument('--ranking-temperature', type=float, default=1.0, help='Temperature for pairwise positive-negative score differences.')
    parser.add_argument('--embedding-lr', type=float, default=1e-4, help='SparseAdam learning rate for trainable node embedding rows.')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0, help='Maximum norm for dense model gradients.')
    parser.add_argument('--hard-negative-interval', type=int, default=5, help='Epoch interval for refreshing the semi-hard negative pool.')
    parser.add_argument('--hard-negative-warmup', type=int, default=5, help='First epoch at which semi-hard negative mining is enabled.')
    parser.add_argument('--hard-negative-candidates', type=int, default=16, help='Filtered random candidates scored for each mined positive edge.')
    parser.add_argument('--hard-negative-ratio', type=float, default=0.5, help='Fraction of each negative group replaced by cached semi-hard negatives.')
    parser.add_argument('--hard-negative-max-edges', type=int, default=20000, help='Maximum positive edges scored during one hard-negative refresh.')
    parser.add_argument('--hard-negative-margin', type=float, default=0.0, help='Required positive-over-negative margin when selecting semi-hard negatives.')
    parser.add_argument('--disable-hard-negative-mining', action='store_true', help='Use filtered random negatives only.')
    parser.add_argument('--score-chunk-size', type=int, default=4096, help='Maximum pair count per cached-embedding scoring chunk.')
    parser.add_argument('--sparse-transformer-topk', type=int, default=256, help='Maximum non-summary query context tokens retained by sparse Transformer attention.')
    parser.add_argument('--sparse-transformer-max-hops', type=int, default=4, help='Maximum hop index represented by the sparse Transformer.')
    parser.add_argument('--sparse-transformer-hop-penalty', type=float, default=0.2, help='Penalty applied to distant context nodes during Top-K selection.')
    parser.add_argument('--sparse-transformer-evidence-weight', type=float, default=0.2, help='Weight of incident evidence strength during Top-K context selection.')
    parser.add_argument('--modality-fusion-modes', type=str, default='gene:weighted,dna:weighted,rna:concat,protein:concat,metabolite:hierarchical', help='Comma-separated modality:fusion map for the four output adapters.')
    parser.add_argument('--moe-num-experts', type=int, default=4, help='Number of experts in each modality output router.')
    parser.add_argument('--moe-top-k', type=int, default=2, help='Experts selected per target pair by the sparse router.')
    parser.add_argument('--moe-hidden-dim', type=int, default=0, help='Hidden size of each MoE expert; 0 uses twice the shared hidden size.')
    parser.add_argument('--moe-aux-loss-weight', type=float, default=0.01, help='Weight of router load-balancing auxiliary loss.')
    parser.add_argument('--embedding-cache-size', type=int, default=2, help='Maximum reusable encoded-embedding cache entries.')
    parser.add_argument('--source-node-type', type=str, default='disease', help='Query/source node type used to identify disease-candidate target relations.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping. Training stops if the metric does not improve for this many epochs.')
    parser.add_argument('--early_stopping_metrics', type=str, default='mrr,hits_at_10,recall_at_50,ndcg_at_50,val_loss,train_loss', help='Comma-separated list of metrics to monitor for early stopping. Ranking metrics use --ranking_ks. Default: mrr,hits_at_10,recall_at_50,ndcg_at_50,val_loss,train_loss')
    parser.add_argument('--checkpoint-metric', type=str, default='mrr', choices=['mrr', 'hits_at_10', 'recall_at_50', 'ndcg_at_50', 'auc_prc', 'auc_roc', 'loss', 'val_loss', 'train_loss', 'full_ranking_mrr', 'full_ranking_hits_at_10', 'full_ranking_recall_at_50', 'full_ranking_ndcg_at_50', 'full_ranking_macro_mrr', 'full_ranking_macro_hits_at_10', 'full_ranking_macro_recall_at_50', 'full_ranking_macro_ndcg_at_50', 'full_ranking_stable_macro_mrr', 'full_ranking_stable_macro_hits_at_10', 'full_ranking_stable_macro_recall_at_50', 'full_ranking_stable_macro_ndcg_at_50'], help='Validation metric used for best-checkpoint selection.')
    parser.add_argument('--report-optimal-threshold-metrics', action='store_true', help='Select one F1-maximizing decision threshold on validation scores and report Accuracy/F1 at both 0.5 and that frozen threshold on the final evaluation split.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001, help='Minimum change in the monitored metric to qualify as an improvement.')
    parser.add_argument('--save_model_path', type=str, default='trained_model.pt', help='Path to save the best trained model.')
    parser.add_argument('--metrics-output', type=str, default=None, help='Optional JSON path for validation/test metrics.')
    parser.add_argument('--enable-full-ranking-eval', action='store_true', help='Compute full filtered disease-to-target ranking metrics on the final test split.')
    parser.add_argument('--enable-full-ranking-validation', action='store_true', help='Compute full filtered disease-to-target ranking during training for checkpoint selection.')
    parser.add_argument('--full-ranking-validation-interval', type=int, default=5, help='Epoch interval for full-ranking validation.')
    parser.add_argument('--full-ranking-source-type', type=str, default='disease', help='Source node type for full ranking evaluation.')
    parser.add_argument('--full-ranking-candidate-batch-size', type=int, default=4096, help='Candidate target batch size for full ranking evaluation.')
    parser.add_argument('--full-ranking-max-edges', type=int, default=0, help='Optional cap on test edges per target type for full ranking evaluation; 0 means no cap.')
    parser.add_argument('--final-eval-split', type=str, default='test', choices=['test', 'validation'], help='Held-out split used by the final evaluation phase.')
    parser.add_argument('--skip-final-test-eval', action='store_true', help='Train and select checkpoints from validation without evaluating the final held-out split.')
    parser.add_argument('--export-ranking-dataset-path', type=str, default=None, help='Optional output root for exact DGL train/valid/test baseline datasets.')
    parser.add_argument('--export-ranking-dataset-prefix', type=str, default='biogt_dgl_exact_seed42_v1', help='Dataset prefix used by exact DGL baseline exports.')
    parser.add_argument('--export-ranking-data-only', action='store_true', help='Exit after exporting exact DGL ranking datasets.')
    parser.add_argument('--restrict-ranking-etypes-to-source-type', action='store_true', help='Restrict train/validation/test target edge types to full-ranking-source-type <-> train_target_types relations and corrupt the target endpoint for reverse edges.')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model. If provided, training will be skipped.')
    parser.add_argument('--train_target_types', type=str, default="protein,gene,rna,metabolite", help='Comma-separated node types to be used as prediction targets during training.')
    parser.add_argument('--eval_input_name', type=str, default="psoriasis", help='Name of the input node for final prediction (e.g., a specific disease name). Requires --eval_input_type and --eval_output_types.')
    parser.add_argument('--eval_input_type', type=str, default="disease", help='Type of the input node for final prediction (e.g., disease). Requires --eval_input_name and --eval_output_types.')
    parser.add_argument('--eval_output_types', type=str, default="protein,gene,rna,metabolite", help='Comma-separated output node types for final prediction (e.g., protein,gene). Requires --eval_input_name and --eval_input_type.')
    parser.add_argument('--eval_top_k', type=int, default=30, help='The number of top predictions to return for each output type.')
    parser.add_argument('--skip-final-prediction', action='store_true', help='Skip final example top-k prediction after metrics evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use. Set to -1 for CPU.')
    parser.add_argument('--local_model_path', type=str, default='./models_cache', help='Path to the local Hugging Face hub cache directory. Use this for offline execution.')
    parser.add_argument('--feature-cache-path', type=str, default=None, help='Optional shared node feature cache directory or .pt file, separate from graph/split cache.')
    parser.add_argument('--fusion-mode', type=str, default='sequential', choices=['sequential', 'concat', 'weighted', 'hierarchical', 'routing'], help='Method to fuse HGCN and Transformer features.')
    parser.add_argument('--scorer-mode', type=str, default='concat_mlp', choices=['concat_mlp', 'interaction_mlp', 'bilinear', 'hybrid_bilinear', 'hybrid_global_bilinear', 'hybrid_kge'], help='Pair scorer used by the disease-target expert heads.')
    parser.add_argument('--global-bilinear-aux-weight', type=float, default=0.0, help='Auxiliary sampled ranking loss weight applied directly to the context-independent global bilinear branch.')
    parser.add_argument('--global-bilinear-score-weight', type=float, default=1.0, help='Inference and joint-loss score weight for the independent bilinear branch in hybrid_global_bilinear or hybrid_kge mode.')
    parser.add_argument('--contextual-score-weight', type=float, default=1.0, help='Inference and joint-loss score weight for the contextual predictor in hybrid_global_bilinear or hybrid_kge mode; use 0 for a validation-only independent-branch diagnostic.')
    parser.add_argument('--global-bilinear-pretrain-epochs', type=int, default=0, help='Train-only full-coverage pretraining epochs for the global bilinear BioGT branch.')
    parser.add_argument('--global-bilinear-pretrain-batch-size', type=int, default=256, help='Positive-edge batch size for full-coverage global bilinear pretraining.')
    parser.add_argument('--global-bilinear-pretrain-negatives', type=int, default=64, help='Relation-scoped train-filtered negatives per positive during global bilinear pretraining.')
    parser.add_argument('--global-bilinear-pretrain-scope', type=str, default='target_relations', choices=['target_relations', 'all_train_relations'], help='Use only target relations or all non-empty train-graph relations for independent KGE pretraining.')
    parser.add_argument('--global-bilinear-pretrain-max-edges-per-relation', type=int, default=0, help='Optional deterministic per-epoch cap for each train relation during independent KGE pretraining; 0 uses every edge.')
    parser.add_argument('--global-bilinear-pretrain-target-finetune-epochs', type=int, default=0, help='Optional train-only target-relation calibration epochs after the broader independent KGE pretraining stage.')
    parser.add_argument('--global-bilinear-pretrain-objective', type=str, default='bce_pairwise', choices=['bce_pairwise', 'sampled_softmax', 'hard_sampled_softmax', 'full_candidate_softmax'], help='Independent KGE pretraining objective; hard_sampled_softmax mines a sampled pool, while full_candidate_softmax ranks against every relation candidate and masks other train positives.')
    parser.add_argument('--global-bilinear-pretrain-negatives-by-target', type=str, default='', help='Optional target-specific train-filtered negative counts, for example gene:2048,metabolite:2048,protein:512,rna:38.')
    parser.add_argument('--global-bilinear-pretrain-hard-negatives', type=int, default=128, help='Number of highest-scoring negatives retained per positive by hard_sampled_softmax.')
    parser.add_argument('--freeze-independent-after-pretrain', action='store_true', help='Freeze the dedicated hybrid_kge embeddings and relation scorer parameters after full-coverage pretraining.')
    parser.add_argument('--kge-relation-model', type=str, default='distmult', choices=['distmult', 'lowrank_bilinear'], help='Relation scorer for the dedicated hybrid_kge branch. lowrank_bilinear uses asymmetric source and target projections.')
    parser.add_argument('--kge-relation-rank', type=int, default=16, help='Projection rank for --kge-relation-model lowrank_bilinear.')
    parser.add_argument('--kge-context-input-weight', type=float, default=0.0, help='Inject frozen hybrid_kge disease/target embeddings into the BioGT contextual encoder input through the existing feature adapters.')
    parser.add_argument('--override-checkpoint-kge-context-input-weight', action='store_true', help='Keep the command-line KGE contextual input weight instead of restoring that non-parameterized control from a loaded checkpoint.')
    parser.add_argument('--kge-target-specific-disease-embeddings', action='store_true', help='Use a separate sparse disease KGE table for each target relation while retaining shared embeddings for non-target relations.')
    parser.add_argument('--kge-target-disease-shared-weight', type=float, default=0.0, help='Mix the shared disease KGE embedding into each target-specific disease embedding with RMS-preserving normalization; requires --kge-target-specific-disease-embeddings.')
    parser.add_argument('--initialize-target-disease-from-shared-checkpoint', action='store_true', help='Upgrade a loaded shared hybrid_kge checkpoint by requiring only target-specific disease-table keys to be absent and copying the loaded shared disease table into each new target table.')
    parser.add_argument('--freeze-shared-kge-for-target-disease-finetune', action='store_true', help='Freeze loaded KGE node tables and relation parameters so only target-specific disease KGE tables update during target-relation pretraining; supports strict shared-checkpoint migration or continuation from an already specialized checkpoint.')
    parser.add_argument('--kge-target-disease-finetune-types', type=str, default='', help='Optional comma-separated target modalities whose target-specific disease KGE tables remain trainable; requires --freeze-shared-kge-for-target-disease-finetune.')
    parser.add_argument('--kge-candidate-bias', action='store_true', help='Add one sparse train-only candidate bias table per target modality and apply it consistently to pair and full-candidate KGE scores.')
    parser.add_argument('--initialize-kge-candidate-bias-from-checkpoint', action='store_true', help='Upgrade a loaded hybrid_kge checkpoint by requiring only candidate-bias table keys to be absent and initializing those new tables to zero.')
    parser.add_argument('--freeze-kge-for-candidate-bias-finetune', action='store_true', help='Freeze loaded KGE embeddings and relation parameters so only selected target candidate-bias tables update during target-relation pretraining.')
    parser.add_argument('--kge-candidate-bias-finetune-types', type=str, default='', help='Optional comma-separated target modalities whose candidate-bias tables remain trainable; requires --freeze-kge-for-candidate-bias-finetune.')
    parser.add_argument('--kge-target-residual-rank', type=int, default=0, help='Rank of a zero-initialized target-relation low-rank residual added to the base hybrid_kge score; zero disables it.')
    parser.add_argument('--initialize-kge-target-residual-from-checkpoint', action='store_true', help='Upgrade a loaded hybrid_kge checkpoint by requiring only target-residual factor keys to be absent and initializing the residual score to exactly zero.')
    parser.add_argument('--freeze-kge-for-target-residual-finetune', action='store_true', help='Freeze loaded KGE embeddings and base relation parameters so only selected target low-rank residual factors update during target-relation pretraining.')
    parser.add_argument('--kge-target-residual-finetune-types', type=str, default='', help='Optional comma-separated target modalities whose low-rank residual factors remain trainable; requires --freeze-kge-for-target-residual-finetune.')
    parser.add_argument('--kge-path-propagation-hops', type=int, default=0, choices=[0, 3, 5], help='Enable a train-graph-only query-conditioned neural diffusion branch scored at path lengths 3 or 3+5; zero disables it.')
    parser.add_argument('--kge-path-cross-modal', action='store_true', help='Aggregate train-only disease similarity through every target modality before scoring the requested path-propagation target; requires path propagation.')
    parser.add_argument('--kge-path-cross-modal-types', type=str, default='', help='Optional comma-separated output modalities that use cross-modal train-only path diffusion while other modalities retain within-modality paths.')
    parser.add_argument('--kge-path-cross-modal-support-types', type=str, default='', help="Optional semicolon-separated per-output support map such as 'gene=protein;metabolite=protein'; mapped outputs use cross-modal train-only diffusion and unmapped cross-modal outputs retain all target supports.")
    parser.add_argument('--initialize-kge-path-propagation-from-checkpoint', action='store_true', help='Upgrade a loaded hybrid_kge checkpoint by requiring only path-gate keys to be absent and initializing their contribution to exactly zero.')
    parser.add_argument('--freeze-kge-for-path-propagation-finetune', action='store_true', help='Freeze the loaded model except selected target path gates during train-only target-relation pretraining.')
    parser.add_argument('--kge-path-propagation-finetune-types', type=str, default='', help='Optional comma-separated target modalities whose query-conditioned path gates remain trainable; requires --freeze-kge-for-path-propagation-finetune.')
    parser.add_argument('--biogt-path-propagation-hops', type=int, default=0, choices=[0, 3, 5], help='Enable the pure BioGT train-only query-conditioned typed meta-path branch at path lengths 3 or 3+5; this branch never creates or loads KGE parameters.')
    parser.add_argument('--biogt-path-support-types', type=str, default='', help="Optional semicolon-separated pure BioGT support map such as 'gene=protein+gene;metabolite=protein'; omitted outputs use all target modalities.")
    parser.add_argument('--initialize-biogt-path-from-checkpoint', action='store_true', help='Strictly upgrade a pure BioGT unified checkpoint by permitting only new BioGT path-gate keys and initializing their contribution to zero; hybrid_kge checkpoints are rejected.')
    parser.add_argument('--biogt-neural-path-dim', type=int, default=0, help='Hidden width for the default-off pure BioGT relation-conditioned train-only neural diffusion branch; requires five-hop BioGT path propagation.')
    parser.add_argument('--initialize-biogt-neural-path-from-checkpoint', action='store_true', help='Atomically upgrade a path-free pure BioGT checkpoint by permitting only scalar-path and neural-diffusion keys, zero-initializing both residuals; hybrid checkpoints are rejected.')
    parser.add_argument('--biogt-relbf-dim', type=int, default=0, help='Hidden width for the default-off pure BioGT three-layer relational Bellman-Ford residual with PNA aggregation; requires a scalar Path5 parent.')
    parser.add_argument('--initialize-biogt-relbf-from-checkpoint', action='store_true', help='Strictly upgrade a pure scalar Path5 checkpoint by permitting only new BioGT RelBF keys and zero-initializing the residual readouts; hybrid checkpoints are rejected.')
    parser.add_argument('--biogt-relbf-jk-readout', action='store_true', help='Enable the default-off pure BioGT RelBF layer-wise JK residual readout; requires --biogt-relbf-dim > 0.')
    parser.add_argument('--initialize-biogt-relbf-jk-from-checkpoint', action='store_true', help='Strictly upgrade a pure RelBF checkpoint by permitting only new zero-initialized JK residual readout keys.')
    parser.add_argument('--biogt-relbf-target-only-finetune', action='store_true', help='With a loaded RelBF checkpoint, freeze shared PNA and every excluded target so only target-specific RelBF modules selected by --biogt-path-pretrain-types update.')
    parser.add_argument('--biogt-relbf-jk-only-finetune', action='store_true', help='With a loaded JK-enabled RelBF checkpoint, freeze base, Path5, and parent RelBF parameters so only selected target JK residuals update.')
    parser.add_argument('--biogt-relbf-edge-attention', action='store_true', help='Enable default-off pure BioGT query-conditioned relation-aware attention on RelBF messages; zero-initialized attention preserves the parent scores exactly.')
    parser.add_argument('--initialize-biogt-relbf-edge-attention-from-checkpoint', action='store_true', help='Strictly upgrade a pure RelBF checkpoint by permitting only new zero-initialized edge-attention query projections.')
    parser.add_argument('--biogt-relbf-edge-attention-only-finetune', action='store_true', help='Freeze Transformer, HGCN, Path5, and parent RelBF parameters so only selected target edge-attention projections update.')
    parser.add_argument('--biogt-relbf-deep-refinement', action='store_true', help='Enable a default-off pure BioGT two-layer relation-specific PNA residual after the parent three-layer RelBF, yielding five-layer query-conditioned propagation.')
    parser.add_argument('--initialize-biogt-relbf-deep-from-checkpoint', action='store_true', help='Strictly upgrade a three-layer pure RelBF checkpoint by permitting only new deep-refinement keys and zero-initializing its residual readouts.')
    parser.add_argument('--biogt-relbf-deep-only-finetune', action='store_true', help='Freeze Transformer, HGCN, Path5, and parent three-layer RelBF so only selected target deep-refinement parameters update.')
    parser.add_argument('--biogt-relbf-indicator-residual', action='store_true', help='Enable a default-off pure BioGT relation-token source-boundary Bellman-Ford/PNA residual alongside the disease-conditioned parent RelBF.')
    parser.add_argument('--initialize-biogt-relbf-indicator-from-checkpoint', action='store_true', help='Strictly upgrade a three-layer pure RelBF checkpoint by permitting only new relation-only indicator keys and zero-initializing its final residual readouts.')
    parser.add_argument('--biogt-relbf-indicator-only-finetune', action='store_true', help='Freeze Transformer, HGCN, Path5, and parent RelBF so only selected target relation-only indicator parameters update.')
    parser.add_argument('--biogt-relbf-indicator-support-gating', action='store_true', help='Enable default-off parent-condition per-support gates on each indicator RelBF layer; requires the indicator residual.')
    parser.add_argument('--initialize-biogt-relbf-indicator-support-gating-from-checkpoint', action='store_true', help='Strictly add only zero-initialized indicator support-gate keys to an existing indicator-residual checkpoint.')
    parser.add_argument('--biogt-relbf-indicator-support-gate-only-finetune', action='store_true', help='Freeze Transformer, HGCN, Path5, parent RelBF, and indicator propagation/readout so only selected target support gates update.')
    parser.add_argument('--biogt-relbf-indicator-semantic-readout', action='store_true', help='Enable a default-off pure BioGT semantic residual readout over indicator path state, candidate state, and frozen parent disease condition.')
    parser.add_argument('--initialize-biogt-relbf-indicator-semantic-from-checkpoint', action='store_true', help='Strictly add only zero-initialized semantic residual readout keys to an existing plain indicator-residual checkpoint.')
    parser.add_argument('--biogt-relbf-indicator-semantic-only-finetune', action='store_true', help='Freeze Transformer, HGCN, Path5, parent RelBF, indicator propagation, and original indicator readout so only selected target semantic readouts update.')
    parser.add_argument('--biogt-relbf-query-relation-adapter-rank', type=int, default=0, help='Rank for the default-off pure BioGT query-conditioned low-rank delta on each RelBF output/support relation and layer; zero preserves the legacy model exactly.')
    parser.add_argument('--initialize-biogt-relbf-query-relation-adapter-from-checkpoint', action='store_true', help='Strictly add only zero-output query-conditioned RelBF adapter keys to an adapter-free pure BioGT checkpoint.')
    parser.add_argument('--biogt-relbf-query-relation-adapter-only-finetune', action='store_true', help='Freeze Transformer, HGCN, Path5, parent RelBF, and indicator parameters so only selected target query-relation adapters update.')
    parser.add_argument('--biogt-relbf-indicator-query-boundary-rank', type=int, default=0, help='Rank for the default-off pure BioGT query-conditioned low-rank delta added only to the indicator source boundary; zero preserves the parent exactly.')
    parser.add_argument('--initialize-biogt-relbf-indicator-query-boundary-from-checkpoint', action='store_true', help='Strictly add only zero-output indicator query-boundary adapter keys to an adapter-free indicator checkpoint.')
    parser.add_argument('--biogt-relbf-indicator-query-boundary-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, parent RelBF, and indicator parent so only selected target query-boundary adapters update.')
    parser.add_argument('--biogt-relbf-indicator-multirule-count', type=int, default=0, help='Number of default-off independent pure BioGT indicator rule source boundaries; zero disables the branch exactly.')
    parser.add_argument('--biogt-relbf-indicator-multirule-readout-rank', type=int, default=0, help='Low-rank width for each query-gated indicator multi-rule residual readout; zero disables the branch exactly.')
    parser.add_argument('--initialize-biogt-relbf-indicator-multirule-from-checkpoint', action='store_true', help='Strictly add only zero-output multi-rule indicator keys to a multi-rule-free indicator checkpoint.')
    parser.add_argument('--biogt-relbf-indicator-multirule-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, parent RelBF, and indicator parent so only selected target multi-rule tokens, gate, and readouts update.')
    parser.add_argument('--biogt-relbf-indicator-relation-transition-layers', type=int, default=0, choices=[0, 1, 2], help='Default-off train-only sparse relation-transition propagation depth for the pure BioGT indicator branch.')
    parser.add_argument('--biogt-relbf-indicator-relation-transition-rank', type=int, default=0, help='Low-rank query-conditioned relation-transition residual width; zero disables the branch exactly.')
    parser.add_argument('--initialize-biogt-relbf-indicator-relation-transition-from-checkpoint', action='store_true', help='Strictly add only zero-output relation-transition keys to a relation-transition-free indicator checkpoint.')
    parser.add_argument('--biogt-relbf-indicator-relation-transition-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, parent RelBF, and indicator parent so only selected target relation-transition parameters update.')
    parser.add_argument('--biogt-relbf-indicator-query-local-relation-transition-layers', type=int, default=0, choices=[0, 1, 2], help='Default-off train-only query-local relation-transition depth for the pure BioGT indicator branch.')
    parser.add_argument('--biogt-relbf-indicator-query-local-relation-transition-rank', type=int, default=0, help='Low-rank width for query-local incidence-modulated relation-transition propagation; zero disables it exactly.')
    parser.add_argument('--initialize-biogt-relbf-indicator-query-local-relation-transition-from-checkpoint', action='store_true', help='Strictly add only zero-output query-local relation-transition keys to a query-local-free indicator checkpoint.')
    parser.add_argument('--biogt-relbf-indicator-query-local-relation-transition-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, parent RelBF, and indicator parent so only selected query-local relation-transition parameters update.')
    parser.add_argument('--biogt-relbf-path-layer-attention-rank', type=int, default=0, help='Low-rank width for default-off candidate-conditioned attention over shallow/middle/deep RelBF path states; zero disables it exactly.')
    parser.add_argument('--initialize-biogt-relbf-path-layer-attention-from-checkpoint', action='store_true', help='Strictly add only zero-output candidate-conditioned path-layer attention keys to an attention-free pure BioGT checkpoint.')
    parser.add_argument('--biogt-relbf-path-layer-attention-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, and parent RelBF so only selected target path-layer attention residuals update.')
    parser.add_argument('--biogt-relbf-path-token-transformer', action='store_true', help='Enable the default-off candidate-conditioned five-token pure BioGT path micro-Transformer with fixed two-head attention.')
    parser.add_argument('--initialize-biogt-relbf-path-token-transformer-from-checkpoint', action='store_true', help='Strictly add only zero-output path-token Transformer keys to a branch-free pure BioGT checkpoint.')
    parser.add_argument('--biogt-relbf-path-token-transformer-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, and parent RelBF so only selected target path-token Transformer parameters update.')
    parser.add_argument('--biogt-relbf-candidate-competition-boundary', action='store_true', help='Enable the default-off fixed-top32 parent-anchor candidate competition boundary residual.')
    parser.add_argument('--initialize-biogt-relbf-candidate-competition-boundary-from-checkpoint', action='store_true', help='Strictly add only zero-output candidate-competition keys to a competition-free pure BioGT checkpoint.')
    parser.add_argument('--biogt-relbf-candidate-competition-boundary-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, and parent RelBF so only selected target candidate-competition parameters update.')
    parser.add_argument('--biogt-relbf-parent-top10-boundary-comparator', action='store_true', help='Enable the default-off frozen-parent rank-10/11 boundary-slot comparator residual.')
    parser.add_argument('--initialize-biogt-relbf-parent-top10-boundary-comparator-from-checkpoint', action='store_true', help='Strictly add only zero-output parent rank-10/11 comparator keys to a comparator-free pure BioGT checkpoint.')
    parser.add_argument('--biogt-relbf-parent-top10-boundary-comparator-only-finetune', action='store_true', help='Freeze HGCN, Transformer, Path5, and parent RelBF so only selected target rank-10/11 comparator parameters update.')
    parser.add_argument('--biogt-relbf-boundary-window-rank-slot-expert', action='store_true', help='Enable the default-off rank-5..20 expert stacked on a frozen parent top10/11 comparator.')
    parser.add_argument('--initialize-biogt-relbf-boundary-window-rank-slot-expert-from-checkpoint', action='store_true', help='Strictly add only zero-output boundary-window expert keys to a comparator-enabled gene8 checkpoint.')
    parser.add_argument('--biogt-relbf-boundary-window-rank-slot-expert-only-finetune', action='store_true', help='Freeze the gene8 parent and train only the selected target boundary-window expert.')
    parser.add_argument('--biogt-relbf-boundary-exchange', action='store_true', help='Enable the default-off centered incumbent/challenger exchange expert over frozen parent ranks 5..20.')
    parser.add_argument('--initialize-biogt-relbf-boundary-exchange-from-checkpoint', action='store_true', help='Strictly add only zero-output boundary-exchange keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-boundary-exchange-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target boundary-exchange expert.')
    parser.add_argument('--biogt-relbf-challenger-boundary-promotion', action='store_true', help='Enable the default-off challenger-only rank11..20 promotion expert against frozen rank5..10 incumbents.')
    parser.add_argument('--initialize-biogt-relbf-challenger-boundary-promotion-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene11 promotion keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-challenger-boundary-promotion-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene11 promotion expert.')
    parser.add_argument('--biogt-relbf-gatekeeper-challenger-paired-margin', action='store_true', help='Enable the default-off Gene12 rank10 versus rank11..20 paired-margin adapter with independent residual heads.')
    parser.add_argument('--initialize-biogt-relbf-gatekeeper-challenger-paired-margin-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene12 paired-margin keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-gatekeeper-challenger-paired-margin-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene12 paired-margin expert.')
    parser.add_argument('--biogt-relbf-boundary-prototype-ordinal-adapter', action='store_true', help='Enable the default-off Gene13 boundary-prototype ordinal adapter over parent ranks 6..16.')
    parser.add_argument('--initialize-biogt-relbf-boundary-prototype-ordinal-adapter-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene13 ordinal-adapter keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-boundary-prototype-ordinal-adapter-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene13 ordinal adapter.')
    parser.add_argument('--biogt-relbf-multi-depth-score-router', action='store_true', help='Enable the default-off Gene14 full-candidate query-conditioned multi-depth score router.')
    parser.add_argument('--initialize-biogt-relbf-multi-depth-score-router-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene14 score-router keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-multi-depth-score-router-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene14 score router.')
    parser.add_argument('--biogt-relbf-dual-stream-trajectory-agreement-calibrator', action='store_true', help='Enable the default-off Gene15 full-candidate main/path trajectory-agreement calibrator.')
    parser.add_argument('--initialize-biogt-relbf-dual-stream-trajectory-agreement-calibrator-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene15 trajectory-agreement keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-dual-stream-trajectory-agreement-calibrator-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene15 trajectory-agreement calibrator.')
    parser.add_argument('--biogt-relbf-multi-anchor-rank-landmark-calibrator', action='store_true', help='Enable the default-off Gene16 full-candidate query-conditioned multi-anchor rank-landmark calibrator.')
    parser.add_argument('--initialize-biogt-relbf-multi-anchor-rank-landmark-calibrator-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene16 rank-landmark keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-multi-anchor-rank-landmark-calibrator-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene16 rank-landmark calibrator.')
    parser.add_argument('--biogt-relbf-query-token-top64-candidate-set-transformer', action='store_true', help='Enable the default-off Gene17 query-token top64 candidate-set Transformer residual.')
    parser.add_argument('--initialize-biogt-relbf-query-token-top64-candidate-set-transformer-from-checkpoint', action='store_true', help='Strictly add only zero-output Gene17 candidate-set Transformer keys to an enabled gene9 checkpoint.')
    parser.add_argument('--biogt-relbf-query-token-top64-candidate-set-transformer-only-finetune', action='store_true', help='Freeze the complete gene9 parent and train only the selected target Gene17 candidate-set Transformer.')
    parser.add_argument('--biogt-path-pretrain-epochs', type=int, default=0, help='Train-only full-coverage epochs for the pure BioGT path gates using exact relation-candidate listwise softmax.')
    parser.add_argument('--biogt-path-pretrain-batch-size', type=int, default=128, help='Positive-edge batch size for pure BioGT path full-candidate pretraining.')
    parser.add_argument('--biogt-path-pretrain-lr', type=float, default=3e-4, help='Learning rate for pure BioGT path gates during full-candidate pretraining.')
    parser.add_argument('--biogt-path-pretrain-temperature', type=float, default=0.5, help='Temperature for pure BioGT path full-candidate softmax.')
    parser.add_argument('--biogt-path-pretrain-hard-negatives', type=int, default=0, help='Top-k exact train-filtered negatives per query for the optional pure BioGT pairwise ranking auxiliary; zero disables it.')
    parser.add_argument('--biogt-path-pretrain-rank-weight', type=float, default=0.0, help='Weight of the optional pure BioGT softplus(hard-negative - positive) ranking auxiliary; zero disables it.')
    parser.add_argument('--biogt-path-pretrain-topk-boundary-weight', type=float, default=0.0, help='Weight of the optional pure BioGT full-candidate Top-K boundary margin loss; zero strictly disables it.')
    parser.add_argument('--biogt-path-pretrain-topk-boundary-k', type=int, default=10, help='Ranking boundary K used by the pure BioGT full-candidate Top-K margin loss.')
    parser.add_argument('--biogt-path-pretrain-topk-boundary-margin', type=float, default=0.0, help='Non-negative positive-over-Kth-negative margin for the pure BioGT Top-K boundary loss.')
    parser.add_argument('--biogt-path-pretrain-lambdarank-weight', type=float, default=0.0, help='Weight of the optional pure BioGT train-only LambdaRank@K auxiliary; zero strictly disables it.')
    parser.add_argument('--biogt-path-pretrain-lambdarank-k', type=int, default=10, help='NDCG cutoff K used by the pure BioGT LambdaRank swap weights.')
    parser.add_argument('--biogt-path-pretrain-lambdarank-temperature', type=float, default=1.0, help='Positive temperature for the pure BioGT LambdaRank pairwise logistic loss.')
    parser.add_argument('--biogt-path-pretrain-soft-hits-weight', type=float, default=0.0, help='Weight of the optional pure BioGT train-only full-candidate Soft-Hits@K surrogate; zero strictly disables it.')
    parser.add_argument('--biogt-path-pretrain-soft-hits-k', type=int, default=10, help='Hits cutoff K used by the pure BioGT full-candidate Soft-Hits surrogate.')
    parser.add_argument('--biogt-path-pretrain-soft-hits-temperature', type=float, default=1.0, help='Positive sigmoid temperature for the pure BioGT full-candidate soft-rank approximation.')
    parser.add_argument('--biogt-path-pretrain-normalized-soft-hits-weight', type=float, default=0.0, help='Weight of the independent pure BioGT train-only candidate-normalized Soft-Hits@K surrogate; zero strictly disables it.')
    parser.add_argument('--biogt-path-pretrain-normalized-soft-hits-k', type=int, default=10, help='Hits cutoff K used by the candidate-normalized pure BioGT Soft-Hits surrogate.')
    parser.add_argument('--biogt-path-pretrain-normalized-soft-hits-temperature', type=float, default=1.0, help='Positive temperature for both exceedance smoothing and the normalized Soft-Hits cutoff surrogate.')
    parser.add_argument('--biogt-path-pretrain-types', type=str, default='', help='Optional comma-separated target modalities included in pure BioGT path pretraining; empty trains every target relation.')
    parser.add_argument('--freeze-biogt-base-for-path-finetune', action='store_true', help='Freeze every loaded BioGT parameter except the selected pure path gates; excluded target path gates are also frozen.')
    parser.add_argument('--biogt-path-pretrain-only', action='store_true', help='After pure BioGT path full-candidate pretraining, skip sampled main training and directly run final validation/full-ranking before saving both final checkpoints.')
    parser.add_argument('--target-sampling-weights', type=str, default='', help='Optional comma-separated target:weight schedule; every target is still sampled at least once per epoch.')
    parser.add_argument('--structural-prior-mode', type=str, default='none', choices=['none', 'ppr', 'joint_ppr'], help='Optional train-graph structural prior fused into full-ranking scores.')
    parser.add_argument('--structural-prior-alpha', type=float, default=0.85, help='Continuation probability for the train-only PPR structural prior.')
    parser.add_argument('--structural-prior-max-iter', type=int, default=50, help='Maximum power iterations for the train-only PPR structural prior.')
    parser.add_argument('--structural-prior-tol', type=float, default=1e-8, help='Convergence tolerance for the train-only PPR structural prior.')
    parser.add_argument('--structural-prior-neural-weight', type=float, default=0.0, help='Weight of the standardized neural residual added to the standardized structural prior.')
    parser.add_argument('--structural-prior-grid', type=str, default='', help='Optional comma-separated alpha:neural_weight grid evaluated in one full-ranking pass.')
    parser.add_argument('--enable-explainability', action='store_true', help="Enable explainability logging when using the 'weighted' fusion mode.")
    parser.add_argument('--explainability-top-k', type=int, default=10, help='Top-K nodes/edges to include in explainability summaries.')
    parser.add_argument('--explainability-max-hops', type=int, default=3, help='Maximum hops to trace when constructing multi-hop explanation paths.')
    parser.add_argument('--explainability-hop-penalty', type=float, default=0.18, help='Base penalty added to every physical hop in explainability path search to reduce fragmented long paths.')
    parser.add_argument('--explainability-virtual-edge-penalty', type=float, default=0.60, help='Additional penalty applied to virtual edges in gradient-guided explainability fallback.')
    parser.add_argument('--explainability-virtual-sim-threshold', type=float, default=0.45, help='Minimum cosine similarity required to create a virtual edge in the fallback graph.')
    parser.add_argument('--explainability-attribution-method', type=str, default='input_x_gradient', choices=['input_x_gradient', 'gradient_norm'], help='Attribution method used in gradient-guided explainability fallback.')

    parser.add_argument('--split-strategy', type=str, default='random', choices=['random', 'node_random', 'category', 'kfold', 'node_kfold', 'disease', 'disease-centric', 'disease_centered', 'independent_cohort', 'cohort', 'cohort_validation', 'exclude', 'type-exclusion', 'exclude_types', 'specific_ids', 'external_edges', 'target_exclude'], help='Edge- or node-induced split strategy to use when creating train/val/test graphs.')
    parser.add_argument('--split-specific-ids-path', type=str, default=None, help="Path to original node IDs (one per line, optionally node_type:original_id) whose incident edges form the test set for 'specific_ids'.")
    parser.add_argument('--split-external-edges-path', type=str, default=None, help="CSV containing disease_id, marker_id, and marker_type columns held out exactly when using 'external_edges'.")
    parser.add_argument('--split-manifest-output', type=str, default=None, help='Optional JSON path for an auditable typed-node split manifest.')
    parser.add_argument('--split-cohort-ids-path', type=str, default=None, help="Optional file of original cohort node IDs to reserve for the independent-cohort test set.")
    parser.add_argument('--split-val-cohort-ids-path', type=str, default=None, help="Optional file of original cohort node IDs to reserve for the independent-cohort validation set.")
    parser.add_argument('--split-holdout-scope', type=str, default='target_edges', choices=['target_edges', 'all_incident'], help="For independent-cohort splitting, hold out only input-output target edges or all incident cohort edges.")
    parser.add_argument('--split-category-map', type=str, default=None, help="Comma-separated list of 'node_type:target_set' pairs (target_set in {train,val,test}) used with the 'category' split strategy.")
    parser.add_argument('--split-kfolds', type=int, default=0, help='Number of folds to create when using kfold or node_kfold.')
    parser.add_argument('--split-kfold-index', type=int, default=0, help='Fold index to treat as the held-out test set when using kfold or node_kfold.')
    parser.add_argument('--split-kfold-val-fold', type=int, default=-1, help='Optional disjoint validation fold for kfold/node_kfold. For edge kfold, -1 derives validation from remaining folds; for node_kfold, -1 selects the next fold.')
    parser.add_argument('--split-disease-node-type', type=str, default='disease', help="Node type name treated as 'disease' when using the disease-centric split strategy.")
    parser.add_argument('--split-exclude-types', type=str, default=None, help='Comma-separated list of node types whose edges should be excluded from validation/test when using the type exclusion split strategy.')
    parser.add_argument('--split-exclude-action', type=str, default='train', choices=['train', 'drop'], help="Action for excluded node types: 'train' forces edges into the training set, 'drop' removes them from all splits.")

    # Ablation study arguments
    parser.add_argument('--disable-hgcn', action='store_true', help='Disable the HGCN component.')
    parser.add_argument('--disable-transformer', action='store_true', help='Disable the Transformer component.')
    parser.add_argument('--disable-moe', action='store_true', help='Disable the Mixture-of-Experts predictors (use a single predictor).')
    parser.add_argument('--disable-mla', action='store_true', help='Disable Multi-head Latent Attention in Transformer (use standard self-attention).')
    parser.add_argument('--disable-feature-gating', action='store_true', help='Disable the feature gating mechanism when fusing HGCN and Transformer outputs.')
    parser.add_argument('--disable-sequence-features', action='store_true', help='Disable using sequence data for node features.')
    parser.add_argument('--default-node-features-only', action='store_true', help='Skip pretrained/text/RDKit feature generation and use learnable default node embeddings only.')
    parser.add_argument('--merged-sequence-features', action='store_true', help='Using sentence transformer for sequence data for node features.')
    parser.add_argument('--disable-pretrained-feature-generation', action='store_true', help='Use trainable default node tables without loading external feature encoders; intended for smoke tests and ablations.')
    parser.add_argument('--sync-model-target-etypes', action='store_true', help='Synchronize model target edge types with dynamically detected training target etypes (off by default for compatibility).')
    parser.add_argument('--use-train-graph-for-eval-message-passing', action='store_true', help='Use the training graph as the message-passing graph during validation/test. Disabled by default to preserve the original validation behavior.')
    parser.add_argument('--disable-isolated-node-filtering', action='store_true', help='Keep nodes that only appear in validation/test edges. Useful for independent-cohort target-edge holdout experiments.')
    parser.add_argument('--regenerate-features-after-filter', action='store_true', help='Regenerate node features after isolated-node filtering (off by default for compatibility).')
    parser.add_argument('--disable-isolated-node-filter', action='store_true', help='Keep held-out and cold-start nodes; the original BioGT HGCN supports zero-degree nodes.')
    parser.add_argument('--target-edge-scope', type=str, default='all_target_incident', choices=['all_target_incident', 'input_output'], help="Which edge types are treated as train/eval targets. 'input_output' limits targets to --eval_input_type <-> --eval_output_types.")
    parser.add_argument('--target-relation-types', type=str, default=None, help="Optional comma-separated relation names to keep among selected target edge types, e.g. protein_disease.")
    parser.add_argument('--disable-evidence-weights', action='store_true', help='Evidence-weight ablation: replace all graph edge weights with 1.0.')
    parser.add_argument('--source-ablation-mode', type=str, default='none', choices=['none', 'exclude', 'include'], help='Source-level ablation mode applied before graph construction.')
    parser.add_argument('--source-ablation-sources', type=str, default=None, help='Comma-separated source names for source-level ablation, matched against Node 1 Source / Node 2 Source.')
    parser.add_argument('--source-ablation-match', type=str, default='any_endpoint', choices=['any_endpoint', 'both_endpoints', 'node1', 'node2'], help='How source-level ablation matches source columns.')
    parser.add_argument('--graph-node-types', type=str, default=None, help='Optional comma-separated node types to keep before graph construction, e.g. protein,disease,gene,metabolite.')
    parser.add_argument('--graph-relation-types', type=str, default=None, help='Optional comma-separated relation names to keep before graph construction, e.g. protein_disease,gene_disease.')
    parser.add_argument('--propagation-steps', type=int, default=3, help='Number of propagation steps for the network_propagation baseline.')
    parser.add_argument('--propagation-restart-prob', type=float, default=0.5, help='Restart probability for the network_propagation baseline.')
    
    # Model selection argument
    parser.add_argument('--model-type', type=str, default='original', 
                       choices=['original', 'gcn', 'gat', 'transformer', 'kg_embedding', 'complex', 'rotate', 'rgcn', 'network_propagation', 'ppr'],
                       help='Type of model to use for training. original: DiseaseAssociationModel (HGCN+Transformer), gcn: GCN-based model, gat: GAT-based model, transformer: Transformer-only model, kg_embedding: DistMult KG embedding baseline, complex: ComplEx KG embedding baseline, rotate: RotatE KG embedding baseline, rgcn: heterogeneous R-GCN baseline, network_propagation: non-parametric RWR-style baseline, ppr: Personalized PageRank baseline.')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.seed)

    # Log all arguments
    logging.info("Effective command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)
    
    # --- Device Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if available == 0:
            device = torch.device('cpu')
            logging.warning("CUDA reported available but no devices found; using CPU")
            logging.info("--- Using CPU ---")
        else:
            gpu_idx = args.gpu
            if gpu_idx < 0 or gpu_idx >= available:
                logging.warning(
                    "Requested GPU index %d is invalid (available: %d). Falling back to GPU 0.",
                    args.gpu, available
                )
                gpu_idx = 0
            device = torch.device(f'cuda:{gpu_idx}')
            try:
                gpu_name = torch.cuda.get_device_name(gpu_idx)
            except AssertionError:
                gpu_name = f"cuda:{gpu_idx}"
            logging.info(f"--- Using GPU[{gpu_idx}]: {gpu_name} ---")
    else:
        device = torch.device('cpu')
        logging.info("--- Using CPU ---")
    
    if args.training_engine == 'unified':
        from Model.unified_training import run_unified_training
        if args.moe_hidden_dim <= 0:
            args.moe_hidden_dim = None
        run_unified_training(args, device)
    else:
        train_model(args, device)
