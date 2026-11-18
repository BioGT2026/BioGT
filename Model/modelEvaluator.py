import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, f1_score
from Model.common import *
import logging
import json
import math
from collections import defaultdict, deque

# Assuming model and node_info structures are consistent with train_test.py
# If modelData.DGLGraphCreator or model.DiseaseAssociationModel are needed for type hints, import them
# from modelData import DGLGraphCreator
# from model import DiseaseAssociationModel

class ModelEvaluator:
    """
    Encapsulates model evaluation logic, including metric computation and prediction generation.
    """
    def __init__(self, model, node_info, device, initial_node_features_cpu=None, rerank_context=None):
        """
        Initializes the evaluator.

        Args:
            model: The trained PyTorch model instance.
            node_info (dict): Dictionary containing node mapping information ('name_map', 'dglid_to_orig').
            device: The torch device (e.g., 'cuda' or 'cpu') the model and graph are on.
            initial_node_features_cpu (dict, optional): A dictionary {ntype: features_tensor_on_cpu}
                                                        for all nodes in the graph. Required if using
                                                        a model.predict_links method that needs initial features.
        """
        self.model = model
        self.node_info = node_info
        self.device = device
        if self.device is None:
            raise ValueError("Device cannot be None. Please specify a valid device (e.g., 'cuda' or 'cpu').")
        else:
            logging.info(f"ModelEvaluator: Using device: {self.device} for evaluation.")

        # Removed device_map check as the model is now single-device.
        self.model.eval()

        if not all(k in self.node_info for k in ['name_map', 'dglid_to_orig']):
            raise ValueError("node_info dictionary is missing required keys ('name_map', 'dglid_to_orig').")
        self.name_map = self.node_info['name_map']
        self.dglid_to_orig = self.node_info['dglid_to_orig']
        self.type_alias_map = dict(self.node_info.get('type_alias_map', {}))

        # Ensure canonical and lowercase aliases are available even if cache lacked them
        for ntype in list(self.name_map.keys()):
            self.type_alias_map.setdefault(ntype, ntype)
            self.type_alias_map.setdefault(ntype.lower(), ntype)

        self.initial_node_features_cpu = initial_node_features_cpu
        self._last_required_target_ranks = {}
        self.rerank_context = rerank_context or {}
        self._graph_connectivity_cache = None
        self._graph_connectivity_cache_key = None
        self._hop_distance_cache = {}
        if self.initial_node_features_cpu is None:
            logging.warning("initial_node_features_cpu not provided to ModelEvaluator. "
                  "predict_top_k might fail if the model's predict_links requires initial features.")

    def get_last_required_target_ranks(self):
        return self._last_required_target_ranks or {}

    def _get_explainability_display_lines(self, res):
        physical_path = res.get('physical_shortest_path')
        final_path = res.get('final_displayed_path') or res.get('path')
        if not physical_path and not final_path:
            return []

        final_path_kind = res.get('final_path_kind')
        if final_path_kind == 'physical':
            selection_reason = 'physical'
        elif final_path_kind == 'gradient_guided':
            selection_reason = 'gradient_guided_fallback'
        else:
            selection_reason = 'unknown'

        hop_limit = getattr(self.model, 'explainability_max_hops', None)
        if physical_path:
            physical_display = physical_path
        elif hop_limit is not None:
            physical_display = f"N/A (no physical path within {hop_limit} hops)"
        else:
            physical_display = "N/A"

        final_display = final_path or "N/A"
        return [
            f"   Path Selection Reason: {selection_reason}",
            f"   Physical Shortest Path: {physical_display}",
            f"   Final Displayed Path: {final_display}",
        ]

    def _rerank_is_enabled(self):
        return float(self.rerank_context.get('lambda_structure', 0.0) or 0.0) > 0

    def prepare_rerank_cache(self, eval_g_cpu):
        if not self._rerank_is_enabled():
            return
        self._ensure_graph_connectivity_cache(eval_g_cpu)

    @staticmethod
    def _is_physical_hop_etype(etype):
        rel_name = str(etype[1]).lower() if len(etype) > 1 else ""
        return 'virtual' not in rel_name

    def _ensure_graph_connectivity_cache(self, eval_g_cpu):
        graph_identity = id(eval_g_cpu)
        if self._graph_connectivity_cache_key == graph_identity and self._graph_connectivity_cache is not None:
            return self._graph_connectivity_cache

        logging.info("Building rerank adjacency cache for graph reuse across predictions...")
        adjacency = defaultdict(set)
        for etype in eval_g_cpu.canonical_etypes:
            if not self._is_physical_hop_etype(etype):
                continue
            src_type, _, dst_type = etype
            src_nodes, dst_nodes = eval_g_cpu.edges(etype=etype)
            if src_nodes.numel() == 0:
                continue
            for src_node, dst_node in zip(src_nodes.detach().cpu().tolist(), dst_nodes.detach().cpu().tolist()):
                src_key = (src_type, int(src_node))
                dst_key = (dst_type, int(dst_node))
                adjacency[src_key].add(dst_key)
                adjacency[dst_key].add(src_key)

        self._graph_connectivity_cache = adjacency
        self._graph_connectivity_cache_key = graph_identity
        self._hop_distance_cache = {}
        logging.info("Rerank adjacency cache ready: %d source nodes with physical neighbors.", len(adjacency))
        return adjacency

    def _get_hop_distances_from_source(self, eval_g_cpu, source_key):
        max_hops = int(self.rerank_context.get('max_hops', 0) or 0)
        cache_key = (id(eval_g_cpu), source_key, max_hops)
        if cache_key in self._hop_distance_cache:
            return self._hop_distance_cache[cache_key]

        adjacency = self._ensure_graph_connectivity_cache(eval_g_cpu)
        queue = deque([(source_key, 0)])
        visited = {source_key: 0}

        while queue:
            node_key, depth = queue.popleft()
            if max_hops > 0 and depth >= max_hops:
                continue
            for neighbor in adjacency.get(node_key, set()):
                if neighbor in visited:
                    continue
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

        self._hop_distance_cache[cache_key] = visited
        return visited

    def _compute_rerank_bonus(self, eval_g_cpu, source_type, source_dgl_id, target_type, target_dgl_id, base_score):
        base_score = float(base_score)
        if not self._rerank_is_enabled():
            return {
                'base_score': base_score,
                'rerank_bonus_total': 0.0,
                'structure_bonus': 0.0,
                'hop_distance': None,
                'final_score': base_score,
            }

        source_key = (source_type, int(source_dgl_id))
        target_key = (target_type, int(target_dgl_id))
        lambda_structure = float(self.rerank_context.get('lambda_structure', 0.0) or 0.0)

        hop_distance = None
        raw_structure_signal = 0.0
        structure_bonus = 0.0
        final_score = min(max(base_score, 0.0), 1.0)
        if lambda_structure > 0:
            hop_map = self._get_hop_distances_from_source(eval_g_cpu, source_key)
            hop_distance = hop_map.get(target_key)
            if hop_distance is not None and hop_distance > 0:
                raw_structure_signal = max(lambda_structure * (1.0 / float(hop_distance)), 0.0)

            if raw_structure_signal > 0:
                structure_bonus = raw_structure_signal / (1.0 + raw_structure_signal)
                final_score = (final_score + raw_structure_signal) / (1.0 + raw_structure_signal)

        total_bonus = final_score - base_score

        return {
            'base_score': base_score,
            'rerank_bonus_total': float(total_bonus),
            'structure_signal_raw': float(raw_structure_signal),
            'structure_bonus': float(structure_bonus),
            'hop_distance': int(hop_distance) if hop_distance is not None else None,
            'final_score': float(final_score),
        }

    def _apply_soft_reranking(self, eval_g_cpu, source_type, source_dgl_id, results_list):
        if not results_list:
            return results_list

        reranked_results = []
        for result in results_list:
            rerank_meta = self._compute_rerank_bonus(
                eval_g_cpu,
                source_type,
                source_dgl_id,
                result['type'],
                result['dgl_id'],
                result['score'],
            )
            enriched_result = dict(result)
            enriched_result.update(rerank_meta)
            enriched_result['score'] = rerank_meta['final_score']
            reranked_results.append(enriched_result)

        reranked_results.sort(key=lambda item: item['score'], reverse=True)
        return reranked_results

    @staticmethod
    def _compute_metric(pos_scores_dict, neg_scores_dict, metric_func, per_etype=False, **kwargs):
        """ Helper function to compute a metric based on positive and negative scores. """
        if not per_etype:
            # Original behavior: aggregate all scores
            all_pos_scores = [s.detach().sigmoid() for s in pos_scores_dict.values() if s is not None and s.numel() > 0]
            all_neg_scores = [s.detach().sigmoid() for s in neg_scores_dict.values() if s is not None and s.numel() > 0]

            if not all_pos_scores and not all_neg_scores:
                logging.warning("(_compute_metric): No valid scores provided for metric calculation.")
                return 0.0
            
            return ModelEvaluator._calculate_metric_for_scores(all_pos_scores, all_neg_scores, metric_func, **kwargs)

        else:
            # New behavior: compute metric per edge type
            metrics_by_etype = {}
            all_etypes = set(pos_scores_dict.keys()) | set(neg_scores_dict.keys())

            for etype in all_etypes:
                pos_scores_etype = [pos_scores_dict[etype]] if etype in pos_scores_dict and pos_scores_dict[etype] is not None and pos_scores_dict[etype].numel() > 0 else []
                neg_scores_etype = [neg_scores_dict[etype]] if etype in neg_scores_dict and neg_scores_dict[etype] is not None and neg_scores_dict[etype].numel() > 0 else []
                
                # Apply sigmoid here before calculating metric
                pos_scores_etype_sig = [s.detach().sigmoid() for s in pos_scores_etype]
                neg_scores_etype_sig = [s.detach().sigmoid() for s in neg_scores_etype]

                metrics_by_etype[etype] = ModelEvaluator._calculate_metric_for_scores(
                    pos_scores_etype_sig, neg_scores_etype_sig, metric_func, **kwargs
                )
            
            return metrics_by_etype

    @staticmethod
    def _calculate_metric_for_scores(pos_scores_list, neg_scores_list, metric_func, **kwargs):
        """Calculates a specific metric given lists of positive and negative score tensors."""
        if not pos_scores_list and not neg_scores_list:
            return 0.0

        pos_scores_tensor = torch.cat(pos_scores_list) if pos_scores_list else torch.empty(0)
        neg_scores_tensor = torch.cat(neg_scores_list) if neg_scores_list else torch.empty(0)

        # The scores are already probabilities (sigmoid applied before calling)
        scores = torch.cat([pos_scores_tensor, neg_scores_tensor]).cpu().numpy()
        labels = np.concatenate([np.ones(len(pos_scores_tensor)), np.zeros(len(neg_scores_tensor))])

        if len(scores) == 0:
             return 0.0
        
        if len(np.unique(labels)) < 2:
             logging.warning(f"(_calculate_metric_for_scores): Only one class present in labels for metric {metric_func.__name__}. Returning 0.0.")
             return 0.0

        try:
            if metric_func == precision_recall_curve:
                precision, recall, _ = precision_recall_curve(labels, scores)
                if len(recall) == 0 or len(precision) == 0: # Handle empty precision/recall arrays
                    return 0.0
                return auc(recall, precision)
            elif metric_func == roc_auc_score:
                 return roc_auc_score(labels, scores)
            elif metric_func == accuracy_score or metric_func == f1_score:
                 threshold = kwargs.get('threshold', 0.5)
                 preds = (scores >= threshold).astype(int)
                 if metric_func == f1_score:
                     return metric_func(labels, preds, zero_division=0, **{k:v for k,v in kwargs.items() if k != 'threshold'})
                 return metric_func(labels, preds, **{k:v for k,v in kwargs.items() if k != 'threshold'})
            else:
                 logging.warning(f"(_calculate_metric_for_scores): Unsupported metric function {metric_func}")
                 return 0.0
        except ValueError as e:
            logging.error(f"Error calculating metric {metric_func.__name__} (ValueError): {e}")
            return 0.0
        except Exception as e_gen:
            logging.error(f"General error calculating metric {metric_func.__name__}: {e_gen}")
            return 0.0

    @staticmethod
    def compute_auc_prc(pos_scores_dict, neg_scores_dict, per_etype=False):
        """ Computes AUC-PRC for link prediction. """
        return ModelEvaluator._compute_metric(pos_scores_dict, neg_scores_dict, precision_recall_curve, per_etype=per_etype)

    @staticmethod
    def compute_auc_roc(pos_scores_dict, neg_scores_dict, per_etype=False):
        """ Computes ROC AUC score for link prediction. """
        return ModelEvaluator._compute_metric(pos_scores_dict, neg_scores_dict, roc_auc_score, per_etype=per_etype)

    @staticmethod
    def compute_accuracy(pos_scores_dict, neg_scores_dict, threshold=0.5, per_etype=False):
        """ Computes Accuracy for link prediction. """
        return ModelEvaluator._compute_metric(pos_scores_dict, neg_scores_dict, accuracy_score, threshold=threshold, per_etype=per_etype)

    @staticmethod
    def compute_f1_score(pos_scores_dict, neg_scores_dict, threshold=0.5, per_etype=False):
        """ Computes F1 Score for link prediction. """
        return ModelEvaluator._compute_metric(pos_scores_dict, neg_scores_dict, f1_score, threshold=threshold, per_etype=per_etype)

    @staticmethod
    def _binary_probabilities_and_labels(pos_scores_dict, neg_scores_dict):
        """Return aggregated sigmoid probabilities and binary labels."""
        positive = [
            score.detach().sigmoid().reshape(-1).cpu()
            for score in pos_scores_dict.values()
            if score is not None and score.numel() > 0
        ]
        negative = [
            score.detach().sigmoid().reshape(-1).cpu()
            for score in neg_scores_dict.values()
            if score is not None and score.numel() > 0
        ]
        if not positive or not negative:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        probabilities = torch.cat(positive + negative).numpy()
        labels = np.concatenate(
            [
                np.ones(sum(item.numel() for item in positive), dtype=np.int64),
                np.zeros(sum(item.numel() for item in negative), dtype=np.int64),
            ]
        )
        return probabilities, labels

    @staticmethod
    def select_f1_optimal_threshold(pos_scores_dict, neg_scores_dict):
        """Select a deterministic F1-maximizing threshold on validation scores."""
        probabilities, labels = ModelEvaluator._binary_probabilities_and_labels(
            pos_scores_dict,
            neg_scores_dict,
        )
        if probabilities.size == 0:
            return {
                "threshold": 0.5,
                "objective": "f1",
                "f1": 0.0,
                "accuracy": 0.0,
                "num_examples": 0,
            }

        _, _, thresholds = precision_recall_curve(labels, probabilities)
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], thresholds)))
        best = None
        for threshold in candidates:
            predictions = (probabilities >= threshold).astype(np.int64)
            f1_value = float(f1_score(labels, predictions, zero_division=0))
            accuracy_value = float(accuracy_score(labels, predictions))
            key = (f1_value, accuracy_value, -abs(float(threshold) - 0.5))
            if best is None or key > best[0]:
                best = (key, float(threshold), f1_value, accuracy_value)

        return {
            "threshold": best[1],
            "objective": "f1",
            "f1": best[2],
            "accuracy": best[3],
            "num_examples": int(probabilities.size),
        }

    @staticmethod
    def normalize_ranking_ks(k_values=None):
        """Normalize K values used by sampled ranking metrics."""
        if k_values is None:
            return [10, 50, 100]
        if isinstance(k_values, str):
            raw_values = [item.strip() for item in k_values.split(',') if item.strip()]
        else:
            raw_values = list(k_values)

        normalized = []
        for raw_value in raw_values:
            try:
                k_value = int(raw_value)
            except (TypeError, ValueError):
                logging.warning("Ignoring invalid ranking K value: %s", raw_value)
                continue
            if k_value > 0 and k_value not in normalized:
                normalized.append(k_value)

        return sorted(normalized) if normalized else [10, 50, 100]

    @staticmethod
    def empty_ranking_metrics(k_values=None):
        """Return a complete zero-filled sampled ranking metric dict."""
        k_values = ModelEvaluator.normalize_ranking_ks(k_values)
        metrics = {
            'mrr': 0.0,
            'mr': 0.0,
            'ranking_num_triples': 0,
        }
        for k_value in k_values:
            metrics[f'hits_at_{k_value}'] = 0.0
            metrics[f'recall_at_{k_value}'] = 0.0
            metrics[f'ndcg_at_{k_value}'] = 0.0
        return metrics

    @staticmethod
    def _ranking_metrics_from_ranks(ranks, k_values=None):
        """Compute ranking metrics from 1-based positive-edge ranks."""
        k_values = ModelEvaluator.normalize_ranking_ks(k_values)
        if not ranks:
            return ModelEvaluator.empty_ranking_metrics(k_values)

        ranks_array = np.asarray(ranks, dtype=np.float64)
        metrics = {
            'mrr': float(np.mean(1.0 / ranks_array)),
            'mr': float(np.mean(ranks_array)),
            'ranking_num_triples': int(len(ranks)),
        }
        for k_value in k_values:
            hits = ranks_array <= float(k_value)
            metrics[f'hits_at_{k_value}'] = float(np.mean(hits))
            # Each evaluated query has one positive edge, so Recall@K equals Hits@K.
            metrics[f'recall_at_{k_value}'] = float(metrics[f'hits_at_{k_value}'])
            ndcg_values = np.where(hits, 1.0 / np.log2(ranks_array + 1.0), 0.0)
            metrics[f'ndcg_at_{k_value}'] = float(np.mean(ndcg_values))
        return metrics

    @staticmethod
    def _sampled_ranks_for_scores(pos_scores, neg_scores, negatives_per_positive=None):
        """Rank each positive score against its sampled negatives."""
        if pos_scores is None or neg_scores is None or pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return []

        pos = pos_scores.detach().sigmoid().flatten().cpu()
        neg = neg_scores.detach().sigmoid().flatten().cpu()
        num_pos = int(pos.numel())
        num_neg = int(neg.numel())
        if num_pos == 0 or num_neg == 0:
            return []

        if negatives_per_positive is None:
            negatives_per_positive = num_neg // num_pos
        negatives_per_positive = int(negatives_per_positive or 0)
        if negatives_per_positive <= 0:
            return []

        expected_neg = num_pos * negatives_per_positive
        if num_neg < expected_neg:
            usable_pos = num_neg // negatives_per_positive
            if usable_pos <= 0:
                return []
            logging.warning(
                "Ranking metric negative-score count is shorter than expected: got %d, expected %d. "
                "Using %d positive edges.",
                num_neg,
                expected_neg,
                usable_pos,
            )
            pos = pos[:usable_pos]
            expected_neg = usable_pos * negatives_per_positive
        elif num_neg > expected_neg:
            logging.warning(
                "Ranking metric negative-score count is longer than expected: got %d, expected %d. "
                "Ignoring extra negatives.",
                num_neg,
                expected_neg,
            )

        neg = neg[:expected_neg].reshape(-1, negatives_per_positive)
        ranks = 1 + (neg > pos[:neg.shape[0]].unsqueeze(1)).sum(dim=1)
        return [int(rank) for rank in ranks.tolist()]

    @staticmethod
    def compute_sampled_ranking_metrics(
        pos_scores_dict,
        neg_scores_dict,
        k_values=None,
        negatives_per_positive=None,
        per_etype=False,
    ):
        """Compute sampled link-prediction ranking metrics from positive and sampled negative scores."""
        k_values = ModelEvaluator.normalize_ranking_ks(k_values)
        all_etypes = set(pos_scores_dict.keys()) | set(neg_scores_dict.keys())

        if per_etype:
            metrics_by_etype = {}
            for etype in all_etypes:
                ranks = ModelEvaluator._sampled_ranks_for_scores(
                    pos_scores_dict.get(etype),
                    neg_scores_dict.get(etype),
                    negatives_per_positive=negatives_per_positive,
                )
                metrics_by_etype[etype] = ModelEvaluator._ranking_metrics_from_ranks(ranks, k_values)
            return metrics_by_etype

        all_ranks = []
        for etype in all_etypes:
            all_ranks.extend(ModelEvaluator._sampled_ranks_for_scores(
                pos_scores_dict.get(etype),
                neg_scores_dict.get(etype),
                negatives_per_positive=negatives_per_positive,
            ))
        return ModelEvaluator._ranking_metrics_from_ranks(all_ranks, k_values)

    def _resolve_node_type(self, requested_type, available_types):
        """Return the canonical node type for the given request if present in the graph."""
        if not requested_type:
            return None

        stripped = requested_type.strip()
        if not stripped:
            return None

        candidates = [stripped]
        lowered = stripped.lower()
        if lowered not in candidates:
            candidates.append(lowered)

        for candidate in candidates:
            canonical = self.type_alias_map.get(candidate)
            if canonical and canonical in self.name_map and canonical in available_types:
                return canonical

        for candidate in candidates:
            if candidate in self.name_map and candidate in available_types:
                return candidate

        return None

    @staticmethod
    def _normalize_original_id(raw_id):
        if raw_id is None:
            return None

        normalized = str(raw_id).strip()
        return normalized or None

    def _build_orig_id_lookup(self, canonical_type):
        lookup = {}
        for dgl_id, orig_id in self.dglid_to_orig.get(canonical_type, {}).items():
            normalized = self._normalize_original_id(orig_id)
            if normalized is None:
                continue
            lookup.setdefault(normalized, int(dgl_id))
        return lookup

    def _build_dgl_to_name_lookup(self, canonical_type):
        return {
            int(dgl_id): name
            for name, dgl_id in self.name_map.get(canonical_type, {}).items()
        }

    def _resolve_input_node(self, eval_g_cpu, input_node_type, input_node_name=None, input_node_id=None):
        available_ntypes = set(eval_g_cpu.ntypes)
        canonical_input_type = self._resolve_node_type(input_node_type, available_ntypes)
        normalized_input_id = self._normalize_original_id(input_node_id)
        resolved_input_name = None
        if isinstance(input_node_name, str):
            stripped_name = input_node_name.strip()
            if stripped_name:
                resolved_input_name = stripped_name

        if not canonical_input_type and resolved_input_name:
            for candidate_type, name_lookup in self.name_map.items():
                if candidate_type in available_ntypes and resolved_input_name in name_lookup:
                    canonical_input_type = candidate_type
                    logging.warning(
                        "Input node type '%s' not found; using '%s' based on node name match.",
                        input_node_type,
                        candidate_type,
                    )
                    break

        if not canonical_input_type and normalized_input_id:
            for candidate_type in available_ntypes:
                candidate_lookup = self._build_orig_id_lookup(candidate_type)
                if normalized_input_id in candidate_lookup:
                    canonical_input_type = candidate_type
                    logging.warning(
                        "Input node type '%s' not found; using '%s' based on original node ID '%s'.",
                        input_node_type,
                        candidate_type,
                        normalized_input_id,
                    )
                    break

        if not canonical_input_type:
            logging.error(f"Input node type '{input_node_type}' not found in graph data.")
            return None, None, None, None

        name_lookup = self.name_map.get(canonical_input_type, {})
        input_node_dgl_id = None

        if resolved_input_name and resolved_input_name in name_lookup:
            input_node_dgl_id = int(name_lookup[resolved_input_name])

        if input_node_dgl_id is None and normalized_input_id:
            input_node_dgl_id = self._build_orig_id_lookup(canonical_input_type).get(normalized_input_id)

        if input_node_dgl_id is None:
            if resolved_input_name:
                logging.error(
                    "Input node '%s' not found for type '%s'.",
                    resolved_input_name,
                    canonical_input_type,
                )
            elif normalized_input_id:
                logging.error(
                    "Input node original ID '%s' not found for type '%s'.",
                    normalized_input_id,
                    canonical_input_type,
                )
            else:
                logging.error("Either input_node_name or input_node_id must be provided for prediction.")
            return None, None, None, None

        dgl_to_name_lookup = self._build_dgl_to_name_lookup(canonical_input_type)
        if resolved_input_name is None:
            resolved_input_name = dgl_to_name_lookup.get(int(input_node_dgl_id))

        if normalized_input_id is None:
            normalized_input_id = self._normalize_original_id(
                self.dglid_to_orig.get(canonical_input_type, {}).get(int(input_node_dgl_id))
            )

        return canonical_input_type, int(input_node_dgl_id), resolved_input_name, normalized_input_id

    def _prepare_eval_graph_inputs(self, eval_g_cpu):
        eval_g_on_device = eval_g_cpu.to(self.device)
        h_initial_on_device = {
            ntype: feat.to(self.device)
            for ntype, feat in self.initial_node_features_cpu.items()
            if ntype in eval_g_on_device.ntypes and eval_g_on_device.num_nodes(ntype) > 0 and feat.shape[0] == eval_g_on_device.num_nodes(ntype)
        }

        missing_feature_ntypes = []
        for ntype_g in eval_g_on_device.ntypes:
            if eval_g_on_device.num_nodes(ntype_g) > 0 and ntype_g not in h_initial_on_device:
                missing_feature_ntypes.append(ntype_g)

        if missing_feature_ntypes:
            error_message = (
                f"CRITICAL ERROR in ModelEvaluator.predict_top_k: Initial features are missing or mismatched for node types "
                f"present in the evaluation graph: {missing_feature_ntypes}. "
                f"These features are required by the model's predict_links method. "
                f"Ensure 'initial_node_features_cpu' (provided to ModelEvaluator) contains valid and correctly shaped features "
                f"for all node types in the graph that have nodes. "
                f"Node types in current eval_g_on_device: {eval_g_on_device.ntypes}. "
                f"Keys in provided initial_node_features_cpu: {list(self.initial_node_features_cpu.keys()) if self.initial_node_features_cpu else 'None'}. "
                f"Keys in successfully processed h_initial_on_device: {list(h_initial_on_device.keys())}."
            )
            logging.error(error_message)
            raise ValueError(error_message)

        return eval_g_on_device, h_initial_on_device

    def predict_top_k(self, eval_g_cpu, input_node_name, input_node_type, output_node_types, top_k=10,
                      target_batch_size=1024, display_output=True, input_node_id=None,
                      required_target_ids_by_type=None, explainability_path_target_ids_by_type=None):
        """
        Predicts the top K associations for a given input node.
        Uses the model's 'predict_links' method which requires initial features for the graph.
        Batches target nodes if they are too numerous to reduce memory.

        Args:
            eval_g_cpu (DGLHeteroGraph): The graph (on CPU) to perform evaluation on.
            input_node_name (str): The name of the input node.
            input_node_type (str): The type of the input node.
            output_node_types (list[str]): A list of target node types to predict associations for.
            top_k (int): The number of top predictions to return and display.
            target_batch_size (int): Max number of target nodes of a single type to process in one call to predict_links.
        """
        self.model.eval()
        self._last_required_target_ranks = {}

        if self.initial_node_features_cpu is None:
            logging.error("`initial_node_features_cpu` is required for `predict_top_k` but was not provided during ModelEvaluator initialization.")
            return {}

        available_ntypes = set(eval_g_cpu.ntypes)
        canonical_input_type, input_node_dgl_id, resolved_input_name, resolved_input_id = self._resolve_input_node(
            eval_g_cpu,
            input_node_type,
            input_node_name=input_node_name,
            input_node_id=input_node_id,
        )
        if not canonical_input_type:
            return {}

        if canonical_input_type != input_node_type:
            logging.info(f"Resolved input node type '{input_node_type}' to canonical type '{canonical_input_type}'.")

        canonical_output_order = []
        canonical_to_requested = {}
        skipped_output_types = []
        for requested_type in output_node_types:
            canonical_target = self._resolve_node_type(requested_type, available_ntypes)
            if not canonical_target:
                logging.warning(f"Output node type '{requested_type}' not found in graph data. Skipping.")
                skipped_output_types.append(requested_type)
                continue
            alias_bucket = canonical_to_requested.setdefault(canonical_target, [])
            if requested_type not in alias_bucket:
                alias_bucket.append(requested_type)
            if canonical_target not in canonical_output_order:
                canonical_output_order.append(canonical_target)

        if not canonical_output_order:
            logging.error("No valid output node types were resolved for prediction.")
            return {requested: [] for requested in output_node_types}

        eval_g_on_device, h_initial_on_device = self._prepare_eval_graph_inputs(eval_g_cpu)

        resolved_input_display = (
            canonical_input_type if canonical_input_type == input_node_type
            else f"{input_node_type} -> {canonical_input_type}"
        )
        input_display_name = resolved_input_name or resolved_input_id or input_node_name or "UNKNOWN_INPUT"
        logging.info(f"Performing prediction for '{input_display_name}' ({resolved_input_display}) on device {self.device}...")
        
        all_predictions_accumulated = {} # To store predictions from all batches

        with torch.no_grad():
            for target_ntype in canonical_output_order:
                if target_ntype not in self.name_map or not self.name_map[target_ntype] or eval_g_cpu.num_nodes(target_ntype) == 0:
                    logging.warning(f"Output node type '{target_ntype}' not found, has no nodes in graph data, or no nodes in eval_g_cpu. Skipping.")
                    all_predictions_accumulated[target_ntype] = torch.empty(1, 0, device=self.device) # Ensure key exists with correct shape
                    continue

                all_dgl_ids_for_ntype = list(self.name_map[target_ntype].values())
                num_targets_for_ntype = len(all_dgl_ids_for_ntype)
                logging.info(f"Processing target type '{target_ntype}' with {num_targets_for_ntype} total nodes.")
                
                batched_scores_for_ntype = []

                for i in range(0, num_targets_for_ntype, target_batch_size):
                    batch_target_dgl_ids = all_dgl_ids_for_ntype[i : i + target_batch_size]
                    if not batch_target_dgl_ids:
                        continue

                    current_batch_output_nodes_dict = {target_ntype: batch_target_dgl_ids}
                    logging.debug(f"  Batch {i // target_batch_size + 1}: Predicting for {len(batch_target_dgl_ids)} nodes of type '{target_ntype}'.")
                    
                    try:
                        batch_predictions = self.model.predict_links(
                            eval_g_on_device,
                            h_initial_on_device,
                            canonical_input_type,
                            [input_node_dgl_id], # Single input node
                            current_batch_output_nodes_dict
                        )
                        
                        if target_ntype in batch_predictions and batch_predictions[target_ntype] is not None and batch_predictions[target_ntype].numel() > 0:
                            # Ensure the batch_prediction has shape (1, num_targets_in_batch)
                            logging.debug(f"  Batch prediction for '{target_ntype}' received with shape: {batch_predictions[target_ntype].shape}")
                            if batch_predictions[target_ntype].shape[0] == 1 and batch_predictions[target_ntype].shape[1] == len(batch_target_dgl_ids):
                                batched_scores_for_ntype.append(batch_predictions[target_ntype])
                            else:
                                logging.warning(f"  Scores for target type '{target_ntype}' batch have unexpected shape: {batch_predictions[target_ntype].shape}. Expected (1, {len(batch_target_dgl_ids)}). Skipping batch.")
                    except Exception as e:
                        logging.error(f"Error during model.predict_links for target type '{target_ntype}', batch {i // target_batch_size + 1}: {e}")
                        import traceback
                        traceback.print_exc()

                if batched_scores_for_ntype:
                    all_predictions_accumulated[target_ntype] = torch.cat(batched_scores_for_ntype, dim=1) 
                else:
                    all_predictions_accumulated[target_ntype] = torch.empty(1,0, device=self.device) 
                logging.info(f"Finished processing batches for '{target_ntype}'. Accumulated scores shape: {all_predictions_accumulated.get(target_ntype, 'None')}")

        logging.info("Prediction finished.")
        
        if display_output:
            header_label = "Reranked Top" if self._rerank_is_enabled() else "Top"
            print(f"--- {header_label} {top_k} Predicted Associations for '{input_display_name}' ({resolved_input_display}) ---")
        
        all_results_by_type = {ntype: [] for ntype in canonical_output_order}
        canonical_top_results = {}

        output_dglid_to_name_map_full = {}
        output_nodes_dict_with_ids_full = {}
        for ntype in canonical_output_order: # Iterate over resolved canonical output types
            if ntype in self.name_map and self.name_map[ntype]:
                output_nodes_dict_with_ids_full[ntype] = list(self.name_map[ntype].values())
                output_dglid_to_name_map_full[ntype] = {v: k for k, v in self.name_map[ntype].items()}
            else: # Ensure keys exist even if type was skipped or empty
                output_nodes_dict_with_ids_full[ntype] = []
                output_dglid_to_name_map_full[ntype] = {}
            logging.debug(f"Prepared full reverse map for '{ntype}': {len(output_dglid_to_name_map_full.get(ntype, {}))} entries.")

        for target_ntype in canonical_output_order:
            scores_tensor = all_predictions_accumulated.get(target_ntype)
            if scores_tensor is None:
                logging.info(f"No prediction scores returned for target type '{target_ntype}'.")
                continue
            # Normalize scores to probabilities using sigmoid before ranking
            scores_tensor = torch.sigmoid(scores_tensor)
            
            if target_ntype not in output_nodes_dict_with_ids_full:
                logging.warning(f"Target node type '{target_ntype}' from predictions not in output_node_types. Skipping.")
                continue

            if scores_tensor is not None and scores_tensor.numel() > 0 and scores_tensor.shape[0] == 1:
                scores_for_single_input = scores_tensor[0].detach().cpu().numpy()
                logging.info(f"Processing results for '{target_ntype}'. Scores found with shape: {scores_for_single_input.shape}")
                
                current_target_dgl_ids_ordered = output_nodes_dict_with_ids_full.get(target_ntype, [])

                if len(scores_for_single_input) != len(current_target_dgl_ids_ordered):
                    logging.warning(f"Score count ({len(scores_for_single_input)}) mismatch with total target ID count ({len(current_target_dgl_ids_ordered)}) for type '{target_ntype}' after batching. Skipping results for this type.")
                    continue

                for i in range(len(current_target_dgl_ids_ordered)):
                    target_dgl_id = current_target_dgl_ids_ordered[i]
                    name = output_dglid_to_name_map_full.get(target_ntype, {}).get(target_dgl_id, f"UNKNOWN_{target_ntype}_{target_dgl_id}")
                    original_id = self.dglid_to_orig.get(target_ntype, {}).get(target_dgl_id, "UNKNOWN_ID")
                    if canonical_input_type == target_ntype and target_dgl_id == input_node_dgl_id: 
                        continue
                    result_entry = {
                        'id': original_id,
                        'name': name,
                        'type': target_ntype,
                        'score': scores_for_single_input[i],
                        'dgl_id': target_dgl_id
                    }
                    all_results_by_type[target_ntype].append(result_entry)
            elif target_ntype in output_nodes_dict_with_ids_full : 
                logging.info(f"No prediction scores available or scores tensor is empty/malformed for target type '{target_ntype}' after processing all batches.")

        canonical_path_target_ids = None
        if explainability_path_target_ids_by_type is not None:
            canonical_path_target_ids = {}
            for requested_type, target_ids in explainability_path_target_ids_by_type.items():
                canonical_target = self._resolve_node_type(requested_type, available_ntypes)
                if canonical_target is None and requested_type in canonical_output_order:
                    canonical_target = requested_type

                if canonical_target is None:
                    logging.warning(
                        "Explainability-path target type '%s' could not be resolved in graph data. Ignoring path restriction for this type.",
                        requested_type,
                    )
                    continue

                normalized_target_ids = {
                    self._normalize_original_id(target_id)
                    for target_id in (target_ids or [])
                    if self._normalize_original_id(target_id) is not None
                }
                canonical_path_target_ids.setdefault(canonical_target, set()).update(normalized_target_ids)

        canonical_required_target_ids = {}
        for requested_type, target_ids in (required_target_ids_by_type or {}).items():
            canonical_target = self._resolve_node_type(requested_type, available_ntypes)
            if canonical_target is None and requested_type in canonical_output_order:
                canonical_target = requested_type

            if canonical_target is None:
                logging.warning(
                    "Required-rank target type '%s' could not be resolved in graph data. Ignoring rank lookup for this type.",
                    requested_type,
                )
                continue

            normalized_target_ids = {
                self._normalize_original_id(target_id)
                for target_id in (target_ids or [])
                if self._normalize_original_id(target_id) is not None
            }
            if normalized_target_ids:
                canonical_required_target_ids.setdefault(canonical_target, set()).update(normalized_target_ids)

        required_target_ranks = {}

        for ntype in canonical_output_order:
            alias_list = canonical_to_requested.get(ntype, [])
            alias_hint = ""
            if alias_list and any(alias != ntype for alias in alias_list):
                alias_hint = f" (requested as: {', '.join(alias_list)})"

            results_list = all_results_by_type.get(ntype, [])
            if results_list:
                required_target_ids = canonical_required_target_ids.get(ntype, set())
                if self._rerank_is_enabled():
                    results_list = self._apply_soft_reranking(
                        eval_g_cpu,
                        canonical_input_type,
                        input_node_dgl_id,
                        results_list,
                    )
                else:
                    results_list.sort(key=lambda x: x['score'], reverse=True)
                for rank_index, res in enumerate(results_list, start=1):
                    res['rank'] = rank_index

                if required_target_ids:
                    rank_lookup = {
                        res['id']: res['rank']
                        for res in results_list
                        if res.get('id') in required_target_ids
                    }
                    required_target_ranks[ntype] = rank_lookup

                    missing_rank_ids = sorted(required_target_ids - set(rank_lookup.keys()))
                    if missing_rank_ids:
                        logging.warning(
                            "%d required target ID(s) for type '%s' were not found in ranked predictions. Example(s): %s",
                            len(missing_rank_ids),
                            ntype,
                            ', '.join(missing_rank_ids[:10]),
                        )

                top_results = results_list[:top_k]
                canonical_top_results[ntype] = top_results
                if display_output:
                    print(f"--- Top {top_k} for type: {ntype}{alias_hint} ---")
                # If explainability enabled, enrich results with dominant_branch info before removing dgl_id
                explain_payload = None
                fusion_weights = {}
                if (
                    getattr(self.model, 'enable_explainability', False)
                    and self.model.enable_explainability
                    and getattr(self.model, 'fusion_mode', None) == 'weighted'
                ):
                    try:
                        explain_payload = self.model.get_last_explainability()
                    except Exception:
                        explain_payload = None

                    # Log a compact summary of dominant_branch info to aid debugging
                    try:
                        if explain_payload:
                            per_type_map = explain_payload.get('per_type', {})
                            edge_map = explain_payload.get('edge_alignment', {})
                            per_type_summary = {k: v.get('dominant_branch') for k, v in per_type_map.items()}
                            edge_summary = {str(k): [e.get('dominant_branch') for e in (v or [])[:5]] for k, v in edge_map.items()}
                            # logging.info("Explainability dominant summary: %s", json.dumps({'per_type': per_type_summary, 'edge_alignment': edge_summary}))
                    except Exception:
                        pass
                if explain_payload:
                    fusion_weights = explain_payload.get('fusion_weights', {}) or {}

                if (
                    getattr(self.model, 'enable_explainability', False)
                    and self.model.enable_explainability
                    and hasattr(self.model, 'get_explainability_path')
                ):
                    allowed_path_target_ids = None
                    if canonical_path_target_ids is not None:
                        allowed_path_target_ids = canonical_path_target_ids.get(ntype, set())

                    for res in top_results:
                        if 'dgl_id' not in res:
                            continue
                        if allowed_path_target_ids is not None and res.get('id') not in allowed_path_target_ids:
                            continue
                        if hasattr(self.model, 'get_explainability_path_details'):
                            path_details = self.model.get_explainability_path_details(
                                eval_g_cpu,
                                canonical_input_type,
                                input_node_dgl_id,
                                res['type'],
                                res['dgl_id'],
                                max_hops=getattr(self.model, 'explainability_max_hops', None),
                                h_initial=self.initial_node_features_cpu
                            )
                            if path_details:
                                res['physical_shortest_path'] = path_details.get('physical_shortest_path')
                                res['final_displayed_path'] = path_details.get('final_displayed_path')
                                res['final_path_kind'] = path_details.get('final_path_kind')
                                if path_details.get('final_displayed_path'):
                                    res['path'] = path_details.get('final_displayed_path')
                        else:
                            path_str = self.model.get_explainability_path(
                                eval_g_cpu,
                                canonical_input_type,
                                input_node_dgl_id,
                                res['type'],
                                res['dgl_id'],
                                max_hops=getattr(self.model, 'explainability_max_hops', None),
                                h_initial=self.initial_node_features_cpu
                            )
                            if path_str:
                                res['path'] = path_str

                # Attach dominant_branch info from explain_payload where available
                if explain_payload:
                    per_type_map = explain_payload.get('per_type', {})
                    edge_align_map = explain_payload.get('edge_alignment', {})
                    for res in top_results:
                        # per-type dominant branch
                        per_dom = None
                        try:
                            per_dom = per_type_map.get(res['type'], {}).get('dominant_branch')
                        except Exception:
                            per_dom = None
                        res['dominant_branch_type'] = per_dom

                        # edge-level dominant branch: search matching etype entries
                        edge_dom = None
                        try:
                            for etype_key, entries in edge_align_map.items():
                                try:
                                    src_type_k, rel_k, dst_type_k = etype_key
                                except Exception:
                                    continue
                                # look for etypes that connect input type and this result type
                                if not ((src_type_k == canonical_input_type and dst_type_k == res['type']) or (dst_type_k == canonical_input_type and src_type_k == res['type'])):
                                    continue
                                for entry in entries:
                                    # match by dgl ids (consider direction)
                                    try:
                                        if (entry.get('src_idx') == int(input_node_dgl_id) and entry.get('dst_idx') == int(res.get('dgl_id'))):
                                            edge_dom = entry.get('dominant_branch')
                                            break
                                        if (entry.get('dst_idx') == int(input_node_dgl_id) and entry.get('src_idx') == int(res.get('dgl_id'))):
                                            edge_dom = entry.get('dominant_branch')
                                            break
                                    except Exception:
                                        continue
                                if edge_dom:
                                    break
                        except Exception:
                            edge_dom = None
                        res['dominant_branch_edge'] = edge_dom

                # Now remove dgl_id (no longer needed) and print results with dominant markers
                for i, res in enumerate(top_results):
                    dom_type = res.get('dominant_branch_type')
                    dom_edge = res.get('dominant_branch_edge')
                    dom_parts = []
                    ratio_suffix = ""
                    try:
                        w = fusion_weights.get(res.get('type'))
                        if w is not None and not math.isnan(float(w)):
                            hgcn_ratio = 1.0 - float(w)
                            tf_ratio = float(w)
                            ratio_suffix = f"(hgcn={hgcn_ratio:.2f}, transformer={tf_ratio:.2f})"
                    except Exception:
                        ratio_suffix = ""

                    if dom_type:
                        dom_parts.append(f"type:{dom_type}{ratio_suffix}")
                    if dom_edge:
                        dom_parts.append(f"edge:{dom_edge}{ratio_suffix}")
                    dom_hint = f" [dominant: {', '.join(dom_parts)}]" if dom_parts else ""
                    score_full_precision = np.format_float_positional(
                        np.float64(res['score']),
                        unique=False,
                        precision=17,
                        trim='-'
                    )
                    if display_output:
                        rerank_hint = ""
                        if self._rerank_is_enabled():
                            bonus_parts = []
                            if res.get('structure_bonus'):
                                hop_part = f"structure={res['structure_bonus']:.4f}"
                                if res.get('hop_distance') is not None:
                                    hop_part += f"@{res['hop_distance']}"
                                if res.get('structure_signal_raw'):
                                    hop_part += f" raw={res['structure_signal_raw']:.4f}"
                                bonus_parts.append(hop_part)
                            if bonus_parts:
                                rerank_hint = (
                                    f" [base={res.get('base_score', 0.0):.6f}, bonus={res.get('rerank_bonus_total', 0.0):.6f}; "
                                    + ", ".join(bonus_parts)
                                    + "]"
                                )
                        print(f"{i+1}. ID: {res['id']}, Name: {res['name']} ({res['type']}) - Score: {score_full_precision}{rerank_hint}{dom_hint}")
                        for explain_line in self._get_explainability_display_lines(res):
                            print(explain_line)
            else:
                canonical_top_results[ntype] = []
                if display_output:
                    print(f"--- Top {top_k} for type: {ntype}{alias_hint} ---")
                    print("No associations found or predicted.")

        final_top_results = {}
        for ntype, top_results in canonical_top_results.items():
            for alias in canonical_to_requested.get(ntype, []):
                final_top_results[alias] = top_results

        for skipped in skipped_output_types:
            final_top_results.setdefault(skipped, [])

        self._last_required_target_ranks = required_target_ranks
        return final_top_results

    def predict_specific_targets(self, eval_g_cpu, input_node_type, output_node_type, target_node_ids,
                                 input_node_name=None, input_node_id=None, display_output=True,
                                 target_rank_lookup=None):
        """Predict scores for user-specified target original IDs, regardless of their top-k rank."""
        self.model.eval()

        if self.initial_node_features_cpu is None:
            logging.error("`initial_node_features_cpu` is required for `predict_specific_targets` but was not provided during ModelEvaluator initialization.")
            return []

        normalized_target_ids = []
        for raw_target_id in target_node_ids or []:
            normalized_target_id = self._normalize_original_id(raw_target_id)
            if normalized_target_id and normalized_target_id not in normalized_target_ids:
                normalized_target_ids.append(normalized_target_id)

        if not normalized_target_ids:
            logging.warning("No valid target node IDs were supplied for forced prediction.")
            return []

        available_ntypes = set(eval_g_cpu.ntypes)
        canonical_input_type, input_node_dgl_id, resolved_input_name, resolved_input_id = self._resolve_input_node(
            eval_g_cpu,
            input_node_type,
            input_node_name=input_node_name,
            input_node_id=input_node_id,
        )
        if not canonical_input_type:
            return []

        canonical_target_type = self._resolve_node_type(output_node_type, available_ntypes)
        if not canonical_target_type:
            logging.error(f"Output node type '{output_node_type}' not found in graph data.")
            return []

        eval_g_on_device, h_initial_on_device = self._prepare_eval_graph_inputs(eval_g_cpu)
        target_orig_to_dgl = self._build_orig_id_lookup(canonical_target_type)
        target_dgl_to_name = self._build_dgl_to_name_lookup(canonical_target_type)

        missing_target_ids = set()
        valid_target_ids = []
        valid_target_dgl_ids = []
        for target_id in normalized_target_ids:
            target_dgl_id = target_orig_to_dgl.get(target_id)
            if target_dgl_id is None:
                missing_target_ids.add(target_id)
                continue
            valid_target_ids.append(target_id)
            valid_target_dgl_ids.append(int(target_dgl_id))

        resolved_input_display = (
            canonical_input_type if canonical_input_type == input_node_type
            else f"{input_node_type} -> {canonical_input_type}"
        )
        input_display_name = resolved_input_name or resolved_input_id or input_node_name or "UNKNOWN_INPUT"

        results_by_target_id = {}
        if valid_target_dgl_ids:
            with torch.no_grad():
                prediction_dict = self.model.predict_links(
                    eval_g_on_device,
                    h_initial_on_device,
                    canonical_input_type,
                    [input_node_dgl_id],
                    {canonical_target_type: valid_target_dgl_ids}
                )

            scores_tensor = prediction_dict.get(canonical_target_type)
            if scores_tensor is not None and scores_tensor.numel() > 0 and scores_tensor.shape[0] == 1:
                scores = torch.sigmoid(scores_tensor[0]).detach().cpu().numpy()
                for target_id, target_dgl_id, score in zip(valid_target_ids, valid_target_dgl_ids, scores):
                    rerank_meta = self._compute_rerank_bonus(
                        eval_g_cpu,
                        canonical_input_type,
                        input_node_dgl_id,
                        canonical_target_type,
                        target_dgl_id,
                        float(score),
                    )
                    entry = {
                        'id': target_id,
                        'name': target_dgl_to_name.get(target_dgl_id, f"UNKNOWN_{canonical_target_type}_{target_dgl_id}"),
                        'type': canonical_target_type,
                        'score': rerank_meta['final_score'],
                        'dgl_id': target_dgl_id,
                        'status': 'ok',
                        'rank': (target_rank_lookup or {}).get(target_id),
                    }
                    entry.update(rerank_meta)
                    if (
                        getattr(self.model, 'enable_explainability', False)
                        and self.model.enable_explainability
                        and hasattr(self.model, 'get_explainability_path')
                    ):
                        if hasattr(self.model, 'get_explainability_path_details'):
                            path_details = self.model.get_explainability_path_details(
                                eval_g_cpu,
                                canonical_input_type,
                                input_node_dgl_id,
                                canonical_target_type,
                                target_dgl_id,
                                max_hops=getattr(self.model, 'explainability_max_hops', None),
                                h_initial=self.initial_node_features_cpu
                            )
                            if path_details:
                                entry['physical_shortest_path'] = path_details.get('physical_shortest_path')
                                entry['final_displayed_path'] = path_details.get('final_displayed_path')
                                entry['final_path_kind'] = path_details.get('final_path_kind')
                                if path_details.get('final_displayed_path'):
                                    entry['path'] = path_details.get('final_displayed_path')
                        else:
                            path_str = self.model.get_explainability_path(
                                eval_g_cpu,
                                canonical_input_type,
                                input_node_dgl_id,
                                canonical_target_type,
                                target_dgl_id,
                                max_hops=getattr(self.model, 'explainability_max_hops', None),
                                h_initial=self.initial_node_features_cpu
                            )
                            if path_str:
                                entry['path'] = path_str
                    results_by_target_id[target_id] = entry
            else:
                logging.warning(
                    "Forced prediction returned empty or malformed scores for input '%s' and target type '%s'.",
                    input_display_name,
                    canonical_target_type,
                )
                for target_id, target_dgl_id in zip(valid_target_ids, valid_target_dgl_ids):
                    results_by_target_id[target_id] = {
                        'id': target_id,
                        'name': target_dgl_to_name.get(target_dgl_id, f"UNKNOWN_{canonical_target_type}_{target_dgl_id}"),
                        'type': canonical_target_type,
                        'score': None,
                        'dgl_id': target_dgl_id,
                        'status': 'score_unavailable',
                        'rank': (target_rank_lookup or {}).get(target_id),
                    }

        ordered_results = []
        for target_id in normalized_target_ids:
            if target_id in results_by_target_id:
                ordered_results.append(results_by_target_id[target_id])
                continue

            ordered_results.append({
                'id': target_id,
                'name': None,
                'type': canonical_target_type,
                'score': None,
                'status': 'target_not_found_in_graph' if target_id in missing_target_ids else 'score_unavailable',
                'rank': (target_rank_lookup or {}).get(target_id),
            })

        if display_output:
            print(f"--- Forced Associations for '{input_display_name}' ({resolved_input_display}) -> type: {canonical_target_type} ---")
            for idx, res in enumerate(ordered_results, start=1):
                display_rank = res.get('rank') if res.get('rank') is not None else idx
                if res.get('score') is None:
                    score_text = 'N/A'
                else:
                    score_text = np.format_float_positional(
                        np.float64(res['score']),
                        unique=False,
                        precision=17,
                        trim='-'
                    )
                name_text = res.get('name') or 'N/A'
                status_text = f" Status: {res.get('status')}" if res.get('status') and res.get('status') != 'ok' else ''
                rank_text = f" Rank: {res.get('rank')}" if res.get('rank') is not None else ''
                rerank_text = ''
                if self._rerank_is_enabled() and res.get('score') is not None:
                    rerank_parts = []
                    if res.get('structure_bonus'):
                        hop_part = f"structure={res['structure_bonus']:.4f}"
                        if res.get('hop_distance') is not None:
                            hop_part += f"@{res['hop_distance']}"
                        if res.get('structure_signal_raw'):
                            hop_part += f" raw={res['structure_signal_raw']:.4f}"
                        rerank_parts.append(hop_part)
                    if rerank_parts:
                        rerank_text = (
                            f" Base: {res.get('base_score', 0.0):.6f} Bonus: {res.get('rerank_bonus_total', 0.0):.6f}"
                            f" ({', '.join(rerank_parts)})"
                        )
                print(
                    f"{display_rank}. ID: {res['id']}, Name: {name_text} ({res['type']}) - Score: {score_text}{rank_text}{status_text}{rerank_text}"
                )
                for explain_line in self._get_explainability_display_lines(res):
                    print(explain_line)

        for res in ordered_results:
            res.pop('dgl_id', None)

        return ordered_results
