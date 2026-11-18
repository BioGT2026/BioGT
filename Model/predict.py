import os
import torch
import dgl
import numpy as np
import argparse
import logging
import traceback
import time
import pandas as pd
import re
from collections import defaultdict, deque

"""
predict.py
------------
预测脚本主入口。功能概览（中文注释）：
- 从给定的节点/边 CSV 构建或加载图（通过 DGLGraphCreator）
- 对用户提供的输入节点名做模糊搜索（支持精确匹配和包含匹配）
- 初始化模型结构（从 graph_creator 获取 node_info 以对齐 embedding）
- 支持从训练时检查点加载权重：
    - 支持纯 state_dict 或包含 metadata 的检查点
    - 尝试使用训练时的 node_info 进行按原始 ID 的 embedding 行重映射
    - 若无法按 ID 映射，回退为拷贝重叠前缀行
- 运行 ModelEvaluator 来生成 top-k 预测（可选可解释性输出）

注意：此文件只添加注释以便阅读，实际逻辑未修改。
"""

# Import from local modules
from Model.modelData import DGLGraphCreator, _build_name_lookup_from_dgl_mapping
from Model.model import DiseaseAssociationModel
from Model.modelEvaluator import ModelEvaluator
from Model.common import setup_logging, log_memory_usage

NODE_NAME_FIELD_CANDIDATES = (
    'name',
    'disease_name',
    'disease',
    'mondo_name',
    'display_name',
    'preferred_name',
    'label',
    'title',
)

MARKER_PAIR_TARGET_TYPE_FIELD_CANDIDATES = (
    'marker_type',
    'target_type',
    'output_type',
)

def _deduplicate_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def _normalize_column_name(column_name):
    return str(column_name).strip().lower().replace(' ', '_')

def _normalize_original_node_id(raw_value):
    if raw_value is None:
        return None

    normalized = str(raw_value).strip()
    return normalized or None

def _build_orig_to_dgl_lookup(node_info, canonical_type):
    lookup = {}
    if not node_info or not canonical_type:
        return lookup

    for dgl_id, original_id in node_info.get('dglid_to_orig', {}).get(canonical_type, {}).items():
        normalized_original_id = _normalize_original_node_id(original_id)
        if normalized_original_id is None:
            continue
        lookup.setdefault(normalized_original_id, int(dgl_id))

    return lookup

def _load_marker_pair_requests(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"The marker pair CSV file was not found at '{csv_path}'.")

    pair_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[''])
    if pair_df.empty:
        raise ValueError(f"The marker pair CSV '{csv_path}' is empty.")

    pair_df.columns = [_normalize_column_name(col) for col in pair_df.columns]
    required_columns = {'disease_id', 'marker_id'}
    missing_columns = required_columns - set(pair_df.columns)
    if missing_columns:
        raise KeyError(
            f"The marker pair CSV '{csv_path}' is missing required columns: {sorted(missing_columns)}. "
            f"Expected at least: {sorted(required_columns)}"
        )

    target_type_column = next(
        (candidate for candidate in MARKER_PAIR_TARGET_TYPE_FIELD_CANDIDATES if candidate in pair_df.columns),
        None,
    )

    pairs_by_disease_id = {}
    observed_target_types = []
    pair_count = 0
    non_mondo_disease_ids = []

    for row in pair_df.to_dict('records'):
        disease_id = _normalize_original_node_id(row.get('disease_id'))
        marker_id = _normalize_original_node_id(row.get('marker_id'))
        marker_type = str(row.get(target_type_column, '')).strip().lower() if target_type_column else ''

        if not disease_id or not marker_id:
            continue

        if not disease_id.upper().startswith('MONDO:') and disease_id not in non_mondo_disease_ids:
            non_mondo_disease_ids.append(disease_id)

        disease_bucket = pairs_by_disease_id.setdefault(
            disease_id,
            {
                'target_ids': [],
                'target_type': marker_type or None,
            }
        )

        if marker_type:
            if disease_bucket.get('target_type') and disease_bucket['target_type'] != marker_type:
                raise ValueError(
                    f"Disease '{disease_id}' in '{csv_path}' is associated with multiple marker types "
                    f"('{disease_bucket['target_type']}' and '{marker_type}'). Please split the CSV by target type."
                )
            disease_bucket['target_type'] = marker_type
            if marker_type not in observed_target_types:
                observed_target_types.append(marker_type)

        if marker_id not in disease_bucket['target_ids']:
            disease_bucket['target_ids'].append(marker_id)
            pair_count += 1

    if not pairs_by_disease_id:
        raise ValueError(f"No valid disease-marker pairs were found in '{csv_path}'.")

    if non_mondo_disease_ids:
        preview = ', '.join(non_mondo_disease_ids[:10])
        logging.warning(
            "Pair CSV '%s' contains %d disease ID(s) that do not start with MONDO:. "
            "Prediction will still resolve them by original node ID, not by disease name. Example(s): %s",
            csv_path,
            len(non_mondo_disease_ids),
            preview,
        )

    return {
        'pairs_by_disease_id': pairs_by_disease_id,
        'target_types': observed_target_types,
        'pair_count': pair_count,
        'disease_ids': list(pairs_by_disease_id.keys()),
    }

def _load_input_names(args, eval_input_type):
    if getattr(args, 'eval_input_mondo_csv', None):
        csv_path = args.eval_input_mondo_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The mondo disease CSV file was not found at '{csv_path}'.")

        csv_df = pd.read_csv(csv_path)
        if csv_df.empty:
            raise ValueError(f"The mondo disease CSV '{csv_path}' is empty.")

        name_column_candidates = [
            'Name', 'name', 'Disease', 'disease', 'disease_name',
            'DiseaseName', 'mondo_name', 'MONDO_NAME'
        ]
        name_col = next((col for col in name_column_candidates if col in csv_df.columns), None)
        if not name_col:
            raise KeyError(
                f"Could not find a disease-name column in '{csv_path}'. "
                f"Expected one of: {name_column_candidates}"
            )

        # Special handling for the common MONDO export schema:
        # ID,Type,Name,Sequence,Comments
        # Prefer Type='Disease' rows and (if present) IDs prefixed with 'MONDO:'.
        source_df = csv_df
        if 'Type' in csv_df.columns:
            type_series = csv_df['Type'].astype(str).str.strip().str.lower()
            preferred_types = {'disease', eval_input_type.lower()}
            filtered_df = csv_df[type_series.isin(preferred_types)]
            if filtered_df.empty:
                logging.warning(
                    "No rows in mondo CSV matched Type in %s. Falling back to all rows from column '%s'.",
                    sorted(list(preferred_types)),
                    name_col
                )
            else:
                source_df = filtered_df

        if 'ID' in source_df.columns:
            id_series = source_df['ID'].astype(str).str.strip().str.upper()
            mondo_id_df = source_df[id_series.str.startswith('MONDO:')]
            if not mondo_id_df.empty:
                source_df = mondo_id_df

        source_series = source_df[name_col]

        names = [str(v).strip() for v in source_series.dropna().tolist() if str(v).strip()]
        names = _deduplicate_preserve_order(names)
        if not names:
            raise ValueError(
                f"No valid disease names were found in '{csv_path}' using column '{name_col}'."
            )

        return names, f"mondo CSV: {csv_path}", name_col

    if not getattr(args, 'eval_input_names', None):
        raise ValueError("Please provide --eval_input_names, --eval_input_mondo_csv, or --eval_input_ids.")

    names = [name.strip() for name in args.eval_input_names.split(',') if name.strip()]
    names = _deduplicate_preserve_order(names)
    if not names:
        raise ValueError("--eval_input_names did not contain any valid names.")

    return names, "--eval_input_names", None

def _load_input_ids(args):
    raw_input_ids = getattr(args, 'eval_input_ids', None)
    if not raw_input_ids:
        return [], None

    tokens = [token.strip() for token in re.split(r'[,\s]+', raw_input_ids) if token.strip()]
    normalized_ids = _deduplicate_preserve_order([
        normalized
        for normalized in (_normalize_original_node_id(token) for token in tokens)
        if normalized
    ])
    if not normalized_ids:
        raise ValueError("--eval_input_ids did not contain any valid original node IDs.")

    return normalized_ids, "--eval_input_ids"

def _parse_original_node_id_list(raw_value, arg_name):
    if not raw_value:
        return []

    tokens = [token.strip() for token in re.split(r'[,\s]+', str(raw_value)) if token.strip()]
    normalized_ids = _deduplicate_preserve_order([
        normalized
        for normalized in (_normalize_original_node_id(token) for token in tokens)
        if normalized
    ])
    if not normalized_ids:
        raise ValueError(f"{arg_name} did not contain any valid original node IDs.")
    return normalized_ids

def _resolve_input_type(node_info, eval_input_type, available_ntypes):
    type_alias_map = node_info.get('type_alias_map', {}) if node_info else {}
    candidates = [
        eval_input_type,
        eval_input_type.lower(),
        type_alias_map.get(eval_input_type),
        type_alias_map.get(eval_input_type.lower())
    ]
    for candidate in candidates:
        if candidate and candidate in available_ntypes:
            return candidate
    return None

def _is_isolated_in_train(g_train_cpu, ntype, dgl_id):
    total_in = 0
    total_out = 0
    for srctype, etype, dsttype in g_train_cpu.canonical_etypes:
        if dsttype == ntype:
            deg = g_train_cpu.in_degrees(dgl_id, etype=(srctype, etype, dsttype))
            total_in += int(deg.item()) if hasattr(deg, 'item') else int(deg)
        if srctype == ntype:
            deg = g_train_cpu.out_degrees(dgl_id, etype=(srctype, etype, dsttype))
            total_out += int(deg.item()) if hasattr(deg, 'item') else int(deg)
    return (total_in + total_out) == 0

def _resolve_any_node_type(node_info, requested_candidates, available_ntypes):
    type_alias_map = node_info.get('type_alias_map', {}) if node_info else {}
    for raw in requested_candidates:
        if not raw:
            continue
        candidates = [raw, raw.lower(), type_alias_map.get(raw), type_alias_map.get(raw.lower())]
        for candidate in candidates:
            if candidate and candidate in available_ntypes:
                return candidate
    return None

def _rebuild_name_lookup_for_type(node_info, canonical_type):
    """Best-effort fallback when node_info['name_map'][type] is missing."""
    if not node_info:
        return {}

    dgl_to_orig = node_info.get('dglid_to_orig', {}).get(canonical_type, {})
    id_to_node_data = node_info.get('id_to_node_data', {})
    return _build_name_lookup_from_dgl_mapping(dgl_to_orig, id_to_node_data)

def _load_id_to_node_data_from_csv(node_csv_path):
    """Load node metadata directly from nodes_csv for cache-recovery fallbacks."""
    if not node_csv_path or not os.path.exists(node_csv_path):
        return {}

    try:
        nodes_df = pd.read_csv(node_csv_path, dtype=str, keep_default_na=False, na_values=[''])
    except Exception as exc:
        logging.warning("Failed to load nodes_csv '%s' for fallback name-map rebuild: %s", node_csv_path, exc)
        return {}

    normalized_columns = [str(col).strip().lower().replace(' ', '_') for col in nodes_df.columns]
    nodes_df.columns = normalized_columns
    if 'id' not in nodes_df.columns:
        logging.warning("nodes_csv '%s' is missing required 'id' column for fallback name-map rebuild.", node_csv_path)
        return {}

    try:
        return nodes_df.set_index('id').to_dict('index')
    except Exception as exc:
        logging.warning("Failed to build id_to_node_data from nodes_csv '%s': %s", node_csv_path, exc)
        return {}

def _extract_node_name_from_row(row_dict):
    if not isinstance(row_dict, dict):
        return None

    normalized = {str(key).strip().lower(): value for key, value in row_dict.items()}
    for field_name in NODE_NAME_FIELD_CANDIDATES:
        raw_value = normalized.get(field_name)
        if raw_value is None:
            continue
        cleaned_value = str(raw_value).strip()
        if cleaned_value and cleaned_value.lower() != 'nan':
            return cleaned_value
    return None

def _rebuild_name_lookup_from_graph_and_csv(g_cpu, canonical_type, node_csv_path):
    """Rebuild name lookup without cache metadata, using graph node ordering and nodes_csv rows."""
    if g_cpu is None or canonical_type not in getattr(g_cpu, 'ntypes', []):
        return {}
    if not node_csv_path or not os.path.exists(node_csv_path):
        return {}

    try:
        nodes_df = pd.read_csv(node_csv_path, dtype=str, keep_default_na=False, na_values=[''])
    except Exception as exc:
        logging.warning("Failed to load nodes_csv '%s' for graph-based name-map rebuild: %s", node_csv_path, exc)
        return {}

    nodes_df.columns = [str(col).strip().lower().replace(' ', '_') for col in nodes_df.columns]
    if 'type' not in nodes_df.columns:
        logging.warning("nodes_csv '%s' is missing required 'type' column for graph-based name-map rebuild.", node_csv_path)
        return {}

    typed_rows = nodes_df[nodes_df['type'].astype(str) == str(canonical_type)].reset_index(drop=True)
    if typed_rows.empty:
        return {}

    if dgl.NID in g_cpu.nodes[canonical_type].data:
        graph_indices = g_cpu.nodes[canonical_type].data[dgl.NID].detach().cpu().tolist()
    else:
        graph_indices = list(range(g_cpu.num_nodes(canonical_type)))

    name_lookup = {}
    for new_dgl_id, csv_row_idx in enumerate(graph_indices):
        try:
            row_idx = int(csv_row_idx)
        except (TypeError, ValueError):
            continue
        if row_idx < 0 or row_idx >= len(typed_rows):
            continue
        node_name = _extract_node_name_from_row(typed_rows.iloc[row_idx].to_dict())
        if node_name:
            name_lookup[node_name] = int(new_dgl_id)

    return name_lookup

def _ensure_name_map_for_type(node_info, g_cpu, canonical_type, node_csv_path):
    """Ensure node_info['name_map'][canonical_type] exists, using progressively weaker fallbacks."""
    if not node_info or not canonical_type:
        return {}

    existing_name_map = node_info.get('name_map', {}).get(canonical_type, {})
    if existing_name_map:
        return existing_name_map

    fallback_name_map = _rebuild_name_lookup_for_type(node_info, canonical_type)
    if not fallback_name_map:
        csv_id_to_node_data = node_info.get('id_to_node_data') or _load_id_to_node_data_from_csv(node_csv_path)
        if csv_id_to_node_data:
            node_info['id_to_node_data'] = csv_id_to_node_data
            fallback_name_map = _build_name_lookup_from_dgl_mapping(
                node_info.get('dglid_to_orig', {}).get(canonical_type, {}),
                csv_id_to_node_data,
            )
            if fallback_name_map:
                logging.warning(
                    "Name map for type '%s' rebuilt from nodes_csv '%s' using cached dglid_to_orig.",
                    canonical_type,
                    node_csv_path,
                )

    if not fallback_name_map:
        fallback_name_map = _rebuild_name_lookup_from_graph_and_csv(
            g_cpu,
            canonical_type,
            node_csv_path,
        )
        if fallback_name_map:
            logging.warning(
                "Name map for type '%s' rebuilt from graph node ordering + nodes_csv '%s'.",
                canonical_type,
                node_csv_path,
            )

    if fallback_name_map:
        node_info.setdefault('name_map', {})[canonical_type] = fallback_name_map
        logging.warning(
            "Name map for type '%s' is missing/empty in node_info; rebuilt %d entries via fallback recovery.",
            canonical_type,
            len(fallback_name_map),
        )
        return fallback_name_map

    return {}

def _collect_reachable_disease_ids_from_drug(g_cpu, disease_type, drug_type, max_hops):
    hop_limit = max(1, int(max_hops))
    adjacency = defaultdict(list)
    for src_type, rel_type, dst_type in g_cpu.canonical_etypes:
        src_nodes, dst_nodes = g_cpu.edges(etype=(src_type, rel_type, dst_type))
        if src_nodes.numel() == 0:
            continue
        src_list = src_nodes.detach().cpu().tolist()
        dst_list = dst_nodes.detach().cpu().tolist()
        for s, d in zip(src_list, dst_list):
            src_key = (src_type, int(s))
            dst_key = (dst_type, int(d))
            adjacency[src_key].append(dst_key)
            adjacency[dst_key].append(src_key)

    if drug_type not in g_cpu.ntypes:
        return set()

    queue = deque()
    visited = set()
    reachable_disease_ids = set()

    for drug_id in range(g_cpu.num_nodes(drug_type)):
        key = (drug_type, int(drug_id))
        queue.append((key, 0))
        visited.add(key)

    while queue:
        node_key, depth = queue.popleft()
        ntype, nid = node_key
        if ntype == disease_type and depth > 0:
            reachable_disease_ids.add(int(nid))
        if depth >= hop_limit:
            continue
        for neighbor in adjacency.get(node_key, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

    return reachable_disease_ids

def _build_undirected_adjacency(g_cpu):
    adjacency = defaultdict(list)
    for src_type, rel_type, dst_type in g_cpu.canonical_etypes:
        src_nodes, dst_nodes = g_cpu.edges(etype=(src_type, rel_type, dst_type))
        if src_nodes.numel() == 0:
            continue
        src_list = src_nodes.detach().cpu().tolist()
        dst_list = dst_nodes.detach().cpu().tolist()
        for s, d in zip(src_list, dst_list):
            src_key = (src_type, int(s))
            dst_key = (dst_type, int(d))
            adjacency[src_key].append(dst_key)
            adjacency[dst_key].append(src_key)
    return adjacency

def _collect_nodes_within_hops(adjacency, start_nodes, max_hops):
    hop_limit = max(1, int(max_hops))
    queue = deque()
    visited = set()

    for node in start_nodes:
        queue.append((node, 0))
        visited.add(node)

    while queue:
        node_key, depth = queue.popleft()
        if depth >= hop_limit:
            continue
        for neighbor in adjacency.get(node_key, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

    return visited

def _collect_diseases_with_drug_to_target_paths(g_cpu, disease_type, drug_type, target_types, max_hops):
    if disease_type not in g_cpu.ntypes or drug_type not in g_cpu.ntypes:
        return set()

    canonical_targets = [t for t in target_types if t in g_cpu.ntypes]
    if not canonical_targets:
        return set()

    adjacency = _build_undirected_adjacency(g_cpu)

    target_starts = []
    for target_type in canonical_targets:
        for node_id in range(g_cpu.num_nodes(target_type)):
            target_starts.append((target_type, int(node_id)))

    if not target_starts:
        return set()

    nodes_reaching_target = _collect_nodes_within_hops(adjacency, target_starts, max_hops)
    valid_drug_ids = {
        nid for (ntype, nid) in nodes_reaching_target
        if ntype == drug_type
    }
    if not valid_drug_ids:
        return set()

    drug_starts = [(drug_type, int(did)) for did in sorted(valid_drug_ids)]
    nodes_reaching_valid_drug = _collect_nodes_within_hops(adjacency, drug_starts, max_hops)

    return {
        nid for (ntype, nid) in nodes_reaching_valid_drug
        if ntype == disease_type
    }

def _parse_path_types_and_hops(path_str):
    if not path_str or not isinstance(path_str, str):
        return [], 0
    segments = [seg.strip() for seg in path_str.split('->') if seg.strip()]
    node_types = []
    for seg in segments:
        # Segment format: "type:id (...)"; only parse the leading type token.
        type_token = seg.split(':', 1)[0].strip().lower()
        if type_token:
            node_types.append(type_token)
    hops = max(0, len(segments) - 1)
    return node_types, hops

def _extract_path_nodes_by_types(path_str, target_types):
    if not path_str or not isinstance(path_str, str):
        return []
    segments = [seg.strip() for seg in path_str.split('->') if seg.strip()]
    matched_nodes = []
    for seg in segments:
        seg_type = seg.split(':', 1)[0].strip().lower()
        if seg_type in target_types:
            matched_nodes.append(seg)
    return matched_nodes

def _build_inference_rerank_context(args, node_info, available_ntypes):
    lambda_hop = float(getattr(args, 'rerank_hop_bonus_lambda', 0.0) or 0.0)
    max_hops = int(getattr(args, 'rerank_max_hops', 0) or 0)

    if lambda_hop <= 0:
        return None

    rerank_context = {
        'lambda_structure': lambda_hop,
        'max_hops': max_hops,
    }
    logging.info(
        "Soft rerank enabled with structure lambda=%.4f max_hops=%s",
        lambda_hop,
        'unbounded' if max_hops <= 0 else max_hops,
    )
    return rerank_context

def predict(args, device, eval_input_type):
    """
    Loads a trained model and performs predictions for a given input node.
    """
    logging.info(f"Entered predict function. Using device: {device}")
    # 说明：
    # - 这个函数负责整个预测流水线：从 CSV 加载图、初始化模型、恢复检查点、
    #   构造全量节点特征并调用 ModelEvaluator 进行 top-k 预测。
    # - 关键数据结构：graph_creator 提供的 node_info（用于 id 映射/名字查找）

    # --- Pre-flight Fuzzy Search for Input Nodes ---
    # 目的：把用户传入的 eval_input_names（逗号分隔）映射为图中真实存在的节点名。
    # 实现策略：先尝试大小写不敏感的精确匹配；若无精确匹配则用包含匹配（fuzzy contains）。
    marker_pair_request_bundle = None
    forced_targets_by_input_id = {}
    prediction_inputs = []
    final_nodes_to_predict = []
    input_ids_to_predict = []
    global_forced_target_ids = []

    try:
        global_forced_target_ids = _parse_original_node_id_list(
            getattr(args, 'force_target_ids', None),
            '--force_target_ids'
        )
        if global_forced_target_ids:
            logging.info(
                "Loaded %d forced target ID(s) from --force_target_ids.",
                len(global_forced_target_ids),
            )

        if getattr(args, 'eval_marker_pairs_csv', None):
            marker_pair_request_bundle = _load_marker_pair_requests(args.eval_marker_pairs_csv)
            logging.info(
                "Loaded %d disease-marker pair(s) spanning %d disease ID(s) from pair CSV: %s",
                marker_pair_request_bundle['pair_count'],
                len(marker_pair_request_bundle['disease_ids']),
                args.eval_marker_pairs_csv,
            )
            logging.info(
                "Pair-CSV mode resolves diseases by original node ID from the ID field (for marker_edge this is the MONDO ID), not by disease name."
            )
            if marker_pair_request_bundle.get('target_types'):
                logging.info(
                    "Observed target marker type(s) in pair CSV: %s",
                    marker_pair_request_bundle['target_types'],
                )
        elif getattr(args, 'eval_input_ids', None):
            input_ids_to_predict, input_source = _load_input_ids(args)
            logging.info(
                "Loaded %d disease input ID(s) from %s.",
                len(input_ids_to_predict),
                input_source,
            )
            logging.info(
                "ID-list mode resolves diseases by original node ID directly, without fuzzy name matching."
            )
        else:
            logging.info(f"Performing pre-flight fuzzy search in '{args.nodes_csv}'...")
            nodes_df = pd.read_csv(args.nodes_csv)

            search_terms, input_source, source_name_col = _load_input_names(args, eval_input_type)
            logging.info(
                "Loaded %d disease input term(s) from %s%s",
                len(search_terms),
                input_source,
                f" (column: {source_name_col})" if source_name_col else ""
            )
            input_type = eval_input_type.lower()

            type_df = nodes_df[nodes_df['Type'].str.lower() == input_type.lower()]

            found_node_names = set()
            for term in search_terms:
                term_lower = term.lower()
                exact_matches = type_df[type_df['Name'].str.lower() == term_lower]

                if not exact_matches.empty:
                    for name in exact_matches['Name']:
                        logging.info(f"Found exact match for '{term}': {name}")
                        found_node_names.add(name)
                else:
                    logging.warning(f"No exact match for '{term}'. Performing fuzzy search...")
                    fuzzy_matches = type_df[type_df['Name'].str.lower().str.contains(term_lower, na=False)]
                    if not fuzzy_matches.empty:
                        for name in fuzzy_matches['Name']:
                            found_node_names.add(name)

            if not found_node_names:
                logging.error(f"Fuzzy search failed. No nodes of type '{input_type}' found for any of: {search_terms}")
                return

            logging.info(f"Search complete. Found {len(found_node_names)} unique matching nodes to predict for:")
            for name in sorted(list(found_node_names)):
                logging.info(f"  - {name}")

            final_nodes_to_predict = sorted(list(found_node_names))

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return
    except KeyError as e:
        if getattr(args, 'eval_marker_pairs_csv', None):
            logging.error(
                "Error: The marker pair CSV '%s' is missing required columns. Missing detail: %s",
                args.eval_marker_pairs_csv,
                e,
            )
        else:
            logging.error(f"Error: The CSV file '{args.nodes_csv}' is missing a required column for fuzzy search. It needs 'Type' and 'Name'. Missing column: {e}")
        return
    except Exception as e_search:
        logging.error(f"An error occurred while preparing prediction inputs: {e_search}")
        traceback.print_exc()
        return

    try:
        logging.info(f"Initializing DGLGraphCreator at {time.strftime('%X')}...")
        # 通过 DGLGraphCreator 构建或加载异构图。
        # - node_info: 包含 dglid_to_orig / id_to_node_data 等映射，用于后续按原始 ID 恢复 embedding
        # - node_features_from_creator: 可能包含 precomputed 的 Embedding/Tensor，用于与训练时一致的初始化
        node_csv_path = os.path.abspath(args.nodes_csv)
        edge_csv_path = os.path.abspath(args.edges_csv)
        if not os.path.exists(node_csv_path):
            logging.error(f"Error: Node CSV not found at: {node_csv_path}")
            return
        if not os.path.exists(edge_csv_path):
            logging.error(f"Error: Edge CSV not found at: {edge_csv_path}")
            return

        graph_creator = DGLGraphCreator(
            node_csv_path=node_csv_path,
            edge_csv_path=edge_csv_path,
            embed_size=args.default_embed_size,
            val_size=0,  # Not needed for prediction
            test_size=0, # Not needed for prediction
            random_state=args.seed,
            freeze_pretrained=args.freeze_pretrained,
            graph_cache_path=args.graph_cache_path,
            device=device,
            local_model_base_path=args.local_model_path,
            generate_embeddings=args.generate_embeddings,
            use_sequence_features=not args.disable_sequence_features,
            filter_isolated_nodes=not args.disable_isolated_filter
        )
        logging.info("Data loading and graph creation/loading complete.")

        g_full_cpu, g_train_cpu, _, _ = graph_creator.get_graphs()
        node_info = graph_creator.get_node_info()
        # Load node features produced/loaded by the graph_creator so that the model
        # initial embeddings align with what was used during training.
        node_features_from_creator = graph_creator.get_node_features()

        canonical_input_type = _resolve_input_type(node_info, eval_input_type, set(g_full_cpu.ntypes))
        if not canonical_input_type:
            logging.error(f"Input node type '{eval_input_type}' not found in graph data.")
            return

        requested_output_types = [s.strip() for s in args.eval_output_types.split(',') if s.strip()]
        canonical_output_types = []
        for requested_type in requested_output_types:
            canonical_target = _resolve_any_node_type(node_info, [requested_type], set(g_full_cpu.ntypes))
            if canonical_target and canonical_target not in canonical_output_types:
                canonical_output_types.append(canonical_target)
        if not canonical_output_types:
            logging.error(
                "No canonical output node types were resolved from eval_output_types='%s'.",
                args.eval_output_types,
            )
            return

        required_name_map_types = [canonical_input_type] + [t for t in canonical_output_types if t != canonical_input_type]
        for required_type in required_name_map_types:
            _ensure_name_map_for_type(node_info, g_full_cpu, required_type, node_csv_path)

        available_name_map = node_info.get('name_map', {}).get(canonical_input_type, {})
        if not available_name_map:
            if marker_pair_request_bundle:
                logging.warning(
                    "No node name map available for type '%s'. Pair-CSV mode will continue using original IDs only.",
                    canonical_input_type,
                )
            else:
                logging.error(f"No node name map available for type '{canonical_input_type}'.")
                return

        input_orig_to_dgl = _build_orig_to_dgl_lookup(node_info, canonical_input_type)
        input_dgl_to_name = {dgl_id: name for name, dgl_id in available_name_map.items()}

        if marker_pair_request_bundle:
            if len(canonical_output_types) != 1:
                logging.error(
                    "Pair-CSV mode requires exactly one eval_output_types value, but got: %s",
                    canonical_output_types,
                )
                return

            resolved_output_type = canonical_output_types[0]
            for requested_pair_type in marker_pair_request_bundle.get('target_types', []):
                resolved_pair_type = _resolve_any_node_type(
                    node_info,
                    [requested_pair_type],
                    set(g_full_cpu.ntypes),
                )
                if resolved_pair_type and resolved_pair_type != resolved_output_type:
                    logging.error(
                        "Pair CSV target type '%s' resolves to '%s', but this run is configured for output type '%s'.",
                        requested_pair_type,
                        resolved_pair_type,
                        resolved_output_type,
                    )
                    return

            missing_input_ids = []
            for disease_id, pair_payload in marker_pair_request_bundle['pairs_by_disease_id'].items():
                normalized_disease_id = _normalize_original_node_id(disease_id)
                input_dgl_id = input_orig_to_dgl.get(normalized_disease_id)
                if input_dgl_id is None:
                    missing_input_ids.append(normalized_disease_id)
                    continue

                prediction_inputs.append({
                    'id': normalized_disease_id,
                    'name': input_dgl_to_name.get(input_dgl_id),
                    'dgl_id': int(input_dgl_id),
                })
                forced_targets_by_input_id[normalized_disease_id] = list(pair_payload.get('target_ids', []))

            if missing_input_ids:
                preview = ", ".join(missing_input_ids[:10])
                logging.warning(
                    "%d disease ID(s) from pair CSV were not found in the graph. Example(s): %s",
                    len(missing_input_ids),
                    preview,
                )

            retained_disease_count = len(prediction_inputs)
            retained_pair_count = sum(len(target_ids) for target_ids in forced_targets_by_input_id.values())
            logging.info(
                "Pair-CSV filtering retained %d disease ID(s) and %d disease-marker pair(s) present in the graph.",
                retained_disease_count,
                retained_pair_count,
            )
        elif input_ids_to_predict:
            missing_input_ids = []
            for disease_id in input_ids_to_predict:
                normalized_disease_id = _normalize_original_node_id(disease_id)
                input_dgl_id = input_orig_to_dgl.get(normalized_disease_id)
                if input_dgl_id is None:
                    missing_input_ids.append(normalized_disease_id)
                    continue

                prediction_inputs.append({
                    'id': normalized_disease_id,
                    'name': input_dgl_to_name.get(input_dgl_id),
                    'dgl_id': int(input_dgl_id),
                })

            if missing_input_ids:
                preview = ", ".join(missing_input_ids[:10])
                logging.warning(
                    "%d disease ID(s) from --eval_input_ids were not found in the graph. Example(s): %s",
                    len(missing_input_ids),
                    preview,
                )

            logging.info(
                "ID-list filtering retained %d disease ID(s) present in the graph.",
                len(prediction_inputs),
            )
        else:
            missing_nodes = sorted(set(final_nodes_to_predict) - set(available_name_map.keys()))
            if missing_nodes:
                preview = ", ".join(missing_nodes[:10])
                logging.warning(
                    "%d input node(s) were not found in the graph after loading. Example(s): %s",
                    len(missing_nodes),
                    preview
                )

            final_nodes_to_predict = [name for name in final_nodes_to_predict if name in available_name_map]

            if args.prefilter_by_drug_hops > 0:
                canonical_drug_type = _resolve_any_node_type(
                    node_info,
                    ['drug', 'compound', 'chemical'],
                    set(g_full_cpu.ntypes)
                )
                if not canonical_drug_type:
                    logging.warning(
                        "--prefilter-by-drug-hops is enabled, but no drug-like node type was found in graph ntypes=%s.",
                        list(g_full_cpu.ntypes)
                    )
                else:
                    reachable_ids = _collect_reachable_disease_ids_from_drug(
                        g_full_cpu,
                        canonical_input_type,
                        canonical_drug_type,
                        args.prefilter_by_drug_hops
                    )
                    before_count = len(final_nodes_to_predict)
                    dglid_to_name = {dgl_id: name for name, dgl_id in available_name_map.items()}
                    reachable_names = {
                        dglid_to_name[dgl_id]
                        for dgl_id in reachable_ids
                        if dgl_id in dglid_to_name
                    }
                    final_nodes_to_predict = [name for name in final_nodes_to_predict if name in reachable_names]
                    logging.info(
                        "Drug-hop prefilter active (drug_type=%s, hops<=%d): %d -> %d input disease nodes.",
                        canonical_drug_type,
                        args.prefilter_by_drug_hops,
                        before_count,
                        len(final_nodes_to_predict)
                    )

            if args.prefilter_by_drug_to_target_hops > 0:
                canonical_drug_type = _resolve_any_node_type(
                    node_info,
                    ['drug', 'compound', 'chemical'],
                    set(g_full_cpu.ntypes)
                )
                canonical_target_types = []
                for requested in requested_output_types:
                    resolved = _resolve_any_node_type(node_info, [requested], set(g_full_cpu.ntypes))
                    if resolved and resolved not in canonical_target_types:
                        canonical_target_types.append(resolved)

                if not canonical_drug_type:
                    logging.warning(
                        "--prefilter-by-drug-to-target-hops is enabled, but no drug-like node type was found."
                    )
                elif not canonical_target_types:
                    logging.warning(
                        "--prefilter-by-drug-to-target-hops is enabled, but no canonical output target types were resolved from eval_output_types='%s'.",
                        args.eval_output_types
                    )
                else:
                    reachable_ids = _collect_diseases_with_drug_to_target_paths(
                        g_full_cpu,
                        canonical_input_type,
                        canonical_drug_type,
                        canonical_target_types,
                        args.prefilter_by_drug_to_target_hops
                    )
                    before_count = len(final_nodes_to_predict)
                    dglid_to_name = {dgl_id: name for name, dgl_id in available_name_map.items()}
                    reachable_names = {
                        dglid_to_name[dgl_id]
                        for dgl_id in reachable_ids
                        if dgl_id in dglid_to_name
                    }
                    final_nodes_to_predict = [name for name in final_nodes_to_predict if name in reachable_names]
                    logging.info(
                        "Drug->target structure prefilter active (drug_type=%s, target_types=%s, hops<=%d each leg): %d -> %d input disease nodes.",
                        canonical_drug_type,
                        canonical_target_types,
                        args.prefilter_by_drug_to_target_hops,
                        before_count,
                        len(final_nodes_to_predict)
                    )

            prediction_inputs = [
                {
                    'id': None,
                    'name': name,
                    'dgl_id': int(available_name_map[name]),
                }
                for name in final_nodes_to_predict
            ]

        if not prediction_inputs:
            logging.error("No valid input nodes remain after matching against graph data.")
            return

    except Exception as e:
        logging.error(f"Error during data loading/graph creation: {e}")
        traceback.print_exc()
        return

    ntypes = g_full_cpu.ntypes
    etypes = g_full_cpu.canonical_etypes
    num_nodes_dict = {ntype: g_full_cpu.num_nodes(ntype) for ntype in ntypes}

    # --- Initialize Model Architecture ---
    # 说明：模型初始化尝试使用 graph_creator 提供的 node 特征来对齐 embedding 大小和初始值。
    # 如果 graph_creator 返回了 embedding 对象或张量，我们会复制其权重到 model 的初始化参数中。
    checkpoint_preview = None
    checkpoint_architecture = None
    if args.load_model_path and os.path.exists(args.load_model_path):
        try:
            checkpoint_preview = torch.load(
                args.load_model_path,
                map_location="cpu",
            )
            if isinstance(checkpoint_preview, dict):
                checkpoint_architecture = checkpoint_preview.get("architecture")
        except Exception as preview_error:
            logging.warning(
                "Could not inspect checkpoint architecture before model init: %s",
                preview_error,
            )

    try:
        # Instantiate the model with the same architecture as during training
        # Prepare node_feature_tensors from the graph creator features (if available)
        node_feature_tensors = {}
        if 'node_features_from_creator' in locals() and node_features_from_creator:
            for ntype, feat in node_features_from_creator.items():
                try:
                    if isinstance(feat, torch.nn.Embedding):
                        node_feature_tensors[ntype] = feat.weight.data.clone()
                    else:
                        node_feature_tensors[ntype] = feat.clone()
                except Exception:
                    logging.debug(f"Could not clone feature for node type {ntype}; skipping embedding init.")

        if checkpoint_architecture == "unified_sparse_v1":
            from Model.unified_model import UnifiedDiseaseAssociationModel

            training_config = checkpoint_preview.get("training_args", {})
            model_config = checkpoint_preview.get("model_config", {})
            checkpoint_targets = [
                node_type
                for node_type in model_config.get(
                    "target_node_types",
                    canonical_output_types,
                )
                if node_type in ntypes
            ]
            model = UnifiedDiseaseAssociationModel(
                node_feature_embeddings=(
                    node_feature_tensors if node_feature_tensors else None
                ),
                hgcn_hidden=training_config.get("hgcn_hidden", args.hgcn_hidden),
                hgcn_out=training_config.get("hgcn_out", args.hgcn_out),
                num_transformer_layers=training_config.get(
                    "num_transformer_layers",
                    args.num_transformer_layers,
                ),
                transformer_heads=training_config.get(
                    "transformer_heads",
                    args.transformer_heads,
                ),
                transformer_ff=training_config.get(
                    "transformer_ff",
                    args.transformer_ff,
                ),
                num_latents=training_config.get("num_latents", args.num_latents),
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=checkpoint_targets,
                num_nodes_dict=num_nodes_dict,
                node_info=node_info,
                dropout=training_config.get("dropout", args.dropout),
                num_gnn_layers=training_config.get(
                    "num_gnn_layers",
                    args.num_gnn_layers,
                ),
                freeze_pretrained=args.freeze_pretrained,
                disable_hgcn=training_config.get(
                    "disable_hgcn",
                    args.disable_hgcn,
                ),
                disable_transformer=training_config.get(
                    "disable_transformer",
                    args.disable_transformer,
                ),
                disable_moe=training_config.get(
                    "disable_moe",
                    args.disable_moe,
                ),
                fusion_mode=training_config.get("fusion_mode", args.fusion_mode),
                modality_fusion_modes=model_config.get(
                    "modality_fusion_modes",
                    training_config.get("modality_fusion_modes"),
                ),
                sparse_transformer_topk=model_config.get(
                    "sparse_transformer_topk",
                    training_config.get("sparse_transformer_topk", 256),
                ),
                sparse_transformer_max_hops=model_config.get(
                    "sparse_transformer_max_hops",
                    training_config.get("sparse_transformer_max_hops", 4),
                ),
                sparse_transformer_hop_penalty=training_config.get(
                    "sparse_transformer_hop_penalty",
                    0.2,
                ),
                sparse_transformer_evidence_weight=training_config.get(
                    "sparse_transformer_evidence_weight",
                    0.2,
                ),
                moe_num_experts=model_config.get(
                    "moe_num_experts",
                    training_config.get("moe_num_experts", 4),
                ),
                moe_top_k=model_config.get(
                    "moe_top_k",
                    training_config.get("moe_top_k", 2),
                ),
                moe_hidden_dim=training_config.get("moe_hidden_dim"),
                embedding_cache_size=training_config.get(
                    "embedding_cache_size",
                    2,
                ),
                graph_backend=model_config.get(
                    "graph_backend",
                    training_config.get("unified_graph_backend", "dgl"),
                ),
            )
            logging.info("Initialized unified_sparse_v1 model from checkpoint metadata.")
        else:
            model = DiseaseAssociationModel(
                node_feature_embeddings=node_feature_tensors if node_feature_tensors else None,
                hgcn_hidden=args.hgcn_hidden,
                hgcn_out=args.hgcn_out,
                num_transformer_layers=args.num_transformer_layers,
                transformer_heads=args.transformer_heads,
                transformer_ff=args.transformer_ff,
                num_latents=args.num_latents,
                ntypes=ntypes,
                etypes=etypes,
                target_node_types=canonical_output_types,
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
                explainability_attribution_method=args.explainability_attribution_method
            )
        logging.info("Model architecture initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing model architecture: {e}")
        traceback.print_exc()
        return

    # --- Load Trained Model State ---
    # 说明：加载检查点采用鲁棒方式：
    # 1) 支持两种检查点格式：纯 state_dict 或 包含 'state_dict' + metadata 的 dict
    # 2) 尝试在检查点或附近的 cache/metadata 中找到训练时的 node_info
    # 3) 如果找到了训练时 node_info，则对 embedding 的行做基于原始 ID 的重映射（优先）
    # 4) 否则回退到拷贝重叠前缀行（rows prefix copy）以避免尺寸不匹配导致 load 失败
    if args.load_model_path and os.path.exists(args.load_model_path):
        try:
            logging.info(f"Loading model state from: {args.load_model_path}...")
            raw = torch.load(args.load_model_path, map_location=device)

            # Support checkpoints that are either a pure state_dict or a dict with metadata
            if isinstance(raw, dict) and 'model_state_dict' in raw and isinstance(raw['model_state_dict'], dict):
                ckpt = raw['model_state_dict']
                ck_node_info = raw.get('node_info') or raw.get('metadata')
            elif isinstance(raw, dict) and 'state_dict' in raw and isinstance(raw['state_dict'], dict):
                ckpt = raw['state_dict']
                ck_node_info = raw.get('node_info') or raw.get('metadata') or raw.get('node_info')
            else:
                ckpt = raw
                ck_node_info = None

            # If checkpoint didn't embed node_info, try to locate metadata files nearby
            if ck_node_info is None:
                try:
                    # Start from experiment dir (one level up from model file's parent)
                    exp_dir = os.path.dirname(os.path.dirname(args.load_model_path))

                    # Search up to a few ancestor directories for common cache/metadata locations
                    search_roots = [exp_dir]
                    parent = exp_dir
                    for _ in range(3):
                        parent = os.path.dirname(parent)
                        if parent and parent not in search_roots:
                            search_roots.append(parent)

                    candidates = []
                    for root in search_roots:
                        candidates.extend([
                            os.path.join(root, 'cache', 'metadata', 'node_info.pt'),
                            os.path.join(root, 'cache', 'node_info.pt'),
                            os.path.join(root, 'metadata', 'node_info.pt'),
                            os.path.join(root, 'metadata.pt'),
                            os.path.join(root, 'node_info.pt'),
                        ])

                    # Deduplicate while preserving order
                    seen = set()
                    candidates_unique = []
                    for c in candidates:
                        if c not in seen:
                            seen.add(c)
                            candidates_unique.append(c)

                    for c in candidates_unique:
                        if os.path.exists(c):
                            try:
                                loaded = torch.load(c, map_location='cpu')
                                # node_info may be stored directly or under 'node_info'
                                if isinstance(loaded, dict) and 'node_info' in loaded:
                                    ck_node_info = loaded['node_info']
                                    break
                                # if file itself is node_info-like
                                if isinstance(loaded, dict) and 'dglid_to_orig' in loaded:
                                    ck_node_info = loaded
                                    break
                            except Exception:
                                # ignore malformed candidate files
                                continue
                except Exception:
                    ck_node_info = None

            # Adapt checkpoint to current model shape when possible (handle node count changes)
            model_state = model.state_dict()
            adapted_state = {}
            mismatches = []

            for k, v in model_state.items():
                if k in ckpt:
                    ck_v = ckpt[k]
                    # If shapes match exactly, use checkpoint value
                    if isinstance(ck_v, torch.Tensor) and ck_v.size() == v.size():
                        adapted_state[k] = ck_v
                    # If both are 2D (embeddings) and embedding dim matches, copy overlapping rows
                    elif (isinstance(ck_v, torch.Tensor) and isinstance(v, torch.Tensor)
                          and ck_v.dim() == 2 and v.dim() == 2 and ck_v.size(1) == v.size(1)):
                        # Try to do name-based remapping if node_info from checkpoint is available
                        remapped = False
                        if ck_node_info and isinstance(ck_node_info, dict):
                            # detect embedding key pattern: node_embeddings.<ntype>.weight
                            parts = k.split('.')
                            if len(parts) >= 3 and parts[0] == 'node_embeddings' and parts[-1] == 'weight':
                                ntype = parts[1]
                                try:
                                    ck_dgl_map = ck_node_info.get('dglid_to_orig', {}).get(ntype, {})
                                    cur_dgl_map = node_info.get('dglid_to_orig', {}).get(ntype, {})
                                    # invert current map: orig_id -> cur_dgl
                                    orig_to_cur = {orig: dglid for dglid, orig in cur_dgl_map.items()}
                                    # initialize with current weights
                                    new_w = v.clone()
                                    copied = 0
                                    for ck_dgl, orig_id in ck_dgl_map.items():
                                        if orig_id in orig_to_cur:
                                            cur_idx = int(orig_to_cur[orig_id])
                                            ck_idx = int(ck_dgl)
                                            if ck_idx < ck_v.size(0) and cur_idx < new_w.size(0):
                                                new_w[cur_idx] = ck_v[ck_idx]
                                                copied += 1
                                    adapted_state[k] = new_w
                                    mismatches.append((k, tuple(ck_v.size()), tuple(v.size()), copied))
                                    remapped = True
                                except Exception:
                                    remapped = False
                        if not remapped:
                            # Fallback: copy overlapping prefix rows
                            rows = min(ck_v.size(0), v.size(0))
                            new_w = v.clone()
                            new_w[:rows] = ck_v[:rows]
                            adapted_state[k] = new_w
                            mismatches.append((k, tuple(ck_v.size()), tuple(v.size()), rows))
                    else:
                        # Incompatible shape: keep initialized value and log
                        adapted_state[k] = v
                        try:
                            ck_shape = tuple(ck_v.size())
                        except Exception:
                            ck_shape = ()
                        mismatches.append((k, ck_shape, tuple(v.size()), 0))
                else:
                    # Key not in checkpoint: keep current init
                    adapted_state[k] = v

            # Load adapted state
            model.load_state_dict(adapted_state)
            # If the checkpoint contained metadata about fusion_mode or explainability, apply it to the model
            try:
                if isinstance(raw, dict):
                    # common places for metadata
                    meta = raw.get('metadata') or raw.get('meta') or raw.get('config') or raw
                    if isinstance(meta, dict):
                        if 'fusion_mode' in meta:
                            try:
                                model.fusion_mode = meta.get('fusion_mode')
                                logging.info(f"Overriding model.fusion_mode from checkpoint metadata: {model.fusion_mode}")
                            except Exception:
                                pass
                        if 'enable_explainability' in meta:
                            try:
                                model.enable_explainability = bool(meta.get('enable_explainability'))
                                logging.info(f"Overriding model.enable_explainability from checkpoint metadata: {model.enable_explainability}")
                            except Exception:
                                pass
                        # Some checkpoints may store args under 'args' or 'training_args'
                        training_args = meta.get('args') or meta.get('training_args') or meta.get('training')
                        if isinstance(training_args, dict):
                            if 'fusion_mode' in training_args and not getattr(model, 'fusion_mode', None):
                                try:
                                    model.fusion_mode = training_args.get('fusion_mode')
                                    logging.info(f"Applied training_args.fusion_mode to model: {model.fusion_mode}")
                                except Exception:
                                    pass
            except Exception:
                pass

            model.to(device)
            model.eval()

            if mismatches:
                logging.warning("Some parameters in the checkpoint did not match current model shapes. "
                                "For embedding layers we copied overlapping rows where possible. Details:")
                for item in mismatches:
                    k, ck_shape, cur_shape, copied = item
                    logging.warning(f"  - {k}: checkpoint={ck_shape}, current={cur_shape}, rows_copied={copied}")
        except Exception as e:
            logging.error(f"Error loading model state from {args.load_model_path}: {e}")
            traceback.print_exc()
            return
    else:
        logging.error(f"Specified load_model_path '{args.load_model_path}' not found or not provided. Cannot proceed with prediction.")
        return

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Loaded model has {num_params:,} total parameters.")
    log_memory_usage("After loading model", device)

    # --- Prepare Features and Evaluator ---
    # 获取模型加载后的全量节点特征（从 model.node_embeddings 中抽取并搬到 CPU）
    # 这些特征会传递给 ModelEvaluator，用于随后基于相似度的检索与可解释性分析。
    full_node_features_cpu = {}
    with torch.no_grad():
        for ntype in g_full_cpu.ntypes:
            if ntype in model.node_embeddings:
                num_nodes = g_full_cpu.num_nodes(ntype)
                if num_nodes > 0:
                    node_ids = torch.arange(num_nodes, device=device)
                    # Move features to CPU for the evaluator
                    full_node_features_cpu[ntype] = model.node_embeddings[ntype](node_ids).cpu()
                else:
                    embedding_dim = model.node_embeddings[ntype].embedding_dim
                    full_node_features_cpu[ntype] = torch.empty(0, embedding_dim, device='cpu')
            else:
                logging.warning(f"Node type '{ntype}' from graph is not in the loaded model's embeddings. It will have no features.")

    # --- Run Prediction ---
    # 说明：对每个通过模糊搜索得到的输入节点（疾病名）调用 ModelEvaluator.predict_top_k
    # evaluator 会负责构造输入索引、计算分数并输出 top-k 以及可解释性路径（若启用）。
    logging.info(
        "Preparing to run predictions for %d disease input(s)%s.",
        len(prediction_inputs),
        " from pair CSV" if marker_pair_request_bundle else "",
    )

    try:
        rerank_context = _build_inference_rerank_context(args, node_info, set(g_full_cpu.ntypes))
        evaluator = ModelEvaluator(model, node_info, device, full_node_features_cpu, rerank_context=rerank_context)
        if len(prediction_inputs) > 1:
            logging.info(
                "Prewarming rerank graph cache once before scoring %d disease inputs.",
                len(prediction_inputs),
            )
            evaluator.prepare_rerank_cache(g_full_cpu)
        eval_output_node_types_list = canonical_output_types

        canonical_drug_type_for_path = _resolve_any_node_type(
            node_info,
            ['drug', 'compound', 'chemical'],
            set(g_full_cpu.ntypes)
        )
        drug_path_tokens = {'drug', 'compound', 'chemical'}
        if canonical_drug_type_for_path:
            drug_path_tokens.add(str(canonical_drug_type_for_path).lower())

        if args.find_top1_protein_with_drug_path:
            if not args.enable_explainability:
                logging.warning("Search mode requires explainability paths; forcing enable_explainability=True behavior is recommended.")
            if args.fusion_mode != 'weighted':
                logging.warning("Search mode relies on weighted-fusion explainability path generation. Current fusion_mode=%s may not emit paths.", args.fusion_mode)

        matched_cases = []
        searched_count = 0

        for input_item in prediction_inputs:
            disease_name = input_item.get('name')
            disease_id = input_item.get('id')
            dgl_id = input_item.get('dgl_id')
            forced_target_ids = list(global_forced_target_ids)
            if disease_id:
                forced_target_ids.extend(forced_targets_by_input_id.get(disease_id, []))
            forced_target_ids = _deduplicate_preserve_order(forced_target_ids)

            disease_label = disease_name or disease_id or 'UNKNOWN_DISEASE'
            is_isolated = _is_isolated_in_train(g_full_cpu, canonical_input_type, dgl_id) if dgl_id is not None else True
            isolated_flag = "yes" if is_isolated else "no"
            logging.info(
                f"\n{'='*20} Running prediction for: '{disease_label}' ({canonical_input_type}, isolated={isolated_flag}) {'='*20}"
            )
            prediction_results = evaluator.predict_top_k(
                eval_g_cpu=g_full_cpu,
                input_node_name=disease_name,
                input_node_type=canonical_input_type,
                output_node_types=eval_output_node_types_list,
                top_k=args.eval_top_k,
                display_output=not args.find_top1_protein_with_drug_path,
                input_node_id=disease_id,
                required_target_ids_by_type={eval_output_node_types_list[0]: forced_target_ids} if forced_target_ids else None,
                explainability_path_target_ids_by_type=None,
            )

            if forced_target_ids:
                target_rank_lookup = evaluator.get_last_required_target_ranks().get(
                    eval_output_node_types_list[0],
                    {},
                )
                evaluator.predict_specific_targets(
                    eval_g_cpu=g_full_cpu,
                    input_node_type=canonical_input_type,
                    output_node_type=eval_output_node_types_list[0],
                    target_node_ids=forced_target_ids,
                    input_node_name=disease_name,
                    input_node_id=disease_id,
                    display_output=True,
                    target_rank_lookup=target_rank_lookup,
                )

            searched_count += 1
            if not args.find_top1_protein_with_drug_path:
                continue

            target_key_for_search = eval_output_node_types_list[0] if eval_output_node_types_list else None
            target_results = prediction_results.get(target_key_for_search, []) if isinstance(prediction_results, dict) and target_key_for_search else []
            if not target_results:
                continue

            top1 = target_results[0]
            path_str = top1.get('path')
            path_types, hops = _parse_path_types_and_hops(path_str)
            has_drug = any(token in drug_path_tokens for token in path_types)

            if has_drug and hops >= args.min_explain_hops:
                top1_path = top1.get('path')
                drug_nodes_in_path = _extract_path_nodes_by_types(top1_path, drug_path_tokens)
                matched_cases.append({
                    'disease': disease_name,
                    'target_id': top1.get('id'),
                    'target_name': top1.get('name'),
                    'target_type': top1.get('type'),
                    'score': top1.get('score'),
                    'hops': hops,
                    'path': top1_path,
                    'drug_nodes': drug_nodes_in_path,
                })
                logging.info(
                    "MATCH found: disease='%s' -> top1 %s='%s' (id=%s, score=%.6f, hops=%d)",
                    disease_label,
                    top1.get('type'),
                    top1.get('name'),
                    top1.get('id'),
                    float(top1.get('score', 0.0)),
                    hops
                )

                if args.max_search_matches > 0 and len(matched_cases) >= args.max_search_matches:
                    logging.info(
                        "Reached max_search_matches=%d, stopping early.",
                        args.max_search_matches
                    )
                    break
            else:
                logging.info(
                    "No qualified top1 result found for disease='%s' with criteria: top1 path contains drug and hops>=%d.",
                    disease_label,
                    args.min_explain_hops
                )

        if args.find_top1_protein_with_drug_path:
            logging.info(
                "Search summary: searched=%d, matched=%d, criteria=[top1 target path contains drug, hops>=%d]",
                searched_count,
                len(matched_cases),
                args.min_explain_hops
            )
            if matched_cases:
                print("\n=== Matched Cases (Top1 target result with drug-in-path and hop threshold) ===")
                for idx, item in enumerate(matched_cases, start=1):
                    score_val = float(item.get('score', 0.0))
                    print(
                        f"{idx}. Disease: {item['disease']} | Target: {item['target_name']} ({item['target_id']}, type={item['target_type']}) "
                        f"| Score: {score_val:.6f} | Hops: {item['hops']} | DrugNodes: {item.get('drug_nodes', [])} | Path: {item['path']}"
                    )
            else:
                print("\n=== Matched Cases ===")
                print("No diseases met the criteria: path contains drug and hop count threshold.")
        logging.info(f"\n{'='*20} All predictions finished. {'='*20}")

    except Exception as e_pred:
        logging.error(f"Error during prediction: {e_pred}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Disease-Biomarker Association Prediction')
    
    # --- Essential Arguments for Prediction ---
    # 必要参数（用于指定模型、节点/边数据及输入名）：
    parser.add_argument('--load_model_path', type=str, required=True, help='Path to load the pre-trained model file.')
    parser.add_argument('--nodes_csv', type=str, required=True, help='Path to node CSV file used during training.')
    parser.add_argument('--edges_csv', type=str, required=True, help='Path to edge CSV file used during training.')
    parser.add_argument('--eval_input_names', type=str, default=None, help='Comma-separated list of input node names for prediction (e.g., "disease1,disease2").')
    parser.add_argument('--eval_input_mondo_csv', type=str, default=None, help='Path to a MONDO disease node CSV (expected columns include ID,Type,Name). If provided, all diseases in the CSV are predicted one-by-one.')
    parser.add_argument('--eval_input_ids', type=str, default=None, help='Comma-separated or whitespace-separated list of original disease node IDs (for example, MONDO IDs). These are resolved directly by original node ID rather than by disease name.')
    parser.add_argument('--force_target_ids', type=str, default=None, help='Comma-separated or whitespace-separated list of original target node IDs to force-score and print for every input disease, regardless of top-k rank.')
    parser.add_argument('--eval_marker_pairs_csv', type=str, default=None, help='Path to a prepared disease-marker pair CSV. Disease rows are resolved by original node ID from the disease_id column rather than by disease name. This mode keeps top-k output but also force-scores every requested disease-marker pair regardless of rank. Requires a single eval_output_types value.')
    parser.add_argument('--eval_input_type', type=str, default="disease", help='Type of the input node for prediction (e.g., "disease").')
    parser.add_argument('--eval_output_types', type=str, default="protein,gene,rna,metabolite", help='Comma-separated output node types for prediction.')
    parser.add_argument('--eval_top_k', type=int, default=30, help='The number of top predictions to return for each output type.')
    parser.add_argument('--find_top1_protein_with_drug_path', action='store_true', help='Search mode: keep only diseases whose top1 result for the first eval output type has an explainability path containing a drug node and satisfying --min_explain_hops.')
    parser.add_argument('--min_explain_hops', type=int, default=4, help='Minimum hop count required for the explainability path in search mode.')
    parser.add_argument('--max_search_matches', type=int, default=0, help='Stop early after this many matches in search mode. 0 means no limit.')
    parser.add_argument('--prefilter-by-drug-hops', type=int, default=0, help='If >0, prefilter input diseases to only those reachable from any drug node within this many hops, reducing disease traversal.')
    parser.add_argument('--prefilter-by-drug-to-target-hops', type=int, default=0, help='If >0, prefilter input diseases to only those with disease->drug and drug->target multi-hop structure (same hop threshold for both legs).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use. Set to -1 for CPU.')

    # --- Model Architecture Arguments (must match the loaded model) ---
    # 模型结构相关参数（必须与训练时的结构保持一致）：
    parser.add_argument('--default_embed_size', type=int, default=64, help='Target embedding size for all node features.')
    parser.add_argument('--hgcn_hidden', type=int, default=64, help='HGCN hidden layer size.')
    parser.add_argument('--hgcn_out', type=int, default=64, help='HGCN output embedding size.')
    parser.add_argument('--num_gnn_layers', type=int, default=3, help='Number of GNN layers in the model.')
    parser.add_argument('--num_transformer_layers', type=int, default=3, help='Number of Transformer layers in the model.')
    parser.add_argument('--transformer_heads', type=int, default=16, help='Number of attention heads in the Transformer.')
    parser.add_argument('--transformer_ff', type=int, default=512, help='Transformer feed-forward layer size.')
    parser.add_argument('--num_latents', type=int, default=32, help='Number of latent vectors for the Perceiver model.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate used in the model.')
    parser.add_argument('--fusion-mode', type=str, default='weighted', choices=['sequential', 'concat', 'weighted', 'hierarchical', 'routing', 'attention', 'film'], help='Method to fuse HGCN and Transformer features.')
    parser.add_argument('--rerank-hop-bonus-lambda', type=float, default=0.0, help='Soft rerank structure lambda. Combined with --rerank-max-hops=1, this only rewards directly connected targets.')
    parser.add_argument('--rerank-max-hops', type=int, default=0, help='Optional maximum hop radius for soft rerank search. 0 means unbounded shortest-path search.')
    
    # --- Ablation Arguments (must match the loaded model) ---
    # 消融开关：用于在预测时复现训练时的 ablation 配置（例如关闭 hgcn 或 transformer）
    parser.add_argument('--disable-hgcn', action='store_true', help='Disable the HGCN component.')
    parser.add_argument('--disable-transformer', action='store_true', help='Disable the Transformer component.')
    parser.add_argument('--disable-moe', action='store_true', help='Disable the Mixture-of-Experts predictors.')
    parser.add_argument('--disable-mla', action='store_true', help='Disable Multi-head Latent Attention in Transformer.')
    parser.add_argument('--disable-feature-gating', action='store_true', help='Disable the feature gating mechanism.')

    # --- Other Arguments ---
    # 其他参数：缓存路径、随机种子、是否冻结预训练 embedding、可解释性开关等
    parser.add_argument('--graph_cache_path', type=str, default=None, help='Path to directory for caching/loading processed graph data.')
    parser.add_argument('--local_model_path', type=str, default='./models_cache', help='Path to the local Hugging Face hub cache directory for offline execution.')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    parser.add_argument('--freeze_pretrained', action='store_true', help='(Should match training) Freeze pre-trained embeddings.')
    parser.add_argument('--generate-embeddings', action='store_true', dest='generate_embeddings', help='Generate specialized node embeddings when cache is missing.')
    parser.add_argument('--no-generate-embeddings', action='store_false', dest='generate_embeddings', help='Disable specialized embedding generation (use default learnable embeddings).')
    parser.add_argument('--disable-sequence-features', action='store_true', help='Disable using sequence data for node features.')
    parser.add_argument('--disable-isolated-filter', action='store_true', help='Disable isolated-node filtering so disconnected nodes remain in the prediction graph.')
    # Explainability: enabled by default; user can disable with --no-explainability
    parser.add_argument('--no-explainability', action='store_false', dest='enable_explainability', help='Disable explainability outputs (default: enabled).')
    parser.add_argument('--explainability-top-k', type=int, default=10, help='Top-K nodes/edges to include in explainability summaries.')
    parser.add_argument('--explainability-max-hops', type=int, default=3, help='Maximum hops to trace when constructing multi-hop explanation paths.')
    parser.add_argument('--explainability-hop-penalty', type=float, default=0.18, help='Base penalty added to every physical hop in explainability path search to reduce fragmented long paths.')
    parser.add_argument('--explainability-virtual-edge-penalty', type=float, default=0.60, help='Additional penalty applied to virtual edges in gradient-guided explainability fallback.')
    parser.add_argument('--explainability-virtual-sim-threshold', type=float, default=0.45, help='Minimum cosine similarity required to create a virtual edge in the fallback graph.')
    parser.add_argument('--explainability-attribution-method', type=str, default='input_x_gradient', choices=['input_x_gradient', 'gradient_norm'], help='Attribution method used in gradient-guided explainability fallback.')
    parser.set_defaults(enable_explainability=True, generate_embeddings=True)

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.seed)

    logging.info("Effective command line arguments for prediction:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)
    
    # --- Device Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logging.info(f"--- Using GPU: {torch.cuda.get_device_name(args.gpu)} ---")
    else:
        device = torch.device('cpu')
        logging.info("--- Using CPU ---")
    
    predict(args, device, args.eval_input_type)

if __name__ == "__main__":
    main()
