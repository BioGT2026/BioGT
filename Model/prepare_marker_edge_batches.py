import argparse
import json
import logging
import os
import time

import pandas as pd


def _normalize_column_name(column_name):
    return str(column_name).strip().lower().replace(' ', '_')


def _normalize_value(raw_value):
    if raw_value is None:
        return None

    normalized = str(raw_value).strip()
    return normalized or None


def _normalize_marker_type(raw_marker_type):
    normalized_marker_type = _normalize_value(raw_marker_type)
    if normalized_marker_type is None:
        return None
    return normalized_marker_type.lower()


def _is_disease_node(node_id, node_type):
    normalized_node_id = (_normalize_value(node_id) or '').upper()
    normalized_node_type = (_normalize_marker_type(node_type) or '')
    return normalized_node_id.startswith('MONDO:') or normalized_node_type == 'disease'


def _extract_marker_pairs(edge_df):
    pair_records = []
    skipped_rows = []

    for row_index, row in edge_df.iterrows():
        node1_id = _normalize_value(row.get('node_1_id'))
        node1_type = _normalize_marker_type(row.get('node_1_data_type'))
        node1_source = _normalize_value(row.get('node_1_source'))
        node2_id = _normalize_value(row.get('node_2_id'))
        node2_type = _normalize_marker_type(row.get('node_2_data_type'))
        node2_source = _normalize_value(row.get('node_2_source'))
        edge_type = _normalize_value(row.get('edge_type'))
        edge_direction = _normalize_value(row.get('edge_direction'))
        edge_weight = _normalize_value(row.get('edge_weight'))

        node1_is_disease = _is_disease_node(node1_id, node1_type)
        node2_is_disease = _is_disease_node(node2_id, node2_type)

        if node1_is_disease == node2_is_disease:
            skipped_rows.append(
                {
                    'row_number': int(row_index) + 2,
                    'reason': 'could_not_identify_single_disease_endpoint',
                }
            )
            continue

        if node1_is_disease:
            disease_id = node1_id
            disease_type = node1_type or 'disease'
            disease_source = node1_source
            marker_id = node2_id
            marker_type = node2_type
            marker_source = node2_source
        else:
            disease_id = node2_id
            disease_type = node2_type or 'disease'
            disease_source = node2_source
            marker_id = node1_id
            marker_type = node1_type
            marker_source = node1_source

        if not disease_id or not marker_id or not marker_type:
            skipped_rows.append(
                {
                    'row_number': int(row_index) + 2,
                    'reason': 'missing_required_pair_values',
                }
            )
            continue

        pair_records.append(
            {
                'disease_id': disease_id,
                'disease_type': disease_type or 'disease',
                'disease_source': disease_source or '',
                'marker_id': marker_id,
                'marker_type': marker_type,
                'marker_source': marker_source or '',
                'edge_type': edge_type or '',
                'edge_direction': edge_direction or '',
                'edge_weight': edge_weight or '',
            }
        )

    if not pair_records:
        raise ValueError('No valid disease-marker pairs could be extracted from marker_edge data.')

    pair_df = pd.DataFrame(pair_records)
    pair_df = pair_df.drop_duplicates().reset_index(drop=True)
    return pair_df, skipped_rows


def _compute_file_signature(file_path):
    stat_result = os.stat(file_path)
    return {
        'size': int(stat_result.st_size),
        'mtime_ns': int(stat_result.st_mtime_ns),
    }


def _load_summary(summary_path):
    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, 'r', encoding='utf-8') as summary_file:
            return json.load(summary_file)
    except Exception as exc:
        logging.warning('Failed to load existing summary from %s: %s', summary_path, exc)
        return None


def _summary_matches_source(summary, marker_edge_csv, source_signature):
    if not isinstance(summary, dict):
        return False

    return (
        summary.get('marker_edge_csv') == marker_edge_csv
        and summary.get('marker_edge_signature') == source_signature
    )


def _cached_outputs_exist(summary):
    required_files = [
        summary.get('unique_disease_ids_path'),
        summary.get('all_pairs_path'),
        summary.get('manifest_path'),
    ]
    if any(not file_path or not os.path.exists(file_path) for file_path in required_files):
        return False

    for row in summary.get('marker_types', []):
        pair_csv = row.get('pair_csv')
        disease_ids_csv = row.get('disease_ids_csv')
        if not pair_csv or not os.path.exists(pair_csv):
            return False
        if not disease_ids_csv or not os.path.exists(disease_ids_csv):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract unique MONDO disease IDs and per-marker-type disease-marker pair batches from marker_edge CSV. The downstream prediction step resolves diseases by disease_id/original node ID rather than disease name.'
    )
    parser.add_argument(
        '--marker_edge_csv',
        type=str,
        default=os.path.join(os.getcwd(), 'marker_edge.csv'),
        help='Path to marker_edge CSV. Defaults to ./marker_edge.csv in the current working directory.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'prediction_logs', 'marker_edge_batches'),
        help='Directory for generated batch files.',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    marker_edge_csv = os.path.abspath(args.marker_edge_csv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(marker_edge_csv):
        raise FileNotFoundError(f"marker_edge CSV not found: {marker_edge_csv}")

    source_signature = _compute_file_signature(marker_edge_csv)
    summary_path = os.path.join(output_dir, 'summary.json')
    existing_summary = _load_summary(summary_path)
    if _summary_matches_source(existing_summary, marker_edge_csv, source_signature) and _cached_outputs_exist(existing_summary):
        logging.info('Reusing cached marker-edge preprocessing outputs from: %s', output_dir)
        logging.info('Cached unique MONDO disease list: %s', existing_summary.get('unique_disease_ids_path'))
        logging.info('Cached pair manifest: %s', existing_summary.get('manifest_path'))
        return

    edge_df = pd.read_csv(marker_edge_csv, dtype=str, keep_default_na=False, na_values=[''])
    if edge_df.empty:
        raise ValueError(f"marker_edge CSV is empty: {marker_edge_csv}")

    edge_df.columns = [_normalize_column_name(col) for col in edge_df.columns]
    required_columns = {
        'node_1_id',
        'node_1_data_type',
        'node_2_id',
        'node_2_data_type',
    }
    missing_columns = required_columns - set(edge_df.columns)
    if missing_columns:
        raise KeyError(
            f"marker_edge CSV is missing required columns: {sorted(missing_columns)}. "
            f"Expected at least: {sorted(required_columns)}"
        )

    pair_df, skipped_rows = _extract_marker_pairs(edge_df)

    unique_disease_ids_df = pd.DataFrame(
        {'disease_id': sorted(pair_df['disease_id'].dropna().astype(str).unique().tolist())}
    )
    unique_disease_ids_path = os.path.join(output_dir, 'unique_mondo_disease_ids.csv')
    unique_disease_ids_df.to_csv(unique_disease_ids_path, index=False)

    all_pairs_path = os.path.join(output_dir, 'marker_pairs_all.csv')
    pair_df.to_csv(all_pairs_path, index=False)

    manifest_rows = []
    for marker_type, group_df in pair_df.groupby('marker_type', sort=True):
        safe_marker_type = ''.join(ch if ch.isalnum() else '_' for ch in str(marker_type).lower()).strip('_') or 'unknown'
        pair_csv_path = os.path.join(output_dir, f'marker_pairs_{safe_marker_type}.csv')
        disease_ids_csv_path = os.path.join(output_dir, f'disease_ids_{safe_marker_type}.csv')
        group_df = group_df.sort_values(['disease_id', 'marker_id']).reset_index(drop=True)
        group_df.to_csv(pair_csv_path, index=False)
        type_disease_ids_df = pd.DataFrame(
            {'disease_id': sorted(group_df['disease_id'].dropna().astype(str).unique().tolist())}
        )
        type_disease_ids_df.to_csv(disease_ids_csv_path, index=False)
        manifest_rows.append(
            {
                'marker_type': marker_type,
                'pair_csv': pair_csv_path,
                'pair_count': int(len(group_df)),
                'disease_count': int(group_df['disease_id'].nunique()),
                'disease_ids_csv': disease_ids_csv_path,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, 'batch_manifest.tsv')
    manifest_df.to_csv(manifest_path, sep='\t', index=False)

    summary = {
        'marker_edge_csv': marker_edge_csv,
        'marker_edge_signature': source_signature,
        'output_dir': output_dir,
        'generated_at_unix': int(time.time()),
        'unique_disease_count': int(unique_disease_ids_df['disease_id'].nunique()),
        'pair_count': int(len(pair_df)),
        'unique_disease_ids_path': unique_disease_ids_path,
        'all_pairs_path': all_pairs_path,
        'manifest_path': manifest_path,
        'marker_types': manifest_rows,
        'skipped_row_count': int(len(skipped_rows)),
        'skipped_rows_preview': skipped_rows[:20],
    }
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    logging.info('Generated unique MONDO disease list: %s', unique_disease_ids_path)
    logging.info('Generated pair manifest: %s', manifest_path)
    logging.info('Generated %d marker-type batch file(s).', len(manifest_rows))
    for row in manifest_rows:
        logging.info(
            "Batch marker_type=%s disease_count=%d pair_count=%d file=%s disease_queue=%s",
            row['marker_type'],
            row['disease_count'],
            row['pair_count'],
            row['pair_csv'],
            row['disease_ids_csv'],
        )


if __name__ == '__main__':
    main()