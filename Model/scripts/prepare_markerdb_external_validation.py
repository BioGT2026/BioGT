#!/usr/bin/env python3
"""Build an auditable MarkerDB-only external edge holdout protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


TARGET_TYPES = {"protein", "gene", "rna", "metabolite"}


def _columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in frame.columns
    ]
    return frame


def _value(value) -> str:
    return str(value or "").strip()


def _node_type(value) -> str:
    return _value(value).lower()


def _is_disease(node_id: str, node_type: str) -> bool:
    return node_id.upper().startswith("MONDO:") or node_type == "disease"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    records = []
    skipped = []
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        left_id = _value(row.get("node_1_id"))
        right_id = _value(row.get("node_2_id"))
        left_type = _node_type(row.get("node_1_data_type"))
        right_type = _node_type(row.get("node_2_data_type"))
        left_disease = _is_disease(left_id, left_type)
        right_disease = _is_disease(right_id, right_type)
        if left_disease == right_disease:
            skipped.append({"row_number": row_number, "reason": "not_one_disease_endpoint"})
            continue
        if left_disease:
            disease_id, marker_id, marker_type = left_id, right_id, right_type
        else:
            disease_id, marker_id, marker_type = right_id, left_id, left_type
        if marker_type not in TARGET_TYPES or not disease_id or not marker_id:
            skipped.append({"row_number": row_number, "reason": "unsupported_or_blank_target"})
            continue
        records.append(
            {
                "disease_id": disease_id,
                "marker_id": marker_id,
                "marker_type": marker_type,
                "markerdb_edge_type": _value(row.get("edge_type")),
            }
        )
    if not records:
        raise ValueError("MarkerDB CSV contains no supported disease-marker pairs")
    pairs = pd.DataFrame(records).drop_duplicates(
        subset=["disease_id", "marker_id", "marker_type"]
    )
    return pairs.sort_values(["marker_type", "disease_id", "marker_id"]), skipped


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((_value(left_id), _value(right_id))))


def prepare(merged_edges: Path, marker_edges: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = _columns(
        pd.read_csv(merged_edges, dtype=str, keep_default_na=False, na_values=[""])
    )
    marker = _columns(
        pd.read_csv(marker_edges, dtype=str, keep_default_na=False, na_values=[""])
    )
    required = {
        "node_1_id", "node_1_data_type", "node_2_id", "node_2_data_type"
    }
    for name, frame in (("merged", merged), ("MarkerDB", marker)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} edge CSV missing columns: {sorted(missing)}")

    external_pairs, skipped = _extract_pairs(marker)
    pair_keys = {
        _pair_key(row.disease_id, row.marker_id)
        for row in external_pairs.itertuples(index=False)
    }
    merged_keys = [
        _pair_key(left, right)
        for left, right in zip(merged["node_1_id"], merged["node_2_id"])
    ]
    marker_mask = pd.Series(
        [key in pair_keys for key in merged_keys],
        index=merged.index,
        dtype=bool,
    )
    clean_graph = merged.loc[~marker_mask].copy()
    removed_graph_edges = merged.loc[marker_mask].copy()
    matched_keys = set(merged_keys[index] for index in merged.index[marker_mask])
    external_pairs = external_pairs.assign(
        present_in_merged_graph=[
            _pair_key(row.disease_id, row.marker_id) in matched_keys
            for row in external_pairs.itertuples(index=False)
        ]
    )
    evaluable = external_pairs.loc[external_pairs["present_in_merged_graph"]].copy()
    if evaluable.empty:
        raise ValueError("No MarkerDB external pairs matched the merged graph")

    clean_path = output_dir / "merged_edge_without_markerdb.csv"
    external_path = output_dir / "markerdb_external_edges.csv"
    removed_path = output_dir / "removed_markerdb_graph_edges.csv"
    clean_graph.to_csv(clean_path, index=False)
    evaluable.to_csv(external_path, index=False)
    removed_graph_edges.to_csv(removed_path, index=False)

    clean_keys = {
        _pair_key(left, right)
        for left, right in zip(clean_graph["node_1_id"], clean_graph["node_2_id"])
    }
    leakage_keys = clean_keys & matched_keys
    if leakage_keys:
        raise AssertionError(
            f"MarkerDB pair leakage remains in clean graph: {len(leakage_keys)} pair(s)"
        )

    audit = {
        "schema_version": 1,
        "protocol": "markerdb_external_edge_holdout",
        "training_graph_excludes_all_evaluable_markerdb_pairs": True,
        "selection_uses_external_edges": False,
        "targets": sorted(TARGET_TYPES),
        "source_files": {
            "merged_edges": {"path": str(merged_edges.resolve()), "sha256": _sha256(merged_edges)},
            "markerdb_edges": {"path": str(marker_edges.resolve()), "sha256": _sha256(marker_edges)},
        },
        "counts": {
            "merged_edges_before": int(len(merged)),
            "merged_edges_after": int(len(clean_graph)),
            "removed_graph_edges": int(len(removed_graph_edges)),
            "markerdb_unique_pairs": int(len(external_pairs)),
            "evaluable_external_pairs": int(len(evaluable)),
            "unmatched_markerdb_pairs": int((~external_pairs["present_in_merged_graph"]).sum()),
            "skipped_markerdb_rows": int(len(skipped)),
            "evaluable_pairs_by_target": dict(Counter(evaluable["marker_type"])),
            "clean_graph_pair_overlap": int(len(leakage_keys)),
        },
        "outputs": {
            "training_graph": {"path": str(clean_path.resolve()), "sha256": _sha256(clean_path)},
            "external_edges": {"path": str(external_path.resolve()), "sha256": _sha256(external_path)},
            "removed_edges": {"path": str(removed_path.resolve()), "sha256": _sha256(removed_path)},
        },
        "skipped_rows_preview": skipped[:20],
    }
    audit_path = output_dir / "markerdb_external_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-edges", type=Path, required=True)
    parser.add_argument("--marker-edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.merged_edges, args.marker_edges, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
