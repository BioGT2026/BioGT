"""Build a small target-disease graph for local training smoke tests."""

import argparse
import csv
import os
import sys
from collections import OrderedDict


csv.field_size_limit(sys.maxsize)


EDGE_HEADER = [
    "Node 1 ID",
    "Node 1 Data Type",
    "Node 1 Source",
    "Edge Type",
    "Edge Direction",
    "Edge Weight",
    "Node 2 ID",
    "Node 2 Data Type",
    "Node 2 Source",
]


def _norm(value):
    return str(value or "").strip().lower()


def _edge_key(row):
    return tuple(row.get(col, "") for col in EDGE_HEADER)


def _is_target_disease_edge(row, target_type):
    src_type = _norm(row.get("Node 1 Data Type"))
    dst_type = _norm(row.get("Node 2 Data Type"))
    rel = _norm(row.get("Edge Type"))
    target_type = _norm(target_type)
    return rel == f"{target_type}_disease" and {
        src_type,
        dst_type,
    } == {target_type, "disease"}


def build_smoke_data(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_nodes = os.path.join(args.output_dir, "merged_node.csv")
    out_edges = os.path.join(args.output_dir, "merged_edge.csv")

    all_node_ids = set()
    with open(args.nodes_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("ID")
            if node_id:
                all_node_ids.add(node_id)

    selected_edges = OrderedDict()
    seed_node_ids = set()
    disease_ids = set()
    target_ids = set()
    max_target_nodes = args.max_target_nodes if args.max_target_nodes is not None else args.max_proteins

    with open(args.edges_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_target_disease_edge(row, args.target_type):
                continue

            src_id = row["Node 1 ID"]
            dst_id = row["Node 2 ID"]
            if src_id not in all_node_ids or dst_id not in all_node_ids:
                continue
            src_type = _norm(row["Node 1 Data Type"])
            dst_type = _norm(row["Node 2 Data Type"])
            disease_id = src_id if src_type == "disease" else dst_id
            target_id = src_id if src_type == _norm(args.target_type) else dst_id

            if disease_id not in disease_ids and len(disease_ids) >= args.max_diseases:
                continue
            if target_id not in target_ids and len(target_ids) >= max_target_nodes:
                continue

            selected_edges.setdefault(_edge_key(row), row)
            disease_ids.add(disease_id)
            target_ids.add(target_id)
            seed_node_ids.update([src_id, dst_id])

            if len(selected_edges) >= args.target_edge_limit:
                break

    if not selected_edges:
        raise RuntimeError(f"No {args.target_type}_disease edges were found for smoke data.")

    context_edges = 0
    with open(args.edges_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_id = row["Node 1 ID"]
            dst_id = row["Node 2 ID"]
            if src_id not in all_node_ids or dst_id not in all_node_ids:
                continue
            if src_id not in seed_node_ids and dst_id not in seed_node_ids:
                continue
            key = _edge_key(row)
            if key in selected_edges:
                continue

            selected_edges[key] = row
            seed_node_ids.update([src_id, dst_id])
            context_edges += 1
            if context_edges >= args.context_edge_limit:
                break

    written_nodes = 0
    with open(args.nodes_csv, newline="") as fin, open(out_nodes, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row.get("ID") in seed_node_ids:
                writer.writerow(row)
                written_nodes += 1

    node_ids_written = set()
    with open(out_nodes, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_ids_written.add(row.get("ID"))

    written_edges = 0
    with open(out_edges, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_HEADER)
        writer.writeheader()
        for row in selected_edges.values():
            if row["Node 1 ID"] in node_ids_written and row["Node 2 ID"] in node_ids_written:
                writer.writerow({col: row.get(col, "") for col in EDGE_HEADER})
                written_edges += 1

    print(f"nodes={written_nodes}")
    print(f"edges={written_edges}")
    print(f"target_type={args.target_type}")
    print(f"target_relation={_norm(args.target_type)}_disease")
    print(f"target_disease_edges={sum(1 for row in selected_edges.values() if _is_target_disease_edge(row, args.target_type))}")
    print(f"diseases={len(disease_ids)}")
    print(f"targets={len(target_ids)}")
    print(f"output_dir={args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare a small target-disease smoke-test graph.")
    parser.add_argument("--nodes-csv", default="../Data/merged_node.csv")
    parser.add_argument("--edges-csv", default="../Data/merged_edge.csv")
    parser.add_argument("--target-type", default="protein", choices=["metabolite", "protein", "rna", "gene"])
    parser.add_argument("--output-dir", default="smoke_data/protein")
    parser.add_argument("--target-edge-limit", type=int, default=120)
    parser.add_argument("--context-edge-limit", type=int, default=1500)
    parser.add_argument("--max-diseases", type=int, default=40)
    parser.add_argument("--max-target-nodes", type=int, default=None)
    parser.add_argument("--max-proteins", type=int, default=120)
    args = parser.parse_args()
    build_smoke_data(args)


if __name__ == "__main__":
    main()
