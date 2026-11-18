import argparse
import csv
from array import array
from collections import deque
from pathlib import Path


csv.field_size_limit(1024 * 1024 * 1024)

BIOMARKER_TYPES = ("protein", "gene", "rna", "metabolite")
OUTPUT_FILENAMES = {
    "protein": "disease_protein_connected_edge_counts.csv",
    "gene": "disease_gene_connected_edge_counts.csv",
    "rna": "disease_rna_connected_edge_counts.csv",
    "metabolite": "disease_metabolite_connected_edge_counts.csv",
}
FILTERED_OUTPUT_FILENAMES = {
    biomarker_type: filename.replace(".csv", "_filtered_by_marker.csv")
    for biomarker_type, filename in OUTPUT_FILENAMES.items()
}


def normalize_text(value):
    return (value or "").strip()


def normalize_type(value):
    return normalize_text(value).lower()


def load_mondo_diseases(mondo_node_path):
    diseases = {}
    with mondo_node_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            disease_id = normalize_text(row.get("ID"))
            disease_type = normalize_type(row.get("Type"))
            if not disease_id.startswith("MONDO:") or disease_type != "disease":
                continue
            diseases[disease_id] = normalize_text(row.get("Name"))
    return diseases


def get_or_create_index(node_id, node_type, node_id_to_index, index_to_node_id, node_types, degrees):
    index = node_id_to_index.get(node_id)
    if index is None:
        index = len(index_to_node_id)
        node_id_to_index[node_id] = index
        index_to_node_id.append(node_id)
        node_types.append(node_type)
        degrees.append(0)
        return index

    if node_type and not node_types[index]:
        node_types[index] = node_type
    return index


def build_graph(merged_edge_path, mondo_diseases):
    node_id_to_index = {}
    index_to_node_id = []
    node_types = []
    degrees = []

    with merged_edge_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            node1_id = normalize_text(row.get("Node 1 ID"))
            node2_id = normalize_text(row.get("Node 2 ID"))
            node1_type = normalize_type(row.get("Node 1 Data Type"))
            node2_type = normalize_type(row.get("Node 2 Data Type"))

            if not node1_id or not node2_id:
                continue

            index1 = get_or_create_index(
                node1_id,
                node1_type,
                node_id_to_index,
                index_to_node_id,
                node_types,
                degrees,
            )
            index2 = get_or_create_index(
                node2_id,
                node2_type,
                node_id_to_index,
                index_to_node_id,
                node_types,
                degrees,
            )

            degrees[index1] += 1
            degrees[index2] += 1

    offsets = array("Q", [0])
    total_degree = 0
    for degree in degrees:
        total_degree += degree
        offsets.append(total_degree)

    adjacency = array("I", [0]) * total_degree
    cursor = array("Q", offsets)

    with merged_edge_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            node1_id = normalize_text(row.get("Node 1 ID"))
            node2_id = normalize_text(row.get("Node 2 ID"))
            if not node1_id or not node2_id:
                continue

            index1 = node_id_to_index[node1_id]
            index2 = node_id_to_index[node2_id]

            adjacency[cursor[index1]] = index2
            cursor[index1] += 1
            adjacency[cursor[index2]] = index1
            cursor[index2] += 1

    disease_indices = []
    biomarker_indices = {biomarker_type: [] for biomarker_type in BIOMARKER_TYPES}
    for index, node_id in enumerate(index_to_node_id):
        node_type = node_types[index]
        if node_type in BIOMARKER_TYPES:
            biomarker_indices[node_type].append(index)
        elif node_type == "disease" and node_id in mondo_diseases:
            disease_indices.append(index)

    return {
        "index_to_node_id": index_to_node_id,
        "offsets": offsets,
        "adjacency": adjacency,
        "disease_indices": disease_indices,
        "biomarker_indices": biomarker_indices,
    }


def count_shortest_paths_to_target_set(graph_data, target_indices):
    node_count = len(graph_data["index_to_node_id"])
    distances = array("i", [-1]) * node_count
    path_counts = array("Q", [0]) * node_count
    queue = deque()

    for target_index in target_indices:
        distances[target_index] = 0
        path_counts[target_index] = 1
        queue.append(target_index)

    offsets = graph_data["offsets"]
    adjacency = graph_data["adjacency"]

    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        current_paths = path_counts[current]
        start = offsets[current]
        end = offsets[current + 1]

        for pointer in range(start, end):
            neighbor = adjacency[pointer]
            if distances[neighbor] == -1:
                distances[neighbor] = current_distance + 1
                path_counts[neighbor] = current_paths
                queue.append(neighbor)
            elif distances[neighbor] == current_distance + 1:
                path_counts[neighbor] += current_paths

    return path_counts


def build_shortest_path_counts(graph_data):
    disease_ids_in_graph = [
        graph_data["index_to_node_id"][disease_index]
        for disease_index in graph_data["disease_indices"]
    ]
    counts_by_type = {biomarker_type: {} for biomarker_type in BIOMARKER_TYPES}

    for biomarker_type in BIOMARKER_TYPES:
        path_counts = count_shortest_paths_to_target_set(
            graph_data,
            graph_data["biomarker_indices"][biomarker_type],
        )
        for disease_index in graph_data["disease_indices"]:
            disease_id = graph_data["index_to_node_id"][disease_index]
            counts_by_type[biomarker_type][disease_id] = int(path_counts[disease_index])

    return counts_by_type, disease_ids_in_graph


def load_marker_filtered_diseases(marker_edge_path, mondo_diseases):
    filtered_diseases = set()

    with marker_edge_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            node1_id = normalize_text(row.get("Node 1 ID"))
            node2_id = normalize_text(row.get("Node 2 ID"))
            node1_type = normalize_type(row.get("Node 1 Data Type"))
            node2_type = normalize_type(row.get("Node 2 Data Type"))

            if node1_type == "disease" and node2_type == "metabolite" and node1_id in mondo_diseases:
                filtered_diseases.add(node1_id)
            elif node2_type == "disease" and node1_type == "metabolite" and node2_id in mondo_diseases:
                filtered_diseases.add(node2_id)

    return filtered_diseases


def build_ranked_rows(mondo_diseases, disease_ids, disease_counts):
    rows = [
        {
            "mondo_id": disease_id,
            "disease_name": mondo_diseases[disease_id],
            "connected_edge_count": disease_counts.get(disease_id, 0),
        }
        for disease_id in disease_ids
    ]
    rows.sort(key=lambda row: (-row["connected_edge_count"], row["mondo_id"]))
    return rows


def write_rows(output_path, rows):
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["mondo_id", "disease_name", "connected_edge_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="统计每个 MONDO 疾病到四类标志物集合的最短路径条数，并输出 marker 过滤结果。"
    )
    parser.add_argument("--mondo-node", default="./disease_connect/mondo_node.csv", type=Path, help="MONDO 节点文件路径。")
    parser.add_argument("--merged-edge", default="./disease_connect/merged_edge.csv", type=Path, help="合并后的知识图谱边文件路径。")
    parser.add_argument("--marker-edge", default="./disease_connect/marker_edge.csv", type=Path, help="marker 边文件路径，用于疾病过滤。")
    parser.add_argument("--output-dir", default=".", type=Path, help="结果输出目录。")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mondo_diseases = load_mondo_diseases(args.mondo_node)
    graph_data = build_graph(args.merged_edge, mondo_diseases)
    counts_by_type, disease_ids_in_graph = build_shortest_path_counts(graph_data)
    marker_filtered_diseases = load_marker_filtered_diseases(args.marker_edge, mondo_diseases)

    for biomarker_type in BIOMARKER_TYPES:
        ranked_rows = build_ranked_rows(mondo_diseases, disease_ids_in_graph, counts_by_type[biomarker_type])
        write_rows(args.output_dir / OUTPUT_FILENAMES[biomarker_type], ranked_rows)

        filtered_rows = [row for row in ranked_rows if row["mondo_id"] in marker_filtered_diseases]
        write_rows(args.output_dir / FILTERED_OUTPUT_FILENAMES[biomarker_type], filtered_rows)

        print(
            f"{biomarker_type}: total={len(ranked_rows)}, filtered={len(filtered_rows)}, "
            f"top_count={ranked_rows[0]['connected_edge_count'] if ranked_rows else 0}"
        )


if __name__ == "__main__":
    main()