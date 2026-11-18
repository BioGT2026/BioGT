"""
给定入口节点 ID、出口节点 ID 和最大跳数，找出在该跳数约束下
能够构成“入口 -> 出口”路径的所有可达边，并输出到日志文件。

默认边文件：disease_connect/merged_edge.csv

示例：
Parkinson disease, mitochondrial (MONDO:0010796) 到 APOD蛋白 (Uniprot:P05090) 的直接边：
python3 find_reachable_edges_within_hops.py \
  --source-id MONDO:0010796 \
  --target-id Uniprot:P02647 \
  --max-hops 5 \
  --require-drug-node

潜在：Uniprot:P02647 (APOA1_HUMA)
潜在：Uniprot:P05090 (APOD_HUMAN)
潜在：Uniprot:Q99497 (DJ-1)
潜在：Uniprot:P37840 (SNCA_HUMAN)

--- Top 1 Predicted Associations for 'Parkinson disease, mitochondrial' (disease) ---
--- Top 1 for type: protein ---
1. ID: Uniprot:P05090, Name: APOD_HUMAN (protein) - Score: 0.9843631386756897 [dominant: type:mixed(hgcn=0.40, transformer=0.60)] Path: disease:1165 (Parkinson disease, mitochondrial) -> drug:797 (Fyarro|Gd-sirolimus|Hyftor|Rapamune|Sirolimus) -> protein:897 (APOD_HUMAN)
"""

import argparse
import csv
import os
from collections import defaultdict, deque
from datetime import datetime


DEFAULT_EDGE_FILE = os.path.join(
    os.path.dirname(__file__), "/home/$USER/dataset/graphWithNodeType/mergeNew", "merged_edge.csv"
    # os.path.dirname(__file__), "disease_connect", "merged_edge.csv"
)


def sanitize_filename_part(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find all edges that can lie on source->target paths within max hops."
    )
    parser.add_argument("--source-id", required=True, help="入口节点 ID")
    parser.add_argument("--target-id", required=True, help="出口节点 ID")
    parser.add_argument("--max-hops", required=True, type=int, help="最大跳数（>=0）")
    parser.add_argument(
        "--edge-file",
        default=DEFAULT_EDGE_FILE,
        help="边文件路径，默认 disease_connect/merged_edge.csv",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="按 Edge Direction 作为有向图处理；默认按无向图处理",
    )
    parser.add_argument(
        "--output-log",
        default=None,
        help="输出日志文件路径；不传则自动按 source/target/hops 生成",
    )
    parser.add_argument(
        "--require-drug-node",
        action="store_true",
        help="仅输出路径中包含药物节点的路径",
    )
    return parser.parse_args()


def get_oriented_arcs(node1: str, node2: str, edge_direction: str, directed: bool):
    """返回该边在图中的有向弧列表 (src, dst)。"""
    if not directed:
        return [(node1, node2), (node2, node1)]

    d = (edge_direction or "").strip().lower()
    if "double" in d or "bi" in d:
        return [(node1, node2), (node2, node1)]
    if "reverse" in d:
        return [(node2, node1)]
    if "single" in d or "direct" in d or "forward" in d:
        return [(node1, node2)]

    # 未知方向时，保守处理为双向，避免漏边。
    return [(node1, node2), (node2, node1)]


def bfs_dist(graph, start: str, max_hops: int):
    dist = {start: 0}
    q = deque([start])

    while q:
        u = q.popleft()
        if dist[u] >= max_hops:
            continue
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def build_graph_and_edges(edge_file: str, directed: bool):
    required_cols = [
        "Node 1 ID",
        "Node 2 ID",
        "Edge Direction",
        "Edge Type",
        "Edge Weight",
        "Node 1 Data Type",
        "Node 2 Data Type",
        "Node 1 Source",
        "Node 2 Source",
    ]

    edges = []
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    node_type_map = {}

    with open(edge_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"边文件缺少必要列: {missing}")

        for row in reader:
            n1 = row["Node 1 ID"]
            n2 = row["Node 2 ID"]
            node_type_map[n1] = row.get("Node 1 Data Type", "")
            node_type_map[n2] = row.get("Node 2 Data Type", "")
            direction = row.get("Edge Direction", "")
            arcs = get_oriented_arcs(n1, n2, direction, directed)

            edge_idx = len(edges)
            edges.append(row)

            for src, dst in arcs:
                graph[src].append((dst, edge_idx))
                reverse_graph[dst].append((src, edge_idx))

    return edges, graph, reverse_graph, node_type_map


def find_reachable_path_edges(source_id: str, target_id: str, max_hops: int, edges, graph, reverse_graph):
    out_graph = defaultdict(list)
    rev_graph = defaultdict(list)

    for u, lst in graph.items():
        out_graph[u] = [v for v, _ in lst]
    for u, lst in reverse_graph.items():
        rev_graph[u] = [v for v, _ in lst]

    dist_from_source = bfs_dist(out_graph, source_id, max_hops)
    dist_to_target = bfs_dist(rev_graph, target_id, max_hops)

    min_hops = dist_from_source.get(target_id)

    selected_edge_indices = set()
    valid_edge_arcs = set()
    for u, outs in graph.items():
        du = dist_from_source.get(u)
        if du is None or du >= max_hops:
            continue

        for v, edge_idx in outs:
            dv = dist_to_target.get(v)
            if dv is None:
                continue
            if du + 1 + dv <= max_hops:
                # 约束1：距离源点要么严格增加，要么相等但距离目标减少
                # 约束2：距离目标必须减少
                dsv = dist_from_source.get(v)
                dtu = dist_to_target.get(u)
                if dsv is None or dtu is None:
                    continue
                
                # 核心约束：防止环路和回退
                # - 如果 dsv > du：允许（向前进）
                # - 如果 dsv == du：必须 dtu > dv（在相同距离的多条路上，必须朝向目标）
                # - 如果 dsv < du：不允许（向回走）
                
                if dsv < du:
                    continue  # 向源点方向走，不允许
                if dsv == du and dtu <= dv:
                    continue  # 在相同源距离下，必须朝向目标
                
                selected_edge_indices.add(edge_idx)
                valid_edge_arcs.add((u, v, edge_idx))

    return {
        "selected_edge_indices": selected_edge_indices,
        "dist_from_source": dist_from_source,
        "dist_to_target": dist_to_target,
        "min_hops": min_hops,
        "valid_edge_arcs": sorted(valid_edge_arcs),
    }


def build_path_adjacency(valid_edge_arcs):
    path_graph = defaultdict(list)
    for u, v, _ in valid_edge_arcs:
        path_graph[u].append(v)

    return dict(path_graph)


def count_complete_paths(path_graph, source_id: str, target_id: str, max_depth: int = 100):
    """Count paths with cycle detection."""
    memo = {}
    visiting = set()

    def dfs(node, depth=0):
        if depth > max_depth:  # Prevent infinite recursion
            return 0
        if node == target_id:
            return 1
        if node in visiting:  # Cycle detected
            return 0
        if node in memo:
            return memo[node]

        visiting.add(node)
        total = 0
        for nxt in path_graph.get(node, []):
            total += dfs(nxt, depth + 1)
        visiting.remove(node)
        memo[node] = total
        return total

    return dfs(source_id)


def format_node(node_id: str, node_type_map: dict) -> str:
    node_type = node_type_map.get(node_id, "Unknown")
    return f"{node_type}（{node_id}）"


def write_log(output_log: str, args: argparse.Namespace, edges, analysis_result, node_type_map):
    selected_indices = sorted(analysis_result["selected_edge_indices"])
    dist_from_source = analysis_result["dist_from_source"]
    valid_edge_arcs = analysis_result["valid_edge_arcs"]
    path_graph = build_path_adjacency(valid_edge_arcs)
    complete_path_count = count_complete_paths(path_graph, args.source_id, args.target_id)
    unique_paths = set()
    displayed_paths = []

    with open(output_log, "w", encoding="utf-8", newline="") as f:
        f.write("=== Reachable Edges Within Hops ===\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"edge_file: {os.path.abspath(args.edge_file)}\n")
        f.write(f"source_id: {args.source_id}\n")
        f.write(f"target_id: {args.target_id}\n")
        f.write(f"max_hops: {args.max_hops}\n")
        f.write(f"directed_mode: {args.directed}\n")
        f.write(f"require_drug_node: {args.require_drug_node}\n")
        f.write(f"source_reachable_nodes_within_hops: {len(dist_from_source)}\n")

        def has_drug_node(path_nodes):
            """检查路径中是否包含药物节点."""
            for node_id in path_nodes:
                node_type = node_type_map.get(node_id, "").lower()
                if "drug" in node_type:
                    return True
            return False

        def collect_paths(node, path_nodes):
            if node == args.target_id:
                path_key = tuple(path_nodes)
                if path_key not in unique_paths:
                    unique_paths.add(path_key)

                    has_drug = has_drug_node(path_nodes)
                    if args.require_drug_node and not has_drug:
                        return

                    displayed_paths.append((list(path_nodes), has_drug))
                return

            for nxt in path_graph.get(node, []):
                collect_paths(nxt, path_nodes + [nxt])

        if complete_path_count > 0:
            collect_paths(args.source_id, [args.source_id])

        displayed_min_hops = min((len(path_nodes) - 1 for path_nodes, _ in displayed_paths), default=None)

        if displayed_min_hops is None:
            f.write("target_reachable_within_hops: False\n")
            f.write("shortest_hops_within_limit: N/A\n")
        else:
            f.write("target_reachable_within_hops: True\n")
            f.write(f"shortest_hops_within_limit: {displayed_min_hops}\n")

        f.write(f"selected_edge_count: {len(selected_indices)}\n")
        f.write(f"path_edge_count: {len(valid_edge_arcs)}\n")
        f.write(f"complete_path_count: {complete_path_count}\n")
        f.write("\n")

        if displayed_min_hops is not None:
            shortest_paths = [
                path_nodes
                for path_nodes, _ in displayed_paths
                if len(path_nodes) - 1 == displayed_min_hops
            ]

            if shortest_paths:
                f.write(
                    f"=== Complete Paths For shortest_hops_within_limit ({displayed_min_hops} hops) ===\n"
                )
                for shortest_path_index, path_nodes in enumerate(shortest_paths, start=1):
                    f.write(
                        "最短路径{}: {}\n".format(
                            shortest_path_index,
                            " -> ".join(format_node(path_node, node_type_map) for path_node in path_nodes),
                        )
                    )
                f.write("\n")

        f.write("=== Complete Source->Target Paths ===\n")

        path_index = 0
        paths_with_drug = 0
        for path_nodes, has_drug in displayed_paths:
            path_index += 1
            if has_drug:
                paths_with_drug += 1

            f.write(
                "路径{}: {}\n".format(
                    path_index,
                    " -> ".join(format_node(path_node, node_type_map) for path_node in path_nodes),
                )
            )

        f.write("\n")
        f.write(f"deduplicated_complete_path_count: {len(displayed_paths)}\n")
        if args.require_drug_node:
            f.write(f"paths_containing_drug_node: {path_index}\n")
        if not args.require_drug_node and path_index > 0:
            f.write(f"paths_containing_drug_node: {paths_with_drug}\n")

    return {
        "complete_path_count": complete_path_count,
        "displayed_min_hops": displayed_min_hops,
        "displayed_path_count": len(displayed_paths),
    }


def main():
    args = parse_args()

    if args.max_hops < 0:
        raise ValueError("--max-hops 必须 >= 0")

    if not os.path.exists(args.edge_file):
        raise FileNotFoundError(f"找不到边文件: {args.edge_file}")

    if args.output_log:
        output_log = args.output_log
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = sanitize_filename_part(args.source_id)
        dst = sanitize_filename_part(args.target_id)
        output_log = os.path.join(
            os.path.dirname(__file__),
            f"reachable_edges_{src}_to_{dst}_h{args.max_hops}_{ts}.log",
        )

    print("[1/3] 加载边并构建图结构...")
    edges, graph, reverse_graph, node_type_map = build_graph_and_edges(args.edge_file, args.directed)
    print(f"      边总数: {len(edges):,}")

    print("[2/3] 计算指定跳数内可达路径边...")
    result = find_reachable_path_edges(
        args.source_id,
        args.target_id,
        args.max_hops,
        edges,
        graph,
        reverse_graph,
    )

    print("[3/3] 写出日志...")
    log_summary = write_log(output_log, args, edges, result, node_type_map)

    min_hops = log_summary["displayed_min_hops"]
    if min_hops is None:
        print("      在给定跳数内，入口到出口不可达。")
    else:
        print(f"      在给定跳数内可达，最短跳数: {min_hops}")

    print(f"      可达路径边数量: {len(result['selected_edge_indices']):,}")
    print(f"      source->target 路径上的边条目数: {len(result['valid_edge_arcs']):,}")
    print(f"      完整路径数量: {log_summary['complete_path_count']:,}")
    if args.require_drug_node:
        print(f"      [过滤模式] 仅输出包含药物节点的路径")
    print("      仅在最终输出阶段对完整路径去重")
    print(f"完成，日志已输出: {output_log}")


if __name__ == "__main__":
    main()
