"""
analyze_mondo_node_enrichment.py

分析 MONDO 疾病节点在知识图谱中的富集程度。
统计指标：
  - total_degree          : 在全图（merged_edge.csv）中的节点度（相连边数）
  - marker_degree         : 在生物标志物子图（marker_edge.csv）中的节点度
    - markerdb_gene_edges   : MarkerDB 子图中与基因边的连通边数
    - markerdb_protein_edges: MarkerDB 子图中与蛋白边的连通边数
    - markerdb_metabolite_edges: MarkerDB 子图中与代谢物边的连通边数
    - markerdb_rna_edges    : MarkerDB 子图中与 RNA 边的连通边数
  - cc_size               : 所属连通分量的节点数（全图）
  - cc_edge_count         : 所属连通分量的边数（全图）
  - gene_conn_edges       : 与基因节点的连通边数（全量）
  - protein_conn_edges    : 与蛋白质节点的连通边数（全量）
  - metabolite_conn_edges : 与代谢物节点的连通边数（全量）
  - rna_conn_edges        : 与 RNA 节点的连通边数（全量）
  - gene_conn_edges_mk    : 与基因节点的连通边数（marker 过滤后）
  - protein_conn_edges_mk : 与蛋白质节点的连通边数（marker 过滤后）
  - metabolite_conn_edges_mk: 与代谢物节点的连通边数（marker 过滤后）
  - rna_conn_edges_mk     : 与 RNA 节点的连通边数（marker 过滤后）

输出：
  mondo_node_enrichment_stats.csv  — 按 total_degree 降序排列的完整统计表
"""

import os
import pandas as pd
import urllib.request
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DC_DIR   = os.path.join(BASE_DIR, "disease_connect")

# ─── 文件路径 ────────────────────────────────────────────────────────────────
MONDO_NODE_FILE  = os.path.join(DC_DIR, "mondo_node.csv")
MERGED_EDGE_FILE = os.path.join(DC_DIR, "merged_edge.csv")
_MARKER_EDGE_CANDIDATES = [
    os.path.join(BASE_DIR, "marker_edge.csv"),
    os.path.join(DC_DIR, "marker_edge.csv"),
]
MARKER_EDGE_FILE = next((p for p in _MARKER_EDGE_CANDIDATES if os.path.exists(p)), _MARKER_EDGE_CANDIDATES[-1])

CONN_FILES = {
    "gene":       os.path.join(DC_DIR, "disease_gene_connected_edge_counts.csv"),
    "protein":    os.path.join(DC_DIR, "disease_protein_connected_edge_counts.csv"),
    "metabolite": os.path.join(DC_DIR, "disease_metabolite_connected_edge_counts.csv"),
    "rna":        os.path.join(DC_DIR, "disease_rna_connected_edge_counts.csv"),
}
CONN_MK_FILES = {
    "gene":       os.path.join(DC_DIR, "disease_gene_connected_edge_counts_filtered_by_marker.csv"),
    "protein":    os.path.join(DC_DIR, "disease_protein_connected_edge_counts_filtered_by_marker.csv"),
    "metabolite": os.path.join(DC_DIR, "disease_metabolite_connected_edge_counts_filtered_by_marker.csv"),
    "rna":        os.path.join(DC_DIR, "disease_rna_connected_edge_counts_filtered_by_marker.csv"),
}

OUTPUT_FILE = os.path.join(BASE_DIR, "mondo_node_enrichment_stats.csv")
MONDO_OBO = os.path.join(BASE_DIR, "mondo.obo")
MONDO_OBO_URL = "https://purl.obolibrary.org/obo/mondo.obo"


def download_mondo_obo(path: str) -> None:
    print(f"下载 MONDO OBO 文件到 {path} ...")
    print("（文件约 50-100 MB，请稍候）")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  进度: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MONDO_OBO_URL, path, reporthook)
    print("\n下载完成。")


def get_leaf_disease_ids(obo_path: str, disease_ids: set) -> set:
    """返回 MONDO 中无子节点（叶节点）的疾病 ID。"""
    valid_disease_terms = set()
    disease_parents_with_children = set()

    current_id = None
    current_parents = []
    is_obsolete = False

    def commit_term():
        if not current_id or is_obsolete:
            return
        if current_id in disease_ids:
            valid_disease_terms.add(current_id)
            for pid in current_parents:
                if pid in disease_ids:
                    disease_parents_with_children.add(pid)

    print("[6/7] 解析 MONDO 结构，识别无子节点疾病...")
    with open(obo_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "[Term]":
                commit_term()
                current_id = None
                current_parents = []
                is_obsolete = False
            elif line.startswith("id: MONDO:"):
                current_id = line[4:]
            elif line.startswith("is_a: MONDO:"):
                parent_id = line.split()[1]
                current_parents.append(parent_id)
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True

    commit_term()

    leaf_ids = valid_disease_terms - disease_parents_with_children
    print(f"      MONDO 疾病术语 {len(valid_disease_terms):,} 个，叶节点疾病 {len(leaf_ids):,} 个")
    return leaf_ids

# ─── 1. 加载 MONDO 疾病节点 ──────────────────────────────────────────────────
print("[1/7] 加载 MONDO 疾病节点...")
mondo_df = pd.read_csv(MONDO_NODE_FILE, usecols=["ID", "Name"])
mondo_df.rename(columns={"ID": "mondo_id", "Name": "disease_name"}, inplace=True)
mondo_ids = set(mondo_df["mondo_id"])
print(f"      共 {len(mondo_ids):,} 个疾病节点")

# ─── 2. 加载预计算的连通边数统计文件 ─────────────────────────────────────────
print("[2/7] 加载预计算连通边数统计...")

def load_conn(path, col_suffix):
    df = pd.read_csv(path, usecols=["mondo_id", "connected_edge_count"])
    df.rename(columns={"connected_edge_count": col_suffix}, inplace=True)
    return df

conn_gene       = load_conn(CONN_FILES["gene"],       "gene_conn_edges")
conn_protein    = load_conn(CONN_FILES["protein"],    "protein_conn_edges")
conn_metabolite = load_conn(CONN_FILES["metabolite"], "metabolite_conn_edges")
conn_rna        = load_conn(CONN_FILES["rna"],        "rna_conn_edges")

conn_gene_mk       = load_conn(CONN_MK_FILES["gene"],       "gene_conn_edges_mk")
conn_protein_mk    = load_conn(CONN_MK_FILES["protein"],    "protein_conn_edges_mk")
conn_metabolite_mk = load_conn(CONN_MK_FILES["metabolite"], "metabolite_conn_edges_mk")
conn_rna_mk        = load_conn(CONN_MK_FILES["rna"],        "rna_conn_edges_mk")

# ─── 3. 从 marker_edge.csv 计算 marker_degree（仅保留权重=1.0边）──────────────
print("[3/7] 计算 marker_degree（marker 子图节点度，仅权重=1.0）...")
marker_df = pd.read_csv(MARKER_EDGE_FILE,
                        usecols=["Node 1 ID", "Node 1 Data Type",
                                 "Node 2 ID", "Node 2 Data Type", "Edge Weight"])

# 仅统计权重为 1.0 的 marker 边（兼容字符串/浮点）
marker_df = marker_df[pd.to_numeric(marker_df["Edge Weight"], errors="coerce") == 1.0]

# 统计每个 MONDO 疾病节点在 marker 子图中出现的次数（即度）
mk_degree = defaultdict(int)
marker_type_to_col = {
    "Gene": "markerdb_gene_edges",
    "Protein": "markerdb_protein_edges",
    "Metabolite": "markerdb_metabolite_edges",
    "RNA": "markerdb_rna_edges",
}
mk_type_counts = {col: defaultdict(int) for col in marker_type_to_col.values()}

for _, row in marker_df.iterrows():
    if row["Node 1 Data Type"] == "Disease" and row["Node 1 ID"] in mondo_ids:
        disease_id = row["Node 1 ID"]
        mk_degree[disease_id] += 1
        counter_col = marker_type_to_col.get(row["Node 2 Data Type"])
        if counter_col:
            mk_type_counts[counter_col][disease_id] += 1
    if row["Node 2 Data Type"] == "Disease" and row["Node 2 ID"] in mondo_ids:
        disease_id = row["Node 2 ID"]
        mk_degree[disease_id] += 1
        counter_col = marker_type_to_col.get(row["Node 1 Data Type"])
        if counter_col:
            mk_type_counts[counter_col][disease_id] += 1

mk_degree_df = pd.DataFrame(
    list(mk_degree.items()), columns=["mondo_id", "marker_degree"]
)
mk_type_df = pd.DataFrame({"mondo_id": list(mondo_ids)})
for col, counter in mk_type_counts.items():
    mk_type_df[col] = mk_type_df["mondo_id"].map(counter).fillna(0).astype(int)

print(f"      权重=1.0 的 marker 子图中涉及 {len(mk_degree_df):,} 个疾病节点")

# ─── 4. 从 merged_edge.csv 计算 total_degree 及连通分量 ─────────────────────
print("[4/7] 加载全图边表并计算 total_degree（可能需要 1-2 分钟）...")

CHUNK_SIZE = 200_000
node1_counts = defaultdict(int)   # 度计数
node2_counts = defaultdict(int)

# Union-Find（并查集）用于连通分量
parent = {}
rank   = {}

def find(x):
    if x not in parent:
        parent[x] = x
        rank[x]   = 0
    root = x
    while parent[root] != root:
        root = parent[root]
    # 路径压缩
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root

def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        rx, ry = ry, rx
    parent[ry] = rx
    if rank[rx] == rank[ry]:
        rank[rx] += 1

total_edges_in_cc = defaultdict(int)  # 每条边计入两端节点所属分量

reader = pd.read_csv(
    MERGED_EDGE_FILE,
    usecols=["Node 1 ID", "Node 1 Data Type", "Node 2 ID", "Node 2 Data Type"],
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

chunk_count = 0
for chunk in reader:
    chunk_count += 1
    if chunk_count % 10 == 0:
        print(f"      已处理 {chunk_count * CHUNK_SIZE:,} 行...")

    for n1, n2 in zip(chunk["Node 1 ID"], chunk["Node 2 ID"]):
        n1, n2 = str(n1), str(n2)
        # 度统计
        node1_counts[n1] += 1
        node2_counts[n2] += 1
        # 构建并查集
        union(n1, n2)

print(f"      全图加载完毕，共遍历约 {chunk_count * CHUNK_SIZE:,} 条边")

# 合并双端度计数
all_degree = defaultdict(int)
for node, cnt in node1_counts.items():
    all_degree[node] += cnt
for node, cnt in node2_counts.items():
    all_degree[node] += cnt

# 统计每个连通分量的节点数
print("[5/7] 统计连通分量大小...")
comp_node_count = defaultdict(int)
all_nodes = set(parent.keys())
for node in all_nodes:
    comp_node_count[find(node)] += 1

# 统计每个连通分量的边数：每条边对其根节点分量 +1
comp_edge_count = defaultdict(int)
# 重新遍历一次 merged_edge 以统计边数（按分量）—— 利用已有 parent 直接查找
reader2 = pd.read_csv(
    MERGED_EDGE_FILE,
    usecols=["Node 1 ID", "Node 2 ID"],
    chunksize=CHUNK_SIZE,
    low_memory=False,
)
for chunk in reader2:
    for n1, n2 in zip(chunk["Node 1 ID"], chunk["Node 2 ID"]):
        root = find(str(n1))
        comp_edge_count[root] += 1

# 为每个疾病节点提取连通分量指标
cc_rows = []
for mid in mondo_ids:
    if mid in parent:
        root = find(mid)
        cc_rows.append({
            "mondo_id"    : mid,
            "total_degree": all_degree.get(mid, 0),
            "cc_size"     : comp_node_count[root],
            "cc_edge_count": comp_edge_count[root],
        })
    else:
        cc_rows.append({
            "mondo_id"    : mid,
            "total_degree": 0,
            "cc_size"     : 1,
            "cc_edge_count": 0,
        })

cc_df = pd.DataFrame(cc_rows)

# ─── 6. 解析 MONDO 树并筛选无子节点疾病 ─────────────────────────────────────
if not os.path.exists(MONDO_OBO):
    download_mondo_obo(MONDO_OBO)
else:
    print(f"[6/7] 已找到本地 MONDO OBO：{MONDO_OBO}")

leaf_mondo_ids = get_leaf_disease_ids(MONDO_OBO, mondo_ids)

# ─── 7. 合并所有指标，筛选并生成排名列 ───────────────────────────────────────
print("[7/7] 合并指标并筛选叶节点疾病后生成排名...")

stats = mondo_df.copy()

for df in [cc_df, mk_degree_df, mk_type_df,
           conn_gene, conn_protein, conn_metabolite, conn_rna,
           conn_gene_mk, conn_protein_mk, conn_metabolite_mk, conn_rna_mk]:
    stats = stats.merge(df, on="mondo_id", how="left")

# 填充缺失值为 0（未出现在某子图中的节点）
fill_cols = [c for c in stats.columns if c not in ("mondo_id", "disease_name")]
stats[fill_cols] = stats[fill_cols].fillna(0).astype(int)

# 仅保留 MONDO 结构中的单疾病（无子节点）
before_leaf_filter = len(stats)
stats = stats[stats["mondo_id"].isin(leaf_mondo_ids)].copy()
print(f"      叶节点过滤：{before_leaf_filter:,} -> {len(stats):,}")

# 生成排名列（降序，名次相同时取最小名次）
rank_targets = [
    "total_degree",
    "marker_degree",
    "markerdb_gene_edges",
    "markerdb_protein_edges",
    "markerdb_metabolite_edges",
    "markerdb_rna_edges",
    "cc_size",
    "cc_edge_count",
    "gene_conn_edges",
    "protein_conn_edges",
    "metabolite_conn_edges",
    "rna_conn_edges",
    "gene_conn_edges_mk",
    "protein_conn_edges_mk",
    "metabolite_conn_edges_mk",
    "rna_conn_edges_mk",
]

for col in rank_targets:
    stats[f"rank_{col}"] = stats[col].rank(method="min", ascending=False).astype(int)

# 按 total_degree 降序输出
stats.sort_values("total_degree", ascending=False, inplace=True)
stats.reset_index(drop=True, inplace=True)

# 整理列顺序：基础信息 → 指标值 → 排名
base_cols  = ["mondo_id", "disease_name"]
metric_cols = rank_targets
rank_cols  = [f"rank_{c}" for c in rank_targets]
stats = stats[base_cols + metric_cols + rank_cols]

stats.to_csv(OUTPUT_FILE, index=False)
print(f"\n完成！结果已保存至：{OUTPUT_FILE}")
print(f"共统计 {len(stats):,} 个疾病节点\n")

# ─── 简要摘要 ─────────────────────────────────────────────────────────────────
print("=== Top 10（按 total_degree 降序）===")
preview_cols = ["mondo_id", "disease_name", "total_degree", "marker_degree",
                "markerdb_gene_edges", "markerdb_protein_edges",
                "markerdb_metabolite_edges", "markerdb_rna_edges",
                "cc_size", "gene_conn_edges", "protein_conn_edges",
                "metabolite_conn_edges", "rna_conn_edges"]
print(stats[preview_cols].head(10).to_string(index=False))
