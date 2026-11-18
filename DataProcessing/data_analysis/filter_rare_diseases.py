"""
从 mondo_node_enrichment_stats.csv 中筛选罕见病条目，
基于 MONDO OBO 文件中 "subset: rare" 标记。
输出文件维持原有的 total_degree 降序排序，
并保留 node enrichment 中的 MarkerDB 分类列。
"""

import csv
import os
import urllib.request

STATS_CSV = os.path.join(os.path.dirname(__file__), "mondo_node_enrichment_stats.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "mondo_rare_disease_enrichment_stats.csv")
OUTPUT_NON_RARE_CSV = os.path.join(os.path.dirname(__file__), "mondo_non_rare_disease_enrichment_stats.csv")
MONDO_OBO = os.path.join(os.path.dirname(__file__), "mondo.obo")
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


def get_rare_disease_ids(obo_path: str) -> set:
    """解析 OBO 文件，返回单一罕见病的 MONDO ID 集合。
    纳入标准：含 'subset: rare'。
    排除标准：含 'subset: disease_grouping' 或 'subset: rare_grouping'（疾病分类节点）。
    """
    rare_ids = set()
    current_id = None
    is_rare = False
    is_grouping = False

    print("解析 MONDO OBO，提取单一罕见病 ID ...")
    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                # 保存上一个 term（有 rare 且非分类节点）
                if current_id and is_rare and not is_grouping:
                    rare_ids.add(current_id)
                current_id = None
                is_rare = False
                is_grouping = False
            elif line.startswith("id: MONDO:"):
                current_id = line[4:]  # "MONDO:XXXXXXX"
            elif line.startswith("subset: rare") and not line.startswith("subset: rare_grouping"):
                is_rare = True
            elif line in ("subset: disease_grouping", "subset: rare_grouping"):
                is_grouping = True
            elif line.startswith("is_obsolete: true"):
                current_id = None  # 跳过废弃条目

    # 处理文件末尾的最后一个 term
    if current_id and is_rare and not is_grouping:
        rare_ids.add(current_id)

    print(f"  找到 {len(rare_ids)} 个单一罕见病条目（已排除分类节点）。")
    return rare_ids


def filter_rare_diseases(stats_csv: str, rare_ids: set, output_csv: str, output_non_rare_csv: str) -> None:
    """筛选罕见病与非罕见病行，分别写入输出文件，维持原排序。"""
    kept = 0
    non_rare_kept = 0
    total = 0
    markerdb_cols = [
        "markerdb_gene_edges",
        "markerdb_protein_edges",
        "markerdb_metabolite_edges",
        "markerdb_rna_edges",
    ]
    markerdb_totals = {c: 0 for c in markerdb_cols}
    available_markerdb_cols = []

    with open(stats_csv, "r", encoding="utf-8", newline="") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout, \
         open(output_non_rare_csv, "w", encoding="utf-8", newline="") as fout_non_rare:

        reader = csv.DictReader(fin)
        available_markerdb_cols = [c for c in markerdb_cols if c in (reader.fieldnames or [])]
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        non_rare_writer = csv.DictWriter(fout_non_rare, fieldnames=reader.fieldnames)
        writer.writeheader()
        non_rare_writer.writeheader()

        for row in reader:
            total += 1
            if row["mondo_id"] in rare_ids:
                writer.writerow(row)
                kept += 1
                for col in available_markerdb_cols:
                    try:
                        markerdb_totals[col] += int(row.get(col, 0) or 0)
                    except ValueError:
                        pass
            else:
                non_rare_writer.writerow(row)
                non_rare_kept += 1

    print(f"筛选完成：{total} 条 -> 保留 {kept} 条罕见病条目，{non_rare_kept} 条非罕见病条目。")
    if available_markerdb_cols:
        print("罕见病子集中的 MarkerDB 分类边总数：")
        for col in available_markerdb_cols:
            print(f"  - {col}: {markerdb_totals[col]}")
    print(f"输出文件：{output_csv}")
    print(f"非罕见病输出文件：{output_non_rare_csv}")


def main():
    # 1. 确保 OBO 文件存在
    if not os.path.exists(MONDO_OBO):
        download_mondo_obo(MONDO_OBO)
    else:
        print(f"已找到本地 MONDO OBO：{MONDO_OBO}")

    # 2. 提取罕见病 ID
    rare_ids = get_rare_disease_ids(MONDO_OBO)

    if not rare_ids:
        print("未找到任何罕见病 ID，请检查 OBO 文件内容。")
        return

    # 3. 筛选并写出结果
    filter_rare_diseases(STATS_CSV, rare_ids, OUTPUT_CSV, OUTPUT_NON_RARE_CSV)


if __name__ == "__main__":
    main()
