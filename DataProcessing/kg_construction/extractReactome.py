import argparse
import csv
import math
import os
from DataProcessing.kg_construction.commonSettings import *

validPathways = {}

nodes = {}
edges = {}
pathway_pathway_count = 0
pathway_protein_count = 0
pathway_gene_count = 0

def parse_description_pathways(description):
    # 使用 NaN 填充默认值
    reactome_pathway_id = math.nan
    reactome_pathway_name = math.nan
    species = math.nan
    
    try:
        reactome_pathway_id = description[0].strip() if description[0] else math.nan
        reactome_pathway_name = description[1].strip() if description[1] else math.nan
        species = description[2].strip() if description[2] else math.nan

    except IndexError:
        # 在遇到字段缺失时，保持字段为 NaN
        pass
    
    return {
        "Reactome Pathway": reactome_pathway_id,
        "Reactome Pathway Name": reactome_pathway_name,
        "Species": species
    }

def parse_description_protein(description):
    # 使用 NaN 填充默认值
    source_db_identifier = math.nan
    reactome_pathway_id = math.nan
    url = math.nan
    event_name = math.nan
    evidence_code = math.nan
    species = math.nan
    
    try:
        source_db_identifier = description[0].strip() if description[0] else math.nan
        reactome_pathway_id = description[1].strip() if description[1] else math.nan
        url = description[2].strip() if description[2] else math.nan
        event_name = description[3].strip() if description[3] else math.nan
        evidence_code = description[4].strip() if description[4] else math.nan
        species = description[5].strip() if description[5] else math.nan

    except IndexError:
        # 在遇到字段缺失时，保持字段为 NaN
        pass
    
    return {
        "Protein Name": source_db_identifier,
        "Reactome Pathway": reactome_pathway_id,
        "URL": url,
        "Event Name": event_name,
        "Evidence Code": evidence_code,
        "Species": species
    }

def parse_description_gene(description):
    # 使用 NaN 填充默认值
    source_db_identifier = math.nan
    reactome_pathway_id = math.nan
    url = math.nan
    event_name = math.nan
    evidence_code = math.nan
    species = math.nan
    
    try:
        source_db_identifier = description[0].strip() if description[0] else math.nan
        reactome_pathway_id = description[1].strip() if description[1] else math.nan
        url = description[2].strip() if description[2] else math.nan
        event_name = description[3].strip() if description[3] else math.nan
        evidence_code = description[4].strip() if description[4] else math.nan
        species = description[5].strip() if description[5] else math.nan

    except IndexError:
        # 在遇到字段缺失时，保持字段为 NaN
        pass
    
    return {
        "Gene ID": source_db_identifier,
        "Reactome Pathway": reactome_pathway_id,
        "URL": url,
        "Event Name": event_name,
        "Evidence Code": evidence_code,
        "Species": species
    }

# 处理 Reactome 文件并写入 CSV
def process_reactome_protein_to_csv(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            description_info = parse_description_protein(row)
            species = description_info.get("Species", "").strip().lower()  # 去空格并转小写
            if species != "homo sapiens":
                # 处理非人类或数据异常情况
                if not species:
                    print("警告：物种信息缺失")
                continue # Skip non-human records

            # Check if Protein Name is NaN
            if description_info.get("Protein Name") is None or isinstance(description_info["Protein Name"], float) and math.isnan(description_info["Protein Name"]):
                path_way = description_info.get("Reactome Pathway")
                print(f"Protein Name is NaN {description_info} for {path_way}. Skipping edge creation.")
                continue
            
            current_weight = evidence_to_weight(description_info.get("Evidence Code", ""))
            if current_weight == 0.0:
                continue
            record_info_edge = {
                "Node 1 ID": UNIPROT_PROTEIN_PREFIX+description_info.get("Protein Name"),
                "Node 1 Data Type": "Protein",
                "Node 1 Source": "Uniprot",
                "Edge Type": "protein_pathway",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": current_weight,
                "Node 2 ID": description_info.get("Reactome Pathway"),
                "Node 2 Data Type": "Pathway",
                "Node 2 Source": "Reactome"
            }
            edges[record_info_edge["Node 1 ID"] + record_info_edge["Node 2 ID"]] = record_info_edge
            global pathway_protein_count
            pathway_protein_count += 1

def process_reactome_gene_to_csv(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            description_info = parse_description_gene(row)
            species = description_info.get("Species", "").strip().lower()  # 去空格并转小写
            if species != "homo sapiens":
                # 处理非人类或数据异常情况
                if not species:
                    print("警告：物种信息缺失")
                continue # Skip non-human records

            # Check if Gene ID is NaN
            if description_info.get("Gene ID") is None or isinstance(description_info["Gene ID"], float) and math.isnan(description_info["Gene ID"]):
                path_way = description_info.get("Reactome Pathway")
                print(f"Gene ID is NaN for {path_way}. Skipping edge creation.")
                continue
            
            current_weight = evidence_to_weight(description_info.get("Evidence Code", ""))
            if current_weight == 0.0:
                continue
            record_info_edge = {
                "Node 1 ID": NCBI_GENE_PREFIX+description_info.get("Gene ID"),
                "Node 1 Data Type": "Gene",
                "Node 1 Source": "NCBI",
                "Edge Type": "gene_pathway",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": current_weight,
                "Node 2 ID": description_info.get("Reactome Pathway"),
                "Node 2 Data Type": "Pathway",
                "Node 2 Source": "Reactome"
            }
            edges[record_info_edge["Node 1 ID"] + record_info_edge["Node 2 ID"]] = record_info_edge
            global pathway_gene_count
            pathway_gene_count += 1

def process_reactome_to_csv(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            description_info = parse_description_pathways(row)
            species = description_info.get("Species", "").strip().lower()  # 去空格并转小写
            if species != "homo sapiens":
                # 处理非人类或数据异常情况
                if not species:
                    print("警告：物种信息缺失")
                continue # Skip non-human records

            record_info_node = {
                "ID": description_info.get("Reactome Pathway", ""),
                "Type": "Pathway",
                "Name": description_info.get("Reactome Pathway Name", ""),
                "Sequence": "",
                "Comments": ""
            }
            nodes[record_info_node.get("ID")] = record_info_node

def process_reactome_relations_to_csv(input_file):
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            original_pathway = row[0]
            target_pathway = row[1]
            if target_pathway not in nodes or original_pathway not in nodes:
                continue
            record_info_edge = {
                "Node 1 ID": original_pathway,
                "Node 1 Data Type": "Pathway",
                "Node 1 Source": "Reactome",
                "Edge Type": "pathway_pathway",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": target_pathway,
                "Node 2 Data Type": "Pathway",
                "Node 2 Source": "Reactome"
            }
            edges[record_info_edge["Node 1 ID"] + record_info_edge["Node 2 ID"]] = record_info_edge
            global pathway_pathway_count 
            pathway_pathway_count += 1

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    reactome_files_path = args.input
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]
    reactome_files = os.listdir(reactome_files_path)
    if not reactome_files:
        print("没有找到 Reactome 文件，请检查输入路径。")
        return
    for reactome_file in reactome_files:
        reactome_file = os.path.join(reactome_files_path, reactome_file)
        if not os.path.isfile(reactome_file):
            print(f"文件 {reactome_file} 不是一个有效的文件，跳过。")
            continue
        print(f"正在处理文件：{reactome_file}")
        if "ReactomePathways.txt" in reactome_file:
            # 处理 Reactome Pathways 文件
            process_reactome_to_csv(reactome_file)
        elif "ReactomePathwaysRelation.txt" in reactome_file:
            # 处理 Reactome Protein Pathways 文件
            process_reactome_relations_to_csv(reactome_file)
        elif "NCBI2Reactome.txt" in reactome_file:
            # 处理 Reactome Pathways 文件
            process_reactome_gene_to_csv(reactome_file)
        elif "UniProt2Reactome.txt" in reactome_file:
            # 处理 Reactome Gene Pathways 文件
            process_reactome_protein_to_csv(reactome_file)
        else:
            print(f"文件 {reactome_file} 不符合预期格式，跳过。")
            continue

    with open(output_node_csv, "w", newline="") as file_node, open(output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)

        writer_node.writeheader()
        writer_edge.writeheader()
        for node in nodes.values():
            writer_node.writerow(node)
        for edge in edges.values():
            writer_edge.writerow(edge)

    print(f"数据已成功导出到 节点记录：{output_node_csv}, 共有 {len(nodes)}; 边记录：{output_edge_csv}, 共有 {len(edges)}. Pathway-Pathway: {pathway_pathway_count}, Pathway-Protein: {pathway_protein_count}, Pathway-Gene: {pathway_gene_count}.")

if __name__ == "__main__":
    main()