from Bio import SeqIO
import argparse
import csv
import math
from collections import defaultdict
from DataProcessing.kg_construction.commonSettings import *
max_pe_level = 5

human_protein_ids = set()  # 用于存储人类蛋白质的ID
edge_type_stats = defaultdict(lambda: defaultdict(int))  # 用于统计边类型数量

def process_biogrid_ppi_map(ppi_map_path, output_edge_csv):
    """加载BioGrid蛋白质相互作用映射文件"""
    edge_type_counter = defaultdict(int)  # 统计边类型
    with open(ppi_map_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        edge_rows = []
        for row in reader:
            if "Homo sapiens" not in row[2] or "Homo sapiens" not in row[3] or row[0] == "-" or row[1] == "-":
                continue
            row0Set = []
            row1Set = []
            if "|" in row[0] or "|" in row[1]:
                # Split row[0] (e.g., "P12345|P67890" → ["P12345", "P67890"])
                if "|" in row[0]:
                    row0Set = row[0].split("|")
                else:
                    row0Set = [row[0]]  # If no "|", keep original value as a single-element list
                # Split row[1] (e.g., "Q98765|Q54321" → ["Q98765", "Q54321"])
                if "|" in row[1]:
                    row1Set = row[1].split("|")
                else:
                    row1Set = [row[1]]  # If no "|", keep original value as a single-element list
            for row0 in row0Set:
                for row1 in row1Set:
                    edge = {
                        "Node 1 ID": UNIPROT_PROTEIN_PREFIX + row0,
                        "Node 1 Data Type": "Protein",
                        "Node 1 Source": "Uniprot",
                        "Edge Type": "protein_protein",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": 1.0,
                        "Node 2 ID": UNIPROT_PROTEIN_PREFIX + row1,
                        "Node 2 Data Type": "Protein",
                        "Node 2 Source": "Uniprot"
                    }
                    edge_rows.append(edge)
                    # 统计边类型
                    edge_key = f"{edge['Node 1 Data Type']}-{edge['Node 2 Data Type']}"
                    edge_type_counter[edge_key] += 1
        print(f"共找到 {len(edge_rows)} 条人类蛋白质相互作用记录。")
        # 打印BioGrid边类型统计
        print("\nBioGrid 边类型统计:")
        for edge_type, count in sorted(edge_type_counter.items()):
            print(f"  {edge_type}: {count}")
        edge_type_stats['BioGrid'] = dict(edge_type_counter)
        # 将边记录写入 CSV 文件
        with open(output_edge_csv, mode="a", newline="", encoding="utf-8") as csvfile_edge:
            writer_edge = csv.DictWriter(csvfile_edge, fieldnames=fieldnames_edge)
            for edge in edge_rows:
                writer_edge.writerow(edge)

def process_intact_ppi_map(ppi_map_path, output_edge_csv):
    """加载蛋白质相互作用映射文件"""
    edge_type_counter = defaultdict(int)  # 统计边类型
    with open(ppi_map_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        edge_rows = []
        for row in reader:
            # Remove tags before ":"
            row0 = row[0].split(":")[1] if ":" in row[0] else row[0]
            row1 = row[1].split(":")[1] if ":" in row[1] else row[1]
            # print(f"Processing row: {row0}-{row1}")
            if row0 in human_protein_ids and row1 in human_protein_ids:
                edge = {
                    "Node 1 ID": UNIPROT_PROTEIN_PREFIX + row0,
                    "Node 1 Data Type": "Protein",
                    "Node 1 Source": "Uniprot",
                    "Edge Type": "protein_protein",
                    "Edge Direction": edge_types.get("DD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": UNIPROT_PROTEIN_PREFIX + row1,
                    "Node 2 Data Type": "Protein",
                    "Node 2 Source": "Uniprot"
                }
                edge_rows.append(edge)
                # 统计边类型
                edge_key = f"{edge['Node 1 Data Type']}-{edge['Node 2 Data Type']}"
                edge_type_counter[edge_key] += 1
        print(f"共找到 {len(edge_rows)} 条人类蛋白质相互作用记录。")
        # 打印Intact边类型统计
        print("\nIntact 边类型统计:")
        for edge_type, count in sorted(edge_type_counter.items()):
            print(f"  {edge_type}: {count}")
        edge_type_stats['Intact'] = dict(edge_type_counter)
        # 将边记录写入 CSV 文件
        with open(output_edge_csv, mode="a", newline="", encoding="utf-8") as csvfile_edge:
            writer_edge = csv.DictWriter(csvfile_edge, fieldnames=fieldnames_edge)
            for edge in edge_rows:
                writer_edge.writerow(edge)

# 解析描述字段
def parse_description(description, record_id):
    # 使用 NaN 填充默认值
    function = math.nan
    organism_species = math.nan
    taxonomy_id = math.nan
    gene_name = math.nan
    protein_existence = math.nan
    sequence_version = math.nan
    
    try:
        description_parts = description.split(" OS=")
        function = description_parts[0].replace(record_id, "").strip() or math.nan
        
        os_part = description_parts[1].split(" OX=")
        organism_species = os_part[0].strip() or math.nan
        
        ox_part = os_part[1].split(" GN=")
        taxonomy_id = ox_part[0].strip() or math.nan
        
        gn_part = ox_part[1].split(" PE=")
        gene_name = gn_part[0].strip() or math.nan
        
        pe_part = gn_part[1].split(" SV=")
        protein_existence = pe_part[0].strip() or math.nan
        
        sequence_version = pe_part[1].strip() or math.nan
    except IndexError:
        # 在遇到字段缺失时，保持字段为 NaN
        pass
    
    return {
        "Function": function,
        "Organism Species": organism_species,
        "Taxonomy ID": taxonomy_id,
        "Gene Name": gene_name,
        "Protein Existence": protein_existence,
        "Sequence Version": sequence_version
    }

def split_uniprot_name_entry(entry):
    parts = entry.split('|')  # 以 '|' 分割
    if len(parts) == 3:
        db_type, accession, entry_name = parts
        return db_type, accession, entry_name
    else:
        raise ValueError("Invalid UniProt name entry format")

# 处理 FASTA 文件并写入 CSV
def process_fasta_to_csv(input_file, output_file_node, output_file_edge):

    # 打开 CSV 文件并写入表头
    with open(output_file_node, mode="w", newline="", encoding="utf-8") as csvfile_node:
        writer_node = csv.DictWriter(csvfile_node, fieldnames=fieldnames_node)
        writer_node.writeheader()
        with open (output_file_edge, mode="w", newline="", encoding="utf-8") as csvfile_edge:
            writer_edge = csv.DictWriter(csvfile_edge, fieldnames=fieldnames_edge)
            writer_edge.writeheader()
        
            # 逐条读取 FASTA 文件
            for record in SeqIO.parse(input_file, "fasta"):
                # 解析描述字段
                description_info = parse_description(record.description, record.id)
                # 拆分name entry
                db_type, accession, entry_name = split_uniprot_name_entry(record.id)

                # 构建记录信息
                comments = f"DB Type: {db_type}, Function: {description_info['Function']}"
                record_info_node = {
                    "ID": UNIPROT_PROTEIN_PREFIX+accession,
                    "Type": "Protein",
                    "Name": entry_name,
                    "Sequence": str(record.seq),
                    "Comments": comments
                }
                species = description_info.get("Organism Species", "").strip().lower()  # 去空格并转小写
                if species != "homo sapiens":
                    # 处理非人类或数据异常情况
                    if not species:
                        print("警告：物种信息缺失")
                    continue # Skip non-human records

                human_protein_ids.add(accession)
                # 将记录写入 CSV 文件
                writer_node.writerow(record_info_node)

                # Check if gene_name is NaN
                if description_info.get("Gene Name") is None or isinstance(description_info["Gene Name"], float) and math.isnan(description_info["Gene Name"]):
                    # print(f"Gene Name is NaN for {record.id}. Skipping edge creation.")
                    continue
                if description_info.get("Protein Existence") is None or isinstance(description_info["Protein Existence"], float) and math.isnan(description_info["Protein Existence"]):
                    print(f"Protein Existence is NaN for {record.id}. Skipping edge creation.")
                    continue
        
                protein_existence = float(description_info.get("Protein Existence")) - 1.0

                protein_existence_edge_weight = 1.0 - (protein_existence / max_pe_level)  # Ensure float division

                record_info_edge = {
                    "Node 1 ID": NCBI_GENE_PREFIX + description_info.get("Gene Name"),  # Default to avoid KeyError
                    "Node 1 Data Type": "Gene",
                    "Node 1 Source": "NCBI",
                    "Edge Type": "gene_protein", # gene -> protein edge
                    "Edge Direction": edge_types.get("SD"),
                    "Edge Weight": protein_existence_edge_weight,
                    "Node 2 ID": UNIPROT_PROTEIN_PREFIX + accession,
                    "Node 2 Data Type": "Protein",
                    "Node 2 Source": "Uniprot"
                }

                # 将记录写入 CSV 文件
                writer_edge.writerow(record_info_edge)

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Uniprot FASTA file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-ip', '--intactppi', type=str, required=True, help="Input Intact ppi file")
    parser.add_argument('-bp', '--biogridppi', type=str, required=True, help="Input BioGrid ppi file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    fasta_file = args.input
    intact_ppi_file = args.intactppi
    biogrid_ppi_file = args.biogridppi
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]

    # 处理 FASTA 文件并写入 CSV
    print(f"正在处理 Uniprot FASTA 文件：{fasta_file} ...")
    process_fasta_to_csv(fasta_file, output_node_csv, output_edge_csv)
    print(f"Total human proteins processed: {len(human_protein_ids)}")
    print(f"正在处理 Intact PPI 文件：{intact_ppi_file} ...")
    process_intact_ppi_map(intact_ppi_file, output_edge_csv)
    print(f"正在处理 BioGrid PPI 文件：{biogrid_ppi_file} ...")
    process_biogrid_ppi_map(biogrid_ppi_file, output_edge_csv)
    
    # 打印总体边类型统计
    print("\n" + "="*50)
    print("总体边类型统计汇总:")
    print("="*50)
    for source in ['Intact', 'BioGrid']:
        if source in edge_type_stats:
            print(f"\n{source}:")
            for edge_type, count in sorted(edge_type_stats[source].items()):
                print(f"  {edge_type}: {count}")
    print("="*50)
    
    print(f"数据已成功导出到 节点记录：{output_node_csv}, 边记录：{output_edge_csv}")

if __name__ == "__main__":
    main()