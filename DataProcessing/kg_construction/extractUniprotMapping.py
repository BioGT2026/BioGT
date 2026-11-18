from Bio import SeqIO
import argparse
import csv
import math
from DataProcessing.kg_construction.commonSettings import *
max_pe_level = 5

def process_protein_map(protein_map_path, writer_edge):
    """加载蛋白质映射文件"""
    with open(protein_map_path, "r") as f:
        # Column 1: uniprot ID, Column 2: database, Column 3: Refseq protein ID
        reader = csv.reader(f, delimiter="\t")
        count = 0
        edge_rows = []
        for row in reader:
            if "RefSeq" in row[1]:
                edge = {
                    "Node 1 ID": row[0],
                    "Node 1 Data Type": "Protein",
                    "Node 1 Source": "Uniprot",
                    "Edge Type": "protein_protein",
                    "Edge Direction": edge_types.get("EQ"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": row[2],
                    "Node 2 Data Type": "Protein",
                    "Node 2 Source": "Refseq"
                }
                edge_rows.append(edge)
            # if "UniProtKB-ID" in row[1]:
            #     edge = {
            #         "Node 1 ID": row[0],
            #         "Node 1 Data Type": "Protein",
            #         "Node 1 Source": "Uniprot",
            #         "Edge Type": "protein_protein",
            #         "Edge Direction": edge_types.get("EQ"),
            #         "Edge Weight": 1.0,
            #         "Node 2 ID": NCBI_GENE_PREFIX + row[2],
            #         "Node 2 Data Type": "Gene",
            #         "Node 2 Source": "NCBI_Gene"
            #     }
            #     edge_rows.append(edge)
        for edge in edge_rows:
            writer_edge.writerow(edge)

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Uniprot idmapping file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV files (edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    mapping_file = args.input
    output_edge_csv = args.output[0]

    process_protein_map(mapping_file, output_edge_csv)

    print(f"数据已成功导出到 边记录：{output_edge_csv}")

if __name__ == "__main__":
    main()