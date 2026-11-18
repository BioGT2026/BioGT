from collections import defaultdict
import argparse
import re
import os
import csv
import numpy as np
import pandas as pd
from DataProcessing.kg_construction.commonSettings import *

# Only creat MAXO nodes and MAXO-HPO + MAXO-MONDO edges
maxo_fields = ['disease_id', 'disease_name', 'source_id', 'maxo_id',	'maxo_name', 'hpo_id', 'relation', 'evidence', 'extension_id', 'extension_name', 'comment', 'other', 'author', 'last_updated', 'created']
gene_disease_fields = ['ncbi_gene_id', 'gene_symbol', 'association_type', 'disease_id', 'source']
gene_phenotype_fields = ['ncbi_gene_id', 'gene_symbol', 'hpo_id', 'hpo_name', 'frequency', 'disease_id']
phenotype_to_genes_fields = ['hpo_id', 'hpo_name', 'ncbi_gene_id', 'gene_symbol', 'disease_id']
filedNameForDiseaseMap = ['MONDO ID', 'MONDO Name', 'OMIM ID']


def process_hpoa_file(file_path):

    # reading annotations file 
    with open(file_path, 'r') as f: 
        data = f.readlines()

    data_str = ''
    for line in data[4:]: 
        data_str += line 

    with open('./phenotype_formatted.hpoa', 'w') as f:
        f.write(data_str)

    print("HPOA文件已处理并保存为 'phenotype_formatted.hpoa'")
    df = pd.read_csv('./phenotype_formatted.hpoa', sep='\t', low_memory=False)

    # Extract database_id, hpo_id, qualifier, and evidence
    data_map = df[['database_id', 'hpo_id', 'qualifier', 'evidence']].copy()
    print(f"提取了数据库ID、HPO ID、限定符和证据列。总数量: {len(data_map)}")
    return data_map


# 处理CSV
def process_to_csv(maxo_file, genes_to_disease_file, genes_to_phenotype_file, phenotype_to_genes_file, omim_to_mondo, hpoa_map, output_file_edge):
    skipped_count = 0
    output_count = 0
    written_edges = set()

    with open(output_file_edge, "w", newline="", encoding='utf-8') as file_edge:
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_edge.writeheader()

        # Process maxo_file for MAXO-HPO and MAXO-Disease edges
        with open(maxo_file, 'r', newline='', encoding='utf-8') as f:
            lines = (line for line in f if not line.startswith('#'))
            reader = csv.DictReader(lines, fieldnames=maxo_fields, delimiter='\t')
            for row in reader:
                maxo_id = row.get('maxo_id', '').strip()
                hpo_id = row.get('hpo_id', '').strip()
                disease_id_str = row.get('disease_id', '').strip()

                if not maxo_id:
                    skipped_count += 1
                    continue

                # HP-Disease edge
                if disease_id_str:
                    mondo_ids = []
                    if disease_id_str.startswith('OMIM:'):
                        omim_id = disease_id_str
                        if omim_id in omim_to_mondo:
                            mondo_ids = omim_to_mondo[omim_id]
                        else:
                            skipped_count += 1
                            continue
                    elif disease_id_str.startswith('MONDO:'):
                        mondo_ids.append(disease_id_str)
                    else:
                        skipped_count += 1
                        continue
                    
                    for mondo_id in mondo_ids:
                        edge_tuple = tuple(sorted((maxo_id, mondo_id))) + ('maxo_disease',)
                        if edge_tuple not in written_edges:
                            if hpo_id.startswith('HP:'):
                                edge_data = {
                                    "Node 1 ID": hpo_id,
                                    "Node 1 Data Type": "Phenotype",
                                    "Node 1 Source": "HPO",
                                    "Edge Type": "phenotype_disease",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": mondo_id,
                                    "Node 2 Data Type": "Disease",
                                    "Node 2 Source": "MONDO"
                                }
                                writer_edge.writerow(edge_data)
                                output_count += 1
                                written_edges.add(edge_tuple)
                            elif hpo_id.startswith('MONDO:'):
                                edge_data = {
                                    "Node 1 ID": hpo_id,
                                    "Node 1 Data Type": "Disease",
                                    "Node 1 Source": "MONDO",
                                    "Edge Type": "disease_disease",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": mondo_id,
                                    "Node 2 Data Type": "Disease",
                                    "Node 2 Source": "MONDO"
                                }
                                writer_edge.writerow(edge_data)
                                output_count += 1
                                written_edges.add(edge_tuple)
                            else:
                                # If hpo_id is not a valid HPO or MONDO ID, skip
                                skipped_count += 1
                                continue

        # Process genes_to_disease_file
        with open(genes_to_disease_file, 'r', newline='', encoding='utf-8') as f:
            lines = (line for line in f if not line.startswith('#'))
            reader = csv.DictReader(lines, fieldnames=gene_disease_fields, delimiter='\t')
            for row in reader:
                disease_id_str = row.get('disease_id', '').strip()
                ncbi_gene_id = row.get('ncbi_gene_id', '').strip()

                if not disease_id_str or not ncbi_gene_id:
                    skipped_count += 1
                    continue

                mondo_ids = []
                if disease_id_str.startswith('OMIM:'):
                    omim_id = disease_id_str
                    if omim_id in omim_to_mondo:
                        mondo_ids = omim_to_mondo[omim_id]
                    else:
                        skipped_count += 1
                        continue
                elif disease_id_str.startswith('MONDO:'):
                    mondo_ids.append(disease_id_str)
                else:
                    skipped_count += 1
                    continue

                for mondo_id in mondo_ids:
                    gene_id_prefixed = f"{ncbi_gene_id}"
                    edge_tuple = tuple(sorted((gene_id_prefixed, mondo_id))) + ('gene_disease',)
                    if edge_tuple not in written_edges:
                        edge_data = {
                            "Node 1 ID": gene_id_prefixed,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_disease",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": 1.0,
                            "Node 2 ID": mondo_id,
                            "Node 2 Data Type": "Disease",
                            "Node 2 Source": "MONDO"
                        }
                        writer_edge.writerow(edge_data)
                        output_count += 1
                        written_edges.add(edge_tuple)

        # Process genes_to_phenotype_file for phenotype-disease and gene-phenotype edges
        with open(genes_to_phenotype_file, 'r', newline='', encoding='utf-8') as f:
            lines = (line for line in f if not line.startswith('#'))
            reader = csv.DictReader(lines, fieldnames=gene_phenotype_fields, delimiter='\t')
            for row in reader:
                hpo_id = row.get('hpo_id', '').strip()
                disease_id_str = row.get('disease_id', '').strip()
                ncbi_gene_id = row.get('ncbi_gene_id', '').strip()

                if not (ncbi_gene_id or hpo_id or disease_id_str):
                    skipped_count += 1
                    continue

                # Gene-Phenotype edge
                if ncbi_gene_id and hpo_id and hpo_id.startswith('HP:'):
                    gene_id_prefixed = f"NCBIGene:{ncbi_gene_id}"
                    edge_tuple = tuple(sorted((gene_id_prefixed, hpo_id))) + ('gene_phenotype',)
                    if edge_tuple not in written_edges:
                        edge_data = {
                            "Node 1 ID": gene_id_prefixed,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_phenotype",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": 1.0,
                            "Node 2 ID": hpo_id,
                            "Node 2 Data Type": "Phenotype",
                            "Node 2 Source": "HPO"
                        }
                        writer_edge.writerow(edge_data)
                        output_count += 1
                        written_edges.add(edge_tuple)

                # Phenotype-Disease or Disease-Disease edge
                if hpo_id and disease_id_str:
                    # Get MONDO IDs for the second part (disease_id_str)
                    mondo_ids2 = []
                    if disease_id_str.startswith('OMIM:'):
                        omim_id = disease_id_str
                        if omim_id in omim_to_mondo:
                            mondo_ids2 = omim_to_mondo[omim_id]
                        else:
                            skipped_count += 1
                            continue
                    elif disease_id_str.startswith('MONDO:'):
                        mondo_ids2.append(disease_id_str)
                    else:
                        skipped_count += 1
                        continue
                    
                    # Case 1: hpo_id is a Phenotype
                    if hpo_id.startswith('HP:'):
                        for mondo_id2 in mondo_ids2:
                            edge_tuple = tuple(sorted((hpo_id, mondo_id2))) + ('phenotype_disease',)
                            if edge_tuple not in written_edges:
                                edge_data = {
                                    "Node 1 ID": hpo_id,
                                    "Node 1 Data Type": "Phenotype",
                                    "Node 1 Source": "HPO",
                                    "Edge Type": "phenotype_disease",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": mondo_id2,
                                    "Node 2 Data Type": "Disease",
                                    "Node 2 Source": "MONDO"
                                }
                                writer_edge.writerow(edge_data)
                                output_count += 1
                                written_edges.add(edge_tuple)
                    
                    # Case 2: hpo_id is a Disease
                    else:
                        mondo_ids1 = []
                        if hpo_id.startswith('OMIM:'):
                            omim_id = hpo_id
                            if omim_id in omim_to_mondo:
                                mondo_ids1 = omim_to_mondo[omim_id]
                            else:
                                skipped_count += 1
                                continue
                        elif hpo_id.startswith('MONDO:'):
                            mondo_ids1.append(hpo_id)
                        else:
                            # It's neither HP, OMIM, nor MONDO. Skip.
                            skipped_count += 1
                            continue
                        
                        for mondo_id1 in mondo_ids1:
                            for mondo_id2 in mondo_ids2:
                                if mondo_id1 == mondo_id2:
                                    continue
                                edge_tuple = tuple(sorted((mondo_id1, mondo_id2))) + ('disease_disease',)
                                if edge_tuple not in written_edges:
                                    edge_data = {
                                        "Node 1 ID": mondo_id1,
                                        "Node 1 Data Type": "Disease",
                                        "Node 1 Source": "MONDO",
                                        "Edge Type": "disease_disease",
                                        "Edge Direction": edge_types.get("DD"),
                                        "Edge Weight": 1.0,
                                        "Node 2 ID": mondo_id2,
                                        "Node 2 Data Type": "Disease",
                                        "Node 2 Source": "MONDO"
                                    }
                                    writer_edge.writerow(edge_data)
                                    output_count += 1
                                    written_edges.add(edge_tuple)

        # Process phenotype_to_genes_file for gene-phenotype edges
        with open(phenotype_to_genes_file, 'r', newline='', encoding='utf-8') as f:
            lines = (line for line in f if not line.startswith('#'))
            reader = csv.DictReader(lines, fieldnames=phenotype_to_genes_fields, delimiter='\t')
            for row in reader:
                hpo_id = row.get('hpo_id', '').strip()
                ncbi_gene_id = row.get('ncbi_gene_id', '').strip()

                if not ncbi_gene_id or not hpo_id:
                    skipped_count += 1
                    continue
                
                gene_id_prefixed = f"NCBIGene:{ncbi_gene_id}"
                edge_tuple = tuple(sorted((gene_id_prefixed, hpo_id))) + ('gene_phenotype',)
                if edge_tuple not in written_edges:
                    edge_data = {
                        "Node 1 ID": gene_id_prefixed,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_phenotype",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": 1.0,
                        "Node 2 ID": hpo_id,
                        "Node 2 Data Type": "Phenotype",
                        "Node 2 Source": "HPO"
                    }
                    writer_edge.writerow(edge_data)
                    output_count += 1
                    written_edges.add(edge_tuple)

        # Process HPOA files for disease-phenotype edges
        count_negative = 0
        count_positive = 0
        skipped_due_to_invalid = 0
        locate_mondo_count = 0
        print(f"开始处理HPOA文件, 总边数: {len(hpoa_map)}")
        for index, row in hpoa_map.iterrows():
            database_id = row['database_id'].strip()
            # print(f"处理行 {index + 1}: 数据库ID: {database_id}")
            hpo_id = row['hpo_id'].strip()
            qualifier = row['qualifier']
            evidence = row['evidence'].strip()

            if not hpo_id or not database_id:
                skipped_count += 1
                skipped_due_to_invalid += 1
                continue

            # Check if the HPO ID is valid
            if not hpo_id.startswith('HP:'):
                skipped_count += 1
                skipped_due_to_invalid += 1
                continue

            # Create edge for disease-phenotype relationship
            mondo_ids = []
            if database_id.startswith('OMIM:'):
                if database_id in omim_to_mondo:
                    mondo_ids = omim_to_mondo[database_id]
                    locate_mondo_count += 1
                else:
                    skipped_count += 1
                    continue
            elif database_id.startswith('MONDO:'):
                mondo_ids.append(database_id)
                locate_mondo_count += 1
            else:
                skipped_due_to_invalid += 1
                # print(f"跳过无效的数据库ID: {database_id}，行索引: {index + 1}")
                continue

            edge_weight = evidence_to_weight(evidence)
            if pd.isna(qualifier):
                count_positive += 1
            else:
                edge_weight = -edge_weight  # Negate the weight for NOT qualifiers
                count_negative += 1

            for mondo_id in mondo_ids:
                edge_tuple = tuple(sorted((mondo_id, hpo_id))) +    ('disease_phenotype',)
                if edge_tuple not in written_edges:
                    edge_data = {
                        "Node 1 ID": mondo_id,
                        "Node 1 Data Type": "Disease",
                        "Node 1 Source": "MONDO",
                        "Edge Type": "disease_phenotype",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": str(edge_weight),
                        "Node 2 ID": hpo_id,
                        "Node 2 Data Type": "Phenotype",
                        "Node 2 Source": "HPO"
                    }
                    writer_edge.writerow(edge_data)
                    output_count += 1
                    written_edges.add(edge_tuple)
        print(f"处理HPOA文件完成。正向边数量: {count_positive}, 负向边数量:     {count_negative}, 跳过无效边数量: {skipped_due_to_invalid}, 成功定位到MONDO ID 数量: {locate_mondo_count}")

    print(f"处理完成。共输出 {output_count} 条边，跳过 {skipped_count} 条边。")

def load_omim_to_mondo_mapping(mondo_omim_file):
    omim_to_mondo = defaultdict(list)
    print(f"加载OMIM到MONDO的映射关系文件: {mondo_omim_file}")
    with open(mondo_omim_file, 'r') as file:
        reader = csv.DictReader(file, delimiter=',', fieldnames=filedNameForDiseaseMap)
        # 跳过标题行
        next(reader)
        for row in reader:
            mondo_id = row.get('MONDO ID', '').strip()
            omim_id = row.get('OMIM ID', '').strip()
            if mondo_id and omim_id:
                omim_to_mondo[omim_id].append(mondo_id)

    return omim_to_mondo

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, nargs=6, required=True, help="Input HPO relationship files")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    maxo_file = args.input[0]
    genes_to_disease_file = args.input[1]
    genes_to_phenotype_file = args.input[2]
    phenotype_to_genes_file = args.input[3]
    hpoa_file = args.input[4]
    mondo_omim_file = args.input[5]
    output_edge_csv = args.output[0]

    # Process hpoa first
    hpoa_map = process_hpoa_file(hpoa_file)
    
    # load OMIM to MONDO mapping
    omim_to_mondo = load_omim_to_mondo_mapping(mondo_omim_file)

    print(f"开始处理HPO文件：{maxo_file}, {genes_to_disease_file}, {genes_to_phenotype_file}, {phenotype_to_genes_file}，输出文件：{output_edge_csv}")
    # 处理文件并写入 CSV
    process_to_csv(maxo_file, genes_to_disease_file, genes_to_phenotype_file, phenotype_to_genes_file, omim_to_mondo, hpoa_map, output_edge_csv)


if __name__ == "__main__":
    main()