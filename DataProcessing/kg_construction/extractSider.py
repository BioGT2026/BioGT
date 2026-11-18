import os
import numpy as np
import pandas as pd
import sys
import argparse
import csv
from DataProcessing.kg_construction.commonSettings import *

fieldNamesHPOToUMLS = ["hpo_id", "umls_id"]
filedNameForDiseaseMap = ['MONDO ID', 'MONDO Name', 'UMLS ID']
filedNameForDrugATC= ['Drug ID', 'Drug Name', 'CAS', 'ATC Code', 'InChi']

def parse_sider_files(atc_file, meddra_file):
    # 读取 ATC 和 MedDRA 文件
    atc_df = pd.read_csv(atc_file, sep='\t', header=None)
    atc_df.columns = ['stitch', 'atc']
    
    meddra_df = pd.read_csv(meddra_file, sep='\t', header=None)
    meddra_df.columns = ['stitch_id_1', 'stitch_id_2', 'UMLS_from_label', 
                         'meddra_concept_type', 'UMLS_from_meddra', 'side_effect_name']
    
    # 过滤 MedDRA 数据
    meddra_df = meddra_df.query('meddra_concept_type=="PT"')
    
    # 合并数据
    side_effects = pd.merge(atc_df, meddra_df, left_on='stitch', right_on='stitch_id_1')
    # Keep the stitch ID for later mapping
    return side_effects[['stitch', 'atc', 'UMLS_from_label', 'UMLS_from_meddra', 'side_effect_name']]

def match_atc_to_drug_and_umls_to_disease(sider_data, hp_umls_df, mondo_umls_df):
    """
    匹配 ATC 和 UMLS ID 到药物和疾病
    :param sider_data: SIDER 数据
    :param hp_umls_df: HPO-UMLS 映射数据
    :param mondo_umls_df: MONDO-UMLS 映射数据
    :return: 匹配后的数据
    """
    # 创建输出 DataFrame
    output_data = []

    # 遍历 SIDER 数据
    count = 0
    stop = -1
    for _, row in sider_data.iterrows():
        print(f"\r当前进度: {(count+1)}", end="", flush=True)
        count += 1
        if count >= stop and stop != -1:
            print(f"\n已处理 {stop} 条记录，停止处理。")
            break
        stitch_id = row['stitch'] # Get stitch id
        atc_code = row['atc']
        umls_id_label = row['UMLS_from_label'] 
        umls_id_meddra = row['UMLS_from_meddra']
        side_effect_name = row['side_effect_name']

        # 收集当前行的唯一 UMLS ID
        umls_ids_to_check = set()
        if pd.notna(umls_id_label):
            umls_ids_to_check.add(umls_id_label)
        if pd.notna(umls_id_meddra):
            umls_ids_to_check.add(umls_id_meddra)

        # 对于每个唯一的 UMLS ID，查找匹配
        for umls_id in umls_ids_to_check:
            # 查找 HPO 和 MONDO 映射
            hpo_match = hp_umls_df[hp_umls_df['umls_id'] == umls_id]
            mondo_match = mondo_umls_df[mondo_umls_df['UMLS ID'] == umls_id]

            if not hpo_match.empty:
                hpo_id = hpo_match['hpo_id'].values[0]
                output_data.append({
                    'Stitch ID': stitch_id, # Add stitch id
                    'ATC Code': atc_code,
                    'HPO ID': hpo_id,
                    'Side Effect Name': side_effect_name,
                    'Source': 'HPO'
                })

            if not mondo_match.empty:
                mondo_id = mondo_match['MONDO ID'].values[0]
                output_data.append({
                    'Stitch ID': stitch_id, # Add stitch id
                    'ATC Code': atc_code,
                    'MONDO ID': mondo_id,
                    'Side Effect Name': side_effect_name,
                    'Source': 'MONDO'
                })

    return pd.DataFrame(output_data)

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process SIDER, HPO, and MONDO data to create drug-disease/phenotype edges.")
    parser.add_argument('-i', '--input', type=str, nargs=5, required=True, 
                        help="Input files in order: Sider ATC, Sider MedDRA, HP-UMLS map, MONDO-UMLS map, DrugBank-ATC map")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV file for edges")

    args = parser.parse_args()

    # 获取输入和输出文件路径
    atc_file, meddra_file, hp_umls_file, mondo_umls_file, db_atc_file = args.input
    output_edge_csv = args.output[0]

    # Load the sider data and process it
    print("Parsing SIDER files...")
    sider_data = parse_sider_files(atc_file, meddra_file)
    print(f"Found {len(sider_data)} initial side effect records.")

    # 读取 HPO-UMLS 和 MONDO-UMLS 映射文件
    print("Loading HPO and MONDO mapping files...")
    hp_umls_df = pd.read_csv(hp_umls_file, sep=',', header=None, names=fieldNamesHPOToUMLS)
    mondo_umls_df = pd.read_csv(mondo_umls_file, sep=',', header=None, names=filedNameForDiseaseMap)
    db_atc_df = pd.read_csv(db_atc_file, sep=',', header=None, names=filedNameForDrugATC)
    print(f"Found {len(hp_umls_df)} HPO-UMLS mappings, {len(mondo_umls_df)} MONDO-UMLS mappings, and {len(db_atc_df)} Drug-ATC mappings.")

    # Call the matching function
    print("Matching SIDER UMLS IDs to HPO and MONDO...")
    matched_data = match_atc_to_drug_and_umls_to_disease(sider_data, hp_umls_df, mondo_umls_df)
    print(f"Found {len(matched_data)} potential drug-disease/phenotype links.")

    if matched_data.empty:
        print("No matches found. Exiting.")
        return

    # --- Drug Matching Enhancement ---
    # 2. Merge matched data with the ATC mapping (db_atc_df)
    # A left join is used to keep all side effects and find corresponding DrugBank IDs.
    final_edges_df = pd.merge(matched_data, db_atc_df, on='ATC Code', how='left')

    # 3. Clean up
    final_edges_df.dropna(subset=['Drug ID'], inplace=True)
    final_edges_df.drop_duplicates(inplace=True)
    print(f"Found {len(final_edges_df)} final edges after mapping to DrugBank IDs using ATC.")

    # 创建输出 CSV 文件
    drug_phenotype_count = 0
    drug_disease_count = 0
    with open(output_edge_csv, "w", newline="", encoding="utf-8") as file_edge:
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_edge.writeheader()

        for _, row in final_edges_df.iterrows():
            drug_id = str(row['Drug ID'])
            target_id_raw = row.get('HPO ID') or row.get('MONDO ID')
            target_type = "Phenotype" if pd.notna(row.get('HPO ID')) else "Disease"
            target_source = "HPO" if target_type == "Phenotype" else "MONDO"
            target_id = str(target_id_raw)

            edge = {
                "Node 1 ID": drug_id,
                "Node 1 Data Type": "Drug",
                "Node 1 Source": "DrugBank",
                "Edge Type": "drug_disease" if target_type == "Disease" else "drug_phenotype",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": target_id,
                "Node 2 Data Type": target_type,
                "Node 2 Source": target_source
            }
            writer_edge.writerow(edge)
            if target_type == "Phenotype":
                drug_phenotype_count += 1
            else:
                drug_disease_count += 1
    print(f"Total drug-phenotype edges written: {drug_phenotype_count}")
    print(f"Total drug-disease edges written: {drug_disease_count}")
    print(f"Processing complete. Edges written to {output_edge_csv}")

if __name__ == "__main__":
    main()
