#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理 tcm_monomer_candidate_drugs.csv 文件中的 path_with_drug 列
将药物别名整理为只取第一个别名，格式为：
disease -> drug -> protein1 -> protein2 -> ...
"""

import csv
import re


def parse_path_with_drug(path_str):
    """
    解析 path_with_drug 字符串，提取疾病、药物和蛋白质信息
    
    输入格式示例（兼容有/无局部 ID 的两种格式）:
    disease:4480 (biotin metabolic disease) -> drug:107 (|drug1|drug2|drug3) -> protein:596 (ALG13_HUMAN) -> protein:899 (APOF_HUMAN)
    disease (biotin metabolic disease) -> drug (|drug1|drug2|drug3) -> protein (ALG13_HUMAN) -> protein (APOF_HUMAN)
    
    输出格式:
    disease (biotin metabolic disease) -> drug (drug1) -> protein (ALG13_HUMAN) -> protein (APOD_HUMAN)

    如果路径段只有裸 ID（例如 disease:8010）且没有括号名称，则从输出路径中剔除。
    """
    if not path_str or not isinstance(path_str, str):
        return path_str

    node_pattern = re.compile(r'^\s*([A-Za-z_]+)(?::\d+|:)?\s*\((.*)\)\s*$')
    bare_id_pattern = re.compile(r'^\s*([A-Za-z_]+):\d+\s*$')
    result_parts = []

    for segment in path_str.split(' -> '):
        segment = segment.strip()
        node_match = node_pattern.match(segment)
        if node_match:
            node_type, node_name = node_match.groups()
            node_name = node_name.strip()

            if node_type == 'drug':
                # 药物之间用 | 分隔，去除开头的 | 并取第一个
                drugs = [d.strip() for d in node_name.split('|') if d.strip()]
                node_name = drugs[0] if drugs else "Unknown"
            elif not node_name:
                node_name = "Unknown"

            result_parts.append(f"{node_type} ({node_name})")
            continue

        bare_id_match = bare_id_pattern.match(segment)
        if bare_id_match:
            continue

        result_parts.append(segment)

    return " -> ".join(result_parts)


def process_csv(input_file, output_file):
    """
    处理 CSV 文件，整理 path_with_drug 列
    """
    print(f"正在处理文件：{input_file}")
    
    # 增加 CSV 字段大小限制
    csv.field_size_limit(10 * 1024 * 1024)  # 10MB
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        row_count = 0
        for row in reader:
            # 处理 path_with_drug 列
            if 'path_with_drug' in row:
                original_path = row['path_with_drug']
                formatted_path = parse_path_with_drug(original_path)
                row['path_with_drug'] = formatted_path
                
                # 打印前几行作为示例
                if row_count < 3:
                    print(f"\n示例 {row_count + 1}:")
                    print(f"  原始：{original_path[:100]}...")
                    print(f"  整理后：{formatted_path}")
            
            # 处理 drug_name 列，只保留第一个药物别名
            if 'drug_name' in row and row['drug_name']:
                drug_aliases = [d.strip() for d in row['drug_name'].split('|') if d.strip()]
                if len(drug_aliases) > 1:
                    row['drug_name'] = drug_aliases[0]
            
            writer.writerow(row)
            row_count += 1
            
            # 每处理 100 行显示一次进度
            if row_count % 100 == 0:
                print(f"已处理 {row_count} 行...")
    
    print(f"\n处理完成！共处理 {row_count} 行")
    print(f"输出文件：{output_file}")


if __name__ == "__main__":
    input_csv = "top1_protein_results_with_drug_paths.csv"
    output_csv = "top1_protein_results_with_drug_paths_formatted.csv"
    
    process_csv(input_csv, output_csv)
    
    print("\n格式说明:")
    print("  类别 (名称) -> 类别 (名称) -> ...")
    print("  不同药物别名之间用 | 隔开，只取第一个")
