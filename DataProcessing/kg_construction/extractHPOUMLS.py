from collections import defaultdict
import argparse
import re
import csv
from DataProcessing.kg_construction.commonSettings import *
fieldNamesHPOToUMLS = ["hpo_id", "umls_id"]

def parse_obo_nested(file_path):
    """
    解析OBO文件生成嵌套字典结构
    :param file_path: OBO文件路径
    :return: 包含所有term的列表，每个term是嵌套字典结构
    """
    terms = []
    current_term = defaultdict(lambda: {'values': [], 'properties': []})
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 新term开始
            if line == '[Term]':
                if current_term:
                    terms.append(_convert_term(current_term))
                current_term = defaultdict(lambda: {'values': [], 'properties': []})
                continue
                
            # 忽略其他区块和空行
            if not line or line.startswith('['):
                continue

            # 解析键值对
            if ': ' in line:
                key, remainder = line.split(': ', 1)
                value, *props = remainder.split(' {', 1)
                value = value.strip()
                
                # 解析属性
                properties = {}
                if props:
                    prop_str = props[0].rstrip('}').replace('"', '')
                    for pair in prop_str.split(','):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            properties[k.strip()] = v.strip()
                            
                # 存储到嵌套结构
                current_term[key]['values'].append(value)
                current_term[key]['properties'].append(properties)
                
        # 添加最后一个term
        if current_term:
            terms.append(_convert_term(current_term))
    
    return terms

def _convert_term(term_dict):
    """将defaultdict转换为标准字典并简化结构"""
    converted = {}
    for key, data in term_dict.items():
        # 如果存在多个值或属性，保留嵌套结构
        if len(data['values']) > 1 or any(data['properties']):
            converted[key] = {
                'values': data['values'],
                'properties': data['properties']
            }
        else:
            # 单个值直接存储
            converted[key] = data['values'][0] if data['values'] else None
    return converted

def extract_ontology_ids(data):
    """
    通用本体ID提取函数，支持多种格式：
    示例输入:
    - "disease_has_infectious_agent NCBITaxon:945 ! Ehrlichia chaffeensis"
    - "GO:0008150 ! biological process"
    - "CL:123456"
    输出: ["NCBITaxon:945"]
    """
    # 统一输入为列表
    entries = [data] if isinstance(data, str) else data if isinstance(data, list) else []
    
    # 正则表达式模式（支持常见本体格式）
    pattern = r'\b([A-Za-z_]+:[0-9]+)\b'
    
    found_ids = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        
        # 步骤1：去除注释部分
        clean_part = entry.split("!", 1)[0].strip()
        
        # 步骤2：正则匹配本体ID
        matches = re.findall(pattern, clean_part)
        if matches:
            # 取最后一个匹配项（适用于"has_part UBERON:000123"这类情况）
            found_ids.extend(matches)
    
    return found_ids


def process_terms(terms, writer_mapping):
    # Get id, name, and definition
    print(f"Total terms: {len(terms)}")
    # Skip first term
    hpo_umls_count = 0
    for term in terms:
        if "id" not in term:
            # Skip terms without ID
            continue
        elif "is_obsolete" in term:
            # Skip obsolete terms
            continue

        term_id = term.get("id")
        term_alt_id = term.get("alt_id")
        term_xref = term.get("xref")

        # Check if id contains an url
        if "HP" not in term_id:
            continue
        # Extract HPO-UMLS mapping from xref
        if term_xref:
            xrefs_to_process = []
            if isinstance(term_xref, dict) and 'values' in term_xref:
                xrefs_to_process = term_xref['values']
            elif isinstance(term_xref, str):
                xrefs_to_process = [term_xref]

            # Normalize term_alt_id to a list to handle single or multiple alt_ids
            alt_ids_to_process = []
            if term_alt_id:
                if isinstance(term_alt_id, dict) and 'values' in term_alt_id:
                    alt_ids_to_process = term_alt_id['values']
                elif isinstance(term_alt_id, str):
                    alt_ids_to_process = [term_alt_id]
            
            for xref in xrefs_to_process:
                if xref.startswith("UMLS:"):
                    umls_id = xref.split(':', 1)[1]

                    # Write mapping for the primary ID
                    writer_mapping.writerow({
                        "hpo_id": term_id,
                        "umls_id": umls_id
                    })
                    hpo_umls_count += 1

                    # Write mapping for all alternate IDs
                    for alt_id in alt_ids_to_process:
                        writer_mapping.writerow({
                            "hpo_id": alt_id,
                            "umls_id": umls_id
                        })
                        hpo_umls_count += 1
    print(f"Total HPO-UMLS mappings found: {hpo_umls_count}")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV file for HPO-UMLS mapping")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    obo_file = args.input
    mapping_output_csv = args.output[0]

    with open(mapping_output_csv, "w", newline="") as file_mapping:
        writer_mapping = csv.DictWriter(file_mapping, fieldnames=fieldNamesHPOToUMLS)
        writer_mapping.writeheader()

        print(f"开始处理HPO OBO文件：{obo_file}")
        all_terms = parse_obo_nested(obo_file)
        process_terms(all_terms, writer_mapping)


if __name__ == "__main__":
    main()