from collections import defaultdict
import argparse
import re
import csv
from DataProcessing.kg_construction.commonSettings import *

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


def process_terms(terms, writer_node, writer_edge):
    # Get id, name, and definition
    total_alt_ids = 0
    total_nodes = 0
    print(f"Total terms: {len(terms)}")
    # Skip first term
    for term in terms:
        if "id" not in term:
            # Skip terms without ID
            continue
        elif "is_obsolete" in term:
            # Skip obsolete terms
            continue

        term_id = term.get("id")
        term_alt_id = term.get("alt_id")
        # Check if id contains an url
        if "HP" not in term_id:
            continue
        term_name = term.get("name")
        term_def = term.get("def")
        term_synonym = term.get("synonym")
        term_comment = term.get("comment")
        term_replaced = term.get("replaced_by")
        if term_replaced:
            # Skip replaced terms
            print(f"Skipping replaced term: {term_id}")
            continue
        # Get realtionships
        term_is_a = term.get("is_a") # A是B的子类 Subset
        if term_def:
            # Merge all definition lines if term_def is a list
            if isinstance(term_def, list):
                term_def = " ".join(term_def.get("values"))
        else:
            term_def = ""
        if term_synonym:
            # Merge all synonyms if term_synonym is a list
            if isinstance(term_synonym, list):
                term_synonym = ", ".join(term_synonym.get("values"))
        else:
            term_synonym = ""
        
        if term_comment:
            if isinstance(term_comment, list):
                term_comment = " ".join(term_comment.get("values"))
        else:
            term_comment = ""
        # print(f"Term ID: {term_id}, Name: {term_name}, Definition: {term_def}")
        
        if term_alt_id:
            if isinstance(term_alt_id, dict):
                term_alt_id = extract_ontology_ids(term_alt_id)
                for alt_id in term_alt_id:
                    total_alt_ids += 1
                    print(f"List Alt Term ID: {alt_id}")
                    # Create node
                    comments = f"Definition: {term_def or ''} Synonym: {term_synonym or ''} Comment: {term_comment or ''}"
                    node = {
                        "ID": alt_id,
                        "Type": "Phenotype",
                        "Name": term_name,
                        "Sequence": "",
                        "Comments": comments
                    }
                    # Write node to CSV
                    # writer_node.writerow(node)
            else:
                # print(f"Alt Term ID: {term_alt_id}")
                # Create node
                total_alt_ids += 1
                comments = f"Definition: {term_def or ''} Synonym: {term_synonym or ''} Comment: {term_comment or ''}"
                node = {
                    "ID": term_alt_id,
                    "Type": "Phenotype",
                    "Name": term_name,
                    "Sequence": "",
                    "Comments": comments
                }
                # Write node to CSV
                # writer_node.writerow(node)

        # Create node
        comments = f"Definition: {term_def or ''} Synonym: {term_synonym or ''} Comment: {term_comment or ''}"
        node = {
            "ID": term_id,
            "Type": "Phenotype",
            "Name": term_name,
            "Sequence": "",
            "Comments": comments
        }
        # Write node to CSV
        total_nodes += 1
        writer_node.writerow(node)

        # Check relationships
        if term_is_a:
            term_is_a = extract_ontology_ids(term_is_a)
            for parent in term_is_a:
                # Create edge
                if "HP:" not in parent:
                    print(f"Error: Term ID: {term_id}, Name: {term_name}, Parent: {parent}")
                edge = {
                    "Node 1 ID": parent,
                    "Node 1 Data Type": "Phenotype",
                    "Node 1 Source": "HPO",
                    "Edge Type": "phenotype_phenotype",
                    "Edge Direction": edge_types.get("SD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": term_id,
                    "Node 2 Data Type": "Phenotype",
                    "Node 2 Source": "HPO"
                }
                # Write edge to CSV
                writer_edge.writerow(edge)
    print(f"Total alternative IDs processed: {total_alt_ids}, total nodes: {total_nodes}")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    obo_file = args.input
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]

    with open(output_node_csv, "w", newline="") as file_node, open(output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()

        print(f"开始处理HPO OBO文件：{obo_file}")

        all_terms = parse_obo_nested(obo_file)
        process_terms(all_terms, writer_node, writer_edge)


if __name__ == "__main__":
    main()