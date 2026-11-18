import argparse
import csv
import re
from collections import defaultdict
from DataProcessing.kg_construction.commonSettings import *  # 确保 commonSettings.py 包含 fieldnames_node, fieldnames_edge, edge_types
goTermToTypeMapping = {}

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

                # 处理可能包含属性的行
                value, *props = remainder.split(' {', 1)
                value = value.strip().replace('"', '')  # 去除可能存在的引号

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
        if len(data['values']) > 1 or any(p for p in data['properties'] if p):  # 检查属性字典是否非空
            converted[key] = {
                'values': data['values'],
                'properties': data['properties']
            }
        else:
            # 单个值直接存储
            converted[key] = data['values'][0] if data['values'] else None
    return converted

def extract_ontology_ids(data):
    """从字符串或列表中提取本体ID (e.g., UBERON:0000955)"""
    entries = []
    if isinstance(data, str):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and 'values' in data:
        entries = data['values']

    # 正则表达式模式（支持常见本体格式）
    pattern = r'\b([A-Za-z_]+[:_][A-Za-z0-9]+)\b'  # 更通用的模式，支持下划线

    found_ids = []
    for entry in entries:
        if isinstance(entry, str):
            # 提取ID部分，忽略注释
            id_part = entry.split(' ! ')[0].strip()
            matches = re.findall(pattern, id_part)
            found_ids.extend(matches)

    return list(set(found_ids))  # 返回去重后的ID列表

def check_data_type(data):
    """根据ID前缀判断数据类型"""
    if not isinstance(data, str):
        return "Unknown"
    if data.startswith("UBERON:"):
        return "Anatomy"
    elif data.startswith("GO:"):
        current_type = goTermToTypeMapping.get(data, "Unknown")
        return current_type
    elif data.startswith("CL:") or data.startswith("DOID:") or data.startswith("OMIM:") or data.startswith("NCBITaxon:") or data.startswith("CHEBI:") or data.startswith("PubChem:") or data.startswith("PATO:") or data.startswith("SO:"):
        return "Unknown"
    # elif data.startswith("MONDO:"):
    #     return "Disease"
    # elif data.startswith("RefSeq:"):
    #     if data in refseqRNATerms:
    #         return "RNA"
    #     else:
    #         return "Unknown"
    elif data.startswith("HP:"):
        return "Phenotype"
    elif data.startswith("DrugBank:"):
        return "Drug"
    # elif data.startswith("Uniprot:"):
    #     return "Protein"
    # elif data.startswith("NCBIGene:"):
    #     return "Gene"
    # 可以根据需要添加更多类型判断
    else:
        return "Unknown"

def process_terms(terms, writer_node, writer_edge):
    """处理提取的术语，写入节点和边"""
    print(f"Total terms to process: {len(terms)}")
    processed_count = 0
    for term in terms:
        if "id" not in term:
            continue
        elif term.get("is_obsolete") == 'true':  # OBO标准中使用 'true'
            continue

        term_id = term.get("id")
        if not term_id or not term_id.startswith("UBERON:"):  # 只处理Uberon自身的术语作为主要节点
            continue

        term_name = term.get("name")
        term_def_data = term.get("def")
        term_synonym_data = term.get("synonym")

        # 提取定义
        term_def = ""
        if isinstance(term_def_data, dict) and 'values' in term_def_data:
            term_def = term_def_data['values'][0] if term_def_data['values'] else ""
        elif isinstance(term_def_data, str):
            term_def = term_def_data

        # 提取同义词
        synonyms_list = []
        if isinstance(term_synonym_data, dict) and 'values' in term_synonym_data:
            synonyms_list = term_synonym_data['values']
        elif isinstance(term_synonym_data, list):  # 如果 _convert_term 未简化
            if term_synonym_data and isinstance(term_synonym_data[0], dict) and 'values' in term_synonym_data[0]:
                synonyms_list = term_synonym_data[0]['values']

        term_synonym_str = "|".join(synonyms_list)

        # 创建节点
        comments = f"Definition: {term_def}|Synonyms: {term_synonym_str}"
        if check_data_type(term_id) != "Anatomy":
            continue  # 只处理Anatomy类型的节点
        node = {
            "ID": term_id,
            "Name": term_name,
            "Type": "Anatomy",
            "Sequence": "",
            "Comments": comments.strip('|')
        }
        writer_node.writerow(node)
        processed_count += 1

        # 处理关系边
        relationships = {
            "is_a": term.get("is_a"),
            "union_of": term.get("union_of"),
            "intersection_of": term.get("intersection_of"),
            "relationship": term.get("relationship"),
            "disjoint_from": term.get("disjoint_from")
        }

        for rel_type, rel_data in relationships.items():
            if rel_data:
                target_ids = extract_ontology_ids(rel_data)
                for target_id in target_ids:
                    if not target_id:
                        continue  # 跳过无效ID

                    source_type = check_data_type(term_id)
                    target_type = check_data_type(target_id)
                    edge_type_str = f"{source_type}_{target_type}"
                    if source_type == "Unknown" or target_type == "Unknown":
                        continue  # 跳过未知类型的边
                    if source_type == "Drug":
                        term_id = term_id.split(":")[1]  # 去掉前缀
                    if target_type == "Drug":
                        target_id = target_id.split(":")[1]
                    # 确定方向和权重
                    if rel_type == "is_a" or rel_type == "union_of":
                        direction = edge_types.get("SD")
                        weight = 1.0
                        source_node = target_id  # is_a: parent -> child
                        target_node = term_id
                    elif rel_type == "disjoint_from":
                        direction = edge_types.get("DD")
                        weight = -1.0
                        source_node = term_id
                        target_node = target_id
                    else:  # intersection_of, relationship
                        direction = edge_types.get("DD")
                        weight = 1.0
                        source_node = term_id
                        target_node = target_id

                    edge = {
                        "Node 1 ID": source_node,
                        "Node 1 Data Type": check_data_type(source_node),
                        "Node 1 Source": "UBERON",
                        "Edge Type": edge_type_str,
                        "Edge Direction": direction,
                        "Edge Weight": weight,
                        "Node 2 ID": target_node,
                        "Node 2 Data Type": check_data_type(target_node),
                        "Node 2 Source": "UBERON"
                    }
                    writer_edge.writerow(edge)

    print(f"Processed and wrote {processed_count} UBERON nodes.")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process an Uberon OBO file and export to CSV nodes and edges.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input Uberon OBO file")
    parser.add_argument('-g', '--go', type=str, required=True, help="Input GO type file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()

    # 获取输入和输出文件路径
    obo_file = args.input
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]

    global goTermToTypeMapping
    # Open GO Type file
    with open(args.go, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            if len(row) >= 2:
                go_id = row[0]
                go_type = row[1]
                goTermToTypeMapping[go_id] = go_type
            else:
                print(f"Skipping row due to insufficient columns: {row}")
    if not goTermToTypeMapping:
        print(f"GO Type mapping为空，请检查文件: {args.go}")
        return
    print(f"GO Type Mapping loaded, total {len(goTermToTypeMapping)} entries.")

    with open(output_node_csv, "w", newline="", encoding="utf-8") as file_node, \
         open(output_edge_csv, "w", newline="", encoding="utf-8") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()

        print(f"开始处理 Uberon OBO 文件：{obo_file}")
        all_terms = parse_obo_nested(obo_file)
        process_terms(all_terms, writer_node, writer_edge)
        print(f"数据已成功导出到 节点记录：{output_node_csv}, 边记录：{output_edge_csv}")

if __name__ == "__main__":
    main()
