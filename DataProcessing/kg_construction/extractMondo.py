from collections import defaultdict
import argparse
import re
import csv
from DataProcessing.kg_construction.commonSettings import *
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDFS

def get_human_diseases_ids(owl_path):
    # 加载 OWL 文件
    g = Graph()
    g.parse(owl_path, format="xml")  # 如果是 .ttl 文件，用 format="turtle"

    # 定义命名空间
    MONDO = Namespace("http://purl.obolibrary.org/obo/MONDO_")
    rdfs = RDFS

    # 查询 MONDO_0700096 的所有子孙类
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

    SELECT DISTINCT ?class ?label
    WHERE {
      ?class rdfs:subClassOf* <http://purl.obolibrary.org/obo/MONDO_0700096> .
      OPTIONAL { ?class rdfs:label ?label }
    }
    ORDER BY ?class
    """

    # 执行查询
    results = g.query(query)
    ids = set()
    for row in results:
        class_uri = row["class"]
        short_id = class_uri.replace("http://purl.obolibrary.org/obo/MONDO_", "MONDO:")
        # print(f"{short_id}")  # 打印ID
        ids.add(short_id)  # 只保留最后的ID部分
    return ids

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

def check_data_type(data):
    if "GO:" in data:
        if data in goTermToTypeMapping:
            return goTermToTypeMapping[data]
        else:
            return "Unknown"
    elif "MONDO:" in data:
        return "Disease"
    elif "UBERON:" in data:
        return "Anatomy"
    elif "HP:" in data:
        return "Phenotype"
    else :
        # Return the content before the colon
        return data.split(":")[0] if ":" in data else "Unknown"

def is_human_disease(term):
    """
    检查一个MONDO术语是否代表人类疾病。
    通过检查 'subset' 属性或 'xref' 属性中是否包含人类物种信息来实现。
    """
    # 检查 'subset' 属性
    subset_data = term.get('subset')
    if subset_data:
        subsets = []
        if isinstance(subset_data, str):
            subsets.append(subset_data)
        elif isinstance(subset_data, dict) and 'values' in subset_data:
            subsets.extend(subset_data['values'])
        
        for s in subsets:
            # MONDO使用 "disease_grouping:human-only disease" 这样的subset
            if 'human' in s.lower():
                return True

    # 检查 'xref' 属性是否链接到人类分类单元 (NCBITaxon:9606)
    xref_data = term.get('xref')
    if xref_data:
        xrefs = []
        if isinstance(xref_data, str):
            xrefs.append(xref_data)
        elif isinstance(xref_data, dict) and 'values' in xref_data:
            xrefs.extend(xref_data['values'])
        
        for xref in xrefs:
            if 'NCBITaxon:9606' in xref:  # NCBITaxon:9606 is for Homo sapiens
                return True
                
    return False

def process_terms(terms, unique_ids, writer_node, writer_edge):
    # Get id, name, and definition
    print(f"Total terms: {len(terms)}")
    
    # 添加统计计数器
    mondo_nodes_written = 0
    total_terms_processed = 0
    skipped_no_id = 0
    skipped_obsolete = 0
    skipped_no_human = 0
    skipped_duplicate = 0
    human_disease_count = 0
    written_nodes = set()
    
    # Skip first term
    for term in terms:
        total_terms_processed += 1
        
        if "id" not in term:
            # Skip terms without ID
            skipped_no_id += 1
            continue
        elif "is_obsolete" in term:
            # Skip obsolete terms
            skipped_obsolete += 1
            continue

        term_id = term.get("id")
        # Check if id contains an url
        if "http" in term_id:
            skipped_no_id += 1
            continue
            
        term_name = term.get("name")
        term_def = term.get("def")
        term_synonym = term.get("synonym")
        # Get realtionships
        term_is_a = term.get("is_a") # A是B的子类 Subset
        term_union_of = term.get("union_of") # A是B和C的并集 Union_of
        term_intersection_of = term.get("intersection_of") #A是B和C的交集  Intersection_of
        term_relationship = term.get("relationship") # A和B有关系 Associates
        term_distjoint_from = term.get("disjoint_from") # A和B不相关 NegativeCorrelation
        if term_def:
            # Merge all definition lines if term_def is a list
            if isinstance(term_def, list):
                term_def = "|".join(term_def.get("values"))
        else:
            term_def = ""
        if term_synonym:
            # Merge all synonyms if term_synonym is a list
            if isinstance(term_synonym, list):
                term_synonym = "|".join(term_synonym.get("values"))
        else:
            term_synonym = ""
        # print(f"Term ID: {term_id}, Name: {term_name}, Definition: {term_def}")
        
        # Create node
        comments = f"Defination: {term_def}, Synonym: {term_synonym}"
        node = {
            "ID": term_id,
            "Type": check_data_type(term_id),
            "Name": term_name,
            "Sequence": "",
            "Comments": comments
        }
        # Write node to CSV
        if "MONDO:" in term_id:
            if term_id not in written_nodes:
                if term_id in unique_ids:
                    writer_node.writerow(node)
                    human_disease_count += 1
                    mondo_nodes_written += 1
                    written_nodes.add(term_id)
                else:
                    skipped_no_human += 1
            else:
                skipped_duplicate += 1

        # Check relationships
        if term_is_a:
            term_is_a = extract_ontology_ids(term_is_a)
            for parent in term_is_a:
                # Create edge
                edge = {
                    "Node 1 ID": parent,
                    "Node 1 Data Type": check_data_type(parent),
                    "Node 1 Source": "MONDO",
                    "Edge Type": check_data_type(parent) + "_" + check_data_type(term_id),
                    "Edge Direction": edge_types.get("SD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": term_id,
                    "Node 2 Data Type": check_data_type(term_id),
                    "Node 2 Source": "MONDO"
                }
                # Write edge to CSV
                writer_edge.writerow(edge)
        if term_union_of:
            term_union_of = extract_ontology_ids(term_union_of)
            for parent in term_union_of:
                # Create edge
                edge = {
                    "Node 1 ID": parent,
                    "Node 1 Data Type": check_data_type(parent),
                    "Node 1 Source": "MONDO",
                    "Edge Type": check_data_type(parent) + "_" + check_data_type(term_id),
                    "Edge Direction": edge_types.get("SD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": term_id,
                    "Node 2 Data Type": check_data_type(term_id),
                    "Node 2 Source": "MONDO"
                }
                # Write edge to CSV
                writer_edge.writerow(edge)
        if term_intersection_of:
            term_intersection_of = extract_ontology_ids(term_intersection_of)
            for parent in term_intersection_of:
                # Create edge
                edge = {
                    "Node 1 ID": term_id,
                    "Node 1 Data Type": check_data_type(term_id),
                    "Node 1 Source": "MONDO",
                    "Edge Type": check_data_type(term_id) + "_" + check_data_type(parent),
                    "Edge Direction": edge_types.get("DD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": parent,
                    "Node 2 Data Type": check_data_type(parent),
                    "Node 2 Source": "MONDO"
                }
                # Write edge to CSV
                writer_edge.writerow(edge)
        if term_relationship:
            term_relationship = extract_ontology_ids(term_relationship)
            for parent in term_relationship:
                # Create edge
                edge = {
                    "Node 1 ID": parent,
                    "Node 1 Data Type": check_data_type(parent),
                    "Node 1 Source": "MONDO",
                    "Edge Type": check_data_type(parent) + "_" + check_data_type(term_id),
                    "Edge Direction": edge_types.get("DD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": term_id,
                    "Node 2 Data Type": check_data_type(term_id),
                    "Node 2 Source": "MONDO"
                }
                # Write edge to CSV
                writer_edge.writerow(edge)
        if term_distjoint_from:
            term_distjoint_from = extract_ontology_ids(term_distjoint_from)
            for parent in term_distjoint_from:
                # Create edge
                edge = {
                    "Node 1 ID": parent,
                    "Node 1 Data Type": check_data_type(parent),
                    "Node 1 Source": "MONDO",
                    "Edge Type": check_data_type(parent) + "_" + check_data_type(term_id),
                    "Edge Direction": edge_types.get("DD"),
                    "Edge Weight": -1.0,
                    "Node 2 ID": term_id,
                    "Node 2 Data Type": check_data_type(term_id),
                    "Node 2 Source": "MONDO"
                }
                # Write edge to CSV
                writer_edge.writerow(edge)
    print(f"总处理terms数量: {total_terms_processed}")
    print(f"跳过无ID的terms: {skipped_no_id}")
    print(f"跳过过时的terms: {skipped_obsolete}")
    print(f"跳过非人类的terms: {skipped_no_human}")
    print(f"跳过重复的MONDO节点: {skipped_duplicate}")
    print(f"有效的MONDO节点写出数量: {mondo_nodes_written}")
    print(f"其中，人类疾病的数量: {human_disease_count}")
    print(f"========================")
    
    return mondo_nodes_written
            
# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, nargs=2, required=True, help="Input obo and owl file")
    parser.add_argument('-g', '--go', type=str, required=True, help="Input go mapping file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    obo_file = args.input[0]
    owl_file = args.input[1]
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

    unique_ids = get_human_diseases_ids(owl_file)
    print(f"Unique human disease IDs extracted: {len(unique_ids)}")
    with open(output_node_csv, "w", newline="") as file_node, open(output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()

        print(f"开始处理MONDO OBO文件：{obo_file}")

        all_terms = parse_obo_nested(obo_file)
        process_terms(all_terms, unique_ids, writer_node, writer_edge)


if __name__ == "__main__":
    main()