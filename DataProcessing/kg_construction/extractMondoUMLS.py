from collections import defaultdict
import argparse
import re
import csv
from DataProcessing.kg_construction.commonSettings import *
filedNameForDiseaseMap = ['MONDO ID', 'MONDO Name', 'UMLS ID']

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

def process_terms(terms, writer_edge):
    # Get id, name, and definition
    print(f"Total terms: {len(terms)}")
    valid_terms_count = 0
    output_edge_count = 0
    edges_without_umls = 0
    # Skip first term
    subset_counts = defaultdict(int)
    count_multiple_subsets = 0
    for term in terms:
        if "id" not in term:
            # Skip terms without ID
            continue
        elif "is_obsolete" in term:
            # Skip obsolete terms
            continue

        term_id = term.get("id")
        # Check if id contains an url
        if "MONDO:" not in term_id:
            continue
        valid_terms_count += 1
        term_name = term.get("name")
        term_xref = term.get("xref")
        subset = term.get("subset")
        term_property_value = term.get("property_value")
        
        if subset:
            # Check if subset is a list or a single value
            values = []
            if isinstance(subset, dict) and 'values' in subset:
                values = subset['values']
                count_multiple_subsets += 1
                print(f"Warning: Term {term_id} has multiple subsets: {values}")
            elif isinstance(subset, str):
                values = [subset]
            
            for s in values:
                subset_counts[s] += 1
        # --- Extract and process UMLS mappings ---
        found_umls_ids = set()
        # 1. From xref
        if term_xref:
            xref_values = []
            if isinstance(term_xref, dict) and 'values' in term_xref:
                xref_values = term_xref['values']
            elif isinstance(term_xref, str):
                xref_values = [term_xref]
            
            for xref_entry in xref_values:
                if xref_entry.startswith("UMLS:"):
                    umls_id = xref_entry.split(" ")[0]
                    found_umls_ids.add(umls_id)
        
        # 2. From property_value
        if term_property_value:
            prop_values = []
            if isinstance(term_property_value, dict) and 'values' in term_property_value:
                prop_values = term_property_value['values']
            elif isinstance(term_property_value, str):
                prop_values = [term_property_value]

            for prop_val_entry in prop_values:
                match = re.search(r'resource/umls/id/(C[0-9]+)', prop_val_entry)
                if match:
                    cui = match.group(1)
                    umls_id = "UMLS:" + cui
                    found_umls_ids.add(umls_id)

        # 3. Create UMLS nodes and edges
        if len(found_umls_ids) > 1:
            print(f"Warning: Multiple UMLS IDs found for term {term_id}: {found_umls_ids}. Only one will be used.")
        for umls_id in found_umls_ids:
            # Create MONDO-UMLS edge
            edge = {
                "MONDO ID": term_id,
                "MONDO Name": term_name,
                "UMLS ID": umls_id
            }
            writer_edge.writerow(edge)
            output_edge_count += 1
        if not found_umls_ids:
            edge = {
                "MONDO ID": term_id,
                "MONDO Name": term_name,
                "UMLS ID": None
            }
            writer_edge.writerow(edge)
            edges_without_umls += 1
    print(f"Valid terms processed: {valid_terms_count}")
    print(f"Edges written to CSV: {output_edge_count}")
    print(f"Edges without UMLS IDs: {edges_without_umls}")
    print("Subset counts:")
    for sub, count in sorted(subset_counts.items()):
        print(f"  {sub}: {count}")
    print(f"Terms with multiple subsets: {count_multiple_subsets}")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input MONDO OBO file")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV files (edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    obo_file = args.input
    output_edge_csv = args.output[0]

    with open(output_edge_csv, "w", newline="") as file_edge:
        writer_edge = csv.DictWriter(file_edge, fieldnames=filedNameForDiseaseMap)
        writer_edge.writeheader()

        print(f"开始处理MONDO OBO文件：{obo_file}")

        all_terms = parse_obo_nested(obo_file)
        process_terms(all_terms, writer_edge)

if __name__ == "__main__":
    main()