from collections import defaultdict, deque
import argparse
import re
import csv
import json
from DataProcessing.kg_construction.commonSettings import *

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
    result = {}
    for key, value_dict in term_dict.items():
        values = value_dict['values']
        properties = value_dict['properties']
        
        if len(values) == 1:
            # 单个值的情况
            result[key] = values[0]
        else:
            # 多个值的情况，保持为列表
            result[key] = values
            
        # 如果有属性，也保存
        if any(properties):
            result[f"{key}_properties"] = properties
    
    return result

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
    """根据ID前缀判断数据类型"""
    if not isinstance(data, str):
        return "Unknown"
    
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
    elif "CHEBI:" in data:
        return "Chemical"
    elif "DrugBank:" in data:
        return "Drug"
    else:
        # Return the content before the colon
        return data.split(":")[0] if ":" in data else "Unknown"

class DiseaseHierarchy:
    """疾病层级结构分析器"""
    def __init__(self):
        self.terms = {}  # ID -> term信息
        self.children = defaultdict(set)  # parent_id -> set of child_ids
        self.parents = defaultdict(set)   # child_id -> set of parent_ids
        self.root_nodes = set()           # 根节点集合
        self.levels = defaultdict(set)    # level -> set of node_ids
        
    def add_term(self, term_id, term_data):
        """添加疾病术语"""
        self.terms[term_id] = term_data
        
    def add_is_a_relation(self, child_id, parent_id):
        """添加is_a关系"""
        self.children[parent_id].add(child_id)
        self.parents[child_id].add(parent_id)
        
    def find_root_nodes(self):
        """找到所有根节点（没有父节点的节点）"""
        all_nodes = set(self.terms.keys())
        nodes_with_parents = set(self.parents.keys())
        self.root_nodes = all_nodes - nodes_with_parents
        print(f"发现 {len(self.root_nodes)} 个根节点")
        
    def build_levels(self):
        """构建层级结构"""
        visited = set()
        
        # 从根节点开始BFS
        queue = deque()
        for root in self.root_nodes:
            queue.append((root, 0))  # (node_id, level)
            
        while queue:
            node_id, level = queue.popleft()
            
            if node_id in visited:
                continue
                
            visited.add(node_id)
            self.levels[level].add(node_id)
            
            # 添加子节点到队列
            for child_id in self.children[node_id]:
                if child_id not in visited:
                    queue.append((child_id, level + 1))
                    
    def get_statistics(self):
        """获取层级统计信息"""
        stats = {
            'total_terms': len(self.terms),
            'total_levels': len(self.levels),
            'root_nodes_count': len(self.root_nodes),
            'levels_distribution': {level: len(nodes) for level, nodes in self.levels.items()}
        }
        return stats
        
    def export_hierarchy(self, output_file):
        """导出层级结构到文件"""
        hierarchy_data = {
            'statistics': self.get_statistics(),
            'root_nodes': list(self.root_nodes),
            'levels': {}
        }
        
        # 按层级组织数据
        for level in sorted(self.levels.keys()):
            level_data = []
            for node_id in self.levels[level]:
                if node_id in self.terms:
                    term_info = self.terms[node_id]
                    level_data.append({
                        'id': node_id,
                        'name': term_info.get('name', ''),
                        'definition': term_info.get('definition', ''),
                        'children_count': len(self.children[node_id]),
                        'parents_count': len(self.parents[node_id])
                    })
            hierarchy_data['levels'][f'level_{level}'] = level_data
            
        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_data, f, ensure_ascii=False, indent=2)
            
    def export_levels_csv(self, output_prefix):
        """分层导出到CSV文件"""
        # 导出统计信息
        stats_file = f"{output_prefix}_statistics.csv"
        with open(stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            stats = self.get_statistics()
            writer.writerow(['Total Terms', stats['total_terms']])
            writer.writerow(['Total Levels', stats['total_levels']])
            writer.writerow(['Root Nodes Count', stats['root_nodes_count']])
            
            # 各层级节点数量分布
            writer.writerow(['', ''])
            writer.writerow(['Level', 'Node Count'])
            for level in sorted(stats['levels_distribution'].keys()):
                writer.writerow([f'Level {level}', stats['levels_distribution'][level]])
                
        # 分层导出详细信息
        for level in sorted(self.levels.keys()):
            level_file = f"{output_prefix}_level_{level}.csv"
            with open(level_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Name', 'Definition', 'Children Count', 'Parents Count', 'Parents', 'Children'])
                
                # 准备排序数据：先收集所有节点信息，然后按Children Count降序排序
                level_nodes = []
                for node_id in self.levels[level]:
                    if node_id in self.terms:
                        term_info = self.terms[node_id]
                        parents_list = list(self.parents[node_id])
                        children_list = list(self.children[node_id])
                        
                        level_nodes.append({
                            'id': node_id,
                            'name': term_info.get('name', ''),
                            'definition': term_info.get('definition', ''),
                            'children_count': len(children_list),
                            'parents_count': len(parents_list),
                            'parents': '|'.join(parents_list),
                            'children': '|'.join(children_list)
                        })
                
                # 按Children Count降序排序，如果Children Count相同则按名称排序
                level_nodes.sort(key=lambda x: (-x['children_count'], x['name']))
                
                # 写入排序后的数据
                for node in level_nodes:
                    writer.writerow([
                        node['id'],
                        node['name'],
                        node['definition'],
                        node['children_count'],
                        node['parents_count'],
                        node['parents'],
                        node['children']
                    ])
def is_human_disease(term_data):
    """判断是否为人类疾病"""
    term_name = term_data.get('name', '').lower()
    term_def = term_data.get('definition', '').lower()
    raw_term = term_data.get('raw_term', {})
    
    # 排除明确的非人类疾病关键词
    animal_keywords = [
        'animal', 'pig', 'swine', 'porcine', 'bovine', 'cattle', 'sheep', 'ovine',
        'goat', 'equine', 'horse', 'chicken', 'poultry', 'avian', 'bird',
        'fish', 'feline', 'cat', 'canine', 'dog', 'mouse', 'rat', 'rodent',
        'primate', 'monkey', 'ape', 'rabbit', 'hamster', 'guinea pig',
        'livestock', 'veterinary', 'zoological', 'wild', 'zoo'
    ]
    
    # 检查名称和定义中是否包含动物关键词
    for keyword in animal_keywords:
        if keyword in term_name or keyword in term_def:
            return False
    
    # 检查cross_reference字段中是否有动物疾病数据库
    xrefs = raw_term.get('xref', [])
    if isinstance(xrefs, list):
        for xref in xrefs:
            if isinstance(xref, str):
                # 排除动物疾病数据库
                if any(db in xref.upper() for db in ['MESH:D', 'DOID:']):
                    # 这些可能包含人类疾病，需要进一步检查
                    continue
                # 排除明确的动物疾病数据库
                if any(db in xref.upper() for db in ['OMIA:', 'ANIMALQTLDB:']):
                    return False
    
    # 如果是MONDO:0000001 (disease)，这是根节点，应该保留
    if raw_term.get('id') == 'MONDO:0000001':
        return True
    
    # 检查是否有human相关的关键词（增加判断准确性）
    human_keywords = [
        'human', 'person', 'patient', 'individual', 'people',
        'man', 'woman', 'child', 'infant', 'adult', 'elderly'
    ]
    
    # 如果明确提到human相关词汇，则认为是人类疾病
    for keyword in human_keywords:
        if keyword in term_name or keyword in term_def:
            return True
    
    # 检查subset字段，如果包含human相关的子集标记
    subset = raw_term.get('subset', [])
    if isinstance(subset, list):
        for s in subset:
            if isinstance(s, str) and 'human' in s.lower():
                return True
    
    # 默认情况下，如果没有明确的动物标识，则认为是人类疾病
    # 因为MONDO主要关注人类疾病
    return True

def is_human_disease_by_lineage(term_id, all_terms, checked_cache=None):
    """通过疾病谱系判断是否为人类疾病"""
    if checked_cache is None:
        checked_cache = set()
    
    if term_id in checked_cache:
        return True  # 避免循环引用
    
    checked_cache.add(term_id)
    
    # 找到对应的term数据
    term_data = None
    for term in all_terms:
        if term.get('id') == term_id:
            term_data = term
            break
    
    if not term_data:
        return True  # 如果找不到，默认为人类疾病
    
    # 检查当前term
    if not is_human_disease({'name': term_data.get('name', ''), 
                            'definition': term_data.get('def', ''),
                            'raw_term': term_data}):
        return False
    
    # 检查父类疾病
    is_a_relations = term_data.get('is_a', [])
    if is_a_relations:
        parent_ids = extract_ontology_ids(is_a_relations)
        for parent_id in parent_ids:
            if parent_id.startswith('MONDO:'):
                if not is_human_disease_by_lineage(parent_id, all_terms, checked_cache):
                    return False
    
    return True

def process_terms(terms, writer_node, writer_edge):
    # Get id, name, and definition
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
        # Check if id contains an url
        if "http" in term_id:
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
            writer_node.writerow(node)

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
        # exit(0)
        
def process_terms_for_hierarchy(terms, hierarchy):
    """处理术语并构建层级结构，只包含人类疾病"""
    print(f"Total terms: {len(terms)}")
    
    # 第一遍：收集所有MONDO术语并过滤人类疾病
    mondo_terms = {}
    human_disease_count = 0
    
    for term in terms:
        if "id" not in term or "is_obsolete" in term:
            continue
            
        term_id = term.get("id")
        if "http" in term_id or not term_id.startswith("MONDO:"):
            continue
            
        term_name = term.get("name", "")
        term_def = term.get("def", "")
        
        # 处理定义
        if isinstance(term_def, dict) and "values" in term_def:
            term_def = "|".join(term_def["values"])
        elif isinstance(term_def, list):
            term_def = "|".join(str(d) for d in term_def)
        else:
            term_def = str(term_def) if term_def else ""
        
        # 检查是否为人类疾病
        term_data_for_check = {
            'name': term_name,
            'definition': term_def,
            'raw_term': term
        }
        
        if is_human_disease(term_data_for_check):
            mondo_terms[term_id] = {
                'name': term_name,
                'definition': term_def,
                'raw_term': term
            }
            
            hierarchy.add_term(term_id, {
                'name': term_name,
                'definition': term_def
            })
            human_disease_count += 1
    
    print(f"Found {len(mondo_terms)} MONDO terms ({human_disease_count} human diseases)")
    
    # 第二遍：构建is_a关系（只在人类疾病之间）
    relation_count = 0
    for term_id, term_data in mondo_terms.items():
        term = term_data['raw_term']
        term_is_a = term.get("is_a")
        
        if term_is_a:
            parent_ids = extract_ontology_ids(term_is_a)
            for parent_id in parent_ids:
                # 只处理MONDO术语之间的关系，并且双方都是人类疾病
                if parent_id.startswith("MONDO:") and parent_id in mondo_terms:
                    hierarchy.add_is_a_relation(term_id, parent_id)
                    relation_count += 1
                    
    print(f"Built {relation_count} is_a relationships among human diseases")
    
    # 构建层级结构
    hierarchy.find_root_nodes()
    hierarchy.build_levels()
    
    return hierarchy

# 主函数
def main():
    parser = argparse.ArgumentParser(description="分析MONDO人类疾病分类层级结构")
    parser.add_argument('-i', '--input', type=str, required=True, help="输入的MONDO OBO文件")
    parser.add_argument('-o', '--output', type=str, required=True, help="输出文件前缀")
    parser.add_argument('--json', action='store_true', help="同时输出JSON格式")
    parser.add_argument('--human-only', action='store_true', default=True, help="只包含人类疾病（默认启用）")

    args = parser.parse_args()
    
    obo_file = args.input
    output_prefix = args.output

    print(f"开始处理MONDO OBO文件：{obo_file}")
    print(f"过滤条件：只包含人类疾病")

    # 解析OBO文件
    all_terms = parse_obo_nested(obo_file)
    
    # 创建层级结构分析器
    hierarchy = DiseaseHierarchy()
    
    # 构建层级结构
    process_terms_for_hierarchy(all_terms, hierarchy)
    
    # 输出统计信息
    stats = hierarchy.get_statistics()
    print("\n=== 人类疾病层级结构统计 ===")
    print(f"人类疾病术语数: {stats['total_terms']}")
    print(f"层级数: {stats['total_levels']}")
    print(f"根节点数: {stats['root_nodes_count']}")
    print("\n各层级节点分布:")
    for level in sorted(stats['levels_distribution'].keys()):
        print(f"  第{level}层: {stats['levels_distribution'][level]}个节点")
    
    # 显示根节点信息（最顶级的人类疾病分类）
    print(f"\n=== 最顶级的人类疾病分类 ===")
    root_diseases = []
    for root_id in sorted(hierarchy.root_nodes):
        if root_id in hierarchy.terms:
            term_info = hierarchy.terms[root_id]
            root_diseases.append((root_id, term_info['name']))
            print(f"{root_id}: {term_info['name']}")
    
    # 导出结果
    hierarchy.export_levels_csv(output_prefix)
    print(f"\n结果已导出到 {output_prefix}_*.csv 文件")
    
    if args.json:
        json_file = f"{output_prefix}_hierarchy.json"
        hierarchy.export_hierarchy(json_file)
        print(f"JSON格式结果已导出到 {json_file}")
    
    print(f"\n共识别出 {len(root_diseases)} 个最顶级的人类疾病分类")

if __name__ == "__main__":
    main()