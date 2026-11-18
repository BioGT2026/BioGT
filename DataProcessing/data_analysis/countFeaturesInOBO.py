from collections import defaultdict
import argparse

def parse_obo(file_path):
    """
    解析OBO文件并返回特征统计信息
    Return:
        tuple (all_terms, feature_stats)
        - all_terms: 包含所有术语数据的列表
        - feature_stats: 每个特征的统计信息（出现次数、值类型）
    """
    all_terms = []
    feature_stats = defaultdict(lambda: {'count': 0, 'value_types': set()})
    
    current_term = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('!'):
                continue
                
            # 开始新术语
            if line == '[Term]':
                if current_term is not None:
                    all_terms.append(current_term)
                current_term = defaultdict(list)
                continue
                
            # 其他类型区块结束处理
            if line.startswith('['):
                if current_term is not None:
                    all_terms.append(current_term)
                current_term = None
                continue
                
            # 解析键值对
            if current_term is not None and ': ' in line:
                key, value = line.split(': ', 1)
                current_term[key].append(value)
                
                # 更新特征统计
                feature_stats[key]['count'] += 1
                value_type = 'xref' if key == 'xref' else \
                            'synonym' if key == 'synonym' else \
                            'structured' if '{' in value else 'simple'
                feature_stats[key]['value_types'].add(value_type)
    
        # 处理最后一个术语
        if current_term is not None:
            all_terms.append(current_term)
            
    return all_terms, feature_stats

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    obo_file = args.input

    print(f"开始处理OBO文件：{obo_file}")

    # 使用示例
    all_terms, feature_stats = parse_obo(obo_file)

    # 输出统计信息
    print("发现总术语数:", len(all_terms))
    print("\n所有可能的特征及其统计:")
    for feat, stats in feature_stats.items():
        print(f"• {feat}:")
        print(f"  出现次数: {stats['count']}次")
        print(f"  值类型: {', '.join(stats['value_types'])}")
        print(f"  示例值: {all_terms[0].get(feat, ['无'])[0][:50]}...")  # 显示首条记录的示例

    # 高级输出：生成特征矩阵
    print("\n特征频率矩阵（前5个术语）:")
    header = list(feature_stats.keys())
    print("ID\t" + "\t".join(header))
    for term in all_terms[:5]:
        id_val = term.get('id', [''])[0]
        presence = [str(len(term.get(h, []))) for h in header]
        print(f"{id_val}\t" + "\t".join(presence))



if __name__ == "__main__":
    main()