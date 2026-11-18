import csv
import argparse
import sys

def find_complete_row(csv_file):
    """
    从CSV文件中查找第一行具有完整字段的记录
    如果没有找到，则返回最后一行
    """
    fieldnames_node = ["ID", "Type", "Name", "Sequence", "Comments"]
    
    first_complete_row = None
    last_row = None
    row_number = 0
    first_complete_row_number = 0
    
    try:
        with open(csv_file, mode="r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            
            # 检查CSV文件是否包含所需的字段
            if not all(field in reader.fieldnames for field in fieldnames_node):
                print(f"错误：CSV文件缺少必需的字段。")
                print(f"需要的字段：{fieldnames_node}")
                print(f"文件中的字段：{reader.fieldnames}")
                return None
            
            for row in reader:
                row_number += 1
                last_row = row  # 保存最后一行
                
                # 检查所有必需字段是否都非空
                if first_complete_row is None:
                    is_complete = True
                    for field in fieldnames_node:
                        value = row.get(field, "").strip()
                        if not value or value.lower() == "nan":
                            is_complete = False
                            break
                    
                    if is_complete:
                        first_complete_row = row
                        first_complete_row_number = row_number
                        break  # 找到第一行完整记录，退出循环
        
        # 输出结果
        if first_complete_row:
            print(f"找到第一行完整记录（第 {first_complete_row_number} 行）：")
            for field in fieldnames_node:
                print(f"  {field}: {first_complete_row[field]}")
            return first_complete_row
        else:
            print(f"未找到完整记录，输出最后一行（第 {row_number} 行）：")
            if last_row:
                for field in fieldnames_node:
                    value = last_row.get(field, "")
                    status = "✓" if value and value.strip() and value.lower() != "nan" else "✗"
                    print(f"  {status} {field}: {value}")
            return last_row
    
    except FileNotFoundError:
        print(f"错误：文件 '{csv_file}' 不存在。")
        return None
    except Exception as e:
        print(f"错误：读取文件时发生异常：{e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="从CSV文件中查找第一行具有完整字段的记录，如果没有则输出最后一行。"
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="输入CSV文件路径"
    )
    
    args = parser.parse_args()
    
    result = find_complete_row(args.input)
    
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

