import xml.etree.ElementTree as ET
import sys

def print_xml_tree(node, depth=0, is_last=False, hierarchy=[]):
    """
    递归打印XML节点树
    :param node: 当前节点
    :param depth: 当前深度
    :param is_last: 是否是父节点的最后一个子节点
    :param hierarchy: 层级关系标记列表
    """
    # 生成前缀符号
    prefix = ""
    for i in range(depth):
        if i < len(hierarchy) and not hierarchy[i]:
            prefix += "│   "
        else:
            prefix += "    "

    # 节点符号
    if depth > 0:
        prefix += "└── " if is_last else "├── "

    # 打印当前节点
    print(f"{prefix}{node.tag}")

    # 递归处理子节点
    children = list(node)
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_xml_tree(child, depth + 1, is_last_child, hierarchy + [is_last])

def main():
    if len(sys.argv) != 2:
        print("Usage: python xml_tree.py <input.xml>")
        sys.exit(1)

    xml_file = sys.argv[1]
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print("XML Structure Tree:")
        print_xml_tree(root)
    except FileNotFoundError:
        print(f"Error: File '{xml_file}' not found")
    except ET.ParseError:
        print(f"Error: Invalid XML format in '{xml_file}'")

if __name__ == "__main__":
    main()
