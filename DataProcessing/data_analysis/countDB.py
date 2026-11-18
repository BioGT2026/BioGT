import csv
import sys
import re
from collections import defaultdict

def countRefseq(file):
    """
    统计RefSeq数据库中每种类型的节点数量
    :param file: 输入的CSV文件路径
    :return: None
    """
    fieldnames = ["ID", "Type", "Name", "Sequence", "Comments"]
    
    with open(file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        count = 0
        count_types = defaultdict(int)
        next(reader)  # 跳过表头行
        for row in reader:
            count += 1
            node_type = row["Type"]
            if node_type:
                count_types[node_type] += 1
        
        for node_type, count in count_types.items():
            print(f"{node_type}: {count}")


def countBioMarker(file):
    """
    统计生物标记数据库中每种类型的节点数量
    :param file: 输入的CSV文件路径
    :return: None
    """
    fieldnames = [
        "Biomarker ID", "Biomarker Name", "Gene Name", "Synonyms",
        "Biomarker Type(es)", "Biomarker Class Biomarker Class", "Function",
        "Gene ID", "HGNC ID", "RefSeq", "Ensembl ID", "UniProt ID", "Sequence",
        "Chromosomal location", "Precursor Accession", "Mature Accession",
        "TTD Target ID", "KEGG ID", "Pubchem CID",
        "Type of approval appropriate for", "Drug mechanism of action",
        "Age Range", "Disease or Use & Patient Population", "Status"
    ]
    
    with open(file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=fieldnames)
        count = 0
        count_types = defaultdict(int)
        next(reader)  # 跳过表头行
        for row in reader:
            count += 1
            node_type = row["Biomarker Class Biomarker Class"]
            if node_type:
                count_types[node_type] += 1
        
        for node_type, count in count_types.items():
            print(f"{node_type}: {count}")

file=sys.argv[1]

# countRefseq(file)
countBioMarker(file)