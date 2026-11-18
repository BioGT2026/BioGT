from Bio import SeqIO
import argparse
import csv
import math
import os
from DataProcessing.kg_construction.commonSettings import *

refseqProteinToUniprotIDMapping = {}
goTermToTypeMapping = {}
human_genes = set()

def process_gene_neighbors_to_csv(input_file, writer_edge):
    #tax_id GeneID  genomic_accession.version       genomic_gi      start_position  end_position  orientation      chromosome      GeneIDs_on_left distance_to_left        GeneIDs_on_right      distance_to_right        overlapping_GeneIDs     assembly
    global human_genes
    total_edge_num = 0
    if len(human_genes) == 0:
        print("Human genes set is empty, please ensure gene_info file is processed first.")
        return
    # 先找到最大距离
    max_distance = 0
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            distance_to_left = row["distance_to_left"]
            distance_to_right = row["distance_to_right"]
            # 处理节点信息
            if tax_id == "9606" or tax_id == 9606: # In case tax_id is an integer
                if distance_to_left != "-":
                    distance_to_left = int(distance_to_left)
                    if distance_to_left > max_distance:
                        max_distance = distance_to_left
                if distance_to_right != "-":
                    distance_to_right = int(distance_to_right)
                    if distance_to_right > max_distance:
                        max_distance = distance_to_right
    print(f"Max distance: {max_distance}")
    # 然后重新打开文件，处理边关系
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            genomic_accession = row["genomic_accession.version"]
            start_position = row["start_position"]
            end_position = row["end_position"]
            orientation = row["orientation"]
            chromosome = row["chromosome"]
            gene_ids_on_left = row["GeneIDs_on_left"]
            distance_to_left = row["distance_to_left"]
            gene_ids_on_right = row["GeneIDs_on_right"]
            distance_to_right = row["distance_to_right"]
            overlapping_gene_ids = row["overlapping_GeneIDs"]
            assembly = row["assembly"]

            # 处理节点信息
            if tax_id == "9606" or tax_id == 9606:
                # Split gene_ids_on_left and gene_ids_on_right into lists
                gene_ids_on_left = gene_ids_on_left.split("|") if gene_ids_on_left != "-" else []
                gene_ids_on_right = gene_ids_on_right.split("|") if gene_ids_on_right != "-" else []
                overlapping_gene_ids = overlapping_gene_ids.split("|") if overlapping_gene_ids != "-" else []
                # Generate distance values
                distance_to_left = int(distance_to_left) if distance_to_left != "-" else max_distance + 1
                distance_to_right = int(distance_to_right) if distance_to_right != "-" else max_distance + 1
                if gene_ids_on_left != []:
                    if distance_to_left > max_distance:
                        continue
                    # Compute weight based on distance, if distance is 0, set weight to 1, otherwise set as 1 - distance/max_distance
                    weight = 1.0 if distance_to_left == 0 else 1.0 - (distance_to_left / max_distance)
                    for gene_id_on_left in gene_ids_on_left:
                        if gene_id_on_left not in human_genes:
                            continue
                        total_edge_num += 1
                        edge_data = {
                            "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_gene",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": weight,
                            "Node 2 ID": NCBI_GENE_PREFIX+gene_id_on_left,
                            "Node 2 Data Type": "Gene",
                            "Node 2 Source": "NCBI"
                        }
                        writer_edge.writerow(edge_data)
                if gene_ids_on_right != []:
                    if distance_to_right > max_distance:
                        continue
                    # Compute weight based on distance, if distance is 0, set weight to 1, otherwise set as 1 - distance/max_distance
                    weight = 1.0 if distance_to_right == 0 else 1.0 - (distance_to_right / max_distance)
                    for gene_id_on_right in gene_ids_on_right:
                        if gene_id_on_right not in human_genes:
                            continue
                        total_edge_num += 1
                        edge_data = {
                            "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_gene",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": weight,
                            "Node 2 ID": NCBI_GENE_PREFIX+gene_id_on_right,
                            "Node 2 Data Type": "Gene",
                            "Node 2 Source": "NCBI"
                        }
                        writer_edge.writerow(edge_data)
                if overlapping_gene_ids != []:
                    for overlapping_gene_id in overlapping_gene_ids:
                        if overlapping_gene_id not in human_genes:
                            continue
                        total_edge_num += 1
                        edge_data = {
                            "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_gene",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": 1.0,
                            "Node 2 ID": NCBI_GENE_PREFIX+overlapping_gene_id,
                            "Node 2 Data Type": "Gene",
                            "Node 2 Source": "NCBI"
                        }
                        writer_edge.writerow(edge_data)
    print(f"Processed {total_edge_num} edges for gene neighbors.")

def process_gene_group_to_csv(input_file, writer_node, writer_edge):
    #tax_id GeneID  relationship    Other_tax_id    Other_GeneID
    # Open the input file as CSV
    total_edge_num = 0
    total_node_num = 0
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            relationship = row["relationship"]
            other_tax_id = row["Other_tax_id"]
            other_gene_id = row["Other_GeneID"]
            # 处理节点信息
            if tax_id == "9606" and other_tax_id == "9606":
                edge_type = edge_types.get("UD")
                if relationship in ["Related pseudogene", "Related functional gene", "Potential readthrough sibling", "Readthrough sibling"]:
                    edge_type = edge_types.get("UD")
                    # 处理边信息
                    edge_data = {
                        "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_gene",
                        "Edge Direction": edge_type,
                        "Edge Weight": 1.0,
                        "Node 2 ID": NCBI_GENE_PREFIX+other_gene_id,
                        "Node 2 Data Type": "Gene",
                        "Node 2 Source": "NCBI"
                    }
                    writer_edge.writerow(edge_data)
                    total_edge_num += 1
                elif relationship in ["Readthrough parent", "Region member", "Region parent"]:
                    edge_type = edge_types.get("SD")
                    # 处理边信息
                    edge_data = {
                        "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_gene",
                        "Edge Direction": edge_type,
                        "Edge Weight": 1.0,
                        "Node 2 ID": NCBI_GENE_PREFIX+other_gene_id,
                        "Node 2 Data Type": "Gene",
                        "Node 2 Source": "NCBI"
                    }
                    writer_edge.writerow(edge_data)
                    total_edge_num += 1
                elif relationship in ["Readthrough child"]:
                    edge_type = edge_types.get("SD")
                    # 处理边信息
                    edge_data = {
                        "Node 1 ID": NCBI_GENE_PREFIX+other_gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_gene",
                        "Edge Direction": edge_type,
                        "Edge Weight": 1.0,
                        "Node 2 ID": NCBI_GENE_PREFIX+gene_id,
                        "Node 2 Data Type": "Gene",
                        "Node 2 Source": "NCBI"
                    }
                    writer_edge.writerow(edge_data)
                    total_edge_num += 1
                else:
                    with open("unknown_relationship_group.log", "a") as log_file:
                        log_file.write(f"Unknown relationship type: {relationship} for GeneID: {gene_id}, Other GeneID: {other_gene_id}\n")
                        # Close 
                        log_file.close()
                    continue
    print(f"Processed {total_edge_num} edges for gene groups.")

def process_gene_orthologs_to_csv(input_file, writer_node, writer_edge):
    #tax_id GeneID  relationship    Other_tax_id    Other_GeneID
    # Open the input file as CSV
    total_edge_num = 0
    total_node_num = 0
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            relationship = row["relationship"]
            other_tax_id = row["Other_tax_id"]
            other_gene_id = row["Other_GeneID"]
            # 处理节点信息
            if tax_id == "9606" and other_tax_id == "9606": # 人类
                if relationship not in ["ortholog"]:
                    with open("unknown_relationship.log", "a") as log_file:
                        log_file.write(f"Unknown relationship type: {relationship} for GeneID: {gene_id}, Other GeneID: {other_gene_id}\n")
                        # Close 
                        log_file.close()
                    continue
                # 处理边信息
                edge_data = {
                    "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                    "Node 1 Data Type": "Gene",
                    "Node 1 Source": "NCBI",
                    "Edge Type": "gene_gene",
                    "Edge Direction": edge_types.get("DD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": NCBI_GENE_PREFIX+other_gene_id,
                    "Node 2 Data Type": "Gene",
                    "Node 2 Source": "NCBI"
                }
                writer_edge.writerow(edge_data)
                total_edge_num += 1
    print(f"Processed {total_edge_num} edges for gene orthologs.")

def process_gene_refseq_to_csv(input_file, writer_node, writer_edge):
    #tax_id GeneID  status  RNA_nucleotide_accession.version        RNA_nucleotide_gi       protein_accession.version      protein_gi      genomic_nucleotide_accession.version    genomic_nucleotide_gi  start_position_on_the_genomic_accession end_position_on_the_genomic_accession   orientation    assembly        mature_peptide_accession.version        mature_peptide_gi       Symbol
    # Open the input file as CSV
    total_edge_num = 0
    total_node_num = 0
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            status = row["status"]
            # ===== NCBI RefSeq 基因注释字段解析 =====
            # 注：以下字段从RefSeq注释文件中提取，可能包含占位符（-/?）表示缺失数据

            # RNA转录本访问号（RefSeq或GenBank格式）
            # 格式：[前缀][编号].版本 例如：NM_001301.3（mRNA）、NR_003051.1（ncRNA）
            # - 前缀含义：NM_（已验证mRNA）、XM_（预测mRNA）、NR_（非编码RNA）
            # - 值为"-"表示该基因没有对应的RNA记录（常见于原核基因）
            rna_accession = row["RNA_nucleotide_accession.version"]  

            # 蛋白质产物访问号（RefSeq格式）
            # 格式示例：NP_001234.1（已验证）、XP_005678.2（预测）、AVV84537.1（自动注释）
            # - 细菌基因通常以非标准前缀（如AVV）开头，表示自动预测模型
            # - 这是分析蛋白质功能的关键字段（可用于BLAST、UniProt查询）
            protein_accession = row["protein_accession.version"]  

            # 基因组序列访问号（染色体/质粒/contig）
            # 前缀含义：
            #   NC_：完整染色体（如NC_000913.3 大肠杆菌K12）
            #   CP_：完整质粒/基因组（常见于细菌，如CP028435.1）
            #   NG_：未完成组装的基因组片段
            genomic_accession = row["genomic_nucleotide_accession.version"]  

            # 基因在基因组上的物理位置（1-based坐标）
            # 注意：
            # 1. 原核基因通常为连续编码，真核基因可能有内
            #    含子
            # 2. 当值为"-"时需通过序列比对重新确定位置
            start_position = row["start_position_on_the_genomic_accession"]  # 起始位置
            end_position = row["end_position_on_the_genomic_accession"]      # 终止位置
            # 基因的链方向
            # 可能值：
            #   "+"：正链（5'->3'方向与基因组一致）
            #   "-"：负链（互补链）
            #   "?"：未知方向（需通过序列分析确定）
            orientation = row["orientation"]
            # 基因组组装信息（人类常用GRCh38，细菌通常为"-"）
            # 示例：GCF_000005845.2（大肠杆菌K12的组装ID）
            assembly = row["assembly"]
            # 成熟肽访问号（翻译后修饰产物的RefSeq编号）
            # - 常见于切除信号肽或前体后的功能性肽段
            # - 细菌中通常为"-"（未单独注释）
            mature_peptide_accession = row["mature_peptide_accession.version"]
            # 官方基因符号（标准命名）
            # 命名规则：
            # - 真核基因：大写+斜体（如TP53）
            # - 原核基因：全小写（如dnaA、gyrB）
            # - 符号可能随研究进展变化（需定期更新）
            symbol = row["Symbol"]
            # 处理节点信息
            if tax_id == "9606": # 人类
                # 处理边信息
                if rna_accession != "-":
                    # print(f"Processing edge for GeneID: {gene_id} with RNA Accession: {rna_accession}")
                    # 处理边信息
                    edge_data = {   
                        "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_rna",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": 1.0,
                        "Node 2 ID": rna_accession,
                        "Node 2 Data Type": "RNA",
                        "Node 2 Source": "RefSeq"
                    }
                    writer_edge.writerow(edge_data)
                    total_edge_num += 1
                if protein_accession != "-":
                    # print(f"Processing edge for GeneID: {gene_id} with Protein Accession: {protein_accession}")
                    # 处理边信息
                    uniprotID = refseqProteinToUniprotIDMapping.get(protein_accession, None)
                    if uniprotID is not None:
                        edge_data = {
                            "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_protein",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": 1.0,
                            "Node 2 ID": UNIPROT_PROTEIN_PREFIX + uniprotID,
                            "Node 2 Data Type": "Protein",
                            "Node 2 Source": "Uniprot"
                        }
                        writer_edge.writerow(edge_data)
                        total_edge_num += 1
    print(f"Processed {total_edge_num} edges for gene RefSeq annotations.")

def process_gene_go_to_csv(input_file, writer_node, writer_edge):
    #tax_id GeneID  GO_ID   Evidence        Qualifier       GO_term PubMed  Category
    # Open the input file as CSV
    total_edge_num = 0
    total_node_num = 0
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            go_id = row["GO_ID"]
            evidence = row["Evidence"]
            qualifier = row["Qualifier"]
            go_term = row["GO_term"]
            pubmed = row["PubMed"]
            category = row["Category"]
            # 处理节点信息 
            if tax_id == "9606": # 人类
                edge_weight = evidence_to_weight(evidence)
                if edge_weight == 0.0:
                    continue # 跳过边权重为0的边

                if go_id in goTermToTypeMapping:
                    current_go_type = goTermToTypeMapping[go_id]
                    # 处理边信息
                    edge_data = {
                        "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_go",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": edge_weight,
                        "Node 2 ID": go_id,
                        "Node 2 Data Type": current_go_type,
                        "Node 2 Source": "GO"
                    }
                    writer_edge.writerow(edge_data)
                    total_edge_num += 1
    print(f"Processed {total_edge_num} edges for gene GO annotations.")

def process_gene_accession_to_csv(input_file, writer_node, writer_edge):
    #tax_id GeneID  status  RNA_nucleotide_accession.version        RNA_nucleotide_gi       protein_accession.version      protein_gi      genomic_nucleotide_accession.version    genomic_nucleotide_gi  start_position_on_the_genomic_accession end_position_on_the_genomic_accession   orientation    assembly        mature_peptide_accession.version        mature_peptide_gi       Symbol
    # Open the input file as CSV
    total_edge_num = 0
    total_node_num = 0
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            status = row["status"]
            # ===== NCBI RefSeq 基因注释字段解析 =====
            # 注：以下字段从RefSeq注释文件中提取，可能包含占位符（-/?）表示缺失数据

            # RNA转录本访问号（RefSeq或GenBank格式）
            # 格式：[前缀][编号].版本 例如：NM_001301.3（mRNA）、NR_003051.1（ncRNA）
            # - 前缀含义：NM_（已验证mRNA）、XM_（预测mRNA）、NR_（非编码RNA）
            # - 值为"-"表示该基因没有对应的RNA记录（常见于原核基因）
            rna_accession = row["RNA_nucleotide_accession.version"]  

            # 蛋白质产物访问号（RefSeq格式）
            # 格式示例：NP_001234.1（已验证）、XP_005678.2（预测）、AVV84537.1（自动注释）
            # - 细菌基因通常以非标准前缀（如AVV）开头，表示自动预测模型
            # - 这是分析蛋白质功能的关键字段（可用于BLAST、UniProt查询）
            protein_accession = row["protein_accession.version"]  

            # 基因组序列访问号（染色体/质粒/contig）
            # 前缀含义：
            #   NC_：完整染色体（如NC_000913.3 大肠杆菌K12）
            #   CP_：完整质粒/基因组（常见于细菌，如CP028435.1）
            #   NG_：未完成组装的基因组片段
            genomic_accession = row["genomic_nucleotide_accession.version"]  

            # 基因在基因组上的物理位置（1-based坐标）
            # 注意：
            # 1. 原核基因通常为连续编码，真核基因可能有内含子
            # 2. 当值为"-"时需通过序列比对重新确定位置
            start_position = row["start_position_on_the_genomic_accession"]  # 起始位置
            end_position = row["end_position_on_the_genomic_accession"]      # 终止位置

            # 基因的链方向
            # 可能值：
            #   "+"：正链（5'->3'方向与基因组一致）
            #   "-"：负链（互补链）
            #   "?"：未知方向（需通过序列分析确定）
            orientation = row["orientation"]  

            # 基因组组装信息（人类常用GRCh38，细菌通常为"-"）
            # 示例：GCF_000005845.2（大肠杆菌K12的组装ID）
            assembly = row["assembly"]  

            # 成熟肽访问号（翻译后修饰产物的RefSeq编号）
            # - 常见于切除信号肽或前体后的功能性肽段
            # - 细菌中通常为"-"（未单独注释）
            mature_peptide_accession = row["mature_peptide_accession.version"]  

            # 官方基因符号（标准命名）
            # 命名规则：
            # - 真核基因：大写+斜体（如TP53）
            # - 原核基因：全小写（如dnaA、gyrB）
            # - 符号可能随研究进展变化（需定期更新）
            symbol = row["Symbol"]  

            if tax_id == "9606":
                # 处理边信息
                if rna_accession != "-":
                    # print(f"Processing edge for GeneID: {gene_id} with RNA Accession: {rna_accession}")
                    # 处理边信息
                    edge_data = {   
                        "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_rna",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": 1.0,
                        "Node 2 ID": rna_accession,
                        "Node 2 Data Type": "RNA",
                        "Node 2 Source": "RefSeq"
                    }
                    writer_edge.writerow(edge_data)
                    total_edge_num += 1
                if protein_accession != "-":
                    # print(f"Processing edge for GeneID: {gene_id} with Protein Accession: {protein_accession}")
                    # 处理边信息
                    uniprotID = refseqProteinToUniprotIDMapping.get(protein_accession, None)
                    if uniprotID is not None:
                        edge_data = {
                            "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                            "Node 1 Data Type": "Gene",
                            "Node 1 Source": "NCBI",
                            "Edge Type": "gene_protein",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": 1.0,
                            "Node 2 ID": UNIPROT_PROTEIN_PREFIX + uniprotID,
                            "Node 2 Data Type": "Protein",
                            "Node 2 Source": "Uniprot"
                        }
                        writer_edge.writerow(edge_data)
                        total_edge_num += 1
    print(f"Processed {total_edge_num} edges for gene accession annotations.")

def process_gene_info_to_csv(input_file, writer_node, writer_edge):
    #tax_id GeneID  Symbol  LocusTag        Synonyms        dbXrefs chromosome      map_location  description      type_of_gene    Symbol_from_nomenclature_authority      Full_name_from_nomenclature_authority  Nomenclature_status     Other_designations      Modification_date       Feature_type
    total_edge_num = 0
    total_node_num = 0
    # Open the input file as CSV
    with open(input_file, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        
        # Iterate through each row in the input file
        for row in reader:
            tax_id = row["#tax_id"]
            gene_id = row["GeneID"]
            symbol = row["Symbol"]
            locus_tag = row["LocusTag"]
            synonyms = row["Synonyms"]
            db_xrefs = row["dbXrefs"]
            chromosome = row["chromosome"]
            map_location = row["map_location"]
            description = row["description"]
            type_of_gene = row["type_of_gene"]
            symbol_from_authority = row["Symbol_from_nomenclature_authority"]
            full_name_from_authority = row["Full_name_from_nomenclature_authority"]
            nomenclature_status = row["Nomenclature_status"]
            other_designations = row["Other_designations"]
            feature_type = row["Feature_type"]
            # 处理节点信息
            if tax_id == "9606" and type_of_gene == "protein-coding": # 人类 + 蛋白质编码基因
                comments_parts = []
                if symbol != "-":
                    comments_parts.append(f"Symbol: {symbol}")
                if locus_tag != "-":
                    comments_parts.append(f"Locus Tag: {locus_tag}")
                if synonyms != "-":
                    comments_parts.append(f"Synonyms: {synonyms}")
                if chromosome != "-":
                    comments_parts.append(f"Chromosome: {chromosome}")
                if map_location != "-":
                    comments_parts.append(f"Map Location: {map_location}")
                if description != "-":
                    comments_parts.append(f"Description: {description}")
                if type_of_gene != "-":
                    comments_parts.append(f"Type of Gene: {type_of_gene}")
                if full_name_from_authority != "-":
                    comments_parts.append(f"Full Name from Authority: {full_name_from_authority}")
                if symbol_from_authority != "-":
                    comments_parts.append(f"Symbol from Authority: {symbol_from_authority}")
                if nomenclature_status != "-":
                    comments_parts.append(f"Nomenclature Status: {nomenclature_status}")
                if other_designations != "-":
                    comments_parts.append(f"Other Designations: {other_designations}")
                if feature_type != "-":
                    comments_parts.append(f"Feature Type: {feature_type}")
                comments = ", ".join(comments_parts)
                node_data = {
                    "ID": NCBI_GENE_PREFIX+gene_id,
                    "Type": "Gene",
                    "Name": symbol,
                    "Sequence": "",
                    "Comments": comments
                }
                human_genes.add(gene_id)  # 添加到人类基因集合中
                total_node_num += 1
                # print(f"Processing node for GeneID: {gene_id} with Name: {full_name_from_authority}")
                writer_node.writerow(node_data)
                # 处理边信息
                # if db_xrefs != "-":
                #     print(f"Processing edge for GeneID: {gene_id} with db_xrefs: {db_xrefs}")
                #     # 处理边信息
                #     data_type_prefix = db_xrefs.split(":")[0]
                #     edge_data = {
                #         "Node 1 ID": NCBI_GENE_PREFIX+gene_id,
                #         "Node 1 Data Type": "Gene",
                #         "Node 1 Source": "NCBI",
                #         "Edge Type": "gene_"+data_type_prefix,  # gene -> data_type_prefix
                #         "Edge Direction": edge_types.get("UD"),
                #         "Edge Weight": 1.0,
                #         "Node 2 ID": db_xrefs,
                #         "Node 2 Data Type": data_type_prefix,
                #         "Node 2 Source": "NCBI"
                #     }
                #     writer_edge.writerow(edge_data)
    print(f"Processed {total_node_num} nodes for gene info annotations.")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Gene log file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file")
    parser.add_argument('-p', '--protein', type=str, required=True, help="Input protein mappping file")
    parser.add_argument('-g', '--go', type=str, required=True, help="Input go mappping file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    input_file_path = args.input
    proteinIDFilePath = args.protein
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]
    # Open GO Type file
    global goTermToTypeMapping
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
    # Open Protein ID file
    global refseqProteinToUniprotIDMapping
    with open(proteinIDFilePath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)  # 跳过标题行
        for row in reader:
            if len(row) >= 6:
                uniprot_id = row[0]
                refseq_id = row[6]
                refseqProteinToUniprotIDMapping[refseq_id] = uniprot_id
            else:
                print(f"Skipping row due to insufficient columns: {row}")

    if not refseqProteinToUniprotIDMapping:
        print(f"Protein ID mapping为空，请检查文件: {proteinIDFilePath}")
        return
    print(f"Protein ID Mapping loaded, total {len(refseqProteinToUniprotIDMapping)} entries.")

    files = ["gene_info", "gene2accession", "gene2go", "gene2refseq", "gene_orthologs", "gene_group", "gene_neighbors"]
    
    with open(output_node_csv, "w", newline="") as file_node, open  (output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()
        
        for file in files:
            print(f"开始处理文件：{file}")
            file_path = os.path.join(input_file_path, file)
            if not os.path.isfile(file_path):
                print(f"文件 {file} 不存在或不是一个有效的文件，请检查路径。")
                continue
            if file_path.endswith("gene2go"):
                process_gene_go_to_csv(file_path, writer_node, writer_edge)
            elif file_path.endswith("gene2accession"):
                process_gene_accession_to_csv(file_path, writer_node, writer_edge)
            elif file_path.endswith("gene_info"):
                process_gene_info_to_csv(file_path, writer_node, writer_edge)
            elif file_path.endswith("gene2refseq"):
                process_gene_refseq_to_csv(file_path, writer_node, writer_edge)
            elif file_path.endswith("gene_orthologs"):
                process_gene_orthologs_to_csv(file_path, writer_node, writer_edge)
            elif file_path.endswith("gene_group"):
                process_gene_group_to_csv(file_path, writer_node, writer_edge)
            elif file_path.endswith("gene_neighbors"):
                process_gene_neighbors_to_csv(file_path, writer_edge)
            else:
                print("Unknown file type. Please provide a valid gene log file.")
            # 打印成功消息
            print(f"数据已成功导出到 节点记录：{output_node_csv}, 边记录：{output_edge_csv}")

if __name__ == "__main__":
    main()