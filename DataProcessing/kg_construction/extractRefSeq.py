import csv
import os
import time
import argparse
import re
import concurrent.futures
import threading
import multiprocessing
from Bio import SeqIO
from DataProcessing.kg_construction.commonSettings import *

csv.field_size_limit(1000 * 1024 * 1024)
extractAllData = False
# 创建线程锁来保护CSV写入
file_lock = threading.Lock()

refseqProteinToUniprotIDMapping = {}
goTermToTypeMapping = {}
    
def parse_complex_fields_in_gbff(data):
    # 处理数据库交叉引用
    if 'Database cross-references' in data:
        dbs = {}
        for item in data['Database cross-references'].split(', '):
            k, v = item.split(':', 1)
            dbs[k.strip()] = v.strip()
        data['cross_references'] = dbs
    
    # 处理分类信息
    if 'taxonomy' in data:
        data['taxonomy'] = [x.strip("' ") for x in 
                           re.findall(r"'(.*?)'", data['taxonomy'])]
    
    # 处理参考文献
    if 'references' in data:
        data['references'] = [ref.strip() for ref in 
                             re.split(r'(?<=}),', data['references'])]
    
    return data

def parse_gbff(gbff_file, result_collector):
    """解析GBFF文件，提取基因组特征信息"""
    has_human = False
    node_rows = []
    edge_rows = []
    
    with open(gbff_file, "r") as handle:
        genesmapping = {}
        for record in SeqIO.parse(handle, "genbank"):
            current_id = getattr(record, 'id', None)
            current_name = getattr(record, 'name', None)
            current_description = getattr(record, 'description', '')
            annotations = getattr(record, 'annotations', {})
            current_organism = getattr(annotations, 'organism', None)
            features = getattr(record, 'features', [])
            # Process the record annotations
            annotations_dict = {
                "molecule_type": annotations.get("molecule_type", "N/A"),
                "topology": annotations.get("topology", "N/A"),
                "data_file_division": annotations.get("data_file_division", "N/A"),
                "accessions": annotations.get("accessions", ["N/A"]),
                "keywords": annotations.get("keywords", ["N/A"]),
                "source": annotations.get("source", "N/A"),
                "organism": annotations.get("organism", "N/A"),
                "taxonomy": annotations.get("taxonomy", ["N/A"]),
                "references": annotations.get("references", ["N/A"]),
                "comment": annotations.get("comment", "N/A"),
                "structured_comment": annotations.get("structured_comment", "N/A"),
                "contig": annotations.get("contig", "N/A"),
            }
            sequence = str(record.seq)
            current_organism = annotations_dict.get("organism", None)
            current_molecule_type = annotations_dict.get("molecule_type", "N/A")
            if (current_organism is not None and "homo sapiens" in current_organism.lower()) or extractAllData:
                has_human = True
                # Merge annotions_dict into a comments string
                comments = "description: " + current_description+"; " + "; ".join([f"{k}: {v}" for k, v in annotations_dict.items() if v != "N/A"])
                node_data = {
                    "ID": current_id,
                    "Type": current_molecule_type,
                    "Name": current_name,
                    "Sequence": sequence,
                    "Comments": comments
                }
                node_rows.append(node_data)
                
                # 遍历所有特征
                for feat in features:
                    if feat.type == "CDS":
                        genes = feat.qualifiers.get("gene", [])
                        products = feat.qualifiers.get("product", [])
                        protein_ids = feat.qualifiers.get("protein_id", [])
                        db_xrefs = feat.qualifiers.get("db_xref", [])
                        go_terms = feat.qualifiers.get("GO_function", [])
                        
                        if (genes != []):
                            for gene in genes:
                                edge = {
                                    "Node 1 ID": current_id,
                                    "Node 1 Data Type": current_molecule_type,
                                    "Node 1 Source": "RefSeq",
                                    "Edge Type": current_molecule_type + "_gene",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": NCBI_GENE_PREFIX + gene,
                                    "Node 2 Data Type": "Gene",
                                    "Node 2 Source": "NCBI"
                                }
                                edge_rows.append(edge)
                                
                        if (go_terms != []):
                            for go_term in go_terms:
                                # Safely extract GO ID
                                match_go_id = re.search(r"GO:\d+", go_term)
                                if not match_go_id:
                                    continue # Skip this term if no GO ID found
                                go_id = match_go_id.group()

                                # Safely extract evidence code
                                match_evidence = re.search(r"\[Evidence (\w+)\]", go_term)
                                if not match_evidence:
                                    evidence_code = "NA" # Assign a default
                                else:
                                    evidence_code = match_evidence.group(1)

                                edge_weight = evidence_to_weight(evidence_code)
                                if edge_weight == 0.0:
                                    continue # Skip the record that has no edge weight

                                edge = {
                                    "Node 1 ID": current_id,
                                    "Node 1 Data Type": current_molecule_type,
                                    "Node 1 Source": "RefSeq",
                                    "Edge Type": current_molecule_type + "_go",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": edge_weight,
                                    "Node 2 ID": go_id,
                                    "Node 2 Data Type": "Phenotype",
                                    "Node 2 Source": "RefSeq" 
                                }
                                edge_rows.append(edge)
                                
                        if (protein_ids != []):
                            for protein_id in protein_ids:
                                edge = {
                                    "Node 1 ID": current_id,
                                    "Node 1 Data Type": current_molecule_type,
                                    "Node 1 Source": "RefSeq",
                                    "Edge Type": current_molecule_type + "_protein",
                                    "Edge Direction": edge_types.get("SD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": UNIPROT_PROTEIN_PREFIX + protein_id,
                                    "Node 2 Data Type": "Protein",
                                    "Node 2 Source": "RefSeq"
                                }
                                edge_rows.append(edge)

    # 存储结果
    result_collector["has_human"] = has_human
    result_collector["nodes"] = node_rows
    result_collector["edges"] = edge_rows
    return has_human, node_rows, edge_rows

def parse_rna_gbff(gbff_file, result_collector):
    """解析GBFF文件，提取基因组特征信息"""
    has_human = False
    node_rows = []
    edge_rows = []
    
    with open(gbff_file, "r") as handle:
        genesmapping = {}
        for record in SeqIO.parse(handle, "genbank"):
            current_id = getattr(record, 'id', None)
            current_name = getattr(record, 'name', None)
            current_description = getattr(record, 'description', '')
            annotations = getattr(record, 'annotations', {})
            current_organism = getattr(annotations, 'organism', None)
            features = getattr(record, 'features', [])
            # Process the record annotations
            annotations_dict = {
                "molecule_type": annotations.get("molecule_type", "N/A"),
                "topology": annotations.get("topology", "N/A"),
                "data_file_division": annotations.get("data_file_division", "N/A"),
                "accessions": annotations.get("accessions", ["N/A"]),
                "keywords": annotations.get("keywords", ["N/A"]),
                "source": annotations.get("source", "N/A"),
                "organism": annotations.get("organism", "N/A"),
                "taxonomy": annotations.get("taxonomy", ["N/A"]),
                "references": annotations.get("references", ["N/A"]),
                "comment": annotations.get("comment", "N/A"),
                "structured_comment": annotations.get("structured_comment", "N/A"),
                "contig": annotations.get("contig", "N/A"),
            }
            sequence = str(record.seq)
            current_organism = annotations_dict.get("organism", None)
            current_molecule_type = annotations_dict.get("molecule_type", "N/A")
            if (current_organism is not None and "homo sapiens" in current_organism.lower()) or extractAllData:
                has_human = True
                # Merge annotions_dict into a comments string
                comments = "description: " + current_description+"; " + "; ".join([f"{k}: {v}" for k, v in annotations_dict.items() if v != "N/A"])
                node_data = {
                    "ID": current_id,
                    "Type": current_molecule_type,
                    "Name": current_name,
                    "Sequence": sequence,
                    "Comments": comments
                }
                node_rows.append(node_data)
                
                # 遍历所有特征
                for feat in features:
                    if feat.type == "CDS":
                        genes = feat.qualifiers.get("gene", [])
                        products = feat.qualifiers.get("product", [])
                        protein_ids = feat.qualifiers.get("protein_id", [])
                        db_xrefs = feat.qualifiers.get("db_xref", [])
                        go_terms = feat.qualifiers.get("GO_function", [])
                        
                        if (db_xrefs != []):
                            for geneID in db_xrefs:
                                match = re.match(r"GeneID:(\d+)", geneID)
                                gene = "NA"
                                if match:
                                    gene = str(match.group(1)) 
                                edge = {
                                    "Node 1 ID": current_id,
                                    "Node 1 Data Type": current_molecule_type,
                                    "Node 1 Source": "RefSeq",
                                    "Edge Type": current_molecule_type + "_gene",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": NCBI_GENE_PREFIX + gene,
                                    "Node 2 Data Type": "Gene",
                                    "Node 2 Source": "NCBI"
                                }
                                edge_rows.append(edge)
                                
                        if (go_terms != []):
                            for go_term in go_terms:
                                # Safely extract GO ID
                                match_go_id = re.search(r"GO:\d+", go_term)
                                if not match_go_id:
                                    continue # Skip this term if no GO ID found
                                go_id = match_go_id.group()

                                # Safely extract evidence code
                                match_evidence = re.search(r"\[Evidence (\w+)\]", go_term)
                                if not match_evidence:
                                    evidence_code = "NA" # Assign a default
                                else:
                                    evidence_code = match_evidence.group(1)

                                edge_weight = evidence_to_weight(evidence_code)
                                if edge_weight == 0.0:
                                    continue # Skip the record that has no edge weight
                                if go_id not in goTermToTypeMapping:
                                    print(f"GO ID {go_id} not found in GO Type mapping, skipping...")
                                    continue
                                current_go_type = goTermToTypeMapping[go_id]

                                edge = {
                                    "Node 1 ID": current_id,
                                    "Node 1 Data Type": current_molecule_type,
                                    "Node 1 Source": "RefSeq",
                                    "Edge Type": current_molecule_type + "_go",
                                    "Edge Direction": edge_types.get("DD"),
                                    "Edge Weight": edge_weight,
                                    "Node 2 ID": go_id,
                                    "Node 2 Data Type": current_go_type,
                                    "Node 2 Source": "GO"
                                }
                                edge_rows.append(edge)
                                
                        if (protein_ids != []):
                            for protein_id in protein_ids:
                                uniprotID = refseqProteinToUniprotIDMapping.get(protein_id, None)
                                if uniprotID is None:
                                    continue
                                edge = {
                                    "Node 1 ID": current_id,
                                    "Node 1 Data Type": current_molecule_type,
                                    "Node 1 Source": "RefSeq",
                                    "Edge Type": current_molecule_type + "_protein",
                                    "Edge Direction": edge_types.get("SD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": UNIPROT_PROTEIN_PREFIX + uniprotID,
                                    "Node 2 Data Type": "Protein",
                                    "Node 2 Source": "Uniprot"
                                }
                                edge_rows.append(edge)

    # 存储结果
    result_collector["has_human"] = has_human
    result_collector["nodes"] = node_rows
    result_collector["edges"] = edge_rows
    return has_human, node_rows, edge_rows

def parse_gpff(gpff_file, result_collector):
    """解析GPFF文件，提取蛋白质信息"""
    has_human = False
    node_rows = []
    edge_rows = []
    
    with open(gpff_file, "r") as handle:
        for record in SeqIO.parse(handle, "genbank"):
            protein_id = getattr(record, 'id', None)
            name = getattr(record, 'name', None)
            description = getattr(record, 'description', '')
            seq = getattr(record, 'seq', None)
            annotations = getattr(record, 'annotations', {})

            # Add checks for essential attributes
            if not protein_id or not seq:
                continue
                
            annotations_dict = {
                "molecule_type": annotations.get("molecule_type", "N/A"),
                "topology": annotations.get("topology", "N/A"),
                "data_file_division": annotations.get("data_file_division", "N/A"),
                "accessions": annotations.get("accessions", ["N/A"]),
                "keywords": annotations.get("keywords", ["N/A"]),
                "source": annotations.get("source", "N/A"),
                "organism": annotations.get("organism", "N/A"),
                "taxonomy": annotations.get("taxonomy", ["N/A"]),
                "references": annotations.get("references", ["N/A"]),
                "comment": annotations.get("comment", "N/A"),
                "structured_comment": annotations.get("structured_comment", "N/A"),
                "contig": annotations.get("contig", "N/A"),
            }
            
            current_organism = annotations_dict.get("organism", "N/A")
            current_molecule_type = annotations_dict.get("molecule_type", "N/A")
            if current_molecule_type != "protein":
                continue
                
            taxonomy = annotations.get("taxonomy", [])
            
            if "homo sapiens" in current_organism.lower() or extractAllData:
                has_human = True
                comments = f"Description: {description}, Annotations: {taxonomy}"
                record_info_node = {
                    "ID": protein_id,
                    "Type": "Protein",
                    "Name": name,
                    "Sequence": seq,
                    "Comments": comments
                }
                node_rows.append(record_info_node)

                for feature in record.features:
                    if feature.type == "CDS":
                        db_xrefs = feature.qualifiers.get("db_xref", [])
                        for db_xref in db_xrefs:
                            match = re.match(r"GeneID:(\d+)", db_xref)
                            if match:
                                gene_id = match.group(1) 
                                edge = {
                                    "Node 1 ID": NCBI_GENE_PREFIX + str(gene_id),
                                    "Node 1 Data Type": "Gene",
                                    "Node 1 Source": "NCBI",
                                    "Edge Type": "gene_protein",
                                    "Edge Direction": edge_types.get("SD"),
                                    "Edge Weight": 1.0,
                                    "Node 2 ID": UNIPROT_PROTEIN_PREFIX + protein_id,
                                    "Node 2 Data Type": "Protein",
                                    "Node 2 Source": "RefSeq"
                                }
                                edge_rows.append(edge)

    # 存储结果
    result_collector["has_human"] = has_human
    result_collector["nodes"] = node_rows
    result_collector["edges"] = edge_rows
    return has_human, node_rows, edge_rows

def parse_fna(fna_file, result_collector):
    """解析FNA文件，提取核酸序列信息"""
    has_human = False
    node_rows = []
    
    with open(fna_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = record.id
            seq_name = record.name
            description = record.description
            seq = str(record.seq)
            molecule_type = "Gene"
            
            if ("homo sapiens" in description.lower()):
                has_human = True
                record_info_node = {
                    "ID": seq_id,
                    "Type": molecule_type,
                    "Name": seq_name,
                    "Sequence": seq,
                    "Comments": description
                }
                node_rows.append(record_info_node)
    
    # 存储结果
    result_collector["has_human"] = has_human
    result_collector["nodes"] = node_rows
    result_collector["edges"] = []
    return has_human, node_rows, []

def process_file(file, input_dir, human_log_lock):
    """多线程处理单个文件的函数"""
    start_time = time.time()
    file_path = os.path.join(input_dir, file)
    result_collector = {"has_human": False, "nodes": [], "edges": []}
    
    try:
        if file.endswith(".gbff"):
            has_human, _, _ = parse_gbff(file_path, result_collector)
        elif file.endswith(".gpff"):
            has_human, _, _ = parse_gpff(file_path, result_collector)
        elif file.endswith(".fna"):
            has_human, _, _ = parse_fna(file_path, result_collector)
        else:
            print(f"不支持的文件类型: {file}")
            return None, 0
        
        if result_collector["has_human"]:
            # 使用锁确保日志文件写入安全
            with human_log_lock:
                with open("human_genome.log", "a") as log_file:
                    log_file.write(f"Found human genome in {file}\n")
            print(f"{file} 数据提取完成(存在人类组学信息)")
        
        elapsed_time = time.time() - start_time
        print(f"Processed {file} in {elapsed_time:.2f} seconds")
        return result_collector, elapsed_time
    
    except Exception as e:
        print(f"处理文件 {file} 时发生错误: {str(e)}")
        return None, 0

def write_results_to_csv(results, writer_node, writer_edge):
    """将结果写入CSV文件"""
    with file_lock:
        for node in results["nodes"]:
            writer_node.writerow(node)
        for edge in results["edges"]:
            writer_edge.writerow(edge)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="输入目录路径")
    parser.add_argument('-p', '--protein', required=True, help="输入Protein ID File路径")
    parser.add_argument('-g', '--go', required=True, help="输入GO Type File路径")
    parser.add_argument('-o', '--output', nargs=2, required=True, help="输出目录路径")
    args = parser.parse_args()

    input_dir = args.input
    proteinIDFilePath = args.protein
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
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]
    
    # 计算要使用的线程数（CPU核心数的一半）
    # max_workers = max(1, multiprocessing.cpu_count() // 2)
    max_workers = 4
    print(f"将使用 {max_workers} 个线程进行并行处理")
    
    # 创建锁用于写入human_genome.log
    human_log_lock = threading.Lock()
    
    # 清空日志文件
    open("human_genome.log", "w").close()
    
    print("处理开始...")
    overall_start_time = time.time()
    
    # 获取要处理的文件列表
    targetFiles = [f for f in os.listdir(input_dir) if f.endswith(".rna.gbff")]
    totalFiles = len(targetFiles)
    print(f"待处理文件数量: {totalFiles}")
        
    # 创建临时目录来存储每个文件的结果
    temp_dir = "temp_results"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 更改处理函数，让它写入临时文件而不是收集结果
    def process_file_to_temp(file_idx, file, input_dir, human_log_lock):
        start_time = time.time()
        file_path = os.path.join(input_dir, file)
        temp_node_file = os.path.join(temp_dir, f"{file_idx}_node.csv")
        temp_edge_file = os.path.join(temp_dir, f"{file_idx}_edge.csv")
        has_human = False
        
        try:
            with open(temp_node_file, "w", newline="") as node_file, open(temp_edge_file, "w", newline="") as edge_file:
                writer_node = csv.DictWriter(node_file,fieldnames=fieldnames_node)
                writer_edge = csv.DictWriter(edge_file,fieldnames=fieldnames_edge)
                writer_node.writeheader()
                writer_edge.writeheader()
                
                result_collector = {"has_human": False, "nodes": [], "edges": []}
                
                # if file.endswith(".rna.gbff"):
                has_human, _, _ = parse_rna_gbff(file_path, result_collector)
                # elif file.endswith(".gpff"):
                #     # has_human, _, _ = parse_gpff(file_path, result_collector)
                #     return False, None, None, 0
                # elif file.endswith(".fna"):
                #     # has_human, _, _ = parse_fna(file_path, result_collector)
                #     return False, None, None, 0
                # else:
                #     # print(f"不支持的文件类型: {file}")
                #     return False, None, None, 0
                # 直接写入临时文件
                for node in result_collector["nodes"]:
                    writer_node.writerow(node)
                for edge in result_collector["edges"]:
                    writer_edge.writerow(edge)
                
                if result_collector["has_human"]:
                    with human_log_lock:
                        with open("human_genome.log", "a") as log_file:
                            log_file.write(f"Found human genome in {file}\n")
                    print(f"{file} 数据提取完成(存在人类组学信息)")
                
            elapsed_time = time.time() - start_time
            print(f"Processed {file} in {elapsed_time:.2f} seconds")
            return has_human, temp_node_file, temp_edge_file, elapsed_time
            
        except Exception as e:
            print(f"处理文件 {file} 时发生错误: {str(e)}")
            # 删除可能部分写入的临时文件
            if os.path.exists(temp_node_file):
                os.remove(temp_node_file)
            if os.path.exists(temp_edge_file):
                os.remove(temp_edge_file)
            return False, None, None, 0
        
    # 使用线程池并行处理所有文件
    processed_results = []
    total_time = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，使用索引标识每个任务
        future_to_file = {
            executor.submit(process_file_to_temp, idx, file, input_dir, human_log_lock): (idx, file) 
            for idx, file in enumerate(targetFiles)
        }
        
        # 收集完成的任务结果
        for future in concurrent.futures.as_completed(future_to_file):
            idx, file = future_to_file[future]
            try:
                has_human, temp_node_file, temp_edge_file, elapsed_time = future.result()
                if temp_node_file and temp_edge_file:  # 如果处理成功
                    processed_results.append((idx, temp_node_file, temp_edge_file))
                    total_time += elapsed_time
                print(f"进度: {len(processed_results)}/{totalFiles}，当前文件处理时间: {elapsed_time:.2f} 秒")
            except Exception as e:
                print(f"文件 {file} 处理失败: {str(e)}")
    
    # 合并所有临时文件到最终输出
    print("合并临时结果文件...")
    with open(output_node_csv, "w", newline="") as final_node_file, open(output_edge_csv, "w", newline="") as final_edge_file:
        writer_node = csv.DictWriter(final_node_file, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(final_edge_file, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()
        
        # 按原始索引顺序合并结果
        for _, temp_node_file, temp_edge_file in sorted(processed_results):
            if os.path.exists(temp_node_file):
                with open(temp_node_file, "r", newline="") as node_file:
                    # 跳过标题行
                    reader = csv.DictReader(node_file)
                    for row in reader:
                        writer_node.writerow(row)
                # os.remove(temp_node_file)  # 删除临时文件
            
            if os.path.exists(temp_edge_file):
                with open(temp_edge_file, "r", newline="") as edge_file:
                    reader = csv.DictReader(edge_file)
                    for row in reader:
                        writer_edge.writerow(row)
                # os.remove(temp_edge_file)  # 删除临时文件
    
    # 清理临时目录
    # if os.path.exists(temp_dir):
    #     os.rmdir(temp_dir)

    overall_elapsed_time = time.time() - overall_start_time
    print(f"所有处理完成，总处理时间: {overall_elapsed_time:.2f} 秒，实际计算时间: {total_time:.2f} 秒")
    print(f"多线程加速比: {total_time/overall_elapsed_time:.2f}x")

if __name__ == "__main__":
    main()
