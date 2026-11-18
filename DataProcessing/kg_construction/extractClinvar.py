import csv
import gc
import argparse
from lxml import etree
from DataProcessing.kg_construction.commonSettings import *
from collections import deque
import itertools # 新增导入

class VariationArchiveProcessor:
    def __init__(self, node_csv, edge_csv):    
        self.writer_node = node_csv
        self.writer_edge = edge_csv
        # 内存优化结构
        self.element_queue = deque(maxlen=100)
        self.gene_cache = set()

    def process_variation_archive(self, elem):
        try:
            # 解析基础信息
            variant_id = elem.get('VariationID') # 保留以防未来使用，但当前逻辑不直接用它创建边
            # variant_name = elem.get('VariationName') # 同上

            # 检查物种（防御性编程）
            variant_species = elem.find('.//Species')
            if variant_species is None or variant_species.text != "Homo sapiens":
                return  # 跳过非人类记录

            # 获取 VariantType（防御性访问）
            variant_type_elem = elem.find('.//VariantType')
            variant_type = variant_type_elem.text if variant_type_elem is not None else ""

            # 获取 Location（精确路径）
            variant_location = elem.find('.//SimpleAllele/Location')
            if variant_location is None:
                variant_location_cytogenetic = "N/A"
                variant_location_sequence37 = "N/A"
                variant_location_sequence38 = "N/A"
            else:
                # CytogeneticLocation 解析
                cytogenetic_elem = variant_location.find('CytogeneticLocation')
                variant_location_cytogenetic = cytogenetic_elem.text if cytogenetic_elem is not None else "N/A"

                # SequenceLocation 解析（GRCh37）
                seq_loc_elem = variant_location.find('.//SequenceLocation[@Assembly="GRCh37"]')
                if seq_loc_elem is not None:
                    chr_num = seq_loc_elem.get('Chr', 'Unknown')
                    start = seq_loc_elem.get('start', 'Unknown')
                    end = seq_loc_elem.get('stop', 'Unknown')
                    variant_location_sequence37 = f"chr{chr_num}:{start}-{end}"
                else:
                    variant_location_sequence37 = "N/A"
                # SequenceLocation 解析（GRCh38）
                seq_loc_elem = variant_location.find('.//SequenceLocation[@Assembly="GRCh38"]')
                if seq_loc_elem is not None:
                    chr_num = seq_loc_elem.get('Chr', 'Unknown')
                    start = seq_loc_elem.get('start', 'Unknown')
                    end = seq_loc_elem.get('stop', 'Unknown')
                    variant_location_sequence38 = f"chr{chr_num}:{start}-{end}"
                else:
                    variant_location_sequence38 = "N/A"

            # 构建评论信息 (如果需要写Clinvar节点，这部分仍然有用)
            # comments = f"Variant type: {variant_type}, Variant cytogenetic location: {variant_location_cytogenetic}, Variant GRCh37 location: {variant_location_sequence37}, Variant GRCh38 location: {variant_location_sequence38}".strip()
            
            # 写入节点 (当前已注释掉，按需启用)
            # node_data = {
            #     "ID": CLINVAR_VARIANT_PREFIX + str(variant_id),
            #     "Type": "Clinvar",
            #     "Name": variant_name,
            #     "Sequence": "",
            #     "Comments": comments
            # }
            # self.writer_node.writerow(node_data)

            associated_genes = []
            associated_proteins = []
            associated_diseases = []

            # 收集基因
            for gene_elem in elem.findall(".//Gene"):
                gene_id = gene_elem.get('GeneID')
                if gene_id:
                    associated_genes.append({
                        "id": NCBI_GENE_PREFIX + str(gene_id),
                        "type": "Gene",
                        "source": "NCBI"
                    })

            # 收集交叉引用中的蛋白质
            for xref in elem.findall(".//XRefList/XRef"):
                xref_id = xref.get('ID')
                xref_source = xref.get('DB')
                if xref_source == "UniProtKB" and xref_id:
                    uniprot_id = xref_id.split(":")[-1]
                    if "#" in uniprot_id:
                        uniprot_id = uniprot_id.split("#")[0]
                    associated_proteins.append({
                        "id": UNIPROT_PROTEIN_PREFIX + uniprot_id,
                        "type": "Protein",
                        "source": "Uniprot"
                    })
            
            # 收集疾病关联（从Trait中的XRef提取MONDO）
            for trait in elem.findall(".//Trait"):
                for xref in trait.findall(".//XRef"):
                    xref_id = xref.get('ID')
                    xref_source = xref.get('DB')
                    if xref_source == "MONDO" and xref_id:
                        associated_diseases.append({
                            "id": xref_id, # MONDO ID is typically not prefixed in common use, adjust if your KG expects it
                            "type": "Disease",
                            "source": "MONDO"
                        })

            # 生成实体间的关联边
            created_edges_for_record = set() # 新增：用于跟踪此记录已创建的边
            edge_direction = edge_types.get("UD", "undirected") # Default to undirected

            # 1. 基因 - 基因 (同一记录下)
            for gene1_info, gene2_info in itertools.combinations(associated_genes, 2):
                if gene1_info["id"] == gene2_info["id"]:
                    continue

                node1_id = gene1_info["id"]
                node2_id = gene2_info["id"]
                edge_type = "gene_gene_interaction"

                canonical_edge = tuple(sorted((node1_id, node2_id))) + (edge_type,)
                if canonical_edge in created_edges_for_record:
                    continue

                self.writer_edge.writerow({
                    "Node 1 ID": node1_id, "Node 1 Data Type": gene1_info["type"], "Node 1 Source": gene1_info["source"],
                    "Edge Type": edge_type, "Edge Direction": edge_direction, "Edge Weight": 1.0,
                    "Node 2 ID": node2_id, "Node 2 Data Type": gene2_info["type"], "Node 2 Source": gene2_info["source"]
                })
                created_edges_for_record.add(canonical_edge)

            # 2. 基因 - 疾病
            for gene_info in associated_genes:
                for disease_info in associated_diseases:
                    if gene_info["id"] == disease_info["id"]:
                        continue

                    node1_id = gene_info["id"]
                    node2_id = disease_info["id"]
                    edge_type = "gene_disease" # 使用用户指定的 Edge Type

                    canonical_edge = tuple(sorted((node1_id, node2_id))) + (edge_type,)
                    if canonical_edge in created_edges_for_record:
                        continue
                    
                    self.writer_edge.writerow({
                        "Node 1 ID": node1_id, "Node 1 Data Type": gene_info["type"], "Node 1 Source": gene_info["source"],
                        "Edge Type": edge_type, "Edge Direction": edge_direction, "Edge Weight": 1.0,
                        "Node 2 ID": node2_id, "Node 2 Data Type": disease_info["type"], "Node 2 Source": disease_info["source"]
                    })
                    created_edges_for_record.add(canonical_edge)

            # 3. 蛋白质 - 疾病
            for protein_info in associated_proteins:
                for disease_info in associated_diseases:
                    if protein_info["id"] == disease_info["id"]:
                        continue

                    node1_id = protein_info["id"]
                    node2_id = disease_info["id"]
                    edge_type = "protein_disease" # 使用用户指定的 Edge Type

                    canonical_edge = tuple(sorted((node1_id, node2_id))) + (edge_type,)
                    if canonical_edge in created_edges_for_record:
                        continue

                    self.writer_edge.writerow({
                        "Node 1 ID": node1_id, "Node 1 Data Type": protein_info["type"], "Node 1 Source": protein_info["source"],
                        "Edge Type": edge_type, "Edge Direction": edge_direction, "Edge Weight": 1.0,
                        "Node 2 ID": node2_id, "Node 2 Data Type": disease_info["type"], "Node 2 Source": disease_info["source"]
                    })
                    created_edges_for_record.add(canonical_edge)

            # 4. 蛋白质 - 蛋白质 (同一记录下)
            for protein1_info, protein2_info in itertools.combinations(associated_proteins, 2):
                if protein1_info["id"] == protein2_info["id"]:
                    continue

                node1_id = protein1_info["id"]
                node2_id = protein2_info["id"]
                edge_type = "protein_protein_interaction"

                canonical_edge = tuple(sorted((node1_id, node2_id))) + (edge_type,)
                if canonical_edge in created_edges_for_record:
                    continue

                self.writer_edge.writerow({
                    "Node 1 ID": node1_id, "Node 1 Data Type": protein1_info["type"], "Node 1 Source": protein1_info["source"],
                    "Edge Type": edge_type, "Edge Direction": edge_direction, "Edge Weight": 1.0,
                    "Node 2 ID": node2_id, "Node 2 Data Type": protein2_info["type"], "Node 2 Source": protein2_info["source"]
                })
                created_edges_for_record.add(canonical_edge)

            # 5. 基因 - 蛋白质
            for gene_info in associated_genes:
                for protein_info in associated_proteins:
                    if gene_info["id"] == protein_info["id"]:
                        continue

                    node1_id = gene_info["id"]
                    node2_id = protein_info["id"]
                    edge_type = "gene_protein_product"

                    canonical_edge = tuple(sorted((node1_id, node2_id))) + (edge_type,)
                    if canonical_edge in created_edges_for_record:
                        continue

                    self.writer_edge.writerow({
                        "Node 1 ID": node1_id, "Node 1 Data Type": gene_info["type"], "Node 1 Source": gene_info["source"],
                        "Edge Type": edge_type, "Edge Direction": edge_direction, "Edge Weight": 1.0,
                        "Node 2 ID": node2_id, "Node 2 Data Type": protein_info["type"], "Node 2 Source": protein_info["source"]
                    })
                    created_edges_for_record.add(canonical_edge)
            
            # 6. 疾病 - 疾病 (同一记录下)
            for disease1_info, disease2_info in itertools.combinations(associated_diseases, 2):
                if disease1_info["id"] == disease2_info["id"]:
                    continue

                node1_id = disease1_info["id"]
                node2_id = disease2_info["id"]
                edge_type = "disease_disease"
                
                canonical_edge = tuple(sorted((node1_id, node2_id))) + (edge_type,)
                if canonical_edge in created_edges_for_record:
                    continue

                self.writer_edge.writerow({
                    "Node 1 ID": node1_id, "Node 1 Data Type": disease1_info["type"], "Node 1 Source": disease1_info["source"],
                    "Edge Type": edge_type, "Edge Direction": edge_direction, "Edge Weight": 1.0,
                    "Node 2 ID": node2_id, "Node 2 Data Type": disease2_info["type"], "Node 2 Source": disease2_info["source"]
                })
                created_edges_for_record.add(canonical_edge)
            
            # 原有的Clinvar与其他实体的边创建逻辑已被移除

        finally:
            self._deep_clean(elem)

    def _deep_clean(self, elem):
        """深度内存清理"""
        parent = elem.getparent()
        if parent is not None:
            if parent.getparent() is not None:
                parent.getparent().remove(parent)
            else:
                parent.clear()
                
        elem.clear()
        del elem
        gc.collect(generation=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', nargs=2, required=True)
    args = parser.parse_args()

    xml_file = args.input
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]

    with open(output_node_csv, "w", newline="") as file_node, open(output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()

        processor = VariationArchiveProcessor(writer_node, writer_edge)
    
        context = etree.iterparse(
            xml_file,
            events=("end",),
            tag="VariationArchive",
            huge_tree=True,
            remove_blank_text=True
        )
        count = 0
        for event, elem in context:
            count += 1
            print(f"Processed {count} elements, current: {elem}.")
            processor.process_variation_archive(elem)
            # 每处理100个元素后清理缓存
            if len(processor.element_queue) % 100 == 0:
                processor.gene_cache.clear()

if __name__ == "__main__":
    main()
