import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys
import argparse
import re
import csv
from DataProcessing.kg_construction.commonSettings import *

fieldNamesMondoToOMIM = ['MONDO ID', 'MONDO Name', 'OMIM ID']
unique_protein_in_edges = set()

namespaces = {
    'hmdb': 'http://www.hmdb.ca'
}

@dataclass
class HmdbPathway:
    name: str
    smpdb_id: Optional[str] = None
    kegg_map_id: Optional[str] = None

@dataclass
class ProteinAssociation:
    uniprot_id: Optional[str] = None
    name: Optional[str] = None
    gene_name: Optional[str] = None
    protein_type: Optional[str] = None  # enzyme/transporter

@dataclass
class DiseaseAssociation:
    name: str
    omim_id: Optional[str] = None

@dataclass
class GoClassification:
    category: Optional[str] = None
    description: Optional[str] = None
    go_id: Optional[str] = None

@dataclass
class MetaboliteAssociation:
    accession: Optional[str] = None
    name: Optional[str] = None

@dataclass
class OntologyTerm:
    term: str
    definition: Optional[str]
    parent_id: Optional[str]
    level: int
    term_type: str
    synonyms: Optional[List[str]] = None
    descendants: List['OntologyTerm'] = field(default_factory=list)

namespaces = {'hmdb': 'http://www.hmdb.ca'}

def parse_metabolite(element) -> Dict:
    """SAX式解析逻辑的ElementTree实现"""
    data = {}
    
    # 基础字段
    data['accession'] = _find_text(element, 'hmdb:accession')
    data['creation_date'] = _find_text(element, 'hmdb:creation_date')
    data['update_date'] = _find_text(element, 'hmdb:update_date')
    data['status'] = _find_text(element, 'hmdb:status')
    data['name'] = _find_text(element, 'hmdb:name')
    data['description'] = _find_text(element, 'hmdb:description')
    # 同义词
    data['synonyms'] = _parse_list(element, 'hmdb:synonyms/hmdb:synonym')
    # 化学信息
    data['formula'] = _find_text(element, 'hmdb:chemical_formula')
    data['average_molecular_weight'] = _to_float(_find_text(element, 'hmdb:average_molecular_weight'))
    data['monisotopic_molecular_weight'] = _to_float(_find_text(element, 'hmdb:monisotopic_molecular_weight'))
    # 命名信息
    data['iupac_name'] = _find_text(element, 'hmdb:iupac_name')
    data['traditional_iupac'] = _find_text(element, 'hmdb:traditional_iupac')
    data['cas_registry_number'] = _find_text(element, 'hmdb:cas_registry_number')
    # 结构信息
    data['smiles'] = _find_text(element, 'hmdb:smiles')
    data['inchi'] = _find_text(element, 'hmdb:inchi')
    data['inchikey'] = _find_text(element, 'hmdb:inchikey')
    # 分类信息
    data['description'] =  _find_taxonomy(element, 'description')
    data['direct_parent'] =  _find_taxonomy(element, 'direct_parent')
    data['kingdom'] =  _find_taxonomy(element, 'kingdom')
    data['super_class'] =  _find_taxonomy(element, 'super_class')
    data['chemical_class'] =  _find_taxonomy(element, 'class')
    data['sub_class'] =  _find_taxonomy(element, 'sub_class')
    data['molecularFramework'] =  _find_taxonomy(element, 'molecular_framework')
    data['alternative_parents'] = _parse_list(element, 'hmdb:taxonomy/hmdb:alternative_parents/hmdb:alternative_parent')
    data['substituents'] = _parse_list(element, 'hmdb:taxonomy/hmdb:substituents/hmdb:substituent')
    # 本体术语
    ontology_elem = element.find('hmdb:ontology', namespaces)
    data['ontology'] = _parse_ontology(ontology_elem) if ontology_elem is not None else None
    # 生物学位置
    data['biofluid_locations'] = _parse_biofluids(element)
    data['tissue_locations'] = _parse_list(element, 'hmdb:tissue_locations/hmdb:tissue')
    # 通路信息
    data['pathways']= _parse_pathways(element)

    # 外部标识符
    data['kegg_id'] = _find_text(element, 'hmdb:kegg_id')
    data['drugbank_ids'] = _parse_list(element, 'hmdb:drugbank_id')
    data['chemspider_id'] = _find_text(element, 'hmdb:chemspider_id')
    data['pubchem_compound_id'] = _find_external_id(element, 'PubChem Compound')
    data['chebi_id'] = _find_external_id(element, 'ChEBI')
    data['chemspider_id'] = _find_text(element, 'hmdb:chemspider_id')
    data['smpdb_ids'] = [p.smpdb_id for p in data.get('pathways', [])]
    
    # 蛋白质关联解析
    protein_associations = []
    for pa in element.findall('hmdb:protein_associations/hmdb:protein', namespaces):
        protein_associations.append(
            ProteinAssociation(
                name=_find_text(pa, 'hmdb:name'),
                uniprot_id=_find_text(pa, 'hmdb:uniprot_id'),
                gene_name=_find_text(pa, 'hmdb:gene_name'),
                protein_type=_find_text(pa, 'hmdb:protein_type')
            )
        )
    data['protein_associations'] = protein_associations
    # 疾病关联解析
    diseases = []
    for dis in element.findall('hmdb:diseases/hmdb:disease', namespaces):
        diseases.append(
            DiseaseAssociation(
                name=_find_text(dis, 'hmdb:name'),
                omim_id=_find_text(dis, 'hmdb:omim_id')
            )
        )
    data['diseases'] = diseases
    # return {k: v for k, v in data.items() if v is not None}
    return {k: v for k, v in data.items()}

# 辅助方法
def _parse_ontology(ontology_elem) -> Optional[OntologyTerm]:
    """解析整个本体结构"""
    if ontology_elem is None:
        return None
    
    root_elem = ontology_elem.find('hmdb:root', namespaces)
    return _parse_ontology_term(root_elem) if root_elem is not None else None

def _parse_ontology_term(term_elem) -> OntologyTerm:
    """递归解析单个本体术语及其后代"""
    # 提取基础字段
    term = _find_text(term_elem, 'hmdb:term') or "Unknown Term"
    definition = _find_text(term_elem, 'hmdb:definition')
    parent_id = _find_text(term_elem, 'hmdb:parent_id')
    synonyms = _parse_list(term_elem, 'hmdb:synonyms/hmdb:synonym')

    # 处理数值型字段
    level = int(_find_text(term_elem, 'hmdb:level') or 0)
    term_type = _find_text(term_elem, 'hmdb:type') or "unclassified"
    

    # 递归解析子术语
    descendants = []
    descendants_elem = term_elem.find('hmdb:descendants', namespaces)
    if descendants_elem is not None:
        for desc_elem in descendants_elem.findall('hmdb:descendant', namespaces):
            desc = _parse_ontology_term(desc_elem)
            if desc is not None:
                descendants.append(desc)

    return OntologyTerm(
        term=term.strip(),
        definition=definition.strip() if definition else None,
        parent_id=parent_id.strip() if parent_id else None,
        level=level,
        term_type=term_type.strip(),
        synonyms=synonyms,
        descendants=descendants
    )

def _find_taxonomy(elem, field: str) -> Optional[str]:
    return _find_text(elem, f'hmdb:taxonomy/hmdb:{field}')

def _find_text(elem, path: str) -> Optional[str]:
    return elem.findtext(f'.//{path}', namespaces=namespaces)

def _find_external_id(elem, resource_name: str) -> Optional[str]:
    path = f"hmdb:external_identifiers/hmdb:external_identifier[hmdb:resource='{resource_name}']/hmdb:identifier"
    return elem.findtext(f'.//{path}', namespaces=namespaces)

def _parse_spectra_ids(elem, spectrum_type: str) -> List[str]:
    """安全解析光谱ID，处理缺失spectrum_type的情况"""
    ids = []
    for s in elem.findall('hmdb:spectra/hmdb:spectrum', namespaces):
        # 安全获取元素
        s_type = s.find('hmdb:spectrum_type', namespaces)
        s_id = s.find('hmdb:spectrum_id', namespaces)
        
        # 双重校验：类型存在且匹配，ID存在
        if (s_type is not None and 
            s_type.text.strip() == spectrum_type and 
            s_id is not None and 
            s_id.text):
            ids.append(s_id.text.strip())
    return ids


# 兼容HMDB不同版本的路径差异
def _parse_biofluids(elem):
    if elem.find('.//hmdb:biofluid_locations', namespaces) is not None:
        print("biofluid_locations exist")
    # 尝试两种不同版本的路径结构
    return _parse_list(elem, 'hmdb:biofluid_locations/hmdb:biofluid') or \
           _parse_list(elem, 'hmdb:biospecimen_locations/hmdb:biospecimen')

def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None

def _parse_list(elem, path: str) -> List[str]:
    return [e.text for e in elem.findall(f'.//{path}', namespaces=namespaces) if e.text]

def _parse_pathways(elem) -> List[HmdbPathway]:
    pathways = []
    for p in elem.findall('.//hmdb:pathway', namespaces=namespaces):
        pathways.append(HmdbPathway(
            name=_find_text(p, 'hmdb:name'),
            smpdb_id=_find_text(p, 'hmdb:smpdb_id'),
            kegg_map_id=_find_text(p, 'hmdb:kegg_map_id')
        ))
    return pathways

def parse_protein(element) -> Dict:
    """SAX-style parsing for a protein element"""
    data = {}
    data['accession'] = _find_text(element, 'hmdb:accession')
    data['uniprot_id'] = _find_text(element, 'hmdb:uniprot_id')
    data['name'] = _find_text(element, 'hmdb:name')
    data['sequence'] = _find_text(element, 'hmdb:protein_properties/hmdb:polypeptide_sequence')
    data['gene_name'] = _find_text(element, 'hmdb:gene_name')
    data['general_function'] = _find_text(element, 'hmdb:general_function')
    data['specific_function'] = _find_text(element, 'hmdb:specific_function')

    pathways = []
    for p in element.findall('.//hmdb:pathway', namespaces=namespaces):
        pathways.append(HmdbPathway(
            name=_find_text(p, 'hmdb:name'),
            smpdb_id=_find_text(p, 'hmdb:smpdb_id'),
            kegg_map_id=_find_text(p, 'hmdb:kegg_map_id')
        ))
    data['pathways'] = pathways

    metabolites = []
    for m in element.findall('.//hmdb:metabolite_associations/hmdb:metabolite', namespaces=namespaces):
        metabolites.append(MetaboliteAssociation(
            accession=_find_text(m, 'hmdb:accession'),
            name=_find_text(m, 'hmdb:name')
        ))
    data['metabolite_associations'] = metabolites

    go_classifications = []
    for go in element.findall('.//hmdb:go_classifications/hmdb:go_class', namespaces=namespaces):
        go_classifications.append(GoClassification(
            category=_find_text(go, 'hmdb:category'),
            description=_find_text(go, 'hmdb:description'),
            go_id=_find_text(go, 'hmdb:go_id')
        ))
    data['go_classifications'] = go_classifications
    
    return data

def process_protein_xml_tree(root, writer_node, writer_edge, written_edges, pathway_map, edge_type_counts):
    processed_go_ids = set()
    for protein in root.findall('hmdb:protein', namespaces=namespaces):
        parsed_data = parse_protein(protein)
        
        if not parsed_data.get('uniprot_id'):
            continue

        # Create protein node
        protein_id = UNIPROT_PROTEIN_PREFIX + parsed_data['uniprot_id']
        # Create edges to metabolites
        for metabolite in parsed_data.get('metabolite_associations', []):
            if not metabolite.accession:
                continue

            relation = tuple(sorted((protein_id, metabolite.accession)))
            if relation in written_edges:
                continue
            written_edges.add(relation)

            edge_data = {
                "Node 1 ID": metabolite.accession,
                "Node 1 Data Type": "Metabolite",
                "Node 1 Source": "HMDB",
                "Edge Type": "metabolite_protein",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": protein_id,
                "Node 2 Data Type": "Protein",
                "Node 2 Source": "Uniprot"
            }
            writer_edge.writerow(edge_data)
            edge_type = edge_data["Edge Type"]
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            unique_protein_in_edges.add(protein_id)

        # Create edges to pathways
        for pathway in parsed_data.get('pathways', []):
            if not pathway.name or not pathway.name.strip():
                continue
            
            reactome_id = pathway_map.get(pathway.name.strip())
            
            if not reactome_id:
                continue

            relation = tuple(sorted((protein_id, reactome_id)))
            if relation in written_edges:
                continue
            written_edges.add(relation)

            edge_data = {
                "Node 1 ID": protein_id,
                "Node 1 Data Type": "Protein",
                "Node 1 Source": "Uniprot",
                "Edge Type": "protein_pathway",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": reactome_id,
                "Node 2 Data Type": "Pathway",
                "Node 2 Source": "Reactome"
            }
            writer_edge.writerow(edge_data)
            edge_type = edge_data["Edge Type"]
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        # Create GO term nodes and edges to them
        for go in parsed_data.get('go_classifications', []):
            if not go.go_id or not go.go_id.strip():
                continue
            
            go_id_prefixed = go.go_id
            go_classification = go.category
            go_classification_no_space = go_classification.replace(" ", "").lower()

            relation = tuple(sorted((protein_id, go_id_prefixed)))
            if relation in written_edges:
                continue
            written_edges.add(relation)

            # Edge protein-GO
            edge_data = {
                "Node 1 ID": protein_id,
                "Node 1 Data Type": "Protein",
                "Node 1 Source": "Uniprot",
                "Edge Type": "protein_"+go_classification_no_space,
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": go_id_prefixed,
                "Node 2 Data Type": go_classification_no_space,
                "Node 2 Source": "GO"
            }
            writer_edge.writerow(edge_data)
            edge_type = edge_data["Edge Type"]
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

def process_xml_tree(root, writer_node, writer_edge, omim_to_mondo_map, written_edges, pathway_map, edge_type_counts):
    for metabolite in root.findall('hmdb:metabolite', namespaces=namespaces):
        parsed_data = parse_metabolite(metabolite)
        comments = f"Description: {parsed_data['description']}, Synonyms: {parsed_data['synonyms']}, Formula: {parsed_data['formula']}, Average Molecular Weight: {parsed_data['average_molecular_weight']}, Monisotopic Molecular Weight: {parsed_data['monisotopic_molecular_weight']}, IUPAC Name: {parsed_data['iupac_name']}, Traditional IUPAC: {parsed_data['traditional_iupac']}, CAS Registry Number: {parsed_data['cas_registry_number']}, InChI: {parsed_data['inchi']}, InChIKey: {parsed_data['inchikey']}, Direct Parent: {parsed_data['direct_parent']}, Kingdom: {parsed_data['kingdom']}, Super Class: {parsed_data['super_class']}, Chemical Class: {parsed_data['chemical_class']}, Sub Class: {parsed_data['sub_class']}, Molecular Framework: {parsed_data['molecularFramework']}, Alternative Parents: {parsed_data['alternative_parents']}, Substituents: {parsed_data['substituents']}, Ontology: {parsed_data['ontology']}, Biofluid Locations: {parsed_data['biofluid_locations']}, Tissue Locations: {parsed_data['tissue_locations']}"
        current_node = {
            "ID": parsed_data['accession'],
            "Type": "Metabolite",
            "Name": parsed_data['name'],
            "Sequence": parsed_data['smiles'],
            "Comments": comments
        }
        writer_node.writerow(current_node)
        # Extract edges for diseases
        for disease in parsed_data.get('diseases', []):
            if disease.omim_id is None or disease.omim_id == "":
                continue
            
            mondo_id = omim_to_mondo_map.get(str(disease.omim_id).strip())
            if not mondo_id:
                continue

            relation = tuple(sorted((parsed_data['accession'], mondo_id)))
            if relation in written_edges:
                continue
            written_edges.add(relation)

            edge_data = {
                "Node 1 ID": parsed_data['accession'],
                "Node 1 Data Type": "Metabolite",
                "Node 1 Source": "HMDB",
                "Edge Type": "metabolite_disease",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": mondo_id,
                "Node 2 Data Type": "Disease",
                "Node 2 Source": "MONDO"
            }
            writer_edge.writerow(edge_data)
            edge_type = edge_data["Edge Type"]
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        # Extract edges for proteins
        for protein in parsed_data.get('protein_associations', []):
            if protein.uniprot_id is None or protein.uniprot_id == "":
                continue
            
            protein_id = UNIPROT_PROTEIN_PREFIX + protein.uniprot_id
            metabolite_id = parsed_data['accession']
            relation = tuple(sorted((protein_id, metabolite_id)))
            if relation in written_edges:
                continue
            written_edges.add(relation)

            edge_data = {
                "Node 1 ID": parsed_data['accession'],
                "Node 1 Data Type": "Metabolite",
                "Node 1 Source": "HMDB",
                "Edge Type": "metabolite_protein",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": protein_id,
                "Node 2 Data Type": "Protein",
                "Node 2 Source": "Uniprot"
            }
            writer_edge.writerow(edge_data)
            edge_type = edge_data["Edge Type"]
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            unique_protein_in_edges.add(protein_id)

        # Create edges to pathways
        for pathway in parsed_data.get('pathways', []):
            if not pathway.name or not pathway.name.strip():
                continue
            
            reactome_id = pathway_map.get(pathway.name.strip())
            
            if not reactome_id:
                continue

            relation = tuple(sorted((parsed_data['accession'], reactome_id)))
            if relation in written_edges:
                continue
            written_edges.add(relation)

            edge_data = {
                "Node 1 ID": parsed_data['accession'],
                "Node 1 Data Type": "Metabolite",
                "Node 1 Source": "HMDB",
                "Edge Type": "metabolite_pathway",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": 1.0,
                "Node 2 ID": reactome_id,
                "Node 2 Data Type": "Pathway",
                "Node 2 Source": "Reactome"
            }
            writer_edge.writerow(edge_data)
            edge_type = edge_data["Edge Type"]
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        # Link to: kegg_id, drugbank_ids, pubchem_compound_id, smpdb_ids
        if parsed_data['drugbank_ids'] is not None:
            for drugbank_id in parsed_data['drugbank_ids']:
                if drugbank_id is None or drugbank_id == "":
                    continue
                
                relation = tuple(sorted((parsed_data['accession'], drugbank_id)))
                if relation in written_edges:
                    continue
                written_edges.add(relation)

                edge_data = {
                    "Node 1 ID": parsed_data['accession'],
                    "Node 1 Data Type": "Metabolite",
                    "Node 1 Source": "HMDB",
                    "Edge Type": "metabolite_drug",
                    "Edge Direction": edge_types.get("DD"),
                    "Edge Weight": 1.0,
                    "Node 2 ID": drugbank_id,
                    "Node 2 Data Type": "Drug",
                    "Node 2 Source": "HMDB"
                }
                writer_edge.writerow(edge_data)
                edge_type = edge_data["Edge Type"]
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

def load_omim_to_mondo_map(disease_file):
    omim_to_mondo = {}
    count = 0
    try:
        with open(disease_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                omim_id = row.get('OMIM ID')
                mondo_id = row.get('MONDO ID')
                if omim_id and mondo_id and omim_id.strip():
                    omim_to_mondo[omim_id.replace("OMIM:", "").strip()] = mondo_id.strip()
                    count += 1
        print(f"Loaded {count} OMIM to MONDO mappings from '{disease_file}'")
    except FileNotFoundError:
        print(f"Error: Disease file '{disease_file}' not found.")
    except Exception as e:
        print(f"Error reading disease file '{disease_file}': {e}")
    return omim_to_mondo

def load_pathway_map(pathway_file):
    pathway_map = {}
    count = 0
    try:
        with open(pathway_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pathway_name = row.get('Name')
                reactome_id = row.get('ID')
                if pathway_name and reactome_id:
                    pathway_map[pathway_name.strip()] = reactome_id.strip()
                    count += 1
        print(f"Loaded {count} pathways from '{pathway_file}'")
    except FileNotFoundError:
        print(f"Error: Pathway file '{pathway_file}' not found.")
    except Exception as e:
        print(f"Error reading pathway file '{pathway_file}': {e}")
    return pathway_map

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, nargs=4, required=True, help="Input metabolic, protein, disease, and pathway file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")

    args = parser.parse_args()
    # 获取输入和输出文件路径
    metabolic_file = args.input[0]
    protein_file = args.input[1]
    disease_file = args.input[2]
    pathway_file = args.input[3]
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]

    omim_to_mondo_map = load_omim_to_mondo_map(disease_file)
    if not omim_to_mondo_map:
        print("Warning: OMIM to MONDO map is empty. No disease edges will be created based on OMIM IDs.")

    pathway_map = load_pathway_map(pathway_file)
    if not pathway_map:
        print("Warning: Pathway map is empty. No protein-pathway edges will be created from HMDB.")

    written_edges = set()
    edge_type_counts = {}

    with open(output_node_csv, "w", newline="") as file_node, open(output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()
        try:
            tree = ET.parse(metabolic_file)
            root = tree.getroot()
            # Count which data fields does not contain data for all metabolites
            process_xml_tree(root, writer_node, writer_edge, omim_to_mondo_map, written_edges, pathway_map, edge_type_counts)
            print("Processing metabolites completed successfully")
        except FileNotFoundError:
            print(f"Error: File '{metabolic_file}' not found")
        except ET.ParseError:
            print(f"Error: Invalid XML format in '{metabolic_file}'")

        try:
            tree = ET.parse(protein_file)
            root = tree.getroot()
            process_protein_xml_tree(root, writer_node, writer_edge, written_edges, pathway_map, edge_type_counts)
            print("Processing proteins completed successfully")
        except FileNotFoundError:
            print(f"Error: File '{protein_file}' not found")
        except ET.ParseError:
            print(f"Error: Invalid XML format in '{protein_file}'")

    print("\n--- Edge Type Statistics ---")
    for edge_type, count in edge_type_counts.items():
        print(f"{edge_type}: {count}")
    print("--------------------------\n")
    print(f"Unique proteins in edges: {len(unique_protein_in_edges)}")

if __name__ == "__main__":
    main()
