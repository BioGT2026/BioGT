import argparse
import csv
import math
import re
import requests
import json
from DataProcessing.kg_construction.commonSettings import *
conditionsToWeightMapping = {
    "Clinical": 1.0,
    "Investigational": 0.9
}

def load_mondo_nodes(mondo_node_file):
    """Loads MONDO nodes and creates a name-to-ID mapping."""
    mapping = {}
    with open(mondo_node_file, 'r', encoding='utf-8', newline='') as infile:
        delimiter = ','
        reader = csv.DictReader(infile, delimiter=delimiter)

        if not reader.fieldnames:
            print(f"File {mondo_node_file} appears to be empty or could not read header. Skipping.")
            return

        actual_header = reader.fieldnames
        if not all(field in actual_header for field in fieldnames_node):
            print(f"File {mondo_node_file} header {actual_header} doesn't contain all expected fields {fieldnames_node}. Processing based on available fields.")
            return mapping

        for row in reader:
            mondo_id = row.get('ID', '').strip()
            disease_name = row.get('Name', '').strip()
            if mondo_id and disease_name:
                mapping[disease_name.lower()] = mondo_id

    print(f"Loaded {len(mapping)} MONDO disease name to ID mappings.")
    return mapping


def load_gene_disease_evidence(gene_disease_evidence_file):
    """Loads gene-disease evidence from a file."""
    evidence_mapping = {}
    if not gene_disease_evidence_file:
        return evidence_mapping
    try:
        with open(gene_disease_evidence_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                data_id = row.get('ID', '').strip()
                disease_name = row.get('Associated Conditions', '').strip()
                if data_id and disease_name:
                    evidence_mapping[(data_id, disease_name)] = 0.9  # Default weight for investigational conditions
    except FileNotFoundError:
        print(f"Warning: Gene-disease evidence file not found at {gene_disease_evidence_file}")
    except Exception as e:
        print(f"An error occurred while reading {gene_disease_evidence_file}: {e}")
    
    print(f"Loaded {len(evidence_mapping)} gene-disease evidence mappings.")
    return evidence_mapping

def load_uniprot_to_gene_mapping(uniprot_mapping_file):
    """Loads UniProt to Gene ID mappings from a file."""
    mapping = {}
    if not uniprot_mapping_file:
        return mapping
    try:
        with open(uniprot_mapping_file, 'r', newline='', encoding='utf-8') as file:
            # Assuming the file has headers and is tab-separated as per the user's example.
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                uniprot_id = row.get('UniProtKB Accession', '').strip()
                gene_id = row.get('Gene ID', '').strip()
                if uniprot_id and gene_id:
                    mapping[uniprot_id] = gene_id
    except FileNotFoundError:
        print(f"Warning: UniProt mapping file not found at {uniprot_mapping_file}")
    except Exception as e:
        print(f"An error occurred while reading {uniprot_mapping_file}: {e}")
    
    print(f"Loaded {len(mapping)} UniProt to Gene ID mappings.")
    return mapping

def load_investigational_conditions(filepath):
    """Loads investigational conditions from a file."""
    conditions = set()
    if not filepath:
        return conditions
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            # Assuming the file has headers and is tab-separated as per the user's example.
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                condition_id = row.get('ID', '').strip()
                condition_name = row.get('Associated Conditions', '').strip()
                if condition_id and condition_name:
                    conditions.add((condition_id, condition_name))
    except FileNotFoundError:
        print(f"Warning: Investigational conditions file not found at {filepath}")
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
    
    print(f"Loaded {len(conditions)} investigational conditions.")
    return conditions

def load_chemical_statuses(chemical_data_file):
    """Loads chemical statuses from the chemical data file."""
    statuses = {}
    try:
        with open(chemical_data_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    # Split line into MDB ID and JSON data
                    parts = line.strip().split('\t', 1)
                    if len(parts) != 2:
                        continue
                    
                    json_data = json.loads(parts[1])
                    compound_info = json_data.get('compound', {})
                    hmdb_id = compound_info.get('hmdb')
                    
                    if not hmdb_id:
                        continue

                    all_statuses = []
                    for conc_type in ['normal_concentration', 'abnormal_concentration']:
                        for item in json_data.get(conc_type, []):
                            status = item.get('status')
                            if status in ["Clinical", "Investigational"]:
                                all_statuses.append(status)
                    
                    if "Clinical" in all_statuses:
                        statuses[hmdb_id] = "Clinical"
                    elif "Investigational" in all_statuses:
                        statuses[hmdb_id] = "Investigational"

                except (json.JSONDecodeError, IndexError):
                    # Ignore lines that are not in the expected format
                    continue
    except FileNotFoundError:
        print(f"Warning: Chemical data file not found at {chemical_data_file}")
    except Exception as e:
        print(f"An error occurred while reading {chemical_data_file}: {e}")
    
    print(f"Loaded statuses for {len(statuses)} chemicals.")
    return statuses

def load_chemical_disease_mapping(chemical_mapping_file, mondo_mapping):
    """Loads chemical-disease mappings from the provided file."""
    mapping = {}
    processed_count = 0
    skipped_count = 0
    try:
        with open(chemical_mapping_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            for row in reader:
                hmdb_id = row.get('hmdb_id', '').strip()
                condition_name = row.get('conditions', '').strip()

                if not hmdb_id or not condition_name:
                    skipped_count += 1
                    continue

                mondo_id = mondo_mapping.get(condition_name.lower())
                if mondo_id:
                    if hmdb_id not in mapping:
                        mapping[hmdb_id] = []
                    # Avoid adding duplicate mondo_ids for the same hmdb_id
                    if mondo_id not in mapping[hmdb_id]:
                        mapping[hmdb_id].append(mondo_id)
                        processed_count += 1
                else:
                    skipped_count += 1
    except FileNotFoundError:
        print(f"Warning: Chemical mapping file not found at {chemical_mapping_file}")
    except Exception as e:
        print(f"An error occurred while reading {chemical_mapping_file}: {e}")

    print(f"Loaded {processed_count} chemical-disease mappings. Skipped {skipped_count} rows due to missing data or MONDO mapping.")
    return mapping

def load_protein_statuses(protein_data_file):
    """Loads protein statuses from the protein data file."""
    statuses = {}
    try:
        with open(protein_data_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    parts = line.strip().split('\t', 1)
                    if len(parts) != 2:
                        continue
                    
                    json_data = json.loads(parts[1])
                    protein_info = json_data.get('protein', {})
                    uniprot_id = protein_info.get('uniprot_id')
                    
                    if not uniprot_id:
                        continue

                    all_statuses = []
                    for conc_type in ['normal_concentration', 'abnormal_concentration']:
                        for item in json_data.get(conc_type, []):
                            status = item.get('status')
                            if status in ["Clinical", "Investigational"]:
                                all_statuses.append(status)
                    
                    if "Clinical" in all_statuses:
                        statuses[uniprot_id] = "Clinical"
                    elif "Investigational" in all_statuses:
                        statuses[uniprot_id] = "Investigational"

                except (json.JSONDecodeError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"Warning: Protein data file not found at {protein_data_file}")
    except Exception as e:
        print(f"An error occurred while reading {protein_data_file}: {e}")
    
    print(f"Loaded statuses for {len(statuses)} proteins.")
    return statuses

def load_protein_disease_mapping(protein_mapping_file, mondo_mapping):
    """Loads protein-disease mappings from the provided file."""
    mapping = {}
    processed_count = 0
    skipped_count = 0
    try:
        with open(protein_mapping_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            for row in reader:
                uniprot_id = row.get('uniprot_id', '').strip()
                condition_name = row.get('conditions', '').strip()

                if not uniprot_id or not condition_name:
                    skipped_count += 1
                    continue

                mondo_id = mondo_mapping.get(condition_name.lower())
                if mondo_id:
                    if uniprot_id not in mapping:
                        mapping[uniprot_id] = []
                    if mondo_id not in mapping[uniprot_id]:
                        mapping[uniprot_id].append(mondo_id)
                        processed_count += 1
                else:
                    skipped_count += 1
    except FileNotFoundError:
        print(f"Warning: Protein mapping file not found at {protein_mapping_file}")
    except Exception as e:
        print(f"An error occurred while reading {protein_mapping_file}: {e}")

    print(f"Loaded {processed_count} protein-disease mappings. Skipped {skipped_count} rows due to missing data or MONDO mapping.")
    return mapping

def load_refseq_mapping(refseq_node_file):
    """Loads RefSeq ID to gene name mappings from the refseq_node.csv file."""
    mapping = {}
    if not refseq_node_file:
        return mapping
    try:
        with open(refseq_node_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if len(row) > 4:
                    refseq_id = row[0].strip()
                    description = row[4].strip()
                    # Extract gene name from description, e.g., "... (MIRLET7F1), ..."
                    match = re.search(r'\((MIR[A-Z0-9-]+)\)', description, re.IGNORECASE)
                    if match:
                        gene_name = match.group(1).upper()
                        mapping[gene_name] = refseq_id
    except FileNotFoundError:
        print(f"Warning: RefSeq node file not found at {refseq_node_file}")
    except Exception as e:
        print(f"An error occurred while reading {refseq_node_file}: {e}")
    
    print(f"Loaded {len(mapping)} RefSeq gene name to ID mappings.")
    return mapping

# Mappings
# Gene: 14030, MDB00595091 595091-14030=
# Gene: 59029, MDB00628831 628831-59029=
def parse_concentration(concentration_str):
    if not concentration_str:
        return None
    # Use regex to find numbers (including floats) in the string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", concentration_str)
    if numbers:
        # Return the first number found
        return float(numbers[0])
    return None

def process_gene_disease(gene_disease_file, mondo_mapping, evidence_to_weight_mapping, writer_edge):
    """Processes gene-disease relationships and creates edges."""
    processed_edges = 0
    num_investigational = 0
    num_clinical = 0
    skip_due_to_missing = 0
    skip_due_to_duplicate = 0
    processed_set = set()
    
    with open(gene_disease_file, 'r', newline='', encoding='utf-8') as tsvfile:
        # Assuming a tab-separated file, adjust if necessary
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            gene_id = row.get('entrez_gene_id')
            condition_name = row.get('conditions', '').strip()
            record_id = row.get('id', '').strip()

            if not gene_id or not condition_name or gene_id == '0':
                skip_due_to_missing += 1
                continue

            mondo_id = mondo_mapping.get(condition_name.lower())
            if mondo_id:
                # Determine weight: 'Investigational' if in the list, otherwise 'Clinical'
                is_investigational = (record_id, condition_name) in evidence_to_weight_mapping
                indication_type = "Investigational" if is_investigational else "Clinical"
                weight = conditionsToWeightMapping.get(indication_type, 1.0)

                if weight > 0.0:
                    if (gene_id, mondo_id, weight) in processed_set:
                        skip_due_to_duplicate += 1
                        continue
                    processed_set.add((gene_id, mondo_id, weight))
                    edge_data = {
                        "Node 1 ID": NCBI_GENE_PREFIX + gene_id,
                        "Node 1 Data Type": "Gene",
                        "Node 1 Source": "NCBI",
                        "Edge Type": "gene_disease",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": weight,
                        "Node 2 ID": mondo_id,
                        "Node 2 Data Type": "Disease",
                        "Node 2 Source": "MONDO"
                    }
                    writer_edge.writerow(edge_data)
                    processed_edges += 1
                    if is_investigational:
                        num_investigational += 1
                    else:
                        num_clinical += 1

    print(f"Processed {processed_edges} gene-disease edges from the gene data file. Get {num_investigational} investigational and {num_clinical} clinical conditions. Note that {skip_due_to_missing} records were skipped due to missing gene ID or condition name, and {skip_due_to_duplicate} records were skipped due to duplicates.")

def process_chemical_data(chemical_disease_mapping, chemical_statuses, writer_edge):
    """Processes chemical data and creates edges."""
    processed_edges = 0
    skipped_edges = 0
    processed_set = set()

    for hmdb_id, mondo_ids in chemical_disease_mapping.items():
        status = chemical_statuses.get(hmdb_id)
        weight = conditionsToWeightMapping.get(status)

        if weight is None:
            # If there's no status, we cannot determine a weight, so we skip.
            skipped_edges += len(mondo_ids)
            continue

        for mondo_id in mondo_ids:
            if (hmdb_id, mondo_id, weight) in processed_set:
                skipped_edges += 1
                continue
            
            processed_set.add((hmdb_id, mondo_id, weight))

            edge_data = {
                "Node 1 ID": hmdb_id,
                "Node 1 Data Type": "Metabolite",
                "Node 1 Source": "HMDB",
                "Edge Type": "metabolite_disease",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": weight,
                "Node 2 ID": mondo_id,
                "Node 2 Data Type": "Disease",
                "Node 2 Source": "MONDO"
            }
            writer_edge.writerow(edge_data)
            processed_edges += 1

    print(f"Processed {processed_edges} chemical-disease edges. Skipped {skipped_edges} edges due to missing status or duplicates.")

def process_protein_data(protein_disease_mapping, protein_statuses, writer_edge):
    """Processes protein data and creates edges."""
    processed_edges = 0
    skipped_edges = 0
    processed_set = set()

    for uniprot_id, mondo_ids in protein_disease_mapping.items():
        status = protein_statuses.get(uniprot_id)
        weight = conditionsToWeightMapping.get(status)

        if weight is None:
            skipped_edges += len(mondo_ids)
            continue

        for mondo_id in mondo_ids:
            if (uniprot_id, mondo_id, weight) in processed_set:
                skipped_edges += 1
                continue
            
            processed_set.add((uniprot_id, mondo_id, weight))

            edge_data = {
                "Node 1 ID": uniprot_id,
                "Node 1 Data Type": "Protein",
                "Node 1 Source": "UniProt",
                "Edge Type": "protein_disease",
                "Edge Direction": edge_types.get("DD"),
                "Edge Weight": weight,
                "Node 2 ID": mondo_id,
                "Node 2 Data Type": "Disease",
                "Node 2 Source": "MONDO"
            }
            writer_edge.writerow(edge_data)
            processed_edges += 1

    print(f"Processed {processed_edges} protein-disease edges. Skipped {skipped_edges} edges due to missing status or duplicates.")


def process_microrna_disease(microrna_file, mondo_mapping, refseq_mapping, writer_edge):
    """Processes miRNA-disease relationships from a CSV file and creates edges."""
    processed_edges = 0
    skipped_due_to_missing = 0
    skipped_due_to_no_mondo = 0
    skipped_due_to_no_refseq = 0
    processed_set = set()

    try:
        with open(microrna_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 3:
                    skipped_due_to_missing += 1
                    continue

                gene_name = row[1].strip().upper()
                disease_name = row[2].strip()

                if not gene_name or not disease_name:
                    skipped_due_to_missing += 1
                    continue

                refseq_id = refseq_mapping.get(gene_name)
                if not refseq_id:
                    skipped_due_to_no_refseq += 1
                    continue

                mondo_id = mondo_mapping.get(disease_name.lower())
                if mondo_id:
                    # Using a default weight of 1.0 for miRNA-disease relationships
                    weight = 1.0
                    if (refseq_id, mondo_id, weight) in processed_set:
                        continue
                    
                    processed_set.add((refseq_id, mondo_id, weight))
                    
                    edge_data = {
                        "Node 1 ID": refseq_id,
                        "Node 1 Data Type": "RNA",
                        "Node 1 Source": "RefSeq",
                        "Edge Type": "rna_disease",
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": weight,
                        "Node 2 ID": mondo_id,
                        "Node 2 Data Type": "Disease",
                        "Node 2 Source": "MONDO"
                    }
                    writer_edge.writerow(edge_data)
                    processed_edges += 1
                else:
                    skipped_due_to_no_mondo += 1
    except FileNotFoundError:
        print(f"Warning: miRNA file not found at {microrna_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading {microrna_file}: {e}")
        return

    print(f"Processed {processed_edges} miRNA-disease edges. Skipped {skipped_due_to_missing} rows due to missing data, {skipped_due_to_no_mondo} due to no MONDO mapping, and {skipped_due_to_no_refseq} due to no RefSeq mapping.")


# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process biomarker data to create a knowledge graph.")
    parser.add_argument('-i', '--input', type=str, nargs=7, required=True, 
                        help="Input files: gene-disease evidence, gene-disease data, "
                             "chemical data, chemical mapping, protein data, protein mapping, rna disease mapping")
    parser.add_argument('--refseq-nodes', type=str, help="Input RefSeq nodes file for miRNA name to RefSeq ID mapping.")
    parser.add_argument('-m', '--mondo-nodes', type=str, required=True, help="Input MONDO nodes file for disease name to ID mapping.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV files (edge)")

    args = parser.parse_args()
    
    # 获取输入和输出文件路径
    gene_disease_evidence = args.input[0]
    gene_disease_file = args.input[1]
    chemical_data_file = args.input[2]
    chemical_data_mapping_file = args.input[3]
    protein_data_file = args.input[4]
    protein_mapping_file = args.input[5]
    microrna_file = args.input[6]
    refseq_node_file = args.refseq_nodes
    mondo_node_file = args.mondo_nodes
    output_edge_csv = args.output

    mondo_mapping = load_mondo_nodes(mondo_node_file)
    refseq_mapping = load_refseq_mapping(refseq_node_file)
    evidence_to_weight_mapping = load_gene_disease_evidence(gene_disease_evidence)
    chemical_disease_mapping = load_chemical_disease_mapping(chemical_data_mapping_file, mondo_mapping)
    chemical_statuses = load_chemical_statuses(chemical_data_file)
    protein_disease_mapping = load_protein_disease_mapping(protein_mapping_file, mondo_mapping)
    protein_statuses = load_protein_statuses(protein_data_file)

    with open(output_edge_csv, "w", newline="") as file_edge:
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_edge.writeheader()

        process_gene_disease(gene_disease_file, mondo_mapping, evidence_to_weight_mapping, writer_edge)
        process_chemical_data(chemical_disease_mapping, chemical_statuses, writer_edge)
        process_protein_data(protein_disease_mapping, protein_statuses, writer_edge)
        if microrna_file:
            process_microrna_disease(microrna_file, mondo_mapping, refseq_mapping, writer_edge)

    print(f"数据已成功导出到 边记录：{output_edge_csv}")

if __name__ == "__main__":
    main()