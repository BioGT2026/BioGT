import csv
import collections
import re
import xml.etree.ElementTree as ET
import pandas
import argparse
import rdkit.Chem
from DataProcessing.kg_construction.commonSettings import *


def load_uniprot_to_ncbi_id_map(gene_id_file):
    """
    Load UniProt -> NCBI GeneID mapping from TSV.
    Expected headers include: uniprot, GeneID
    """
    mapping = {}
    total_rows = 0

    with open(gene_id_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            print(f"Warning: gene_id_file has no header: {gene_id_file}")
            return mapping

        fieldname_map = {name.lower().lstrip('#'): name for name in reader.fieldnames}
        uniprot_col = fieldname_map.get("uniprot")
        gene_id_col = fieldname_map.get("geneid")

        if not uniprot_col or not gene_id_col:
            print(f"Warning: required columns 'uniprot' and 'GeneID' not found in {gene_id_file}.")
            return mapping

        for row in reader:
            total_rows += 1
            uniprot_id = str(row.get(uniprot_col, "")).strip()
            gene_id = str(row.get(gene_id_col, "")).strip()
            if not uniprot_id or not gene_id or uniprot_id == "-" or gene_id == "-":
                continue
            mapping[uniprot_id] = gene_id

    print(f"Loaded {len(mapping)} UniProt->GeneID mappings from {total_rows} rows in {gene_id_file}.")
    return mapping

def load_structure(path):
    """
    Load a structure file in SDF format
    :param path: The path to the SDF file
    :return: A list of RDKit molecules
    """
    supplier = rdkit.Chem.SDMolSupplier(path)
    molecules = [mol for mol in supplier if mol is not None]
    print(f"Number of molecules: {len(molecules)}")
    return molecules

def process_molecules(molecules):
    """
    Process molecules to calculate fingerprints
    :param molecules: A list of RDKit molecules
    :return: A dictionary of fingerprints
    """
    fingerprints = dict()
    for mol in molecules:
        drugbank_id = mol.GetProp('DATABASE_ID')
        # Calculate the smiles string
        smiles = rdkit.Chem.MolToSmiles(mol)
        fingerprints[drugbank_id] = smiles
    return fingerprints

def extract_protein_info(root, ns):
    """
    Extract protein information from a DrugBank XML element
    :param drug: An ElementTree element
    :param ns: The XML namespace
    :return: A list of dictionaries
    """

    # Extract protein information
    protein_rows = list()
    for i, drug in enumerate(root):
        drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
        for category in ['target', 'enzyme', 'carrier', 'transporter']:
            proteins = drug.findall('{ns}{cat}s/{ns}{cat}'.format(ns=ns, cat=category))
            for protein in proteins:
                organism = protein.findtext('{}organism'.format(ns))
                if organism != 'Humans':
                    continue
                row = {'drugbank_id': drugbank_id, 'category': category}
                row['organism'] = organism
                row['known_action'] = protein.findtext('{}known-action'.format(ns))
                actions = protein.findall('{ns}actions/{ns}action'.format(ns=ns))
                row['actions'] = '|'.join(action.text for action in actions)
                uniprot_ids = [polypep.text for polypep in protein.findall(
                    "{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier".format(ns=ns))]
                
                ref_text = protein.findtext("{ns}references[@format='textile']".format(ns=ns))
                pmids = re.findall(r'pubmed/([0-9]+)', ref_text) if ref_text else []
                row['pubmed_ids'] = '|'.join(pmids)

                for uniprot_id in uniprot_ids:
                    new_row = row.copy()
                    new_row['uniprot_id'] = uniprot_id
                    protein_rows.append(new_row)

    protein_df = pandas.DataFrame.from_dict(protein_rows)
    print(f"Extracted {len(protein_df)} protein entries from DrugBank.")

    # Save protein information to TSV
    columns = ['drugbank_id', 'category', 'uniprot_id', 'organism', 'known_action', 'actions', 'pubmed_ids']
    protein_df = protein_df[columns]

    return protein_df

def process_xml(xml_path,fingerprints,writer_node, writer_edge, gene_id_file, inchi_to_puchem_id_map, approved_only=False):
    """
    Process a DrugBank XML file to extract drug and protein information
    :param xml_path: The path to the DrugBank XML file
    :param inchi_to_puchem_id_map: A mapping of InChI strings to PubChem IDs
    :return: None
    """
    without_fp_count = 0
    with_fp_count = 0
    approved_count = 0
    investigational_count = 0
    # Load the DrugBank XML file
    with open(xml_path) as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define the XML namespace
    ns = '{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"
    uniprot_template = "{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier"

    edge_counts = collections.Counter()
    uniprot_to_ncbi_id = load_uniprot_to_ncbi_id_map(gene_id_file)
    protein_table = extract_protein_info(root, ns)
    human_drug_ids = set(protein_table['drugbank_id'])
    print(f"Found {len(human_drug_ids)} human drugs.")

    process_drug_ids = human_drug_ids
    if approved_only:
        approved_drug_ids = set()
        for drug in root:
            drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
            if drugbank_id in human_drug_ids:
                groups = [group.text for group in drug.findall("{ns}groups/{ns}group".format(ns=ns))]
                if 'approved' in groups:
                    approved_drug_ids.add(drugbank_id)
        process_drug_ids = approved_drug_ids
        print(f"Filtering for approved drugs. Found {len(process_drug_ids)} approved human drugs.")

    if approved_only:
        protein_table = protein_table[protein_table['drugbank_id'].isin(process_drug_ids)]

    num_nodes_count = 0
    unmapped_uniprot_count = 0
    seen_drug_gene_edges = set()
    # Extract drug information
    for i, drug in enumerate(root):
        drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
        if drugbank_id not in process_drug_ids:
            continue

        row = collections.OrderedDict()
        assert drug.tag == ns + 'drug'
        row['Type'] = drug.get('type')
        row['ID'] = drug.findtext(ns + "drugbank-id[@primary='true']")
        # Add the SMILES string to the row, if it exists
        if row['ID'] in fingerprints:
            row['Sequence'] = fingerprints.get(row['ID'])
            with_fp_count += 1
        else:
            row['Sequence'] = ""
            without_fp_count += 1
        row['name'] = drug.findtext(ns + "name")
        row['Description'] = drug.findtext(ns + "description")
        row['Groups'] = [group.text for group in drug.findall("{ns}groups/{ns}group".format(ns=ns))]
        
        groups = row['Groups']
        if 'approved' in groups:
            approved_count += 1
        if 'investigational' in groups:
            investigational_count += 1
            
        row['Atc_codes'] = [code.get('code') for code in drug.findall("{ns}atc-codes/{ns}atc-code".format(ns=ns))]
        row['Categories'] = [x.findtext(ns + 'category') for x in drug.findall("{ns}categories/{ns}category".format(ns=ns))]
        row['Inchi'] = drug.findtext(inchi_template.format(ns=ns))
        row['Inchikey'] = drug.findtext(inchikey_template.format(ns=ns))
    
        # Add drug aliases
        aliases = {
            elem.text for elem in 
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns=ns)) +
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}products/{ns}product/{ns}name".format(ns=ns))
        }
        aliases.add(row['name'])
        # Transfer the set to a string with the separator '|'
        row['Names'] = '|'.join(sorted(aliases))

        # Write the drug to the node file
        comments = f"Description: {row.get('Description', '')} Type: {row.get('Type', '')} Groups: {safe_join(row.get('Groups'))} Atc_codes: {safe_join(row.get('Atc_codes'))} Categories: {safe_join(row.get('Categories'))}"
        node = {
            "ID": row['ID'],
            "Type": "Drug",
            "Name": row['Names'],
            "Sequence": row['Sequence'],
            "Comments": comments
        }
        writer_node.writerow(node)
        
        # Extract drug-gene relationships
        for category in ['target', 'enzyme', 'carrier', 'transporter']:
            for protein in drug.findall('{ns}{cat}s/{ns}{cat}'.format(ns=ns, cat=category)):
                organism = protein.findtext('{}organism'.format(ns))
                if organism == 'Humans':
                    uniprot_ids = [polypep.text for polypep in protein.findall(uniprot_template.format(ns=ns))]
                    for uniprot_id in uniprot_ids:
                        mapped_gene_id = uniprot_to_ncbi_id.get(uniprot_id)
                        if not mapped_gene_id:
                            unmapped_uniprot_count += 1
                            continue
                        edge_key = (drugbank_id, mapped_gene_id)
                        if edge_key in seen_drug_gene_edges:
                            continue
                        seen_drug_gene_edges.add(edge_key)
                        edge = {
                            "Node 1 ID": drugbank_id,
                            "Node 1 Data Type": "Drug",
                            "Node 1 Source": "DrugBank",
                            "Edge Type": "drug_gene",
                            "Edge Direction": edge_types.get("DD"),
                            "Edge Weight": 1.0,
                            "Node 2 ID": NCBI_GENE_PREFIX + mapped_gene_id,
                            "Node 2 Data Type": "Gene",
                            "Node 2 Source": "NCBI"
                        }
                        writer_edge.writerow(edge)
                        edge_counts["drug_gene"] += 1


    for _, protein_row in protein_table.iterrows():
        if protein_row['uniprot_id'] is None:
            continue
        edge = {
            "Node 1 ID": protein_row['drugbank_id'],
            "Node 1 Data Type": "Drug",
            "Node 1 Source": "DrugBank",
            "Edge Type": "drug_protein",
            "Edge Direction": edge_types.get("DD"),
            "Edge Weight": 1.0,
            "Node 2 ID": UNIPROT_PROTEIN_PREFIX+protein_row['uniprot_id'],
            "Node 2 Data Type": "Protein",
            "Node 2 Source": "Uniprot"
        }
        writer_edge.writerow(edge)
        edge_counts["drug_protein"] += 1

    print(f"Processed {len(human_drug_ids)} drugs, with fingerprints: {with_fp_count}, without fingerprints: {without_fp_count}")
    print(f"FDA-approved drugs: {approved_count}")
    print(f"Investigational drugs: {investigational_count}")
    print(f"Unmapped UniProt IDs skipped in drug_gene edges: {unmapped_uniprot_count}")
    return edge_counts

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a Reactome txt file and export to CSV.")
    parser.add_argument('-i', '--input', type=str, nargs=3, required=True, help="Input (xml, sdf, tsv) file")
    parser.add_argument('-o', '--output', type=str, nargs=2, required=True, help="Output CSV files (node, edge)")
    parser.add_argument('--approved_only', action='store_true', help="Only extract FDA-approved drugs and their edges.")

    args = parser.parse_args()
    # 获取输入和输出文件路径
    xml_file = args.input[0]
    sdf_file = args.input[1]
    gene_id_file = args.input[2]
    output_node_csv = args.output[0]
    output_edge_csv = args.output[1]
    inchi_to_puchem_id_map = {}

    with open(output_node_csv, "w", newline="") as file_node, open(output_edge_csv, "w", newline="") as file_edge:
        writer_node = csv.DictWriter(file_node, fieldnames=fieldnames_node)
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_node.writeheader()
        writer_edge.writeheader()
        # Load the structure file
        molecules = load_structure(sdf_file)
        # Process the molecules
        fingerprints = process_molecules(molecules)
        print(f"Number of molecular fingerprints: {len(fingerprints)}")
        # Process the XML file
        edge_counts = process_xml(xml_file, fingerprints, writer_node, writer_edge, gene_id_file, inchi_to_puchem_id_map, args.approved_only)
        print(f"数据已成功导出到节点记录：{output_node_csv}, 边记录：{output_edge_csv}")
        print("Edge counts:")
        for edge_type, count in edge_counts.items():
            print(f"  {edge_type}: {count}")

if __name__ == "__main__":
    main()
