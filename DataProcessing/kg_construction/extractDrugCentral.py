import csv
import collections
import os
import pandas
import sys
import argparse
import itertools
from DataProcessing.kg_construction.commonSettings import *
filedNameForDrugATC = ['Drug ID', 'Drug Name', 'CAS', 'ATC Code', 'InChi']
filedNameForDiseaseMap = ['MONDO ID', 'MONDO Name', 'UMLS ID']
fieldNameForDrugDisease = ['cd_id','cd_formula','cd_molweight','id','clogp','alogs','cas_reg_no','tpsa','lipinski','name','no_formulations','stem','molfile','mrdef','enhanced_stereo','arom_c','sp3_c','sp2_c','sp_c','halogen','hetero_sp2_c','rotb','molimg','o_n','oh_nh','inchi','smiles','rgb','fda_labels','inchikey','status','id','struct_id','concept_id','relationship_name','concept_name','umls_cui','snomed_full_name','cui_semantic_type','snomed_conceptid'] # We will use 'cas_reg_no' to find the drug id, than use 'relationship_name' to determine the type of relationship. We also use 'umls_cui' to find the disease id in mondo.

def process_mapping(drug_cas_data, drug_disease_data, mondo_umls_data, writer_edge):
    """
    Process mappings to create drug-disease relationships.
    The link is Drug -> ATC -> UMLS -> MONDO.
    :param drug_cas_data: DataFrame with drug to ATC code mappings.
    :param drug_disease_data: DataFrame with ATC to disease (UMLS) relationships.
    :param mondo_umls_data: DataFrame with MONDO to UMLS mappings.
    :param writer_edge: CSV writer for edge data.
    :return: None
    """
    # 1. Build mappings from drug data.
    cas_to_drug_ids = collections.defaultdict(list)
    name_to_drug_ids = collections.defaultdict(list)
    for _, row in drug_cas_data.iterrows():
        drug_id = row.get('Drug ID')
        cas_codes_str = row.get('CAS')
        drug_name = row.get('Drug Name')
        
        if pandas.notna(drug_id):
            # Map by CAS code
            if pandas.notna(cas_codes_str):
                cas_codes = str(cas_codes_str).split('|')
                for code in cas_codes:
                    code = code.strip()
                    if code:
                        cas_to_drug_ids[code].append(drug_id)
            # Map by Drug Name
            if pandas.notna(drug_name):
                name_key = str(drug_name).strip().lower()
                if name_key:
                    name_to_drug_ids[name_key].append(drug_id)

    print(f"Built index for {len(cas_to_drug_ids)} unique cas codes from drug-cas mapping.")
    print(f"Built index for {len(name_to_drug_ids)} unique drug names from drug-cas mapping.")

    # 2. Build a mapping from UMLS CUI to MONDO ID.
    # A single UMLS CUI can map to multiple MONDO IDs, so we store a list.
    umls_to_mondo_ids = collections.defaultdict(list)
    for _, row in mondo_umls_data.iterrows():
        umls_cui = row.get('UMLS ID')
        mondo_id = row.get('MONDO ID')
        if pandas.notna(umls_cui) and pandas.notna(mondo_id):
            # Assuming UMLS_CUI in the file might have a prefix or not. Let's be safe.
            if ':' in str(umls_cui):
                umls_cui = str(umls_cui).split(':')[-1]
            umls_to_mondo_ids[str(umls_cui).strip()].append(mondo_id)
    print(f"Built index for {len(umls_to_mondo_ids)} unique UMLS CUIs from MONDO-UMLS mapping.")

    # Build a mapping from MONDO Name to MONDO ID.
    name_to_mondo_ids = collections.defaultdict(list)
    for _, row in mondo_umls_data.iterrows():
        mondo_name = row.get('MONDO Name')
        mondo_id = row.get('MONDO ID')
        if pandas.notna(mondo_name) and pandas.notna(mondo_id):
            name_to_mondo_ids[str(mondo_name).strip().lower()].append(mondo_id)
    print(f"Built index for {len(name_to_mondo_ids)} unique MONDO names from MONDO mapping.")


    # 3. Iterate through the drug-disease data (ATC-UMLS) and create edges.
    edges_written = 0
    edges_found = 0
    drug_not_found = 0
    disease_not_found = 0
    for _, row in drug_disease_data.iterrows():
        edges_found += 1
        cas_code = row.get('cas_reg_no')
        drug_name_from_central = row.get('name')
        umls_cui = row.get('umls_cui')
        concept_name = row.get('concept_name')
        relationship_type = row.get('relationship_name', 'drug_disease_association')

        if (pandas.notna(cas_code) or pandas.notna(drug_name_from_central)) and (pandas.notna(umls_cui) or pandas.notna(concept_name)):
            mondo_ids = []
            # Find corresponding MONDO IDs for the UMLS CUI.
            if pandas.notna(umls_cui) and str(umls_cui) in umls_to_mondo_ids:
                mondo_ids.extend(umls_to_mondo_ids[str(umls_cui)])

            # Try to match by concept name
            if pandas.notna(concept_name):
                name_key = str(concept_name).strip().lower()
                if name_key in name_to_mondo_ids:
                    mondo_ids.extend(name_to_mondo_ids[name_key])

            # Find corresponding Drug IDs using both CAS and Name.
            drug_ids = []
            if pandas.notna(cas_code) and cas_code in cas_to_drug_ids:
                drug_ids.extend(cas_to_drug_ids[cas_code])
            
            if pandas.notna(drug_name_from_central):
                name_key = str(drug_name_from_central).strip().lower()
                if name_key in name_to_drug_ids:
                    drug_ids.extend(name_to_drug_ids[name_key])

            if drug_ids:
                # Remove duplicates
                drug_ids = list(set(drug_ids))
            else:
                print(f"Warning: No drug IDs found for CAS code '{cas_code}' or drug name '{drug_name_from_central}'", file=sys.stdout)
                drug_not_found += 1
                continue

            if mondo_ids:
                # Remove duplicates
                mondo_ids = list(set(mondo_ids))
            else:
                print(f"Warning: No MONDO IDs found for UMLS CUI '{umls_cui}' or concept name '{concept_name}'", file=sys.stdout)
                disease_not_found += 1
                continue

            # Create an edge for each combination of Drug and MONDO ID.
            for drug_id in drug_ids:
                for mondo_id in mondo_ids:
                    edge = {
                        "Node 1 ID": drug_id,
                        "Node 1 Data Type": "Drug",
                        "Node 1 Source": "DrugBank", # Source of drug_atc data
                        "Edge Type": relationship_type.lower().replace(' ', '-'),
                        "Edge Direction": edge_types.get("DD"),
                        "Edge Weight": 1.0,
                        "Node 2 ID": mondo_id, # Use MONDO ID here
                        "Node 2 Data Type": "Disease",
                        "Node 2 Source": "MONDO", # Source of the relationship
                    }
                    writer_edge.writerow(edge)
                    edges_written += 1
    
    print(f"Found {edges_found}, not found {drug_not_found} drugs, {disease_not_found} diseases; Wrote {edges_written} drug-disease edges.")

def main():
    # Using argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Create drug-disease relationships from DrugCentral, Drug-ATC, and MONDO-UMLS mappings.")
    # Update to take 3 input files
    parser.add_argument('-i', '--input', type=str, nargs=3, required=True, help="Input CSV files (drug_disease.csv, drug_atc.csv, mondo_umls.csv)")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV file for edges")

    args = parser.parse_args()
    
    drug_disease_file = args.input[0]
    drug_atc_file = args.input[1]
    mondo_umls_file = args.input[2] # New input file
    output_edge_csv = args.output[0]

    # Load the data using pandas, assuming files have headers.
    try:
        drug_disease_data = pandas.read_csv(drug_disease_file)
        drug_cas_data = pandas.read_csv(drug_atc_file)
        mondo_umls_data = pandas.read_csv(mondo_umls_file) # Load new data
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input CSV files: {e}", file=sys.stderr)
        sys.exit(1)

    if not all(col in drug_disease_data.columns for col in fieldNameForDrugDisease):
        print(f"Error: Drug-disease file ({drug_disease_file}) is missing one or more required columns: {fieldNameForDrugDisease}", file=sys.stderr)
        sys.exit(1)
    
    if not all(col in drug_cas_data.columns for col in filedNameForDrugATC):
        print(f"Error: Drug-ATC file ({drug_atc_file}) is missing one or more required columns: {filedNameForDrugATC}", file=sys.stderr)
        sys.exit(1)

    if not all(col in mondo_umls_data.columns for col in filedNameForDiseaseMap):
        print(f"Error: MONDO-UMLS file ({mondo_umls_file}) is missing one or more required columns: {filedNameForDiseaseMap}", file=sys.stderr)
        sys.exit(1)

    with open(output_edge_csv, "w", newline="") as file_edge:
        writer_edge = csv.DictWriter(file_edge, fieldnames=fieldnames_edge)
        writer_edge.writeheader()
        process_mapping(drug_cas_data, drug_disease_data, mondo_umls_data, writer_edge)
        print(f"Data successfully exported to edge file: {output_edge_csv}")

if __name__ == "__main__":
    main()
