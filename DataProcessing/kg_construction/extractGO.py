import csv
from goatools.obo_parser import GODag
from DataProcessing.kg_construction.commonSettings import *
import argparse # Added import
import os # Added import
from collections import Counter # Added import

# --- Configuration ---
# These will be filled based on user input or defaults
GO_OBO_FILE = ""  # Path to the Gene Ontology OBO file
GENE_ASSOCIATION_FILE = None  # Optional: Path to the gene association file (e.g., GAF)
OUTPUT_NODES_CSV = "go_nodes.csv"
OUTPUT_EDGES_CSV = "go_edges.csv"

# GO Namespaces to be processed
TARGET_NAMESPACES = {
    "biological_process",
    "molecular_function",
    "cellular_component"
}

def write_node(writer, node_id, node_type, name, sequence="", comments=""):
    """Writes a node to the CSV file."""
    writer.writerow({
        "ID": node_id,
        "Type": node_type,
        "Name": name,
        "Sequence": sequence,
        "Comments": comments
    })

def write_edge(writer, node1_id, node1_type, node1_source,
               edge_type_code, edge_direction, edge_weight,
               node2_id, node2_type, node2_source):
    """Writes an edge to the CSV file."""
    writer.writerow({
        "Node 1 ID": node1_id,
        "Node 1 Data Type": node1_type,
        "Node 1 Source": node1_source,
        "Edge Type": edge_type_code,
        "Edge Direction": edge_direction,
        "Edge Weight": edge_weight,
        "Node 2 ID": node2_id,
        "Node 2 Data Type": node2_type,
        "Node 2 Source": node2_source
    })

def process_go_terms(godag, nodes_writer, edges_writer):
    """
    Processes GO terms from the GODag object, writing nodes and ALL their relationships.
    """
    processed_nodes = set()
    node_type_map = {
        "biological_process": "BiologicalProcess",
        "molecular_function": "MolecularFunction",
        "cellular_component": "CellularComponent"
    }
    count_node_type_numbers = {node_type: 0 for node_type in node_type_map.values()}
    obsolete_term_count = 0  # Counter for obsolete terms

    relationship_directions = {
        "is_a": edge_types["SD"],
        "part_of": edge_types["SD"],
        "regulates": edge_types["SD"],
        "negatively_regulates": edge_types["SD"],
        "positively_regulates": edge_types["SD"],
        "has_part": edge_types["SD"],
        "occurs_in": edge_types["SD"],
        "intersection_of": edge_types["SD"]  # Added intersection_of
    }
    
    all_found_relationship_types = [] # Changed to list to count occurrences

    for _, go_term in godag.items():
        # First, check if the term is obsolete and count it.
        print(f"Processing GO Term: {go_term.item_id} - {go_term.name} (Namespace: {go_term.namespace}, is obsolete: {go_term.is_obsolete})")
        if go_term.is_obsolete:
            obsolete_term_count += 1
            continue  # Skip obsolete terms from further processing
        # Then, check if the term belongs to the target namespaces.
        if go_term.namespace not in TARGET_NAMESPACES:
            continue

        go_id = go_term.item_id  # Use item_id for consistency with GODag
        current_node_type = node_type_map.get(go_term.namespace, "GOTerm")

        # Skip duplicated canonical GO IDs (can happen due to alt_id mappings in GODag)
        # to avoid duplicated edges and stale node type reuse.
        if go_id in processed_nodes:
            continue

        count_node_type_numbers[current_node_type] += 1
        comments_list = [f"Depth: {go_term.depth}, Level: {go_term.level}"]
        # Add definition if available
        if hasattr(go_term, 'defn') and go_term.defn:
            comments_list.append(f"Def: {go_term.defn}")
        elif hasattr(go_term, 'def') and getattr(go_term, 'def'): # some obo parsers might use 'def'
                comments_list.append(f"Def: {getattr(go_term, 'def')}")

        # Add synonyms if available
        synonyms_str = ""
        if hasattr(go_term, 'synonym') and go_term.synonym:
            # go_term.synonym is usually a set of Synonym objects or strings
            # We need to extract the synonym string itself
            processed_syns = []
            for syn_item in go_term.synonym:
                if hasattr(syn_item, 'name'): # If Synonym object
                    processed_syns.append(syn_item.name)
                elif isinstance(syn_item, str): # If plain string
                    processed_syns.append(syn_item)
            if processed_syns:
                synonyms_str = "; ".join(processed_syns)
                comments_list.append(f"Synonyms: {synonyms_str}")
        
        node_comments = " | ".join(comments_list)
        write_node(nodes_writer, go_term.id, current_node_type, go_term.name, comments=node_comments)
        processed_nodes.add(go_id)

        # Handle 'is_a' relationships (parents)
        # go_term IS_A parent_term  (edge: go_term -> parent_term)
        for parent_term in go_term.parents:
            if parent_term.id in godag and not godag[parent_term.id].is_obsolete and \
               godag[parent_term.id].namespace in TARGET_NAMESPACES:
                
                parent_node_type = node_type_map.get(godag[parent_term.id].namespace, "GOTerm")
                edge_dir = relationship_directions.get("is_a", edge_types["SD"])
                
                write_edge(edges_writer,
                           node1_id=go_term.id, 
                           node1_type=current_node_type, 
                           node1_source="GO",
                           edge_type_code="is_a",
                           edge_direction=edge_dir,
                           edge_weight=1.0, 
                           node2_id=parent_term.id,
                           node2_type=parent_node_type,
                           node2_source="GO")
                all_found_relationship_types.append("is_a") # Ensure "is_a" is recorded

        # Handle 'is_parent_of' relationships (children of the current go_term)
        # This creates edges from go_term (as parent) to its children.
        for child_term_obj in go_term.children:
            if child_term_obj.id in godag and \
               not godag[child_term_obj.id].is_obsolete and \
               godag[child_term_obj.id].namespace in TARGET_NAMESPACES:
                
                child_node_type = node_type_map.get(godag[child_term_obj.id].namespace, "GOTerm")
                
                # Edge: go_term (parent) -> child_term_obj (child)
                write_edge(edges_writer,
                           node1_id=go_term.id,         # Parent node (current go_term)
                           node1_type=current_node_type, 
                           node1_source="GO",
                           edge_type_code="is_parent_of", # Explicitly define this relationship type
                           edge_direction=edge_types["SD"], # Assuming directed
                           edge_weight=1.0, 
                           node2_id=child_term_obj.id,  # Child node
                           node2_type=child_node_type,
                           node2_source="GO")
                all_found_relationship_types.append("is_parent_of")


        # Handle 'intersection_of' relationships
        # go_term IS_INTERSECTION_OF related_term (edge: go_term -> related_term)
        if hasattr(go_term, 'intersection_of') and go_term.intersection_of:
            for related_term_obj in go_term.intersection_of:
                # The items in intersection_of can sometimes be relationship type strings
                # or actual GOTerm objects. We are interested in GOTerm objects.
                if not hasattr(related_term_obj, 'id'): # Skip if not a GOTerm object (e.g. a relationship type string)
                    if isinstance(related_term_obj, str):
                        all_found_relationship_types.append(related_term_obj) # Record the string if it's a relationship type
                    continue

                related_term_id = related_term_obj.id
                if related_term_id in godag and not godag[related_term_id].is_obsolete and \
                   godag[related_term_id].namespace in TARGET_NAMESPACES:
                    
                    related_node_type = node_type_map.get(godag[related_term_id].namespace, "GOTerm")
                    edge_dir = relationship_directions.get("intersection_of", edge_types["SD"])
                    
                    write_edge(edges_writer,
                               node1_id=go_term.id, node1_type=current_node_type, node1_source="GO",
                               edge_type_code="intersection_of",
                               edge_direction=edge_dir,
                               edge_weight=1.0,
                               node2_id=related_term_id,
                               node2_type=related_node_type,
                               node2_source="GO")
                    all_found_relationship_types.append("intersection_of")

        # Handle other relationships from the 'relationship' attribute
        if hasattr(go_term, 'relationship') and go_term.relationship:
            for relationship_type, related_terms_set in go_term.relationship.items():
                if relationship_type == "is_a":
                    all_found_relationship_types.append("is_a") 
                    continue 

                for related_term_obj in related_terms_set:
                    related_term_id = related_term_obj.id
                    if related_term_id in godag and not godag[related_term_id].is_obsolete and \
                       godag[related_term_id].namespace in TARGET_NAMESPACES:
                        
                        related_node_type = node_type_map.get(godag[related_term_id].namespace, "GOTerm")
                        edge_dir = relationship_directions.get(relationship_type, edge_types["SD"]) 

                        write_edge(edges_writer,
                                   node1_id=go_term.id, node1_type=current_node_type, node1_source="GO",
                                   edge_type_code=relationship_type,
                                   edge_direction=edge_dir,
                                   edge_weight=1.0, 
                                   node2_id=related_term_id,
                                   node2_type=related_node_type,
                                   node2_source="GO")
                        all_found_relationship_types.append(relationship_type)
        
        # Handle xrefs, specifically for Reactome pathways
        if hasattr(go_term, 'xref') and go_term.xref:
            reactome_ids_to_process = set() # Use a set to store unique Reactome IDs before writing edges

            if isinstance(go_term.xref, dict):
                # Case 1: go_term.xref is a dict like {'Reactome': {"R-HSA-123 \"Desc\"", ...}}
                if 'Reactome' in go_term.xref:
                    for xref_val_str in go_term.xref['Reactome']:
                        if xref_val_str: # Ensure not None or empty
                            try:
                                reactome_id = xref_val_str.split(' ')[0]
                                reactome_ids_to_process.add(reactome_id)
                            except Exception as e:
                                print(f"Warning: Error parsing Reactome xref value (dict path) '{xref_val_str}' for GO ID {go_term.id}: {e}")
            elif isinstance(go_term.xref, set):
                # Case 2: go_term.xref is a set of strings like {"Reactome:R-HSA-123", "OtherDB:ID"}
                for xref_full_str in go_term.xref:
                    if isinstance(xref_full_str, str) and xref_full_str.startswith("Reactome:"):
                        try:
                            val_part = xref_full_str.split(":", 1)[1] # Get part after "Reactome:"
                            reactome_id = val_part.split(" ")[0]      # Get ID before any description
                            reactome_ids_to_process.add(reactome_id)
                        except IndexError:
                            print(f"Warning: Malformed Reactome xref string (set path, index error): '{xref_full_str}' for GO ID {go_term.id}")
                        except Exception as e:
                            print(f"Warning: Error parsing Reactome xref string (set path) '{xref_full_str}' for GO ID {go_term.id}: {e}")
            # else:
                # print(f"DEBUG: GO:{go_term.id}, xref is of unhandled type: {type(go_term.xref)}, content: {go_term.xref}")

            # Write edges for all collected unique Reactome IDs
            for r_id in reactome_ids_to_process:
                if r_id: # Ensure not empty
                    write_edge(edges_writer,
                               node1_id=go_term.id, 
                               node1_type=current_node_type, 
                               node1_source="GO",
                               edge_type_code=current_node_type+"_pathway",
                               edge_direction=edge_types["SD"], 
                               edge_weight=1.0, 
                               node2_id=r_id, # Reactome ID without prefix
                               node2_type="Pathway",      
                               node2_source="Reactome")   
                    all_found_relationship_types.append(current_node_type+"_pathway")

    print("Finished processing GO terms. All unique relationship types encountered in OBO (including xrefs):")
    relationship_counts = Counter(all_found_relationship_types)
    for rel_type, count in sorted(relationship_counts.items()):
        print(f"  - {rel_type} : {count}")
    # FIX: Iterate over dictionary items to print counts
    for node_type, count in count_node_type_numbers.items():
        print(f"Total {node_type} nodes processed: {count}")

    print(f"Total unique GO IDs processed: {len(processed_nodes)}")
    print(f"Total obsolete GO terms processed: {obsolete_term_count}")

def process_gene_associations(gaf_file_path, godag, nodes_writer, edges_writer, processed_nodes):
    """
    Processes a gene association file (GAF) to extract gene/protein-GO term relationships.
    This is a simplified GAF parser. For robust parsing, consider `goatools.anno.gafpea.GafReader`.
    """
    # print(f"Processing gene association file: {gaf_file_path}")
    count_unique_go_ids = set()  # To count unique GO IDs processed from GAF
    count_unique_genes = set() # To count unique genes processed from GAF
    count_skipped_due_to_less_fields = 0 # To count skipped lines due to less than expected fields
    try:
        with open(gaf_file_path, 'r') as f_gaf:
            for line in f_gaf:
                if line.startswith('!') or line.strip() == "":  # Skip comments and empty lines
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 17: # GAF 2.2 has at least 17 fields
                    # print(f"Skipping malformed GAF line (not enough fields): {line.strip()}")
                    count_skipped_due_to_less_fields += 1
                    continue

                db_object_id = fields[1]    # DB Object ID (e.g., UniProtKB ID for protein, gene ID)
                db_object_symbol = fields[2] # DB Object Symbol (e.g., gene name)
                qualifier_str = fields[3]      # Qualifier (e.g., NOT, contributes_to, colocalizes_with)
                go_id = fields[4]          # GO ID
                # db_reference = fields[5]   # DB:Reference (|DB:Reference)
                evidence_code = fields[6]  # Evidence Code (e.g., IEA, IDA, IMP)
                # with_from = fields[7]      # With (or) From
                aspect = fields[8]         # Aspect (P, F, C)
                # db_object_name = fields[9] # DB Object Name (|Synonym)
                # db_object_synonym = fields[10] # DB Object Synonym (|Synonym)
                db_object_type = fields[11] # DB Object Type (e.g., protein, gene, complex)
                # taxon = fields[12]         # Taxon(|taxon)
                # date = fields[13]          # Date
                # assigned_by = fields[14]   # Assigned By
                # annotation_extension = fields[15] # Annotation Extension
                # gene_product_form_id = fields[16] # Gene Product Form ID (GAF 2.1 and later)
                node1_source = fields[0] # DB (e.g., UniProtKB, SGD, MGI)

                count_unique_go_ids.add(go_id)  # Track unique GO IDs processed
                count_unique_genes.add(db_object_id) # Track unique gene/protein IDs processed

                if "NOT" in qualifier_str.upper(): # Skip NOT annotations specifically if it's the only qualifier or part of a set
                    # More precise check: if all qualifiers are NOT, or if NOT is present without other positive qualifiers.
                    # For simplicity, if "NOT" is anywhere, we skip. This matches common GAF interpretation for direct annotations.
                    continue

                if go_id not in godag or godag[go_id].is_obsolete:
                    print(f"Skipping association with obsolete or unknown GO ID: {go_id}")
                    continue

                go_term = godag[go_id]
                if go_term.namespace not in TARGET_NAMESPACES:
                    continue

                # Determine gene/protein node ID and type
                node1_id = ""
                node1_type = ""
                node1_name = db_object_symbol

                if db_object_type == "protein" or "protein" in db_object_type or node1_source.upper() == "UNIPROTKB":
                    node1_id = f"{UNIPROT_PROTEIN_PREFIX}{db_object_id}"
                    node1_type = "Protein"
                elif db_object_type.lower() == "gene" or "gene" in db_object_type.lower(): # MODIFIED HERE
                    # Assuming NCBI Gene if not specified, adjust if needed. GAF source (fields[0]) might give a hint.
                    node1_id = f"{NCBI_GENE_PREFIX}{db_object_id}"
                    node1_type = "Gene"
                # elif db_object_type == "complex":
                #     node1_id = f"Complex:{db_object_id}" # Or some other prefix
                #     node1_type = "Complex"
                else:
                    # print(f"Skipping unknown DB Object Type: {db_object_type} for ID {db_object_id}")
                    continue # Or handle as a generic entity

                if not node1_id: # Should not happen if logic above is correct
                    continue

                # Write gene/protein node if not already processed
                if node1_id not in processed_nodes:
                    # write_node(nodes_writer, node1_id, node1_type, node1_name, comments=f"SourceDB: {node1_source}")
                    processed_nodes.add(node1_id)

                # Write edge between gene/protein and GO term
                edge_w = evidence_to_weight(evidence_code)
                go_term_type = go_term.namespace.replace("_", " ").title()

                # Determine specific edge type from qualifier or aspect
                specific_edge_type = "associated_with" # Default
                # Parse qualifiers, excluding any "NOT" entries if they are part of a list
                parsed_qualifiers = [q.strip() for q in qualifier_str.split('|') if q.strip() and "NOT" not in q.upper()]

                if parsed_qualifiers:
                    specific_edge_type = parsed_qualifiers[0] # Use the first valid qualifier
                else:
                    # Fallback to aspect if qualifier is empty or only contained NOT (which would have been skipped earlier)
                    if aspect == 'F': # Molecular Function
                        specific_edge_type = "enables"      # Gene/protein enables a molecular function
                    elif aspect == 'P': # Biological Process
                        specific_edge_type = "involved_in" # Gene/protein is involved in a biological process
                    elif aspect == 'C': # Cellular Component
                        specific_edge_type = "located_in"  # Gene/protein is located in a cellular component
                    # else, it remains "associated_with"

                write_edge(edges_writer,
                           node1_id=node1_id, 
                           node1_type=node1_type, 
                           node1_source=node1_source,
                           edge_type_code=specific_edge_type,  # Use determined specific edge type
                           edge_direction=edge_types["SD"], # SingleDirected from Gene/Protein to GO Term
                           edge_weight=edge_w,
                           node2_id=go_id, 
                           node2_type=go_term_type, 
                           node2_source="GO")

    except FileNotFoundError:
        print(f"Error: Gene association file not found at {gaf_file_path}")
    except Exception as e:
        print(f"An error occurred while processing the GAF file: {e}")
    print(f"Processed {len(count_unique_go_ids)} unique GO IDs from GAF file.")
    print(f"Processed {len(count_unique_genes)} unique gene/protein IDs from GAF file.")
    print(f"Skipped {count_skipped_due_to_less_fields} lines due to insufficient fields in GAF file.")

def main():
    """Main function to orchestrate GO data extraction."""
    global GO_OBO_FILE, GENE_ASSOCIATION_FILE, OUTPUT_NODES_CSV, OUTPUT_EDGES_CSV

    parser = argparse.ArgumentParser(description="Extract GO data from OBO and optionally GAF files, and export to CSV.")
    parser.add_argument('--obo', type=str, required=True, help="Path to the Gene Ontology OBO file (e.g., go-basic.obo)")
    parser.add_argument('--gaf', type=str, required=False, help="Optional: Path to the Gene Association File (GAF)")
    parser.add_argument('--nodes_out', type=str, default="go_nodes.csv", help="Output path for the nodes CSV file (default: go_nodes.csv)")
    parser.add_argument('--edges_out', type=str, default="go_edges.csv", help="Output path for the edges CSV file (default: go_edges.csv)")

    args = parser.parse_args()

    GO_OBO_FILE = args.obo
    GENE_ASSOCIATION_FILE = args.gaf
    OUTPUT_NODES_CSV = args.nodes_out
    OUTPUT_EDGES_CSV = args.edges_out

    if not os.path.exists(GO_OBO_FILE):
        print(f"Error: GO OBO file not found at {GO_OBO_FILE}")
        return

    if GENE_ASSOCIATION_FILE and not os.path.exists(GENE_ASSOCIATION_FILE):
        print(f"Warning: Gene Association File not found at {GENE_ASSOCIATION_FILE}. Proceeding without it.")
        GENE_ASSOCIATION_FILE = None


    print(f"Loading GO OBO file from: {GO_OBO_FILE}")
    try:
        # Ensure all relevant optional attributes are parsed
        godag = GODag(GO_OBO_FILE, optional_attrs={'def', 'synonym', 'xref', 'relationship', 'alt_id'})
    except Exception as e:
        print(f"Error loading GO OBO file: {e}")
        return

    processed_gaf_nodes = set()

    try:
        with open(OUTPUT_NODES_CSV, 'w', newline='', encoding='utf-8') as f_nodes, \
             open(OUTPUT_EDGES_CSV, 'w', newline='', encoding='utf-8') as f_edges:

            nodes_writer = csv.DictWriter(f_nodes, fieldnames=fieldnames_node)
            edges_writer = csv.DictWriter(f_edges, fieldnames=fieldnames_edge)

            nodes_writer.writeheader()
            edges_writer.writeheader()

            print("Processing GO terms...")
            process_go_terms(godag, nodes_writer, edges_writer)
            # print(f"GO terms processing complete. Nodes written to {OUTPUT_NODES_CSV}, Edges to {OUTPUT_EDGES_CSV}")

            if GENE_ASSOCIATION_FILE:
                print(f"Processing gene associations from: {GENE_ASSOCIATION_FILE}")
                process_gene_associations(GENE_ASSOCIATION_FILE, godag, nodes_writer, edges_writer, processed_gaf_nodes)
                print(f"Gene association processing complete.")
            else:
                print("No gene association file provided or found. Skipping gene/protein associations.")

    except IOError as e:
        print(f"Error opening or writing CSV files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()