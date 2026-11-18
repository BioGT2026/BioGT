import csv
import os
import sys
import argparse
from collections import defaultdict
from DataProcessing.kg_construction.commonSettings import *
import logging
import glob # Import glob

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
csv.field_size_limit(1000 * 1024 * 1024)

FIELDNAMES_NODE = fieldnames_node
FIELDNAMES_EDGE = fieldnames_edge
MERGE_DELIMITER = '; '
translated_gene_names_count = 0

# --- Statistics Tracking ---
source_statistics = {
    'nodes': {},  # {file_path: {'original': count, 'merged': count, 'by_type': {type: {'original': count, 'merged': count}}}}
    'edges': {},  # {file_path: {'original': count, 'merged': count, 'by_type': {edge_type: {'original': count, 'merged': count}}}}
    'marker_edges': {'original': 0, 'merged': 0, 'by_type': {}}  # edge_type: {'original': count, 'merged': count}
}

# --- Helper Functions ---

def replace_data_type(existing_type):
    """Replaces the existing value with the new value if the new value is not empty."""
    if existing_type is None or existing_type == "":
        return "" # Return empty if input is empty or None

    # Convert to lowercase for case-insensitive matching
    etype_lower = existing_type.lower()
    # Updated list for 'misc' mapping (removed 'organism')
    if etype_lower in ["uniprot", "protein"]:
        return "protein"
    if etype_lower in ["rna", "mrna", "rrna"]:
        return "rna"
    if etype_lower in ["gene"]:
        return "gene"
    if etype_lower in ["drug"]:
        return "drug"
    if etype_lower in ["clinvar"]:
        return "variant"
    if etype_lower in ["disease"]:
        return "disease"
    if etype_lower in ["cellularcomponent"]:
        return "cellular_component"
    if etype_lower in ["biologicalprocess"]:
        return "biological_process"
    if etype_lower in ["molecularfunction"]:
        return "molecular_function"
    if etype_lower in ["phenotype"]:
        return "phenotype"
    if etype_lower in ["pathway"]:
        return "pathway"
    if etype_lower in ["anatomy"]:
        return "anatomy"
    if etype_lower in ["metabolite"]:
        return "metabolite"
    # Default: return the original type if no specific mapping found
    logging.debug(f"No specific mapping for type '{existing_type}', keeping original.")
    return "Unknown"

def merge_field(existing_value, new_value, delimiter=MERGE_DELIMITER):
    """Merges two string fields, combining unique non-empty parts."""
    existing_str = existing_value if existing_value else ""
    new_str = new_value if new_value else ""

    existing_parts = set(p.strip() for p in existing_str.split(delimiter) if p.strip())
    new_parts = set(p.strip() for p in new_str.split(delimiter) if p.strip())

    all_parts = sorted(list(existing_parts.union(new_parts)))
    return delimiter.join(all_parts)

def process_node_files(node_files):
    """
    Reads multiple node CSV files (assuming comma-separated with header)
    and merges entries based on ID.
    """
    merged_nodes = {}
    nodes_skipped_missing_data = 0
    nodes_deduplicated = 0
    unique_node_types = set()
    unique_replaced_types = set()
    geneNameToIDMapping = {}
    for file_path in node_files:
        # Initialize statistics for this file
        source_statistics['nodes'][file_path] = {'original': 0, 'merged': 0, 'by_type': {}}
        logging.info(f"Processing node file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as infile:
                delimiter = ','

                reader = csv.DictReader(infile, delimiter=delimiter)

                if not reader.fieldnames:
                    logging.warning(f"File {file_path} appears to be empty or could not read header. Skipping.")
                    continue

                actual_header = reader.fieldnames
                if not all(field in actual_header for field in FIELDNAMES_NODE):
                    logging.warning(f"File {file_path} header {actual_header} doesn't contain all expected fields {FIELDNAMES_NODE}. Processing based on available fields.")

                line_num = 1
                for row in reader:
                    line_num += 1
                    source_statistics['nodes'][file_path]['original'] += 1
                    try:
                        node_id = row.get("ID")
                        node_name = row.get("Name")
                        node_type = row.get("Type")
                        node_seq = row.get("Sequence", "")
                        node_comments = row.get("Comments", "")
                        unique_node_types.add(node_type)
                        
                        # Record original node type for statistics
                        if node_type:
                            unified_type_for_stats = replace_data_type(node_type)
                            if unified_type_for_stats and unified_type_for_stats.lower() != 'unknown':
                                if unified_type_for_stats not in source_statistics['nodes'][file_path]['by_type']:
                                    source_statistics['nodes'][file_path]['by_type'][unified_type_for_stats] = {'original': 0, 'merged': 0}
                                source_statistics['nodes'][file_path]['by_type'][unified_type_for_stats]['original'] += 1

                        if not node_id:
                            logging.warning(f"Skipping row {line_num} in {file_path}: Missing 'ID'. Row: {row}")
                            nodes_skipped_missing_data += 1
                            continue
                        if not node_name:
                            logging.warning(f"Row {line_num} in {file_path} (ID: {node_id}): Missing 'Name'. Using empty string.")
                            node_name = ""

                        current_node_data = {
                            "ID": node_id,
                            "Name": node_name,
                            "Type": replace_data_type(node_type),  # Replace data type
                            "Sequence": node_seq if node_seq else "",
                            "Comments": node_comments if node_comments else ""
                        }
                        # Update the geneNameToIDMapping
                        if node_id.startswith(NCBI_GENE_PREFIX):
                            id_part_after_prefix = node_id[len(NCBI_GENE_PREFIX):]
                            if id_part_after_prefix.isdigit():
                                # This is a valid "NCBIGene:NUMERIC_ID"
                                if node_name in geneNameToIDMapping:
                                    if geneNameToIDMapping[node_name] != node_id:
                                        logging.warning(
                                                f"Duplicate gene name '{node_name}' encountered during node processing. "
                                                f"Existing mapped ID: '{geneNameToIDMapping[node_name]}', "
                                                f"New ID from node {node_id}: '{node_id}'. "
                                                f"Keeping the first encountered mapping for this name from node files."
                                        )
                                else:
                                    geneNameToIDMapping[node_name] = node_id
                                    logging.debug(f"Mapped gene name '{node_name}' to ID '{node_id}' from node file.")
                            else:
                                logging.debug(f"Gene node ID '{node_id}' for name '{node_name}' does not have a numeric part after prefix. Not adding to geneNameToIDMapping from this node.")
                        else:
                            logging.debug(f"Gene node ID '{node_id}' for name '{node_name}' does not start with '{NCBI_GENE_PREFIX}'. Not adding to geneNameToIDMapping from this node.")

                        unique_replaced_types.add(current_node_data["Type"])
                        if current_node_data["Type"].lower() == 'unknown':
                            logging.debug(f"Skipping node ID {node_id} from {file_path}: Replaced type is 'unknown'.")
                            nodes_skipped_missing_data += 1
                            continue
                        if node_id not in merged_nodes:
                            merged_nodes[node_id] = current_node_data
                            source_statistics['nodes'][file_path]['merged'] += 1
                            # Track merged node type for this file
                            node_unified_type = current_node_data["Type"]
                            if node_unified_type not in source_statistics['nodes'][file_path]['by_type']:
                                source_statistics['nodes'][file_path]['by_type'][node_unified_type] = {'original': 0, 'merged': 0}
                            source_statistics['nodes'][file_path]['by_type'][node_unified_type]['merged'] += 1
                        else:
                            nodes_deduplicated += 1
                            existing_node = merged_nodes[node_id]
                            is_identical = (existing_node["Name"] == current_node_data["Name"] and
                                            (existing_node["Sequence"] or "") == (current_node_data["Sequence"] or "") and
                                            (existing_node["Comments"] or "") == (current_node_data["Comments"] or ""))

                            if not is_identical:
                                if existing_node["Name"] == current_node_data["Name"]:
                                    merged_seq = merge_field(existing_node["Sequence"], current_node_data["Sequence"])
                                    merged_comments = merge_field(existing_node["Comments"], current_node_data["Comments"])
                                    merged_nodes[node_id]["Sequence"] = merged_seq
                                    merged_nodes[node_id]["Comments"] = merged_comments
                                    logging.debug(f"Merged node ID {node_id} from {file_path}.")
                                else:
                                    logging.warning(f"Node ID {node_id} has conflicting names: '{existing_node['Name']}' vs '{current_node_data['Name']}' in {file_path}. Keeping the first encountered name.")
                    except Exception as e:
                        logging.error(f"Error processing row {line_num} in {file_path}: {e} - Row data: {row}")

        except FileNotFoundError:
            logging.error(f"Node file not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to read or process node file {file_path}: {e}")

    logging.info(f"Finished processing node files. Found {len(merged_nodes)} unique node IDs. Skipped {nodes_skipped_missing_data} rows due to missing data. Deduplicated {nodes_deduplicated} nodes, unique node types: {unique_node_types}, unique replaced data types: {unique_replaced_types}.")
    return merged_nodes, geneNameToIDMapping

def load_marker_edges(marker_edge_file, geneNameToIDMapping, nodes_data):
    """
    Loads and processes the marker edge file.
    Returns a dictionary of marker edges and a set of canonical edge types
    based on the types of the connected nodes.
    """
    if not marker_edge_file or not os.path.isfile(marker_edge_file):
        logging.info("Marker edge file not provided or not found. Skipping.")
        return {}, set()

    marker_edges = {}
    inferred_marker_edge_types = set()
    source_statistics['marker_edges']['by_type'] = {}
    logging.info(f"Processing marker edge file: {marker_edge_file}")
    try:
        with open(marker_edge_file, 'r', encoding='utf-8', newline='') as infile:
            reader = csv.DictReader(infile, delimiter=',')
            if not reader.fieldnames:
                logging.warning(f"Marker edge file {marker_edge_file} is empty or has no header. Skipping.")
                return {}, set()

            for row in reader:
                source_statistics['marker_edges']['original'] += 1
                node1_id_raw = row.get("Node 1 ID")
                node2_id_raw = row.get("Node 2 ID")

                if not node1_id_raw or not node2_id_raw:
                    logging.warning(f"Skipping row in marker edge file due to missing Node ID: {row}")
                    continue

                node1_id = translateNCBIGeneIDs(node1_id_raw, geneNameToIDMapping)
                node2_id = translateNCBIGeneIDs(node2_id_raw, geneNameToIDMapping)
                
                # Infer edge type from node types
                node1_type = nodes_data.get(node1_id, {}).get('Type') or replace_data_type(row.get("Node 1 Data Type"))
                node2_type = nodes_data.get(node2_id, {}).get('Type') or replace_data_type(row.get("Node 2 Data Type"))

                if not node1_type or not node2_type or node1_type == 'Unknown' or node2_type == 'Unknown':
                    logging.warning(f"Could not determine types for marker edge ({node1_id}, {node2_id}). Skipping for type inference. Types found: ({node1_type}, {node2_type})")
                else:
                    # Create a canonical edge type key by sorting the node types
                    canonical_edge_type = tuple(sorted((node1_type, node2_type)))
                    inferred_marker_edge_types.add(canonical_edge_type)

                edge_key = (node1_id, node2_id)
                if edge_key not in marker_edges:
                    marker_edges[edge_key] = row
                    source_statistics['marker_edges']['merged'] += 1
                    
                    # Record statistics for this marker edge
                    if node1_type and node2_type and node1_type != 'Unknown' and node2_type != 'Unknown':
                        canonical_edge_type = tuple(sorted((node1_type, node2_type)))
                        if canonical_edge_type not in source_statistics['marker_edges']['by_type']:
                            source_statistics['marker_edges']['by_type'][canonical_edge_type] = {'original': 0, 'merged': 0}
                        source_statistics['marker_edges']['by_type'][canonical_edge_type]['original'] += 1
                        source_statistics['marker_edges']['by_type'][canonical_edge_type]['merged'] += 1

    except Exception as e:
        logging.error(f"Failed to read or process marker edge file {marker_edge_file}: {e}")

    logging.info(f"Loaded {len(marker_edges)} marker edges and inferred {len(inferred_marker_edge_types)} unique marker edge type pairs.")
    return marker_edges, inferred_marker_edge_types


def translateNCBIGeneIDs(originalID, geneNameToIDMapping):
    """
    Translates an NCBI Gene ID or gene name to its numeric NCBI Gene ID.
    - If originalID is "NCBIGene:123", returns "NCBIGene:123".
    - If originalID is "NCBIGene:GENENAME" (where GENENAME is not numeric):
        - Tries to find GENENAME in geneNameToIDMapping. If mapped to "NCBIGene:123", returns "NCBIGene:123".
        - Else, report an error here.
    """
    if not isinstance(originalID, str):
        logging.warning(f"translateNCBIGeneIDs received non-string input: {originalID}. Returning as is.")
        return originalID

    if originalID.startswith(NCBI_GENE_PREFIX):
        id_part = originalID[len(NCBI_GENE_PREFIX):] # Get the part after "NCBIGene:"

        if not id_part: # Handle cases like "NCBIGene:"
            logging.error(f"Malformed NCBI Gene ID '{originalID}' (empty after prefix). Returning original ID.")
            return originalID

        if id_part.isdigit():
            # Docstring says: "If originalID is "NCBIGene:123", returns "NCBIGene:123"."
            return originalID
        else:
            # id_part is GENENAME
            gene_name_to_lookup = id_part
            if gene_name_to_lookup in geneNameToIDMapping:
                mapped_id = geneNameToIDMapping[gene_name_to_lookup] # e.g., "NCBIGene:123"
                
                # Validate if mapped_id is in the expected format "NCBIGene:NUMERIC"
                if isinstance(mapped_id, str) and mapped_id.startswith(NCBI_GENE_PREFIX):
                    numeric_part_of_mapped_id = mapped_id[len(NCBI_GENE_PREFIX):]
                    if numeric_part_of_mapped_id.isdigit():
                        logging.debug(f"Translated '{originalID}' (name part: '{gene_name_to_lookup}') to '{mapped_id}' via geneNameToIDMapping.")
                        global translated_gene_names_count
                        translated_gene_names_count += 1
                        return mapped_id # Return the full mapped ID "NCBIGene:123"
                    else:
                        logging.warning(f"Gene name '{gene_name_to_lookup}' (from '{originalID}') mapped to a non-numeric NCBI ID '{mapped_id}' in geneNameToIDMapping. Returning original ID.")
                        return originalID
                else:
                    logging.error(f"Gene name '{gene_name_to_lookup}' (from '{originalID}') mapped to an invalid format '{mapped_id}' in geneNameToIDMapping. Expected 'NCBIGene:NUMERIC_ID' format. Returning original ID.")
                    return originalID
            else:
                logging.error(f"Gene name '{gene_name_to_lookup}' (derived from '{originalID}') not found in geneNameToIDMapping. Cannot translate. Returning original ID.")
                return originalID
    else:
        # If originalID does not start with NCBI_GENE_PREFIX.
        # The docstring examples all start with the prefix.
        # Returning originalID as a fallback.
        logging.debug(f"ID '{originalID}' does not start with '{NCBI_GENE_PREFIX}'. Returning original ID.")
        return originalID


def process_edge_files(edge_files, merged_nodes_data, geneNameToIDMapping, marker_edges, inferred_marker_edge_types):
    """
    Reads multiple edge CSV files (assuming comma-separated with header)
    and keeps unique edges based on (Node 1 ID, Node 2 ID, Edge Direction).
    Skips edges where either original node type is 'unknown'.
    Replaces node types using replace_data_type.
    Updates edge's node types based on merged_nodes_data if available.
    """
    unique_edges = []
    seen_edge_keys = set()
    skipped_unknown_count = 0
    edges_skipped_missing_data = 0
    edges_skipped_missing_node = 0
    edges_deduplicated = 0
    edges_replaced_data_type = 0
    skipped_due_to_marker = 0
    modified_weight_counts = defaultdict(int)
    edge_type_sources = defaultdict(set)

    # Add marker edges to seen_edge_keys first to give them priority
    for (n1, n2), edge_data in marker_edges.items():
        edge_key = (n1, n2)
        if edge_key not in seen_edge_keys:
            # We assume marker edges are pre-validated and don't need all checks here
            # but we should ensure their node types are also updated.
            node1_id = edge_data.get("Node 1 ID")
            node2_id = edge_data.get("Node 2 ID")
            
            if node1_id in merged_nodes_data:
                edge_data["Node 1 Data Type"] = merged_nodes_data[node1_id]['Type']
            if node2_id in merged_nodes_data:
                edge_data["Node 2 Data Type"] = merged_nodes_data[node2_id]['Type']

            unique_edges.append(edge_data)
            seen_edge_keys.add(edge_key)


    for file_path in edge_files:
        # Initialize statistics for this edge file
        source_statistics['edges'][file_path] = {'original': 0, 'merged': 0, 'by_type': {}}
        logging.info(f"Processing edge file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as infile:
                delimiter = ','
                reader = csv.DictReader(infile, delimiter=delimiter)

                if not reader.fieldnames:
                    logging.warning(f"File {file_path} appears to be empty or could not read header. Skipping.")
                    continue

                actual_header = reader.fieldnames
                if not all(field in actual_header for field in FIELDNAMES_EDGE):
                    logging.warning(f"File {file_path} header {actual_header} doesn't contain all expected fields {FIELDNAMES_EDGE}. Processing based on available fields.")

                line_num = 1
                for row in reader:
                    line_num += 1
                    source_statistics['edges'][file_path]['original'] += 1
                    try:
                        node1_id_raw = row.get("Node 1 ID")
                        node2_id_raw = row.get("Node 2 ID")
                        direction = row.get("Edge Direction")
                        original_node1_type_from_edge = row.get("Node 1 Data Type") # Get original types from edge file
                        original_node2_type_from_edge = row.get("Node 2 Data Type")
                        edge_type = row.get("Edge Type")
                        edge_weight_raw = row.get("Edge Weight")

                        if not node1_id_raw or not node2_id_raw:
                            logging.warning(f"Skipping row {line_num} in {file_path}: Missing key fields (Node 1 ID, Node 2 ID, or Edge Direction). Row: {row}")
                            edges_skipped_missing_data += 1
                            continue
                        if not original_node1_type_from_edge or not original_node2_type_from_edge:
                            logging.warning(f"Skipping row {line_num} in {file_path} (Nodes: {node1_id_raw}, {node2_id_raw}): Missing 'Node 1 Data Type' or 'Node 2 Data Type'. Row: {row}")
                            edges_skipped_missing_data += 1
                            continue
                        
                        # Record original edge type for statistics (before any filtering)
                        # Use replaced types to ensure consistency with final merged data
                        replaced_node1_type_from_edge = replace_data_type(original_node1_type_from_edge)
                        replaced_node2_type_from_edge = replace_data_type(original_node2_type_from_edge)
                        
                        # Only record if types are valid (not unknown)
                        if replaced_node1_type_from_edge.lower() != 'unknown' and replaced_node2_type_from_edge.lower() != 'unknown':
                            inferred_type_from_row = tuple(sorted((replaced_node1_type_from_edge, replaced_node2_type_from_edge)))
                            if inferred_type_from_row not in source_statistics['edges'][file_path]['by_type']:
                                source_statistics['edges'][file_path]['by_type'][inferred_type_from_row] = {'original': 0, 'merged': 0}
                            source_statistics['edges'][file_path]['by_type'][inferred_type_from_row]['original'] += 1
                        
                        # Clean IDs (consistent with how virtual nodes might be keyed if spaces were an issue)
                        node1_id = translateNCBIGeneIDs(node1_id_raw, geneNameToIDMapping) # Assuming IDs in merged_nodes_data and edge files are consistent after node processing
                        node2_id = translateNCBIGeneIDs(node2_id_raw, geneNameToIDMapping) # Or apply .replace(" ", "_") if necessary for consistency

                        # --- Check if edge exists in marker_edges ---
                        if (node1_id, node2_id) in marker_edges or (node2_id, node1_id) in marker_edges:
                            logging.debug(f"Skipping edge ({node1_id}, {node2_id}) because it exists in the marker edge file.")
                            skipped_due_to_marker += 1
                            continue

                        # --- Replace types based on edge file's information ---
                        # (Already done above for statistics)
                        # --- End Replace types ---

                        # --- Filtering based on replaced types from edge file ---
                        if replaced_node1_type_from_edge.lower() == 'unknown' or replaced_node2_type_from_edge.lower() == 'unknown':
                            logging.debug(f"Skipping edge ({node1_id}, {node2_id}) from {file_path}: Replaced type from edge file is 'unknown' for one or both nodes ('{replaced_node1_type_from_edge}', '{replaced_node2_type_from_edge}').")
                            skipped_unknown_count += 1
                            continue
                        # --- End Filtering ---

                        edge_key_1 = (node1_id, node2_id)
                        edge_key_2 = (node2_id, node1_id)

                        if edge_key_1 not in seen_edge_keys and edge_key_2 not in seen_edge_keys:
                            seen_edge_keys.add(edge_key_1)

                            edge_data = {field: row.get(field, "") for field in FIELDNAMES_EDGE}
                            
                            # Initially set node IDs and types in edge_data from the current edge row
                            edge_data["Node 1 ID"] = node1_id
                            edge_data["Node 2 ID"] = node2_id
                            edge_data["Node 1 Data Type"] = replaced_node1_type_from_edge
                            edge_data["Node 2 Data Type"] = replaced_node2_type_from_edge

                            # --- Update node types in edge_data using merged_nodes_data if available ---
                            authoritative_node1_type = merged_nodes_data.get(node1_id, {}).get('Type', replaced_node1_type_from_edge)
                            authoritative_node2_type = merged_nodes_data.get(node2_id, {}).get('Type', replaced_node2_type_from_edge)
                            
                            if node1_id not in merged_nodes_data:
                                edges_skipped_missing_node += 1
                                continue # Skip those without nodes
                            edge_data["Node 1 Data Type"] = authoritative_node1_type

                            if node2_id not in merged_nodes_data:
                                edges_skipped_missing_node += 1
                                continue # Skip those without nodes
                            edge_data["Node 2 Data Type"] = authoritative_node2_type
                            # --- End Update node types ---

                            # Adjust weight if the inferred edge type is in marker_edge_types
                            inferred_edge_type = tuple(sorted((authoritative_node1_type, authoritative_node2_type)))
                            if inferred_edge_type in inferred_marker_edge_types:
                                try:
                                    original_weight = float(edge_weight_raw)
                                    adjusted_weight = original_weight * 0.8
                                    edge_data["Edge Weight"] = str(adjusted_weight)
                                    logging.debug(f"Adjusted weight for edge ({node1_id}, {node2_id}) with inferred type '{inferred_edge_type}' to {adjusted_weight}")
                                    modified_weight_counts[inferred_edge_type] += 1
                                except (ValueError, TypeError):
                                    logging.warning(f"Could not convert weight '{edge_weight_raw}' to float for edge ({node1_id}, {node2_id}). Using original value.")

                            edge_type_sources[inferred_edge_type].add(file_path)
                            unique_edges.append(edge_data)
                            source_statistics['edges'][file_path]['merged'] += 1
                            # Track edge types for this file
                            if inferred_edge_type not in source_statistics['edges'][file_path]['by_type']:
                                source_statistics['edges'][file_path]['by_type'][inferred_edge_type] = {'original': 0, 'merged': 0}
                            source_statistics['edges'][file_path]['by_type'][inferred_edge_type]['merged'] += 1
                        else:
                            logging.debug(f"Duplicate edge ({node1_id}, {node2_id}) found in {file_path}, skipping.")
                            edges_deduplicated += 1
                    except Exception as e:
                        logging.error(f"Error processing row {line_num} in {file_path}: {e} - Row data: {row}")

        except FileNotFoundError:
            logging.error(f"Edge file not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to read or process edge file {file_path}: {e}")

    if skipped_unknown_count > 0:
        logging.info(f"Skipped {skipped_unknown_count} edges because one or both original node types (from edge file, after replacement) were 'unknown'.")
    logging.info(f"Finished processing edge files. Found {len(unique_edges)} unique, valid edges. Skipped {edges_skipped_missing_data} rows due to missing data and {edges_skipped_missing_node} rows due to missing nodes. Deduplicated {edges_deduplicated} edges, replaced {edges_replaced_data_type} node types based on merged nodes data.")
    logging.info(f"Skipped {skipped_due_to_marker} edges because they were already present in the marker edge file.")
    if modified_weight_counts:
        logging.info("Counts of edges with modified weights by type:")
        for edge_type, count in modified_weight_counts.items():
            logging.info(f"  - Type {edge_type}: {count} edges")
    if edge_type_sources:
        logging.info("Source files for each edge type:")
        for edge_type, files in sorted(edge_type_sources.items()):
            logging.info(f"  - Inferred Edge Type '{edge_type}': {', '.join(sorted(list(files)))}")
    return unique_edges

def write_output_csv(data, output_path, fieldnames):
    """Writes data to a CSV file using comma delimiter."""
    logging.info(f"Writing {len(data)} rows to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter=',')
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Successfully wrote output to {output_path}")
    except Exception as e:
        logging.error(f"Failed to write output file {output_path}: {e}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Merge node and edge data from multiple CSV files.")
    parser.add_argument('-n', '--node-files', nargs='+', required=True, help="Paths to input node CSV files (can be space-separated paths or glob patterns within quotes).")
    parser.add_argument('-e', '--edge-files', nargs='+', required=True, help="Paths to input edge CSV files (can be space-separated paths or glob patterns within quotes).")
    parser.add_argument('--marker-edge-file', help="Path to the marker edge CSV file for prioritized processing.")
    parser.add_argument('--out-nodes', required=True, help="Path for the merged output node CSV file.")
    parser.add_argument('--out-edges', required=True, help="Path for the unique output edge CSV file.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    def expand_file_arguments(file_arg_list):
        expanded_files = []
        if not isinstance(file_arg_list, list): # Should be a list due to nargs='+'
            if isinstance(file_arg_list, str): # Fallback if somehow it's a single string
                file_arg_list = [file_arg_list]
            else:
                logging.error(f"Unexpected type for file arguments: {type(file_arg_list)}. Expected list.")
                return []

        for item in file_arg_list:
            print(f"Expanding file argument: {item}")
            # Item could be a glob pattern, a single file, or space-separated files
            # Try glob first for items containing wildcards
            if '*' in item or '?' in item or '[' in item:
                glob_expanded = glob.glob(item)
                if glob_expanded:
                    expanded_files.extend(glob_expanded)
                    logging.debug(f"Glob pattern '{item}' expanded to: {glob_expanded}")
                    continue 
                # If glob didn't expand (e.g. literal name with wildcard char, or no matches)
                # and it contains spaces, it might be a list of files
                
            # If not a glob or glob yielded no results, check for space-separated paths
            # This handles the case where a user might pass "file1.csv file2.csv" as a single argument string
            potential_paths = item.split()
            if len(potential_paths) > 1:
                # Check if these look like actual files or if it was a single filename with spaces
                # This is a heuristic. A filename can legitimately contain spaces.
                # A more robust way is to expect user to provide multiple arguments or use shell globbing.
                # For now, we assume if spaces are present in a single arg item, it's multiple paths.
                are_all_files = True
                temp_expanded_from_split = []
                for p_path in potential_paths:
                    # Further glob expansion for parts if they are patterns
                    if '*' in p_path or '?' in p_path or '[' in p_path:
                        glob_expanded_part = glob.glob(p_path)
                        if glob_expanded_part:
                            temp_expanded_from_split.extend(glob_expanded_part)
                        else: # Glob part didn't expand, treat as literal
                            temp_expanded_from_split.append(p_path)
                    else:
                        temp_expanded_from_split.append(p_path)
                
                expanded_files.extend(temp_expanded_from_split)
                logging.debug(f"Argument '{item}' (split and potentially globbed) expanded to: {temp_expanded_from_split}")

            else: # Single item, no wildcards recognized by glob, or glob returned empty, and no spaces
                expanded_files.append(item)
                logging.debug(f"Argument '{item}' treated as a single path.")
        
        # Deduplicate and ensure files exist (optional, as called functions do this)
        final_files = []
        for f_path in expanded_files:
            if not os.path.isfile(f_path):
                logging.warning(f"File path '{f_path}' from arguments does not exist or is not a file. It will be skipped by processing functions.")
            # Still add it, let processing functions handle non-existence if they re-check
            if f_path not in final_files: # Basic deduplication
                 final_files.append(f_path)
        return final_files

    actual_node_files = expand_file_arguments(args.node_files)
    actual_edge_files = expand_file_arguments(args.edge_files)

    if not actual_node_files:
        logging.error("No valid node files found after expanding arguments. Exiting.")
        return
    if not actual_edge_files:
        logging.error("No valid edge files found after expanding arguments. Exiting.")
        return
    
    logging.info(f"Effective node files to process: {actual_node_files}")
    logging.info(f"Effective edge files to process: {actual_edge_files}")

    nodes_data, geneNameToIDMapping = process_node_files(actual_node_files)
    logging.info(f"Processed gene name to ID map size: {len(geneNameToIDMapping)}")
    
    # Load marker edges
    marker_edges, inferred_marker_edge_types = load_marker_edges(args.marker_edge_file, geneNameToIDMapping, nodes_data)

    # Process edges, potentially creating virtual nodes
    # The types in unique_edges_data will now be updated based on nodes_data
    unique_edges_data = process_edge_files(actual_edge_files, nodes_data, geneNameToIDMapping, marker_edges, inferred_marker_edge_types)
    write_output_csv(unique_edges_data, args.out_edges, FIELDNAMES_EDGE)
    nodes_to_write = list(nodes_data.values())
    write_output_csv(nodes_to_write, args.out_nodes, FIELDNAMES_NODE)
    logging.info(f"Data merging process completed, unique nodes num: {len(nodes_to_write)} and unique edges num: {len(unique_edges_data)}.")
    global translated_gene_names_count
    logging.info(f"Translated {translated_gene_names_count} gene name nodes to gene id nodes in edges.")
    
    # Print detailed statistics to both logger and stdout
    separator_line = "=" * 80
    section_line = "-" * 80
    
    stats_output = []
    stats_output.append("\n" + separator_line)
    stats_output.append("数据合并统计报告 / Data Merging Statistics Report")
    stats_output.append(separator_line)
    
    # Node statistics
    stats_output.append("\n【节点统计 / Node Statistics】")
    stats_output.append(section_line)
    total_original_nodes = 0
    total_merged_nodes = 0
    all_node_types = {}
    
    if source_statistics['nodes']:
        for file_path, stats in sorted(source_statistics['nodes'].items()):
            file_name = os.path.basename(file_path)
            stats_output.append(f"\n  数据源 / Source: {file_name}")
            stats_output.append(f"    原始节点数 / Original nodes: {stats['original']:,}")
            stats_output.append(f"    合并节点数 / Merged nodes: {stats['merged']:,}")
            merge_rate = stats['merged']/stats['original']*100 if stats['original'] > 0 else 0
            stats_output.append(f"    合并率 / Merge rate: {merge_rate:.2f}%")
            
            # Show node type distribution for this source
            if stats.get('by_type'):
                stats_output.append(f"    按节点类型分类 / By node type:")
                sorted_node_types = sorted(stats['by_type'].items(), key=lambda x: x[1]['merged'], reverse=True)
                for node_type, counts in sorted_node_types:
                    original = counts['original']
                    merged = counts['merged']
                    rate = merged / original * 100 if original > 0 else 0
                    stats_output.append(f"      {node_type}: {merged:,}/{original:,} ({rate:.2f}%)")
            
            total_original_nodes += stats['original']
            total_merged_nodes += stats['merged']
            
            # Aggregate all node types
            for node_type, counts in stats.get('by_type', {}).items():
                all_node_types[node_type] = all_node_types.get(node_type, 0) + counts['merged']
        
        stats_output.append(f"\n  总计 / Total:")
        stats_output.append(f"    原始节点总数 / Total original nodes: {total_original_nodes:,}")
        stats_output.append(f"    最终节点总数 / Total merged nodes: {len(nodes_to_write):,}")
        dedup_rate = (1 - len(nodes_to_write)/total_original_nodes)*100 if total_original_nodes > 0 else 0
        stats_output.append(f"    去重率 / Deduplication rate: {dedup_rate:.2f}%")
        
        # Overall node type distribution
        if all_node_types:
            stats_output.append("\n  最终数据中的节点类型分布 / Final node type distribution:")
            for node_type, count in sorted(all_node_types.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(nodes_to_write) * 100 if len(nodes_to_write) > 0 else 0
                stats_output.append(f"    {node_type}: {count:,} ({percentage:.2f}%)")
    else:
        stats_output.append("  No node statistics available.")
    
    logging.info(f"Node statistics: {len(source_statistics['nodes'])} sources processed, {total_original_nodes:,} original nodes, {len(nodes_to_write):,} final nodes")
    logging.info(f"Node statistics: {len(source_statistics['nodes'])} sources processed, {total_original_nodes:,} original nodes, {len(nodes_to_write):,} final nodes")
    
    # Marker edge statistics
    if source_statistics['marker_edges']['original'] > 0:
        stats_output.append("\n【Marker边统计 / Marker Edge Statistics】")
        stats_output.append(section_line)
        stats_output.append(f"  原始边数 / Original edges: {source_statistics['marker_edges']['original']:,}")
        stats_output.append(f"  合并边数 / Merged edges: {source_statistics['marker_edges']['merged']:,}")
        marker_merge_rate = source_statistics['marker_edges']['merged']/source_statistics['marker_edges']['original']*100 if source_statistics['marker_edges']['original'] > 0 else 0
        stats_output.append(f"  合并率 / Merge rate: {marker_merge_rate:.2f}%")
        
        if source_statistics['marker_edges']['by_type']:
            stats_output.append(f"\n  按节点类型分类 / By node type:")
            for edge_type, counts in sorted(source_statistics['marker_edges']['by_type'].items(), key=lambda x: x[1]['merged'], reverse=True):
                original = counts['original']
                merged = counts['merged']
                rate = merged / original * 100 if original > 0 else 0
                stats_output.append(f"    {edge_type[0]} <-> {edge_type[1]}: {merged:,}/{original:,} ({rate:.2f}%)")
        
        logging.info(f"Marker edge statistics: {source_statistics['marker_edges']['original']:,} original, {source_statistics['marker_edges']['merged']:,} merged")
    
    # Edge statistics
    stats_output.append("\n【边统计 / Edge Statistics】")
    stats_output.append(section_line)
    total_original_edges = 0
    total_merged_edges = 0
    all_edge_types = {}
    
    if source_statistics['edges']:
        for file_path, stats in sorted(source_statistics['edges'].items()):
            file_name = os.path.basename(file_path)
            stats_output.append(f"\n  数据源 / Source: {file_name}")
            stats_output.append(f"    原始边数 / Original edges: {stats['original']:,}")
            stats_output.append(f"    合并边数 / Merged edges: {stats['merged']:,}")
            edge_merge_rate = stats['merged']/stats['original']*100 if stats['original'] > 0 else 0
            stats_output.append(f"    合并率 / Merge rate: {edge_merge_rate:.2f}%")
            
            if stats['by_type']:
                stats_output.append(f"    按节点类型分类 / By node type:")
                sorted_types = sorted(stats['by_type'].items(), key=lambda x: x[1]['merged'], reverse=True)
                for edge_type, counts in sorted_types:
                    original = counts['original']
                    merged = counts['merged']
                    rate = merged / original * 100 if original > 0 else 0
                    stats_output.append(f"      {edge_type[0]} <-> {edge_type[1]}: {merged:,}/{original:,} ({rate:.2f}%)")
            
            total_original_edges += stats['original']
            total_merged_edges += stats['merged']
            
            # Aggregate all edge types
            for edge_type, counts in stats['by_type'].items():
                if edge_type not in all_edge_types:
                    all_edge_types[edge_type] = 0
                all_edge_types[edge_type] += counts['merged']
        
        # Add marker edge types to total
        for edge_type, counts in source_statistics['marker_edges']['by_type'].items():
            if edge_type not in all_edge_types:
                all_edge_types[edge_type] = 0
            all_edge_types[edge_type] += counts['merged']
        
        stats_output.append(f"\n  总计 / Total:")
        stats_output.append(f"    原始边总数 / Total original edges: {total_original_edges:,}")
        stats_output.append(f"    最终边总数 / Total merged edges: {len(unique_edges_data):,}")
        edge_dedup_rate = (1 - len(unique_edges_data)/total_original_edges)*100 if total_original_edges > 0 else 0
        stats_output.append(f"    去重率 / Deduplication rate: {edge_dedup_rate:.2f}%")
        
        # Overall edge type distribution
        stats_output.append("\n  最终数据中的边类型分布 / Final edge type distribution:")
        for edge_type, count in sorted(all_edge_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(unique_edges_data) * 100 if len(unique_edges_data) > 0 else 0
            stats_output.append(f"    {edge_type[0]} <-> {edge_type[1]}: {count:,} ({percentage:.2f}%)")
        
        logging.info(f"Edge statistics: {len(source_statistics['edges'])} sources processed, {total_original_edges:,} original edges, {len(unique_edges_data):,} final edges")
    else:
        stats_output.append("  No edge statistics available.")
    
    stats_output.append("\n" + separator_line)
    stats_output.append("统计报告完成 / Statistics Report Completed")
    stats_output.append(separator_line + "\n")
    
    # Print all statistics at once
    for line in stats_output:
        print(line)

if __name__ == "__main__":
    main()
