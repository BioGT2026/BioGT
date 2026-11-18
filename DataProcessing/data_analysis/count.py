import sys
import os
import re
import csv
import glob # Import glob module
from collections import defaultdict # Import defaultdict for easier aggregation
csv.field_size_limit(1000 * 1024 * 1024)

def count_node_types(file_path):
    """
    Count the number of nodes in a given file based on the 'Type' column.
    :param file_path: Path to the node CSV file.
    :return: A dictionary with node types as keys and their counts as values, or None on error.
    """
    node_types = {}
    print(f"--- Processing node file: {file_path} ---") # Indicate which file is being processed

    # Check if the file exists (redundant if called after glob, but good practice)
    if not os.path.isfile(file_path):
        print(f"Error: Node file {file_path} does not exist.")
        return None # Return None to indicate failure

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file: # Added encoding
            # Handle potential BOM (Byte Order Mark) if present
            reader = csv.DictReader(file)
            if 'Type' not in reader.fieldnames:
                print(f"Error: Column 'Type' not found in header of {file_path}. Header: {reader.fieldnames}. Counting rows as 'default' nodes.")
                # Count rows if 'Type' column is missing
                row_count = 0
                for row in reader:
                    row_count += 1
                if row_count > 0:
                    node_types["default"] = row_count
                    print(f"Node type: default, Count: {row_count}")
                else:
                    print("File is empty or header-only.")

            else:
                for row in reader:
                    # Handle potential missing 'Type' value in a row
                    node_type = row.get('Type', '').strip().lower() # Use .get() for safety and strip whitespace
                    if not node_type: # Treat empty type names as 'unknown' or skip? Let's use 'unknown'
                        node_type = 'unknown'
                    node_types[node_type] = node_types.get(node_type, 0) + 1

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None # Return None on error

    # No need to print "No node types found" here, main will handle summary
    return node_types

def count_edge_types(file_path):
    """
    Count the number of edges in a given file based on node types.
    :param file_path: Path to the edge CSV file.
    :return: A dictionary with edge types (src,dst) as keys and counts as values, or None on error.
    """
    edge_types = {}
    print(f"--- Processing edge file: {file_path} ---") # Indicate which file is being processed

    # Check if the file exists (redundant if called after glob, but good practice)
    if not os.path.isfile(file_path):
        print(f"Error: Edge file {file_path} does not exist.")
        return None # Return None to indicate failure

    required_cols = ['Node 1 Data Type', 'Node 2 Data Type', 'Edge Direction']

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file: # Added encoding
            reader = csv.DictReader(file)

            # Check for required columns
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: Missing required columns in {file_path}: {missing_cols}. Header: {reader.fieldnames}")
                return None

            for row in reader:
                # Use .get() for safety and strip whitespace
                src_type = row.get('Node 1 Data Type', '').strip().lower()
                dst_type = row.get('Node 2 Data Type', '').strip().lower()

                # Normalize edge key by sorting the node types to merge bidirectional edges
                # This treats (A, B) and (B, A) as the same edge type
                edge_key = tuple(sorted([src_type, dst_type]))
                edge_types[edge_key] = edge_types.get(edge_key, 0) + 1

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None # Return None on error

    # No need to print "No edge types found" here, main will handle summary
    return edge_types

def main():
    """
    Main function to find and count node/edge types for all relevant CSV files
    within a specified directory, aggregating counts across all found files.
    """
    if len(sys.argv) != 2: # Expect exactly script name + one directory path
        print("Usage: python count.py <directory-path>")
        print("  Example: python count.py ./my_data_folder")
        print("  (This will find all '*_node.csv' and '*_edge.csv' files within that folder)")
        sys.exit(1)

    target_directory = sys.argv[1] # Get the directory path

    # Validate the directory path
    if not os.path.isdir(target_directory):
        print(f"Error: Provided path '{target_directory}' is not a valid directory.")
        sys.exit(1)

    print(f"\n===== Searching for node and edge CSV files in: {target_directory} =====")

    # Find all node files matching the pattern *_node.csv within the directory
    node_files_pattern = os.path.join(target_directory, '*_node.csv')
    node_files_found = glob.glob(node_files_pattern)

    # Find all edge files matching the pattern *_edge.csv within the directory
    edge_files_pattern = os.path.join(target_directory, '*_edge.csv')
    edge_files_found = glob.glob(edge_files_pattern)

    if not node_files_found and not edge_files_found:
        print(f"No '*_node.csv' or '*_edge.csv' files found in '{target_directory}'.")
        sys.exit(0)

    # Initialize dictionaries to store total counts
    total_node_counts = defaultdict(int)
    total_edge_counts = defaultdict(int)
    processed_node_files = 0
    processed_edge_files = 0

    # Process all found node files
    print(f"\n--- Processing {len(node_files_found)} Node Files ---")
    for node_file in sorted(node_files_found): # Sort for consistent processing order
        node_counts = count_node_types(node_file)
        if node_counts is not None:
            processed_node_files += 1
            for node_type, count in node_counts.items():
                total_node_counts[node_type] += count
        else:
            print(f"Skipping aggregation for failed file: {node_file}")

    # Process all found edge files
    print(f"\n--- Processing {len(edge_files_found)} Edge Files ---")
    for edge_file in sorted(edge_files_found): # Sort for consistent processing order
        edge_counts = count_edge_types(edge_file)
        if edge_counts is not None:
            processed_edge_files += 1
            for edge_key, count in edge_counts.items():
                total_edge_counts[edge_key] += count
        else:
            print(f"Skipping aggregation for failed file: {edge_file}")

    # --- Print Final Aggregated Results ---
    print("\n" + "="*40)
    print("      TOTAL AGGREGATED RESULTS")
    print("="*40)

    if processed_node_files == 0 and processed_edge_files == 0:
         print("\nNo node or edge files were successfully processed.")
    else:
         print(f"\nAggregated results from {processed_node_files} node file(s) and {processed_edge_files} edge file(s).")

         # Print Total Node Counts
         print("\nTotal Node Counts:")
         if total_node_counts:
             total_nodes = sum(total_node_counts.values())
             print(f"  (Total Nodes: {total_nodes})")
             for node_type, count in sorted(total_node_counts.items()):
                 print(f"  {node_type}: {count}")
         else:
             print("  (No nodes found or processed successfully)")

         # Print Total Edge Counts
         print("\nTotal Edge Counts:")
         if total_edge_counts:
             total_edges = sum(total_edge_counts.values())
             print(f"  (Total Edges: {total_edges})")
             # Sort by first type, then second type
             for edge_key, count in sorted(total_edge_counts.items()):
                 print(f"  ({edge_key[0]} <-> {edge_key[1]}): {count}")
         else:
             print("  (No edges found or processed successfully)")

    print("\n" + "="*40)
    print("          PROCESSING COMPLETE")
    print("="*40)


if __name__ == "__main__":
    main()