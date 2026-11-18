import pandas as pd
import os
import sys
import logging
from pathlib import Path
import csv

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data_for_import(node_file_path, edge_file_path, output_dir):
    """
    Transforms and validates raw CSV files into a format suitable for neo4j-admin import.
    
    Args:
        node_file_path (str): Path to the merged_node.csv file.
        edge_file_path (str): Path to the merged_edge.csv file.
        output_dir (str): Directory to save the formatted files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_path.resolve()}")

    # --- 1. Process and Validate Nodes ---
    logging.info(f"Reading node file: {node_file_path}")
    try:
        node_df = pd.read_csv(node_file_path, dtype=str).fillna('')
    except FileNotFoundError:
        logging.error(f"Node file not found at {node_file_path}")
        sys.exit(1)

    # Get all unique node IDs for validation
    all_node_ids = set(node_df['ID'].unique())
    logging.info(f"Found {len(all_node_ids)} unique node IDs.")

    # --- 2. Process and Validate Edges ---
    logging.info(f"Reading edge file: {edge_file_path}")
    try:
        edge_df = pd.read_csv(edge_file_path, dtype=str).fillna('')
    except FileNotFoundError:
        logging.error(f"Edge file not found at {edge_file_path}")
        sys.exit(1)

    # --- DATA VALIDATION ---
    logging.info("Starting data validation...")
    
    initial_edge_count = len(edge_df)
    logging.info(f"Initial number of edges: {initial_edge_count}")

    # Validate that all edge IDs exist in the node list
    valid_start_ids = edge_df['Node 1 ID'].isin(all_node_ids)
    valid_end_ids = edge_df['Node 2 ID'].isin(all_node_ids)
    
    invalid_edges_df = edge_df[~(valid_start_ids & valid_end_ids)]
    if not invalid_edges_df.empty:
        logging.warning(f"Found {len(invalid_edges_df)} edges pointing to non-existent nodes. These will be skipped.")
        logging.warning(f"Examples of invalid start/end IDs: {invalid_edges_df['Node 1 ID'].iloc[0]}, {invalid_edges_df['Node 2 ID'].iloc[0]}")
        edge_df = edge_df[valid_start_ids & valid_end_ids]

    # Validate weight column and filter out rows with non-numeric weights
    edge_df['Edge Weight'] = pd.to_numeric(edge_df['Edge Weight'], errors='coerce')
    invalid_weights_mask = edge_df['Edge Weight'].isnull()
    if invalid_weights_mask.any():
        num_invalid_weights = invalid_weights_mask.sum()
        logging.warning(f"Found {num_invalid_weights} edges with non-numeric 'Edge Weight'. These will be skipped.")
        edge_df = edge_df.dropna(subset=['Edge Weight'])
        
    final_edge_count = len(edge_df)
    if final_edge_count < initial_edge_count:
        logging.info(f"Removed a total of {initial_edge_count - final_edge_count} invalid edges.")
    
    logging.info("Data validation and cleaning complete.")
    # --- END VALIDATION ---


    # --- 3. Format and Write Node Files ---
    node_df.rename(columns={
        'ID': 'ID:ID',
        'Type': ':LABEL',
        'Name': 'Name:string',
        'Sequence': 'Sequence:string',
        'Comments': 'Comments:string'
    }, inplace=True)
    
    labels = node_df[':LABEL'].unique()
    for label in labels:
        if not label:
            logging.warning("Skipping nodes with no label.")
            continue
        
        label_df = node_df[node_df[':LABEL'] == label]
        output_node_df = label_df.drop(columns=[':LABEL'])
        node_file_path = output_path / f"import-nodes-{label}.csv"
        output_node_df.to_csv(node_file_path, index=False, quoting=csv.QUOTE_ALL)
        logging.info(f"Created node file for label '{label}': {node_file_path.name}")

    # --- 4. Format and Write Edge File ---
    edge_df.rename(columns={
        'Node 1 ID': ':START_ID',
        'Node 2 ID': ':END_ID',
        'Edge Type': ':TYPE',
        'Edge Weight': 'weight:double',
        'Edge Direction': 'direction:string',
        'Node 1 Source': 'source:string'
    }, inplace=True)

    rel_df = edge_df[[':START_ID', ':END_ID', ':TYPE', 'weight:double', 'direction:string', 'source:string']]
    rel_file_path = output_path / "import-rels.csv"
    rel_df.to_csv(rel_file_path, index=False, quoting=csv.QUOTE_ALL)
    logging.info(f"Created relationship file: {rel_file_path.name}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python prepare_for_admin_import.py <path_to_nodes.csv> <path_to_edges.csv> <output_directory>")
        sys.exit(1)
    
    node_file = os.path.expanduser(sys.argv[1])
    edge_file = os.path.expanduser(sys.argv[2])
    output_dir = os.path.expanduser(sys.argv[3])
    
    prepare_data_for_import(node_file, edge_file, output_dir)
    logging.info("Data preparation complete.")

if __name__ == "__main__":
    main()
