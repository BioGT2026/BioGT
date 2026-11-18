import pandas as pd
from neo4j import GraphDatabase
import os
import sys
import logging

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Neo4j connection details - PLEASE UPDATE THESE
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "nancy981208") # <-- IMPORTANT: Change this!

# --- Main Graph Class ---
class GraphCreator:
    """
    A class to create a graph in Neo4j from CSV files.
    """
    def __init__(self, uri, user, password):
        """
        Initializes the connection to the Neo4j database.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logging.info("Successfully connected to Neo4j.")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            sys.exit(1)

    def close(self):
        """
        Closes the database connection.
        """
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed.")

    def create_constraints(self, node_labels):
        """
        Creates uniqueness constraints on the 'ID' property for each node label.
        This is crucial for performance and data integrity.
        """
        with self.driver.session() as session:
            for label in node_labels:
                try:
                    logging.info(f"Creating constraint for label: {label}")
                    session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.ID IS UNIQUE")
                except Exception as e:
                    logging.error(f"Error creating constraint for {label}: {e}")

    def import_nodes(self, node_file_path):
        """
        Imports nodes from the specified CSV file without using APOC.
        """
        logging.info(f"Starting node import from {node_file_path}...")
        try:
            # Get all unique node types to create constraints first
            all_node_types = pd.read_csv(node_file_path, usecols=['Type'])['Type'].unique()
            self.create_constraints(all_node_types)

            nodes_df = pd.read_csv(node_file_path, chunksize=5000)

            with self.driver.session() as session:
                for chunk in nodes_df:
                    chunk = chunk.where(pd.notnull(chunk), None)  # Replace NaN with None
                    # Group by node type to create nodes with the correct label
                    for node_type, group in chunk.groupby('Type'):
                        if pd.isna(node_type):
                            continue
                        node_list = group.to_dict('records')
                        
                        # Use MERGE to create nodes with dynamic labels
                        query = f"""
                        UNWIND $nodes AS node_data
                        MERGE (n:{node_type} {{ID: node_data.ID}})
                        ON CREATE SET
                            n.Name = node_data.Name,
                            n.Sequence = node_data.Sequence,
                            n.Comments = node_data.Comments
                        ON MATCH SET
                            n.Name = node_data.Name,
                            n.Sequence = node_data.Sequence,
                            n.Comments = node_data.Comments
                        """
                        session.run(query, nodes=node_list)

            logging.info("Node import completed successfully.")
        except FileNotFoundError:
            logging.error(f"Node file not found: {node_file_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred during node import: {e}")
            sys.exit(1)

    def import_edges(self, edge_file_path):
        """
        Imports relationships from the specified CSV file without using APOC.
        """
        logging.info(f"Starting edge import from {edge_file_path}...")
        try:
            edges_df = pd.read_csv(edge_file_path, chunksize=5000)
            with self.driver.session() as session:
                for chunk in edges_df:
                    chunk = chunk.where(pd.notnull(chunk), None)
                    # Group by edge type to create relationships with the correct type
                    for edge_type, group in chunk.groupby('Edge Type'):
                        if pd.isna(edge_type):
                            continue
                        
                        # Sanitize edge type for Cypher
                        sanitized_edge_type = f"`{edge_type}`"
                        edge_list = group.to_dict('records')

                        query = f"""
                        UNWIND $edges AS edge_data
                        MATCH (a {{ID: edge_data['Node 1 ID']}})
                        MATCH (b {{ID: edge_data['Node 2 ID']}})
                        MERGE (a)-[r:{sanitized_edge_type}]->(b)
                        SET r.weight = toFloat(edge_data['Edge Weight']),
                            r.direction = edge_data['Edge Direction'],
                            r.source = edge_data['Node 1 Source']
                        """
                        session.run(query, edges=edge_list)

            logging.info("Edge import completed successfully.")
        except FileNotFoundError:
            logging.error(f"Edge file not found: {edge_file_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred during edge import: {e}")
            sys.exit(1)

def main():
    """
    Main function to run the graph creation process.
    """
    if len(sys.argv) != 3:
        print("Usage: python createGraph.py <path_to_nodes_csv> <path_to_edges_csv>")
        sys.exit(1)

    node_file = os.path.expanduser(sys.argv[1])
    edge_file = os.path.expanduser(sys.argv[2])

    graph_creator = GraphCreator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Import data
    graph_creator.import_nodes(node_file)
    graph_creator.import_edges(edge_file)

    # Clean up
    graph_creator.close()

if __name__ == "__main__":
    main()
