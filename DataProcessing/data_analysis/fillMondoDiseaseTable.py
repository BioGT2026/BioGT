import pandas as pd
import os

def process_files(data_processing_dir):
    """
    Merges disease categories from mondo_disease_categories.csv into mondo_node.csv.

    Args:
        data_processing_dir (str): The path to the DataProcessing directory.
    """
    # Define file paths
    categories_file = os.path.join(data_processing_dir, 'mondo_disease_categories.csv')
    nodes_file = os.path.join(data_processing_dir, 'mondo_node.csv')
    output_file = os.path.join(data_processing_dir, 'mondo_node_with_categories.csv')

    # 1. Read mondo_disease_categories.csv and create a mapping
    try:
        categories_df = pd.read_csv(categories_file)
    except FileNotFoundError:
        print(f"Error: {categories_file} not found.")
        return

    disease_categories = categories_df.groupby('Disease_ID')['Category'].apply(lambda x: ';'.join(sorted(list(set(x))))).to_dict()

    # 2. Read mondo_node.csv and add the new 'Category' column
    try:
        nodes_df = pd.read_csv(nodes_file, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: {nodes_file} not found.")
        return

    # Map the categories to the nodes dataframe
    nodes_df['Category'] = nodes_df['ID'].map(disease_categories)

    # Fill NaN values with an empty string if no category is found
    nodes_df['Category'] = nodes_df['Category'].fillna('')

    # 3. Reorder columns and save to a new file
    # New column order: 'ID', 'Type', 'Category', 'Name', 'Comments'
    cols = nodes_df.columns.tolist()
    if 'Category' in cols:
        cols.insert(2, cols.pop(cols.index('Category')))
        nodes_df = nodes_df[cols]

    nodes_df.to_csv(output_file, index=False)
    print(f"Processing complete. Output saved to {output_file}")

if __name__ == '__main__':
    # Assuming the script is in the DataProcessing directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_files(current_dir)
