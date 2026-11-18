import sys
import os
from DataProcessing.data_analysis.countDO.parse_obo import parse_obo
from DataProcessing.data_analysis.countDO.excel_writer import build_tree, get_paths, write_excel
from DataProcessing.data_analysis.countDO.summary_writer import write_summary_txt, count_leaf_nodes

def main():
    """
    Main function to parse an OBO file, build a hierarchy,
    and write it to an Excel file.
    """
    if len(sys.argv) < 2:
        print("Usage: python countDOLevels.py <path_to_obo_file> [output_excel_file] [output_summary_file]")
        sys.exit(1)

    obo_file_path = os.path.expanduser(sys.argv[1])
    output_file = sys.argv[2] if len(sys.argv) > 2 else "disease_hierarchy.xlsx"
    summary_output_file = sys.argv[3] if len(sys.argv) > 3 else "disease_summary.txt"

    try:
        names, parents, obsolete_terms = parse_obo(obo_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {obo_file_path}")
        sys.exit(1)

    print(f"Debug: Found {len(names)} names, {len(parents)} parent relationships, and {len(obsolete_terms)} obsolete terms.")
    if not names and not parents:
        print("Debug: No data was parsed from the OBO file. Please check the file format and encoding.")
        sys.exit(1)

    # Build the tree from parent relationships
    tree, roots = build_tree(parents)
    
    # Find all paths from roots to leaves
    paths = get_paths(tree, roots, names, obsolete_terms)

    # Write the paths to an Excel file
    write_excel(paths, output_file)

    # Write the summary to a text file
    write_summary_txt(tree, roots, names, obsolete_terms, summary_output_file)

if __name__ == "__main__":
    main()
