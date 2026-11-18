import pandas as pd

def build_tree(parents):
    """Builds a tree structure from the parent-child relationships."""
    tree = {term: [] for term in parents}
    roots = []
    all_children = set()
    for child, parent_list in parents.items():
        for parent in parent_list:
            if parent in tree:
                tree[parent].append(child)
                all_children.add(child)

    # Find roots (nodes that are never children)
    all_nodes = set(tree.keys())
    roots = list(all_nodes - all_children)
    
    # As a fallback, if no clear roots are found (e.g. circular dependencies),
    # consider nodes with no parents as roots.
    if not roots:
        roots = [term for term, parent_list in parents.items() if not parent_list]

    return tree, roots

def get_paths(tree, roots, names, obsolete_terms):
    """Generates all paths from roots to leaves."""
    all_paths = []
    for root in roots:
        if root not in obsolete_terms and root.startswith("DOID:"):
            # Start traversal from each root
            stack = [(root, [names.get(root, root)])]
            while stack:
                node, path = stack.pop()
                children = tree.get(node, [])
                
                # Filter children
                valid_children = [
                    child for child in children 
                    if child not in obsolete_terms and child.startswith("DOID:")
                ]

                if not valid_children:
                    # It's a leaf node in the filtered tree
                    all_paths.append(path)
                    continue

                for child in reversed(valid_children): # Reverse to maintain order
                    new_path = path + [names.get(child, child)]
                    stack.append((child, new_path))
    return all_paths

def write_excel(paths, output_file):
    """Writes the hierarchical paths to an Excel file with merged cells."""
    if not paths:
        print("No paths to write to Excel.")
        return

    df = pd.DataFrame(paths)
    
    # Rename columns to represent levels
    df.columns = [f'Level {i}' for i in range(df.shape[1])]

    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Disease Hierarchy', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['Disease Hierarchy']
    
    # Format for merged cells
    merge_format = workbook.add_format({
        'valign': 'vcenter',
        'align': 'left'
    })

    # Merge cells
    for col_num in range(df.shape[1]):
        row_num = 1 # Start from the first data row
        while row_num <= df.shape[0]:
            cell_value = df.iloc[row_num - 1, col_num]
            if pd.isna(cell_value):
                row_num += 1
                continue
            
            count = 1
            # Count how many rows to merge
            for i in range(row_num, df.shape[0]):
                if df.iloc[i, col_num] == cell_value:
                    # Also check if the previous column is the same, to merge correctly
                    if col_num > 0 and df.iloc[i, col_num-1] != df.iloc[row_num-1, col_num-1]:
                        break
                    count += 1
                else:
                    break
            
            if count > 1:
                worksheet.merge_range(row_num, col_num, row_num + count - 1, col_num, cell_value, merge_format)
            
            row_num += count

    # Adjust column widths
    for i, col in enumerate(df.columns):
        max_len = df[col].astype(str).map(len).max()
        worksheet.set_column(i, i, max_len + 2)

    writer.close()
    print(f"Successfully wrote hierarchy to {output_file}")
