import pandas as pd

def count_leaf_nodes(node, tree, obsolete_terms, memo):
    """Recursively counts leaf nodes under a given node with memoization."""
    if node in memo:
        return memo[node]

    children = [
        child for child in tree.get(node, [])
        if child not in obsolete_terms and child.startswith("DOID:")
    ]

    if not children:
        memo[node] = 1
        return 1

    count = 0
    for child in children:
        count += count_leaf_nodes(child, tree, obsolete_terms, memo)
    
    memo[node] = count
    return count

def write_summary_txt(tree, roots, names, obsolete_terms, output_file):
    """Writes the disease hierarchy summary to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Debug: Found {len(names)} names, {len(tree)} parent relationships, and {len(obsolete_terms)} obsolete terms.\n")
        f.write("Disease Hierarchy Levels:\n")

        leaf_count_memo = {}
        
        # We need to traverse the tree level by level to print in order
        queue = []
        for root in roots:
            if root not in obsolete_terms and root.startswith("DOID:"):
                queue.append((root, 0)) # (node, level)

        visited = set(item[0] for item in queue)
        all_nodes_by_level = []

        head = 0
        while head < len(queue):
            node, level = queue[head]
            head += 1
            
            all_nodes_by_level.append((node, level))

            children = sorted([
                child for child in tree.get(node, [])
                if child not in obsolete_terms and child.startswith("DOID:") and child not in visited
            ])
            
            for child in children:
                visited.add(child)
                queue.append((child, level + 1))
        
        # Sort by level, then by name for consistent output
        all_nodes_by_level.sort(key=lambda x: (x[1], names.get(x[0], x[0])))

        for node, level in all_nodes_by_level:
            term_name = names.get(node, "Unknown Name")
            
            # Count immediate children
            immediate_children = [
                child for child in tree.get(node, [])
                if child not in obsolete_terms and child.startswith("DOID:")
            ]
            next_level_count = len(immediate_children)
            
            # Count all leaf nodes under this node
            total_leaf_count = count_leaf_nodes(node, tree, obsolete_terms, leaf_count_memo)
            
            f.write(f"Level {level}: {term_name} ({node}) | Next Level Children: {next_level_count} | Total Leaf Nodes: {total_leaf_count}\n")

    print(f"Successfully wrote summary to {output_file}")
