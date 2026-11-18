import re

def parse_obo(file_path):
    """
    Parses an OBO file and extracts terms, their names, parent-child relationships,
    and obsolete status.

    Args:
        file_path (str): The path to the OBO file.

    Returns:
        tuple: A tuple containing three items:
               - names: A dictionary mapping term IDs to term names.
               - parents: A dictionary mapping term IDs to a list of their parent IDs.
               - obsolete_terms: A set of term IDs that are marked as obsolete.
    """
    names = {}
    parents = {}
    obsolete_terms = set()
    current_term_id = None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == '[Term]':
                current_term_id = None # Reset for the new term
            elif line.startswith('id:'):
                # A new term block has started
                current_term_id = line.split('id: ')[1]
                if current_term_id not in parents:
                    parents[current_term_id] = []
            elif line.startswith('name:'):
                if current_term_id:
                    names[current_term_id] = line.split('name: ')[1]
            elif line.startswith('is_a:'):
                if current_term_id:
                    parent_id = re.split(r'\s|!', line)[1]
                    parents[current_term_id].append(parent_id)
            elif line.startswith('is_obsolete: true'):
                if current_term_id:
                    obsolete_terms.add(current_term_id)
    return names, parents, obsolete_terms
