import xml.etree.ElementTree as ET
import csv
import os
from collections import defaultdict, deque

# Configuration
OWL_FILE = os.path.join(os.path.dirname(__file__), "mondo-simple.owl")
OUTPUT_FILE = "mondo_disease_categories.csv"

# Category Mapping
# (Category Name, [List of Root MONDO IDs])
CATEGORIES = [
    ("Hereditary Diseases", ["MONDO_0003847"]),
    ("Infectious Diseases", ["MONDO_0005550"]),
    ("Immunological Diseases", ["MONDO_0005046"]),
    ("Neoplastic Diseases", ["MONDO_0005070"]),
    ("Traumatic/Toxic Diseases", ["MONDO_0021178", "MONDO_0029000"]), # Injury, Poisoning
    ("Metabolic/Nutritional Diseases", ["MONDO_0005066", "MONDO_0005137"]),
    ("Degenerative/Age-related Diseases", ["MONDO_0005559"]), # Neurodegenerative (limited but safest root found)
    ("Idiopathic Diseases", ["MONDO_0700007"]),
    ("Congenital/Developmental Diseases", ["MONDO_0021140", "MONDO_0005287"]), # Congenital, Dev. Disability
    ("Psychological/Psychiatric Diseases", ["MONDO_0002025"])
]

NAMESPACES = {
    'owl': 'http://www.w3.org/2002/07/owl#',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
}

def load_ontology(file_path):
    print("Loading ontology...")
    # Graph: parent -> children
    children_map = defaultdict(set)
    
    # We need to map short ID to full URI for internal processing, or just handle URIs
    # The subClassOf resource usually is full URI.
    
    context = ET.iterparse(file_path, events=("start", "end"))
    context = iter(context)
    _, root = next(context)

    for event, elem in context:
        if event == "end" and elem.tag == "{http://www.w3.org/2002/07/owl#}Class":
            about = elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
            if about:
                # Find subClassOf
                for sub in elem.findall("rdfs:subClassOf", NAMESPACES):
                    resource = sub.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                    if resource:
                        # resource is parent, about is child
                        # Store as short IDs if possible to save memory/simplify
                        parent_id = resource.split('/')[-1]
                        child_id = about.split('/')[-1]
                        if "MONDO_" in parent_id and "MONDO_" in child_id:
                             children_map[parent_id].add(child_id)
            
            root.clear()
            
    return children_map

def get_descendants(start_node, children_map):
    descendants = set()
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        if node in children_map:
            for child in children_map[node]:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
    return descendants

def main():
    children_map = load_ontology(OWL_FILE)
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Disease_ID"])
        
        for category, roots in CATEGORIES:
            print(f"Processing {category}...")
            all_descendants = set()
            for root_id in roots:
                # Add root itself? Usually yes.
                all_descendants.add(root_id)
                # Add descendants
                desc = get_descendants(root_id, children_map)
                all_descendants.update(desc)
            
            print(f"  - Found {len(all_descendants)} diseases in {category}")
            
            for d_id in all_descendants:
                # Convert MONDO_123 to MONDO:123 if standard format is needed, 
                # but usually MONDO_123 is the ID part. User asked for MONDO ID.
                # Standard format usually MONDO:12345 or just the ID.
                # I'll keep MONDO_ format or convert to MONDO: if that's standard.
                # The user request says "MONDO ID". I'll use MONDO:XXXX format strictly.
                formatted_id = d_id.replace("_", ":")
                writer.writerow([category, formatted_id])

    print(f"Done. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
