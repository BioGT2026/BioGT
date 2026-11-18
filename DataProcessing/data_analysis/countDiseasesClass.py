import sys
import csv
import argparse
from rdflib import Graph, URIRef, RDFS, OWL
from collections import defaultdict

def get_human_diseases_ids(g):
    """
    从已加载的图中查询 MONDO_0700096 的所有子孙类（人类疾病）
    """
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

    SELECT DISTINCT ?class
    WHERE {
      ?class rdfs:subClassOf* <http://purl.obolibrary.org/obo/MONDO_0700096> .
    }
    """
    
    results = g.query(query)
    human_disease_uris = set()
    for row in results:
        human_disease_uris.add(row["class"])
    return human_disease_uris

def count_disease_categories(owl_file, output_csv=None):
    """
    Parses an OWL file and counts diseases based on predefined categories
    by leveraging the ontology hierarchy.
    If output_csv is provided, also exports results to CSV file.
    """
    g = Graph()
    print(f"Loading and parsing {owl_file}...")
    g.parse(owl_file, format="xml")
    print("Parsing complete.")

    # 获取所有人类疾病ID
    print("Extracting human disease IDs...")
    human_disease_uris = get_human_diseases_ids(g)
    print(f"Found {len(human_disease_uris)} human disease URIs")

    # Define keywords to find the top-level category URIs
    # This list is expanded based on the user-provided top-level terms from the ontology.
    category_keywords = {
        "Hereditary Diseases": [
            "hereditary disease", "chromosomal disorder", "mitochondrial disease", 
            "premature aging syndrome", "syndromic disease"
        ],
        "Infectious Diseases": [
            "infectious disease", "post-infectious disorder"
        ],
        "Immunological Diseases": [
            "immune system disorder", "inflammatory disease", "disease related to transplantation",
            "connective tissue disorder" # Often has immunological basis
        ],
        "Neoplastic Diseases": [
            "cancer or benign tumor", "neoplasm"
        ],
        "Traumatic/Toxic Diseases": [
            "poisoning", "radiation-induced disorder", "iatrogenic disease", 
            "occupational disorder"
        ],
        "Metabolic/Nutritional Diseases": [
            "metabolic disease", "nutritional disorder", "disorder of glycosylation"
        ],
        "Degenerative/Age-related Diseases": [
            "degenerative", "age-related" # Kept from before, might need refinement
        ],
        "Idiopathic Diseases": [
            "idiopathic disease"
        ],
        "Congenital/Developmental Diseases": [
            "disorder of development or morphogenesis", "congenital disorder", "perinatal disease"
        ],
        "Psychological/Psychiatric Diseases": [
            "psychiatric disorder", "sleep disorder", "mental disorder"
        ]
        # ,
        # "Acute Disease": [
        #     "acute disease"
        # ]
        # ,
        # "Disorders by Body System": [
        #     "auditory system disorder", "breast disorder", "cardiovascular disorder",
        #     "digestive system disorder", "disorder of orbital region", "disorder of visual system",
        #     "endocrine system disorder", "hematologic disorder", "integumentary system disorder",
        #     "mouth disorder", "musculoskeletal system disorder", "nervous system disorder",
        #     "otorhinolaryngologic disease", "reproductive system disorder", 
        #     "respiratory system disorder", "upper digestive tract disorder", "urinary system disorder",
        #     "obstetric disorder", "ulcer disease" # Grouping ulcer here as it affects systems
        # ]
    }

    print("Finding category URIs...")
    category_uris = defaultdict(set)
    for s, p, o in g.triples((None, RDFS.label, None)):
        label = str(o).lower()
        for category, keywords in category_keywords.items():
            if any(keyword in label for keyword in keywords):
                category_uris[category].add(s)

    # This will hold all superclasses for each class
    subclass_map = defaultdict(set)
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            subclass_map[s].add(o)

    memo = {}
    def get_all_superclasses(uri):
        if uri in memo:
            return memo[uri]
        parents = subclass_map.get(uri, set())
        all_parents = set(parents)
        for parent in parents:
            all_parents.update(get_all_superclasses(parent))
        memo[uri] = all_parents
        return all_parents

    UNCATEGORIZED_CATEGORY = "Uncategorized Diseases"
    counts = defaultdict(int)
    for category in category_keywords.keys():
        counts[category] = 0
    counts[UNCATEGORIZED_CATEGORY] = 0
    classified_diseases = defaultdict(set)  # 使用set而非list，便于去重和计算
    uncategorized_diseases = []  # 存储未分类疾病的详细信息
    unique_diseases = set()
    disease_category_count = defaultdict(int)  # 记录每个疾病属于多少个类别
    all_disease_uris = {s for s, p, o in g if isinstance(s, URIRef) and "http://purl.obolibrary.org/obo/MONDO_" in str(s)}
    
    # 只保留人类疾病
    all_disease_uris = all_disease_uris.intersection(human_disease_uris)
    print(f"Filtering to human diseases only: {len(all_disease_uris)} diseases")

    print("Analyzing disease classes using hierarchy...")
    for disease_uri in all_disease_uris:
        unique_diseases.add(disease_uri)
        superclasses = get_all_superclasses(disease_uri)
        superclasses_with_self = set(superclasses)
        superclasses_with_self.add(disease_uri)
        disease_id = str(disease_uri).replace("http://purl.obolibrary.org/obo/", "")
        
        # 检查该疾病是否属于任何分类
        found_any_category = False
        for category, uris in category_uris.items():
            if not uris.isdisjoint(superclasses_with_self):
                counts[category] += 1
                classified_diseases[category].add(disease_id)
                disease_category_count[disease_id] += 1
                found_any_category = True
        
        # 如果不属于任何分类，添加到未分类
        if not found_any_category:
            counts[UNCATEGORIZED_CATEGORY] += 1
            classified_diseases[UNCATEGORIZED_CATEGORY].add(disease_id)
            disease_label = g.value(subject=disease_uri, predicate=RDFS.label)
            disease_label = str(disease_label) if disease_label else "Unknown"
            uncategorized_diseases.append((disease_id, disease_label))

    print("\n--- Disease Category Counts ---")
    print(f"{'Category':<40} {'Total':<8} {'Unique':<8} {'Shared':<8}")
    print("-" * 64)
    
    total_classified = 0
    total_unique_diseases = set()
    
    for category in sorted(counts.keys()):
        if category == UNCATEGORIZED_CATEGORY:
            continue
        
        category_diseases = classified_diseases[category]
        # 计算该类别独有的疾病（只属于该类别的）
        unique_to_category = sum(1 for d in category_diseases if disease_category_count[d] == 1)
        # 计算该类别共享的疾病（属于多个类别的）
        shared_in_category = len(category_diseases) - unique_to_category
        
        print(f"{category:<40} {counts[category]:<8} {unique_to_category:<8} {shared_in_category:<8}")
        total_classified += counts[category]
        total_unique_diseases.update(category_diseases)
    
    total_uncategorized = counts[UNCATEGORIZED_CATEGORY]
    print("-" * 64)
    print(f"{'Uncategorized Diseases':<40} {total_uncategorized:<8}")
    
    print("\n-----------------------------")
    print(f"Total count (with duplicates): {total_classified}")
    print(f"Total unique classified diseases: {len(total_unique_diseases)}")
    print(f"Total uncategorized diseases: {total_uncategorized}")
    print(f"Total unique diseases (classified + uncategorized): {len(total_unique_diseases) + total_uncategorized}")
    print(f"Total unique disease records found: {len(unique_diseases)}")
    
    # 统计共享疾病信息
    multi_category_diseases = {d: disease_category_count[d] for d in total_unique_diseases if disease_category_count[d] > 1}
    if multi_category_diseases:
        print(f"\nDiseases in multiple categories: {len(multi_category_diseases)}")
        # 统计每个类别数量的分布
        category_distribution = defaultdict(int)
        for count in multi_category_diseases.values():
            category_distribution[count] += 1
        for num_categories in sorted(category_distribution.keys()):
            print(f"  - In {num_categories} categories: {category_distribution[num_categories]}")
    
    print("-----------------------------\n")
    
    # 输出未分类疾病的详细信息
    if uncategorized_diseases:
        print("--- Uncategorized Diseases Details ---")
        print(f"Total: {len(uncategorized_diseases)} diseases\n")
        for disease_id, disease_label in sorted(uncategorized_diseases):
            print(f"{disease_id:20s} | {disease_label}")
        print("-----------------------------\n")
    # 导出到 CSV 如果输出文件已指定
    if output_csv:
        print(f"Exporting results to {output_csv}...")
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Disease_ID"])
            
            for category in sorted(counts.keys()):
                for disease_id in sorted(classified_diseases[category]):
                    # 转换 MONDO_12345 为 MONDO:12345
                    formatted_id = disease_id.replace("_", ":")
                    writer.writerow([category, formatted_id])
                
                # 统计已导出的数量
                print(f"  - {category}: {len(classified_diseases[category])} diseases exported")
        
        print(f"Results exported successfully to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count and classify diseases by category from MONDO ontology")
    parser.add_argument('owl_file', help="Path to MONDO OWL file")
    parser.add_argument('-o', '--output', type=str, help="Optional output CSV file path")
    
    args = parser.parse_args()
    
    # Increase recursion limit for deep ontologies
    sys.setrecursionlimit(2000000)
    count_disease_categories(args.owl_file, args.output)
