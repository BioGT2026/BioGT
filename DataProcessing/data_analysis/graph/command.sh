#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPORT_DIR="$SCRIPT_DIR/admin_import_files"

# Clear old generated import files first.
mkdir -p "$IMPORT_DIR"
rm -rf "$IMPORT_DIR"/*

# Prepare Neo4j admin-import files.
python "$SCRIPT_DIR/prepare_for_admin_import.py" \
  ~/Data/merge/merged_node.csv \
  ~/Data/merge/merged_edge.csv \
  "$IMPORT_DIR"

sudo neo4j-admin database import full \
  --multiline-fields=true \
  --nodes="biological_process=$IMPORT_DIR/import-nodes-biological_process.csv" \
  --nodes="cellular_component=$IMPORT_DIR/import-nodes-cellular_component.csv" \
  --nodes="disease=$IMPORT_DIR/import-nodes-disease.csv" \
  --nodes="drug=$IMPORT_DIR/import-nodes-drug.csv" \
  --nodes="gene=$IMPORT_DIR/import-nodes-gene.csv" \
  --nodes="metabolite=$IMPORT_DIR/import-nodes-metabolite.csv" \
  --nodes="molecular_function=$IMPORT_DIR/import-nodes-molecular_function.csv" \
  --nodes="pathway=$IMPORT_DIR/import-nodes-pathway.csv" \
  --nodes="phenotype=$IMPORT_DIR/import-nodes-phenotype.csv" \
  --nodes="protein=$IMPORT_DIR/import-nodes-protein.csv" \
  --nodes="rna=$IMPORT_DIR/import-nodes-rna.csv" \
  --relationships="$IMPORT_DIR/import-rels.csv" \
  --id-type=STRING \
  --overwrite-destination \
  neo4j


sudo chown -R neo4j:neo4j /var/lib/neo4j/data/databases/neo4j
sudo systemctl start neo4j

cat <<'CYPHER_QUERIES'
Example protein query:

MATCH path = (d:disease {ID: 'MONDO:0005083'})-[*1..3]-(p:protein)
WITH d, p, relationships(path) as rels, length(path) as hop_count
WITH d, p, rels, hop_count,
     reduce(total_weight = 0, r IN rels | total_weight + coalesce(r.weight, 0)) as total_weight
WITH p.ID as protein_id, 
     d, p, rels, hop_count, total_weight
ORDER BY hop_count ASC, total_weight DESC
WITH protein_id, collect({
  disease_id: d.ID,
  disease_name: d.name,
  protein_name: p.name,
  hop_count: hop_count,
  total_weight: total_weight,
  relationship_types: [r IN rels | type(r)],
  rels: rels
})[0] as best_path
RETURN best_path.disease_id as disease_id,
       best_path.disease_name as disease_name,
       protein_id,
       best_path.protein_name as protein_name,
       best_path.hop_count as hop_count,
       best_path.total_weight as total_weight,
       best_path.relationship_types as relationship_types
ORDER BY best_path.hop_count, best_path.total_weight DESC


MATCH path = (d:disease {ID: 'MONDO:0005083'})-[*1..3]-(p:gene)
WITH d, p, relationships(path) as rels, length(path) as hop_count
WITH d, p, rels, hop_count,
     reduce(total_weight = 0, r IN rels | total_weight + coalesce(r.weight, 0)) as total_weight
WITH p.ID as gene_id, 
     d, p, rels, hop_count, total_weight
ORDER BY hop_count ASC, total_weight DESC
WITH gene_id, collect({
  disease_id: d.ID,
  disease_name: d.name,
  gene_name: p.name,
  hop_count: hop_count,
  total_weight: total_weight,
  relationship_types: [r IN rels | type(r)],
  rels: rels
})[0] as best_path
RETURN best_path.disease_id as disease_id,
       best_path.disease_name as disease_name,
       gene_id,
       best_path.gene_name as gene_name,
       best_path.hop_count as hop_count,
       best_path.total_weight as total_weight,
       best_path.relationship_types as relationship_types
ORDER BY best_path.hop_count, best_path.total_weight DESC
CYPHER_QUERIES
