# Data Analysis and Inspection Scripts

This directory contains scripts for inspecting generated KG CSV files, ontology files, and intermediate statistics. These scripts are not part of the main KG extraction pipeline; they are used to understand data coverage, disease category structure, connectivity, and intermediate result quality.

Run scripts from the repository root when possible, or set `PYTHONPATH` to the repository root before running Python entry points that import project modules.

## Connectivity and Enrichment Analysis

| Script | Purpose |
| --- | --- |
| `analyze_disease_marker_connectivity.py` | Counts shortest-path connectivity from each MONDO disease to gene, RNA, protein, and metabolite marker sets, and writes marker-filtered outputs. |
| `analyze_mondo_node_enrichment.py` | Aggregates MONDO disease degree, marker-subgraph degree, connected-component metrics, MarkerDB category edges, and target-type connectivity metrics. |
| `filter_rare_diseases.py` | Splits enrichment statistics into rare-disease and non-rare-disease outputs using the MONDO OBO `subset: rare` annotation. |

## Ontology and Category Analysis

| Script or Directory | Purpose |
| --- | --- |
| `countDiseasesClass.py` | Reads MONDO OWL, counts human diseases by predefined high-level categories, and can export a disease-category CSV. |
| `extractDiseasesByClass.py` | Extracts descendant disease IDs for predefined disease categories from a MONDO simple OWL hierarchy. |
| `extractMondoClass.py` | Analyzes the MONDO human-disease hierarchy and exports level-wise CSV summaries plus optional JSON. |
| `fillMondoDiseaseTable.py` | Joins a disease-category table back into `mondo_node.csv` and writes a disease node table with category annotations. |
| `countFeaturesInOBO.py` | Counts OBO feature frequencies and prints example values for ontology-format inspection. |
| `countDO/` | Parses Disease Ontology OBO, builds a hierarchy tree, and exports Excel path tables plus text summaries. |

## CSV and Data Quality Utilities

| Script | Purpose |
| --- | --- |
| `count.py` | Scans a directory for `*_node.csv` and `*_edge.csv` files and aggregates node-type and endpoint-type counts. |
| `countDB.py` | Performs simple type counts for RefSeq or BiomarkerDB raw/intermediate files. |
| `findFullInCSV.py` | Finds the first complete row in a standard node CSV, or reports the last row when no complete row exists. |
| `fetchXMLStructure.py` | Prints the tree structure of an XML file to help inspect upstream XML formats. |

## Neo4j Graph Visualization

| Path | Purpose |
| --- | --- |
| `graph/createGraph.py` | Imports merged node and edge CSV files into Neo4j through the Neo4j Python driver for interactive visualization and Cypher exploration. |
| `graph/prepare_for_admin_import.py` | Converts merged CSV files into node and relationship files for `neo4j-admin database import`. |
| `graph/command.sh` | Example `neo4j-admin` import command and Cypher query scratchpad. |

## Notes

Some scripts use fixed default input or output file names. Prefer passing explicit paths, or check the path constants at the top of the script before running it.