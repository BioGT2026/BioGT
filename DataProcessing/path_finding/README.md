# Potential Path Finding and Candidate Screening Scripts

This directory contains scripts for path search, path formatting, and candidate screening over an already constructed KG, prediction logs, or candidate drug/marker relationships. These scripts help discover or curate potential disease-marker and drug-marker paths; they are not part of raw KG source extraction.

Run scripts from the repository root when possible, or set `PYTHONPATH` to the repository root before running Python entry points that import project modules.

## Path Search

| Script | Purpose |
| --- | --- |
| `find_reachable_edges_within_hops.py` | Given a source node, target node, and maximum hop count, finds edges that can appear on source-to-target paths and writes complete path logs. |
| `find_drug_connected_disease_marker_edges.py` | Searches short paths from diseases to candidate markers that pass through a specified drug, with options for marker type, maximum path length, GO-node constraints, and MarkerDB exclusion. |

## Prediction Log and Candidate Post-processing

| Script | Purpose |
| --- | --- |
| `extract_top1_protein_drug_paths.py` | Extracts Top 1 protein prediction results that contain drug nodes from prediction logs and writes them to CSV. |
| `format_path_with_drug.py` | Cleans the `path_with_drug` field in candidate CSV files by keeping the primary drug alias and standardizing path display. |
| `screen_tcm_monomer_candidates.py` | Screens potential traditional Chinese medicine monomer candidates using drug-name keywords, exclusion rules, and optional MONDO disease ID lookup. |

## Notes

These scripts usually depend on `merged_edge.csv`, `marker_edge.csv`, or prediction logs. If the default paths do not match the current data layout, pass input files explicitly through command-line arguments.