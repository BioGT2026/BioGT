# Knowledge Graph Construction Data Processing Scripts

This directory contains the data download, source extraction, CSV normalization, merge, and optional feature-generation scripts used to build BioGT-KG. The scripts primarily produce standardized node and edge CSV files for downstream model training.

Run `ExtractData.sh` from this directory. The shell entry point automatically adds the repository root to `PYTHONPATH`, so Python scripts can import `DataProcessing.kg_construction.commonSettings`.

## Main Entry Points

| Script | Purpose |
| --- | --- |
| `ExtractData.sh` | Pipeline wrapper that calls source-specific extractors and provides the `all` and `merge` stages. |
| `mergeData.py` | Merges source-specific node and edge CSV files, normalizes node types, deduplicates records, and handles marker edges with priority. |
| `generateNodeEmbedding.py` | Generates or rewrites merged node feature fields from sequence, SMILES, text annotations, or learned embeddings. |
| `commonSettings.py` | Shared KG CSV schemas, edge-direction constants, ID prefixes, and evidence-to-weight conversion logic. |

## Source Extraction Scripts

| Script | Purpose |
| --- | --- |
| `downloadBioMarkerDBData.py` | Downloads MarkerDB records through the MarkerDB API and writes gene, chemical, and protein source files. |
| `extractBioMarkerAndMerge.py` | Converts MarkerDB gene, chemical, protein, and miRNA records into disease-marker edges using MONDO and RefSeq mappings. |
| `extractClinvar.py` | Extracts variant nodes and related edges from ClinVar XML. |
| `extractDrugCAS.py` | Extracts DrugBank-to-CAS/ATC-style drug identifier mappings from DrugBank XML. |
| `extractDrugCentral.py` | Builds drug-disease edges from DrugCentral, DrugBank CAS/ATC mappings, and MONDO-UMLS mappings. |
| `extractDrugs.py` | Extracts DrugBank drug nodes and drug-gene or drug-protein relationships from XML, SDF, and GeneID inputs. |
| `extractGO.py` | Extracts GO nodes and gene/protein-to-GO annotation edges from GO OBO and GAF files. |
| `extractGene.py` | Extracts NCBI Gene nodes and gene-refseq, gene-go, gene-neighbor, and related edges. |
| `extractHMDB.py` | Extracts metabolite nodes and disease, pathway, or protein relationships from HMDB XML inputs. |
| `extractHPO.py` | Extracts HPO phenotype nodes and HPO hierarchy edges from HPO OBO. |
| `extractHPORelations.py` | Extracts phenotype, disease, and gene relationship edges from HPO annotations and gene-disease resources. |
| `extractHPOUMLS.py` | Extracts HPO-to-UMLS mappings for cross-source alignment. |
| `extractMondo.py` | Extracts MONDO disease nodes, disease hierarchy edges, and cross-ontology relationships from MONDO OBO/OWL. |
| `extractMondoOMIM.py` | Extracts MONDO-to-OMIM disease mappings from MONDO OBO. |
| `extractMondoUMLS.py` | Extracts MONDO-to-UMLS disease mappings from MONDO OBO. |
| `extractPubChem.py` | Extracts compound or metabolite-like node information from PubChem SDF files. |
| `extractReactome.py` | Extracts Reactome pathway nodes and pathway-protein, pathway-gene, and pathway-pathway edges. |
| `extractRefSeq.py` | Extracts RNA, protein, and sequence nodes plus RefSeq-related edges from RefSeq GenBank and FASTA files. |
| `extractSider.py` | Builds drug-phenotype and drug-disease style edges from SIDER plus HPO-UMLS, MONDO-UMLS, and drug mappings. |
| `extractUberon.py` | Extracts anatomy nodes and anatomy/GO-related edges from Uberon OBO. |
| `extractUniprot.py` | Extracts UniProt protein nodes and protein-protein or protein-gene relationships from FASTA and PPI files. |
| `extractUniprotMapping.py` | Extracts UniProt-to-RefSeq and related protein identifier mapping edges from UniProt idmapping data. |

## Typical Workflow

```bash
cd DataProcessing/kg_construction
bash ExtractData.sh all
bash ExtractData.sh merge
```

Direct Python entry points are safest from the repository root so package imports resolve consistently:

```bash
python DataProcessing/kg_construction/mergeData.py \
  -n "/path/to/graphWithNodeType/*_node.csv" \
  -e "/path/to/graphWithNodeType/*_edge.csv" \
  --marker-edge-file "/path/to/graphWithNodeType/marker_edge.csv" \
  --out-nodes "/path/to/graphWithNodeType/merge/merged_node.csv" \
  --out-edges "/path/to/graphWithNodeType/merge/merged_edge.csv"
```