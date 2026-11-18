# BioGT: A knowledge graph–grounded predictor for pan-disease, multi-layer biomarker discovery

> System: BioGT
> Knowledge graph: BioGT-KG
> Prediction model: BioGT-Predictor
> Project type: Biomedical knowledge graph construction and disease-biomarker link prediction

We implements BioGT, including construction of the BioGT-KG heterogeneous biomedical knowledge graph and the BioGT-Predictor link-prediction model for disease-centered biomarker discovery. The codebase connects disease, phenotype, drug, gene, RNA, protein, metabolite, pathway, molecular-function, biological-process, cellular-component, anatomy, and variant records from public biomedical resources. The model side predicts disease associations for four biomarker target types: gene, RNA, protein, and metabolite.

The repository is organized around two workflows:

1. Build a biomedical knowledge graph from source databases and merge it into node and edge CSV files.
2. Train and run prediction models over the merged knowledge graph.

The default training engine now uses one shared encoder with four modality
outputs, target-edge context mini-batches, ranking loss, semi-hard negatives,
query-centered sparse attention, and Top-K router MoE. The previous full-graph
implementation remains available with `--training-engine legacy`. 

## Data Assets

To keep repository size small, this repo does not directly store large data artifacts in `Data/`.
Please download the released data package from Google Drive and place it locally:

- Google Drive: [BioGT Data Package](https://drive.google.com/drive/folders/10_XDrQJMh_zyKrc12sEyuWQEuVgmyHwU?usp=sharing).

The downloadable package includes:

- Knowledge graph node CSV files and edge CSV files, including the merged graph used by the model.
- Trained model weights for the four biomarker targets: `gene`, `rna`, `protein`, and `metabolite`.
- Optional intermediate source-specific node and edge CSV files when redistribution is allowed by the upstream data source.

Recommended local layout after download:

```text
Data/
  merged_node.csv
  merged_edge.csv
  marker_edge.csv
  weights/
    biogt_gene.pt
    biogt_rna.pt
    biogt_protein.pt
    biogt_metabolite.pt
```

If your local setup uses different file names, pass the actual paths to `--nodes_csv`, `--edges_csv`, `--load_model_path`, and the wrapper script arguments described below.

## Repository Layout

```text
  Data/                    # Local downloaded assets (not tracked in Git)
  DataProcessing/kg_construction/ # BioGT-KG source extraction, merge, and feature scripts
  DataProcessing/data_analysis/    # KG statistics, ontology inspection, and Neo4j visualization helpers
  Model/                   # DGL model training, prediction, baselines, and result parsing
  README.md                # This guide
```

Key implementation files:

| Area | Files | Purpose |
| --- | --- | --- |
| KG construction data processing | `DataProcessing/kg_construction/` | Convert upstream biomedical resources into node and edge CSV files, merge source-specific CSVs, and generate optional node features. |
| Source extraction | `DataProcessing/kg_construction/extract*.py`, `DataProcessing/kg_construction/ExtractData.sh` | Convert upstream biomedical resources into node and edge CSV files. |
| KG merge | `DataProcessing/kg_construction/mergeData.py` | Deduplicate nodes, normalize node types, prioritize marker edges, and write merged graph CSVs. |
| Node features | `DataProcessing/kg_construction/generateNodeEmbedding.py`, `Model/modelData.py` | Generate or cache node representations from sequence, SMILES, comments, and learnable embeddings. |
| KG data statistics | `DataProcessing/data_analysis/count.py`, `DataProcessing/data_analysis/analyze_mondo_node_enrichment.py` | Summarize node/edge type counts and disease-level connectivity or enrichment metrics over merged KG CSV files. |
| Neo4j graph visualization | `DataProcessing/data_analysis/graph/createGraph.py`, `DataProcessing/data_analysis/graph/prepare_for_admin_import.py` | Load merged CSVs into Neo4j for interactive graph visualization and Cypher exploration. |
| Main model | `Model/unified_model.py`, `Model/model.py` | Default shared four-output sparse model and the legacy compatibility model. |
| Baselines | `Model/gcn_model.py`, `Model/gat_model.py`, `Model/transformer_model.py` | Alternative GCN, GAT, and Transformer-only architectures. |
| Training | `Model/unified_training.py`, `Model/sampling.py`, `Model/train_test_single.py`, `Model/run_training.sh`, `Model/graph_splitter.py` | Target-edge sampled training, fixed evaluation, hard-negative mining, splits, and graph caching. |
| Prediction | `Model/predict.py`, `Model/run_predict.sh`, `Model/prepare_marker_edge_batches.py` | Disease-to-biomarker prediction, MONDO-ID batch mode, marker-edge batch mode, and explainability output. |
| Result export | `Model/ResultsProcessing/export_prediction_results_to_csv.py` | Convert prediction logs into CSV summaries. |

## Knowledge Graph Data Model

### Node CSV schema

All node CSV files use this schema:

```text
ID,Type,Name,Sequence,Comments
```

- `ID`: Stable external identifier, such as `MONDO:...`, `NCBIGene:...`, `Uniprot:...`, `HMDB...`, or a database-specific identifier.
- `Type`: Canonical node type used by the model, such as `disease`, `gene`, `rna`, `protein`, `metabolite`, `drug`, `phenotype`, `pathway`, `molecular_function`, `biological_process`, `cellular_component`, `anatomy`, or `variant`.
- `Name`: Human-readable display name.
- `Sequence`: Biological sequence, SMILES string, or an empty value when no sequence-like feature is available.
- `Comments`: Free-text annotation used for text features and provenance.

### Edge CSV schema

All edge CSV files use this schema:

```text
Node 1 ID,Node 1 Data Type,Node 1 Source,Edge Type,Edge Direction,Edge Weight,Node 2 ID,Node 2 Data Type,Node 2 Source
```

- `Edge Type` records the relation family, such as `gene_protein`, `drug_disease`, `protein_pathway`, or `phenotype_disease`.
- `Edge Direction` uses the project constants in `DataProcessing/kg_construction/commonSettings.py`: `Undirected`, `SingleDirected`, `DoubleDirected`, or `Equal`.
- `Edge Weight` stores relation confidence or relation strength. Evidence-derived weights are used where available.

### Marker edge CSV

`marker_edge.csv` is a special disease-marker edge file. During merge and prediction it is treated as a high-priority source for known disease-marker relationships. The prediction pipeline can also split it into per-target batches and force-score known disease-marker pairs.

## Installation

The model setup has been tested around Python 3.8, PyTorch, DGL, RDKit, Hugging Face Transformers, and pandas. Use a CUDA-compatible PyTorch and DGL pair when training on GPU.

```bash
conda create -n bio_env python=3.8
conda activate bio_env

# Example CUDA 11.7 setup. Adjust for your CUDA version.
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install a DGL wheel compatible with your CUDA/PyTorch stack.
pip install dgl

# Shared model and data-processing dependencies.
pip install -r Model/requirements.txt
pip install rdkit biopython obonet requests sentencepiece rdkit-pypi neo4j psutil huggingface_hub
```

Optional: pre-download Hugging Face models for offline or cluster execution:

```bash
cd Model
python download_models.py --path ./models_cache
```

## Build the Knowledge Graph

### 1. Prepare raw source data

Place raw datasets under the data root expected by `DataProcessing/kg_construction/ExtractData.sh`, or edit the path variables near the top of that script. See `DataProcessing/kg_construction/README.md` for the KG construction data-processing script inventory.

Current defaults are host-aware:

- GPU-node path: `/home/$USER/dataset`
- Generated source-specific CSV folder: `graphWithNodeType`
- Merged CSV folder: `graphWithNodeType/merge`

The extraction scripts cover resources including MONDO, HPO, MarkerDB, HMDB, DrugCentral, SIDER, Reactome, DrugBank, GO, UniProt, NCBI Gene, RefSeq, ClinVar, PubChem, and Uberon. Some upstream datasets require registration or manual download.

### 2. Extract source-specific node and edge CSV files

Run all currently configured extraction stages:

```bash
cd DataProcessing/kg_construction
bash ExtractData.sh all
```

Run a single source when you only need one stage:

```bash
cd DataProcessing/kg_construction
bash ExtractData.sh mondo
bash ExtractData.sh hpo
bash ExtractData.sh marker
```

The configured `all` pipeline currently keeps source order for cross-source dependencies, such as MONDO mappings used by drug and marker extractors.

### 3. Merge source-specific files into the model graph

Use the wrapper:

```bash
cd DataProcessing/kg_construction
bash ExtractData.sh merge
```

Or call the merge script directly from the repository root:

```bash
python DataProcessing/kg_construction/mergeData.py \
  -n "/path/to/graphWithNodeType/*_node.csv" \
  -e "/path/to/graphWithNodeType/*_edge.csv" \
  --marker-edge-file "/path/to/graphWithNodeType/marker_edge.csv" \
  --out-nodes "/path/to/graphWithNodeType/merge/merged_node.csv" \
  --out-edges "/path/to/graphWithNodeType/merge/merged_edge.csv"
```

Expected outputs:

```text
merged_node.csv
merged_edge.csv
```

### 4. Optional: generate node feature CSVs

The DGL model can generate and cache features during training. If you want a standalone feature CSV, run:

```bash
python DataProcessing/kg_construction/generateNodeEmbedding.py \
  -n /path/to/merged_node.csv \
  --out-nodes /path/to/merged_node_features.csv \
  --generate-features \
  --embed-size 128 \
  --local-model-path ~/.cache/huggingface/hub \
  --device cuda:0
```

### 5. Optional: visualize the graph in Neo4j

The Neo4j helpers live under `DataProcessing/data_analysis/graph/` because they support KG inspection and visualization after the graph has already been built.

For driver-based import:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
python DataProcessing/data_analysis/graph/createGraph.py /path/to/merged_node.csv /path/to/merged_edge.csv
```

For `neo4j-admin` import, first prepare import files:

```bash
python DataProcessing/data_analysis/graph/prepare_for_admin_import.py \
  /path/to/merged_node.csv \
  /path/to/merged_edge.csv \
  DataProcessing/data_analysis/graph/admin_import_files
```

Then adapt the `neo4j-admin database import full` command in `DataProcessing/data_analysis/graph/command.sh` to your Neo4j installation. After import, use Neo4j Browser or Bloom to inspect node neighborhoods, relation types, and disease-centered subgraphs.

### 6. Optional: summarize KG data statistics

Use the analysis scripts to inspect the generated KG before model training or visualization. For a quick node and edge type summary:

```bash
python DataProcessing/data_analysis/count.py /path/to/graphWithNodeType/merge
```

For disease-centered coverage and marker-connectivity summaries, see `DataProcessing/data_analysis/analyze_disease_marker_connectivity.py` and `DataProcessing/data_analysis/analyze_mondo_node_enrichment.py`.

## Model Overview

BioGT-Predictor is implemented as a DGL-based heterogeneous graph link-prediction pipeline over BioGT-KG. The main model combines:

- Heterogeneous graph neural-network message passing for local biomedical graph structure.
- Transformer-style global feature fusion for broader cross-node interactions.
- Mixture-of-experts style relation predictors for target edge types.
- Optional explainability output through fusion diagnostics and graph-path summaries.

Supported model choices:

| `--model-type` | Implementation | Use case |
| --- | --- | --- |
| `original` | `Model/model.py` | Main BioGT-Predictor model with HGCN plus Transformer fusion. |
| `gcn` | `Model/gcn_model.py` | Lightweight heterogeneous GCN baseline. |
| `gat` | `Model/gat_model.py` | Attention-based heterogeneous graph baseline. |
| `transformer` | `Model/transformer_model.py` | Transformer-only baseline. |

The four primary prediction targets in the training scripts are:

```text
gene,rna,protein,metabolite
```

The article discusses a DNA/gene-like molecular target; in this codebase the corresponding target type is `gene`.

## Train Models

### Direct single-target training

The most explicit entry point is `Model/train_test_single.py`:

```bash
cd Model

TARGET=protein
DATA_DIR=../Data
RUN_DIR="./experiments/target-${TARGET}_split-random_fusion-weighted_hops-10"
mkdir -p "$RUN_DIR/cache"

python train_test_single.py \
  --nodes_csv "$DATA_DIR/merged_node.csv" \
  --edges_csv "$DATA_DIR/merged_edge.csv" \
  --gpu 0 \
  --model-type original \
  --epochs 30000 \
  --lr 0.0001 \
  --val_split 0.1 \
  --test_split 0.1 \
  --default_embed_size 64 \
  --hgcn_hidden 64 \
  --hgcn_out 64 \
  --transformer_heads 16 \
  --num_transformer_layers 3 \
  --num_gnn_layers 3 \
  --transformer_ff 512 \
  --num_latents 32 \
  --dropout 0.1 \
  --k_neg_samples 1 \
  --early_stopping_patience 100 \
  --early_stopping_metrics "train_loss,val_loss" \
  --train_target_types "$TARGET" \
  --eval_input_name "psoriasis" \
  --eval_input_type "disease" \
  --eval_output_types "$TARGET" \
  --fusion-mode weighted \
  --disable-sequence-features \
  --save_model_path "$RUN_DIR/model_final_best.pt" \
  --graph_cache_path "$RUN_DIR/cache"
```

Train all four biomarker targets:

```bash
cd Model

for TARGET in gene rna protein metabolite; do
  DATA_DIR=../Data
  RUN_DIR="./experiments/target-${TARGET}_split-random_fusion-weighted_hops-10"
  mkdir -p "$RUN_DIR/cache"

  python train_test_single.py \
    --nodes_csv "$DATA_DIR/merged_node.csv" \
    --edges_csv "$DATA_DIR/merged_edge.csv" \
    --gpu 0 \
    --model-type original \
    --epochs 30000 \
    --lr 0.0001 \
    --val_split 0.1 \
    --test_split 0.1 \
    --default_embed_size 64 \
    --hgcn_hidden 64 \
    --hgcn_out 64 \
    --transformer_heads 16 \
    --num_transformer_layers 3 \
    --num_gnn_layers 3 \
    --transformer_ff 512 \
    --num_latents 32 \
    --dropout 0.1 \
    --k_neg_samples 1 \
    --early_stopping_patience 100 \
    --early_stopping_metrics "train_loss,val_loss" \
    --train_target_types "$TARGET" \
    --eval_input_name "psoriasis" \
    --eval_input_type "disease" \
    --eval_output_types "$TARGET" \
    --fusion-mode weighted \
    --disable-sequence-features \
    --save_model_path "$RUN_DIR/model_final_best.pt" \
    --graph_cache_path "$RUN_DIR/cache"
done
```

The graph cache must be reused for prediction with the same node and edge CSV files. If you change the graph, remove the old cache or use a new cache directory.

### Batch training wrapper

`Model/run_training.sh` wraps repeated experiments over target types, model types, fusion modes, split strategies, and optional ablations.

Before using it, edit the configuration block at the top of the script:

- `train_target_set`
- `model_type_set`
- `fusion_mode_set`
- `split_method_set`
- `data_path`
- `output_root`
- `local_model_path`

Then run:

```bash
cd Model
bash run_training.sh
```

The wrapper is convenient for cluster runs, but its output directory naming may differ from `run_predict.sh`. Either keep the direct-training layout above or update `build_model_layout()` in `run_predict.sh` to match your experiment folders.

## Run Predictions

### Direct prediction with `predict.py`

Use this form when you know the exact model file and graph cache:

```bash
cd Model

python -u predict.py \
  --load_model_path ./experiments/target-protein_split-random_fusion-weighted_hops-10/model_final_best.pt \
  --nodes_csv ../Data/merged_node.csv \
  --edges_csv ../Data/merged_edge.csv \
  --graph_cache_path ./experiments/target-protein_split-random_fusion-weighted_hops-10/cache \
  --local_model_path ./models_cache \
  --eval_input_names "Alzheimer's disease,Parkinson's disease" \
  --eval_input_type disease \
  --eval_output_types protein \
  --eval_top_k 30 \
  --gpu 0 \
  --default_embed_size 64 \
  --hgcn_hidden 64 \
  --hgcn_out 64 \
  --num_transformer_layers 3 \
  --num_gnn_layers 3 \
  --transformer_heads 16 \
  --transformer_ff 512 \
  --num_latents 32 \
  --dropout 0.1 \
  --fusion-mode weighted \
  --disable-sequence-features
```

Prediction input modes:

| Mode | Argument | Example |
| --- | --- | --- |
| Disease names | `--eval_input_names` | `"diabetes,psoriasis"` |
| MONDO disease CSV | `--eval_input_mondo_csv` | `../Data/mondo_node.csv` |
| MONDO ID list | `--eval_input_ids` | `MONDO:0010796,MONDO:0005086` |
| Disease-marker pair CSV | `--eval_marker_pairs_csv` | `prediction_logs/marker_edge_cache/.../marker_pairs_gene.csv` |

### Wrapper prediction with `run_predict.sh`

Default wrapper usage:

```bash
cd Model
./run_predict.sh original protein "Alzheimer's disease" ./experiments 30 ../Data ./models_cache
```

The wrapper expects each target model at:

```text
./experiments/target-<target>_split-random_fusion-weighted_hops-10/model_final_best.pt
```

Batch prediction over a MONDO disease CSV:

```bash
cd Model
./run_predict.sh original protein "" ./experiments 30 ../Data ./models_cache ../Data/mondo_node.csv
```

Prediction by direct MONDO IDs:

```bash
cd Model
./run_predict.sh original metabolite "" ./experiments 30 ../Data ./models_cache "MONDO:0010796,MONDO:0005086"
```

Marker-edge batch mode:

```bash
cd Model
./run_predict.sh original gene "" ./experiments 30 ../Data ./models_cache "" false true ../Data/marker_edge.csv
```

In marker-edge mode, `prepare_marker_edge_batches.py` extracts unique MONDO disease IDs and per-marker-type disease-marker pair CSV files. The pipeline then runs a separate prediction batch for each marker type.

### Export prediction logs to CSV

```bash
cd Model
python ResultsProcessing/export_prediction_results_to_csv.py \
  --input-log prediction_logs/predict_protein_mondo_batch_YYYYMMDD_HHMMSS.log \
  --output-csv prediction_results.csv
```

## Reproduce Article-Style Evaluation

The article evaluates disease-centered biomarker link prediction with metrics such as AUC-PRC, AUC-ROC, accuracy, and F1. To reproduce the same style of evaluation:

1. Build or obtain `merged_node.csv` and `merged_edge.csv`.
2. Train one target-specific model for each of `gene`, `rna`, `protein`, and `metabolite`.
3. Use a consistent split strategy across targets, such as `random`, `disease-centric`, `kfold`, or `specific_ids`.
4. Keep model hyperparameters aligned across targets unless performing an ablation.
5. Read the training logs for validation and test metrics.
6. Run prediction on held-out disease nodes or MONDO disease lists.
7. Export prediction logs to CSV for downstream analysis.

Useful split options in `train_test_single.py` include:

| Split strategy | Key arguments | Use case |
| --- | --- | --- |
| `random` | `--val_split`, `--test_split` | General edge-level validation. |
| `disease-centric` | `--split-disease-node-type disease` | Hold out disease-centered neighborhoods. |
| `kfold` | `--split-kfolds`, `--split-kfold-index` | Cross-validation. |
| `specific_ids` | `--split-specific-ids-path` | Hold out selected disease IDs or category files. |
| `target_exclude` | Target-specific options in the splitter | Stress-test target generalization. |

## Concept-to-Code Mapping

| Article concept | Code location | Notes |
| --- | --- | --- |
| Biomedical KG construction from curated resources | `DataProcessing/kg_construction/extract*.py`, `DataProcessing/kg_construction/ExtractData.sh` | Source-specific extractors create standardized CSV files. |
| KG node and edge normalization | `DataProcessing/kg_construction/commonSettings.py`, `DataProcessing/kg_construction/mergeData.py` | Canonical node types and edge schemas are enforced before model training. |
| MarkerDB and known marker relations | `DataProcessing/kg_construction/extractBioMarkerAndMerge.py`, `Model/marker_edge.csv` | Marker edges can be prioritized during merge and force-scored during prediction. |
| KG data statistics | `DataProcessing/data_analysis/count.py`, `DataProcessing/data_analysis/analyze_mondo_node_enrichment.py` | Summarizes merged graph composition and disease-level graph coverage. |
| Neo4j KG visualization | `DataProcessing/data_analysis/graph/createGraph.py`, `DataProcessing/data_analysis/graph/prepare_for_admin_import.py` | Imports merged KG CSV files into Neo4j for visual browsing and Cypher queries. |
| Disease-centered link prediction | `Model/train_test_single.py`, `Model/modelEvaluator.py`, `Model/predict.py` | Trains and evaluates disease-to-target association models. |
| Main BioGT-Predictor architecture | `Model/model.py` | HGCN plus Transformer fusion, relation predictors, and explainability hooks. |
| Baseline architectures | `Model/gcn_model.py`, `Model/gat_model.py`, `Model/transformer_model.py` | Alternative model families for comparison. |
| Split and holdout protocols | `Model/graph_splitter.py`, `Model/prepare_category_splits.py` | Random, category, k-fold, disease-centric, and specific-ID splits. |
| Prediction explainability | `Model/modelEvaluator.py` | Weighted shortest-path and gradient-guided path summaries. |
| Batch disease and marker prediction | `Model/run_predict.sh`, `Model/prepare_marker_edge_batches.py` | MONDO CSV, MONDO ID, and marker-edge driven batch modes. |

## Troubleshooting

### `Data/` is empty

The expected state is to have released files under `Data/`. If the directory is empty in your environment, either populate it with your BioGT-KG data package or build the KG with `DataProcessing/kg_construction/` and place `merged_node.csv`, `merged_edge.csv`, `marker_edge.csv`, and target-specific weights under `Data/`.

### Extraction scripts cannot find raw data

Edit the path variables in `DataProcessing/kg_construction/ExtractData.sh`, especially `data_path_prefix`, `output_path_prefix`, and `merged_data_path_prefix`. The script currently contains host-specific defaults.

### Prediction cannot find a model file

Check the expected layout used by `Model/run_predict.sh`:

```text
experiments/target-<target>_split-random_fusion-weighted_hops-10/model_final_best.pt
```

If your training run saved a different file name or directory, either move the model, pass the path directly to `predict.py`, or update `build_model_layout()` in `run_predict.sh`.

### DGL or PyTorch CUDA errors

Install PyTorch and DGL versions built for the same CUDA runtime. If GPU execution is not available, use `--gpu -1` for CPU execution, though large graphs may be slow.

### Hugging Face model downloads fail on a cluster

Run `python Model/download_models.py --path ./Model/models_cache` on a machine with internet access, then copy the cache to the cluster and pass `--local_model_path ./models_cache`.

### Stale graph cache after changing CSV files

Delete the old cache directory or choose a new `--graph_cache_path`. The cached DGL graph must match the node and edge CSV files used for training or prediction.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).

Please also check upstream data-source licenses before redistributing derived KG CSV files or trained weights.
