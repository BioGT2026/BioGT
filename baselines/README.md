# BioGT Baseline Integration

This directory vendors or wraps the nine baseline models selected for the
BioGT experiments:

1. PPR/RWR graph propagation
2. TransE
3. DistMult
4. ComplEx
5. RotatE
6. R-GCN
7. CompGCN
8. HGT
9. NBFNet

The current manuscript repository does not contain the full BioGT training
pipeline, so this package uses a neutral triple/split interface. Export BioGT-KG
edges and experiment splits from the model code, then convert them with
`scripts/prepare_biogt_baseline_data.py`.

## External source layout

- `external/KnowledgeGraphEmbedding`: MIT upstream implementation for TransE,
  DistMult, ComplEx, and RotatE.
- `external/CompGCN`: Apache-2.0 upstream implementation for CompGCN. The local
  copy is patched to preserve BioGT entity/relation case instead of lowercasing
  IDs.
- `external/NBFNet-PyG`: official author-maintained PyG implementation.
- `external/relational-gcn`: MIT original R-GCN reference code. It is retained
  for provenance, but its old Keras/Theano dependency stack and entity
  classification focus make it unsuitable as the main BioGT link-prediction
  runner.

See `sources.tsv` for exact URLs and licenses.

Run `python baselines/scripts/download_baselines.py` to recreate the pinned
external sources and apply the tracked BioGT patches.

## Expected input

Prepare a TSV/CSV edge file with at least these columns:

- source entity ID
- relation ID
- target entity ID
- optional split column with values such as `train`, `valid`, `test`

Example:

```powershell
python baselines/scripts/prepare_biogt_baseline_data.py `
  --edges exported_biogt_edges.tsv `
  --dataset biogt_edge_masked `
  --source-col source_id `
  --relation-col relation_type `
  --target-col target_id `
  --split-col split `
  --install-external
```

If no split column is present, the script creates deterministic train/valid/test
splits with `--valid-frac`, `--test-frac`, and `--seed`.

## Run all baselines once for four outputs

The joint runner uses the four exact transductive datasets to create one
multi-relation training input. Each model is trained or executed once and writes
protein, gene, RNA, and metabolite results:

```powershell
python baselines/scripts/run_all_capped_baselines.py --gpu 0
```

See `JOINT4_CAPABILITY.md` for implementation provenance and fairness rules.

## Run individual baselines

KGE models:

```powershell
python baselines/scripts/run_kge_baseline.py --dataset biogt_edge_masked --model TransE --cuda --gpu 0
python baselines/scripts/run_kge_baseline.py --dataset biogt_edge_masked --model DistMult --cuda --gpu 0
python baselines/scripts/run_kge_baseline.py --dataset biogt_edge_masked --model ComplEx --cuda --gpu 0
python baselines/scripts/run_kge_baseline.py --dataset biogt_edge_masked --model RotatE --cuda --gpu 0
```

CompGCN:

```powershell
python baselines/scripts/run_compgcn_baseline.py --dataset biogt_edge_masked --gpu 0
```

PPR/RWR:

```powershell
python baselines/scripts/run_ppr_baseline.py --dataset biogt_edge_masked --alpha 0.85
```

R-GCN:

```powershell
python baselines/scripts/run_rgcn_baseline.py --dataset biogt_edge_masked --cuda --gpu 0
```

Use the same split files for all baselines. Report ranking metrics on the
filtered test triples: MRR, MR, Hits@K, Recall@K, and NDCG@K. The local PPR/RWR
and R-GCN wrappers use `K={1,3,10,50,100}` by default. For the manuscript's
candidate-prioritization task, separately rank all eligible candidates in each
molecular layer.
