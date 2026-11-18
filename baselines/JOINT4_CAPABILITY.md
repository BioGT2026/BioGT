# Joint Four-Target Baseline Capability

All retained baselines support one joint BioGT-KG input and one run that writes
separate protein, gene, RNA, and metabolite ranking results.

| Baseline | Joint execution | Implementation | Fairness treatment |
|---|---|---|---|
| PPR/RWR | One graph propagation | Local sparse PPR | Relation-scoped filtered candidate ranking |
| TransE | One training run | Pinned upstream KGE | Relation-balanced batches and relation-scoped train-only filtered negatives |
| DistMult | One training run | Pinned upstream KGE | Relation-balanced batches and relation-scoped train-only filtered negatives |
| ComplEx | One training run | Pinned upstream KGE | Relation-balanced batches and relation-scoped train-only filtered negatives |
| RotatE | One training run | Pinned upstream KGE | Relation-balanced batches and relation-scoped train-only filtered negatives |
| R-GCN | One training run | Local PyTorch link-prediction adaptation | Inverse message edges, relation-balanced train-only filtered negatives |
| CompGCN | One training run | Pinned upstream CompGCN | Forward task loss only, typed split-blind BCE candidate mask |
| HGT | One training run | PyG `HGTConv` | Typed graph, inverse message edges, relation-balanced train-only filtered negatives |
| NBFNet | One training run | Official NBFNet-PyG | Inverse message edges, official Bellman-Ford layers, typed train-only filtered negatives |

The previous local `ScalarNBFNet` and hand-written `HGT-style` implementations
are not used for paper results. They were replaced because they were not faithful
official implementations.

Every metrics file contains:

- `metrics_by_target`: the four independently filtered target results.
- `metrics_macro`: equal-weight average over the four target types.
- `metrics`: query-level micro average, retained for diagnostics only.

BioGT-versus-baseline gates use per-target metrics, not the gene-dominated micro
average.

## Split protocol

- Training graphs, positive labels, and negative-sampling filters are train-only.
- Validation selects PPR alpha or a trainable model checkpoint; it never
  contributes gradients.
- The selected or final checkpoint is frozen before test evaluation.
- Final metrics contain test queries only. Filtered test ranking removes known
  train/validation positives while deliberately ignoring every other test
  label.
- Entity dictionaries and candidate lists are transductive public vocabulary;
  they expose node identity, not held-out association labels.
