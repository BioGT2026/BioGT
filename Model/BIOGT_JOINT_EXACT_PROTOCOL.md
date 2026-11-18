# BioGT Joint Exact Protocol

## Purpose

This protocol trains one shared BioGT model and reports protein, gene, RNA,
and metabolite ranking results from the same run. It uses the same seed-42
random split and relation-scoped candidate sets as the joint capped baseline
suite.

## Fairness Contract

- One split: seed 42, validation fraction 0.0625, test fraction 0.0625.
- One shared four-output model.
- Training target edges prefer the canonical disease-to-target orientation so
  the pair scorer uses the same source/tail order as baseline ranking.
- Validation selects checkpoints and all tuning decisions.
- After reloading the selected checkpoint, validation-only runs evaluate that
  checkpoint on validation again; the test split remains untouched.
- The held-out test split is evaluated once after the final configuration is
  frozen.
- Ranking is full, filtered, relation-scoped, and uses average tie ranks.
- Training, hard-negative mining, and sampled validation draw negatives from
  the same relation-scoped candidates used by full ranking, rather than every
  node sharing the broad biological type.
- Validation filtering uses training positives only; final test filtering uses
  all known positives, matching the baseline test protocol.
- Required metrics are MRR, Hits@10, Recall@50, and NDCG@50, per target and
  as an equal-target macro average.
- Node and split caches must come from the exact transductive v2 export.
- Before smoke, validation, or final test, `verify_joint_biogt_cache.py`
  compares split counts, candidate counts, and canonical pair overlap against
  the joint baseline summary. It also checks feature-cache schema v2 plus
  per-node-type embedding counts and dimensions, then writes
  `cache_protocol_audit.json`.

## CUDA Backend Change

The remote DGL 2.1.0 installation has a CPU-only core, while PyTorch CUDA 13
supports the RTX 5090. The unified model therefore supports
`--unified-graph-backend torch`:

- DGL loads and samples compact context graphs on CPU.
- Relation edge indices are transferred per batch.
- Relation-normalized graph convolution, Transformer layers, modality
  adapters, MoE heads, node embeddings, and loss computation execute in
  PyTorch on CUDA.
- The original DGL graph-convolution backend remains available for CPU
  regression tests.

## Default Validation Configuration

| Setting | Value |
| --- | --- |
| Hidden/output dimension | 64 |
| GNN layers | 3 |
| Transformer layers | 3 |
| Transformer heads | 8 |
| Transformer FF dimension | 256 |
| MoE experts / active experts | 4 / 2 |
| Batch size | 32 |
| Validation batch size | 32 |
| Steps per epoch | 112 |
| Maximum epochs | 100 |
| Negative samples | 16 |
| Ranking loss weight | 0.3 |
| Neighbor fanouts | 5,3,2 |
| Validation interval | 5 epochs |
| Checkpoint metric | four-target macro full-ranking MRR |
| Test during training | disabled |

Any later architecture or optimizer changes must use a new named run
directory under `Model/experiments_biogt_joint_exact_v3/` and be
recorded beside its validation metrics. Test metrics must not be used to
choose the next configuration.

The first attempted `batch=128, fanouts=15,10,5` configuration exposed a
context-selection bug: when query seeds exceeded the sparse-transformer top-k,
the zero remaining budget selected every non-seed context node. The corrected
model keeps all query seeds and selects no additional non-seeds in that case.
The failed configuration remains archived as an OOM record and is not a
performance result.

All four targets are trained and reported. Model selection uses the exact
four-target validation macro MRR, then NDCG@50, Hits@10, and Recall@50. Every
target must have at least one valid ranking query, and the selector verifies
that the reported macro equals the arithmetic mean of protein, gene, RNA, and
metabolite metrics. RNA remains explicitly disclosed as a two-query,
low-confidence component rather than being silently removed from the primary
objective.

KGE, global bilinear, PPR, and NBFNet are independent baselines or disclosed
ablations. A run eligible for pure BioGT selection must use `concat_mlp` or
`interaction_mlp`, must not load a `hybrid_kge` checkpoint, and must not add
baseline scores, embeddings, or post-hoc structural masking. The optional
BioGT path branch is built only from the training graph and uses BioGT-owned
node features and trainable gates. When upgrading a frozen pure BioGT
checkpoint, `--freeze-biogt-base-for-path-finetune` must be used so that only
the declared target path gates receive updates. Gate-only runs use
`--biogt-path-pretrain-only` to proceed directly from exact full-candidate
pretraining to validation and checkpoint saving without an unrelated sampled
main-training update.

The default-off neural diffusion extension uses
`--biogt-neural-path-dim`. It remains a pure BioGT component: a BioGT disease
query embedding is the boundary condition, relation-specific transformations
propagate vector states over compact train-only disease-target adjacencies,
and BioGT candidate embeddings provide the readout. For the predicted output
modality, every query's own train edges are masked during propagation and
readout so a positive edge cannot become a path shortcut. Exact candidate
universes define compact matrix columns but validation or test edges never
enter the propagation topology. A strict migration uses
`--initialize-biogt-neural-path-from-checkpoint`; only neural-diffusion keys
may be missing, and candidate projections start at zero contribution so the
parent ranking is preserved before training. Neural listwise training
normalizes batch losses so each target modality has equal total weight.

The default-off relational Bellman-Ford extension uses
`--biogt-relbf-dim` and remains a pure BioGT component. It applies three
relation-conditioned message-passing layers with PNA aggregation over only
the four disease-target training relations and their inverse directions.
BioGT disease embeddings define query boundary states, BioGT candidate
embeddings participate in the readout, and no KGE or baseline checkpoint,
embedding, or score is permitted. During train-only listwise optimization,
all forward and inverse edges on the predicted output relation incident to
the current disease query are dynamically removed from both the frozen scalar
path parent and the RelBF residual. Validation uses the complete training
graph and filtered known positives. Strict zero-residual migration uses
`--initialize-biogt-relbf-from-checkpoint`; only `biogt_relbf_*` keys may be
missing, the final residual readout starts at zero, and
`--freeze-biogt-base-for-path-finetune` leaves only the disclosed RelBF
parameters trainable. Because the three PNA propagation layers are shared,
RelBF training must include all four target modalities; selective target-only
fine-tuning is rejected rather than silently changing excluded modalities.
After a joint RelBF checkpoint exists, the default-off
`--biogt-relbf-target-only-finetune` control may instead freeze the shared PNA
stack and train only the target-specific projections, FiLM, relation token,
and readout selected by `--biogt-path-pretrain-types`. This continuation cannot
change excluded target scores and remains train-only/validation-selected.

The default-off `--biogt-relbf-edge-attention` extension makes RelBF message
weights query-conditioned and relation-aware without replacing the
Transformer+HGCN streams. Each output relation, support relation, and RelBF
layer owns a small query projection. Its dot product with the normalized
source message produces a `2 * sigmoid(logit)` edge weight, so a zero
projection is exactly the identity weight one. Strict migration from an
existing pure RelBF checkpoint uses
`--initialize-biogt-relbf-edge-attention-from-checkpoint`; only the new
attention projection keys may be missing and all are reset to zero. With
`--biogt-relbf-edge-attention-only-finetune`, the Transformer, HGCN, scalar
Path5, parent RelBF, and excluded output-relation attention parameters remain
frozen. The existing dynamic removal of every current-query output edge is
applied before aggregation, including both propagation directions. Pair and
full-candidate scoring call the same RelBF routine. Exported graph and
checkpoint metadata record the attention mode, identity initialization,
support modalities, and train-only masking policy. No baseline artifact or
held-out edge participates.

The default-off `--biogt-relbf-deep-refinement` extension appends two
BioGT-owned relational Bellman-Ford layers after the three-layer RelBF output.
Layer four aggregates relation-specific target states back to disease nodes
with PNA; layer five propagates the refined disease state to the requested
relation's exact candidate universe. Every output modality owns its deep
message transforms, PNA updates, FiLM conditioning, normalization, and
residual readout, allowing one modality to be tuned without changing another.
The final residual readout is zero initialized. Strict migration with
`--initialize-biogt-relbf-deep-from-checkpoint` permits only
`biogt_relbf_deep_*` keys and therefore preserves pair and full-candidate
scores exactly. With `--biogt-relbf-deep-only-finetune`, Transformer, HGCN,
scalar Path5, and the complete parent three-layer RelBF stay frozen. Training
masks both directions of the current train query relation, while evaluation
uses only the complete train graph. Metadata records five layers, the
`target_to_disease_to_target` refinement path, relation-specific candidates,
and zero-residual migration. No baseline artifact or test edge participates.

The default-off `--biogt-relbf-indicator-residual` extension adds a separate
three-layer relation-only Bellman-Ford/PNA residual beside the existing
disease-conditioned RelBF score. Its source boundary is the requested target
relation token written only at the current query disease; it never uses a
disease-ID embedding. Propagation uses only the audited train graph, masks both
directions of the current query relation during training, and scores only the
declared relation-specific candidate list. Pair and exact full-candidate
scoring call the same residual implementation. Exported metadata labels this
branch `relation_only_query_conditioned_bellman_ford_pna_v1` and the parent
path `disease_conditioned_three_layer_relbf` for interpretation.

Strict migration requires
`--initialize-biogt-relbf-indicator-from-checkpoint`; only
`biogt_relbf_indicator_*` keys may be absent, and final residual readouts are
reset to zero so parent scores remain unchanged up to floating-point
tolerance. `--biogt-relbf-indicator-only-finetune` freezes Transformer, HGCN,
Path5, parent RelBF, and excluded target branches. This branch cannot combine
with the frozen JK, edge-attention, or deep-refinement axes and does not load
KGE, PPR, NBFNet, or test artifacts.

The stricter default-off `--biogt-relbf-indicator-support-gating` extension
requires both the indicator residual and a positive RelBF dimension. For every
output modality, support modality, and indicator layer it applies
`2 * sigmoid(Linear(parent_condition))` to messages. Supports use the stable
sorted path-support order. All gate weights and biases are zero initialized,
so `--initialize-biogt-relbf-indicator-support-gating-from-checkpoint` permits
only the new gate keys and preserves scores exactly. The matching
`--biogt-relbf-indicator-support-gate-only-finetune` mode freezes Transformer,
HGCN, Path5, the disease-conditioned RelBF, and the full indicator propagation
and readout branch; only gates for explicitly selected target modalities train.

The default-off `--biogt-relbf-indicator-semantic-readout` adds a pure BioGT
residual scorer over the train-only relation-indicator path state, candidate
BioGT state, and frozen disease-conditioned parent RelBF condition. Its input
is `[indicator, indicator*candidate, indicator*condition,
candidate*condition]`; the `4d -> d -> 1` readout has an exactly zero final
layer, so strict migration preserves every parent score. The branch requires
the indicator residual, is mutually exclusive with support gating, and its
semantic-only fine-tune freezes Transformer/HGCN, Path5, parent RelBF,
indicator propagation, and the original indicator readout. It consumes no
KGE, PPR, NBFNet, test, or other baseline artifact.

The default-off `--biogt-relbf-query-relation-adapter-rank R` adds a pure
BioGT low-rank query-gated delta to each existing RelBF output-relation,
support-relation, and layer message transform. `R=0` creates no parameters and
does not enter the scoring path, preserving legacy checkpoint output exactly.
For `R>0`, the message and query are projected into rank `R`, multiplied, and
projected back through an exactly zero-initialized output map. Strict migration
therefore permits only `biogt_relbf_query_relation_adapters.*` missing keys and
preserves parent scores. Adapter-only fine-tuning freezes Transformer, HGCN,
Path5, parent RelBF, and indicator parameters. It reuses the disclosed
`train_graph_only` compact graph and dynamic query-edge masking, and records
relation/support/layer attribution metadata in `biogt_path_graph`.

The default-off `--biogt-relbf-indicator-query-boundary-rank R` extends the
indicator source boundary with a zero-output low-rank residual conditioned on
the detached parent RelBF query condition. Rank zero is a strict no-op. The
delta is injected only at the query disease source and then follows the
existing three-layer train-only indicator propagation, preserving path
attribution. Checkpoint migration permits only
`biogt_relbf_indicator_query_boundary_adapters.*` missing keys, and the
dedicated adapter-only freeze mode leaves HGCN, Transformer, Path5, parent
RelBF, and the parent indicator branch frozen.

The default-off indicator multi-rule residual is enabled only when both
`--biogt-relbf-indicator-multirule-count H` and
`--biogt-relbf-indicator-multirule-readout-rank R` are positive. Each rule has
an independent relation token injected only at the query disease source, then
reuses the frozen three-layer `train_graph_only` indicator message transforms,
PNA updates, and norms in a memory-bounded per-rule loop. Independent low-rank
readouts produce rule residuals; a softmax gate over the detached parent RelBF
condition mixes them. Every final rule readout is exactly zero-initialized, so
the default, `H=0`, `R=0`, and migrated model are score-identical to the parent.
Metadata records rule labels, gate semantics, source, and per-rule path
attribution. Strict migration permits only
`biogt_relbf_indicator_multirule_residuals.*` keys. Multi-rule-only fine-tuning
unfreezes only the selected target's rule tokens, gate, and readouts while
freezing HGCN, Transformer, Path5, parent RelBF, and the parent indicator.

The default-off relation-transition residual is enabled only when
`--biogt-relbf-indicator-relation-transition-layers L` is 1 or 2 and
`--biogt-relbf-indicator-relation-transition-rank R` is positive. It derives
a sparse directed relation graph exclusively from the compact training graph.
Its nodes are the forward and inverse target-relation directions; an edge
exists when two training relations form a directed two-hop walk through a
shared middle node. Row-normalized walk counts are rebuilt at runtime and are
never checkpoint data. The detached target-relation query condition is placed
at the target relation boundary and propagated over this graph. A low-rank,
zero-output readout produces query-specific multiplicative weights for the
existing indicator RelBF/PNA messages. The initial multiplier is exactly one,
so strict migration permits only
`biogt_relbf_indicator_relation_transition_residuals.*` keys and preserves the
parent score bit-for-bit. Metadata exports relation labels, sparse transitions,
training-walk counts, normalization, boundary relation, and the latest mean
query-conditioned weights. The dedicated Gene freeze mode trains only this
module while preserving HGCN, Transformer, Path5, parent RelBF, the parent
indicator, and its explainable path states.

The mutually exclusive default-off query-local relation-transition residual is
enabled only when
`--biogt-relbf-indicator-query-local-relation-transition-layers L` is 1 or 2
and `--biogt-relbf-indicator-query-local-relation-transition-rank R` is
positive. It retains the row-normalized global relation-transition graph built
only from the compact training graph, but derives a separate support-modality
incidence vector for every query disease. Incidence is stabilized with
`log1p`, normalized by the per-query relation mean, and converted to bounded
relation gates that modulate each propagation layer. During training, the
target output relation's forward and inverse local incidence are zeroed under
the same dynamic query-relation policy used by indicator RelBF/PNA, preventing
the current positive edge from entering the query-local gate. Validation and
test edges are never consumed when constructing either the global prior or the
local incidence. The final multiplier projection is zero initialized, so
strict migration permits only
`biogt_relbf_indicator_query_local_relation_transition_residuals.*` keys and
preserves parent logits exactly. Query-local-only fine-tuning can be restricted
to Gene while HGCN, Transformer, Path5, parent RelBF, and the parent indicator
remain frozen. Exported attribution records each relation label, support
source, dynamic-mask state, normalized per-query incidence, mean local gate,
and mean aggregate multiplier. Global and query-local relation-transition
modes cannot be enabled together.

The optional pure BioGT ranking auxiliary is strictly default-off. It is
enabled only when both `--biogt-path-pretrain-hard-negatives` and
`--biogt-path-pretrain-rank-weight` are positive. During the existing exact
`full_candidate_softmax` path pretraining, every query first masks all of its
known training positives from the exact relation-candidate universe. The
highest-scoring remaining `k` candidates are selected from the current BioGT
logits, and their selected scores retain gradients. The auxiliary is the mean
`softplus(hard_negative - positive)` over every selected negative and every
training positive for that query; its weighted value is added to, rather than
replacing, the original listwise softmax. No validation/test edge or baseline
checkpoint, embedding, score, or prediction participates. Metrics record the
combined, listwise, and pairwise epoch losses, the selected-pair counts, `k`,
and the auxiliary weight. Checkpoint `training_args`, launcher status, and the
generated remote command preserve the same two controls.

The optional pure BioGT Top-K boundary objective is independently and
strictly default-off through
`--biogt-path-pretrain-topk-boundary-weight 0`. It is evaluated only on the
same train-only `full_candidate_softmax` logits and known-training-positive
mask described above. For each query, the Kth highest unmasked negative is
compared with every training positive using
`softplus(kth_negative + margin - positive)`; when fewer than K negatives are
available, the weakest available negative is used, and a query with no valid
negative is skipped. The default K is 10. Weight, K, margin, epoch losses, and
selected-boundary counts are retained in metrics; CLI values are also retained
in checkpoint `training_args` and launcher status. This objective has no KGE,
PPR, NBFNet, validation-edge, or test-edge dependency.

The pure BioGT LambdaRank@K auxiliary is separately default-off through
`--biogt-path-pretrain-lambdarank-weight 0`. It uses only the same train-edge
positives, masked exact relation candidates, and current BioGT path/RelBF
scores. For each positive, the highest-scoring valid negatives are paired with
it using a temperature-scaled logistic loss weighted by the binary-relevance
change in NDCG@K caused by swapping their current ranks. Swaps crossing the K
boundary receive weight while pairs wholly below K receive zero weight. Other
train positives for the query remain masked, and short or negative-free
candidate lists are handled without adding invalid pairs. Weight, K,
temperature, epoch loss, and selected-pair counts are retained in metrics,
checkpoint `training_args`, and launcher status. It does not alter pair or
full-candidate scoring and has no baseline or held-out-edge dependency.

The pure BioGT Soft-Hits@K auxiliary is independently and strictly default-off
through `--biogt-path-pretrain-soft-hits-weight 0`. For every train positive,
it computes a differentiable soft rank against every full-candidate negative
remaining after all known positives for the same query are masked:
`1 + sum(sigmoid((negative - positive) / temperature))`. The mean stable
cutoff surrogate `softplus((soft_rank - (K + 0.5)) / K)` is added after the
existing listwise, hard-pair, Top-K boundary, and LambdaRank terms. This is a
train-only full-candidate objective that directly corresponds to strict
validation Hits@K; it is not post-hoc score adjustment and never reads
validation/test edges or baseline artifacts. Weight, K, temperature, epoch
loss, and positive count are recorded in logs, metrics/status, and checkpoint
training arguments. Masked positives receive no gradient from this term, and
candidate sets smaller than K remain valid.

The candidate-normalized Soft-Hits@K repair is a separate, strictly
default-off objective controlled by
`--biogt-path-pretrain-normalized-soft-hits-weight 0`; it does not change the
original Soft-Hits definition or old checkpoint semantics. After masking all
same-query train positives, each positive uses
`mean(sigmoid((negative - positive) / temperature))` as the fraction of full
candidates expected to outrank it. The target fraction is
`min(K, negative_count + 1) / (negative_count + 1)`, and the stable O(1)-scale
surrogate is `softplus((exceedance_fraction - cutoff_fraction) / temperature)`.
Loss is averaged over valid positives; negative-free rows contribute no term,
and candidate sets shorter than K remain finite. Weight, K, temperature,
epoch loss, valid-positive count, and compared-negative count are retained in
logs, metrics/status, and checkpoint training arguments. The objective uses
only train-graph candidates and is neither post-hoc calibration nor a source
of validation, test, or baseline information.

## Final Evaluation

Load the selected best checkpoint with `--epochs 0` and evaluate all four
targets. Record the final-test artifacts and do not repeat a successful test
in the same final-test directory.

## RNA Ceiling

The exact RNA candidate set contains 39 entities and the test split contains
only two RNA edges. Recall@50 is therefore identically 1.0 for every valid
ranking. DistMult additionally reaches 1.0 for RNA MRR, Hits@10, and NDCG@50.
All four requested RNA strict-greater gates are mathematically infeasible
because every metric is bounded above by 1.0. These criteria must remain
marked infeasible unless the comparison protocol is explicitly changed, for
example to allow a tie at the mathematical ceiling and report uncertainty
from the two-edge test set.
