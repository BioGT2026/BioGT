#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f "pbs/pbs_runtime_env.sh" && -z "${BIOMARKER_RUNTIME_ENV_READY:-}" ]]; then
  source pbs/pbs_runtime_env.sh
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv312/bin/python" ]]; then
    PYTHON_BIN=".venv312/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
DATA_PATH="${DATA_PATH:-../Data}"
NODES_CSV="${NODES_CSV:-$DATA_PATH/merged_node.csv}"
EDGES_CSV="${EDGES_CSV:-$DATA_PATH/merged_edge.csv}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-./models_cache}"
TARGET_TYPES_CSV="${TARGET_TYPES:-${TARGET_TYPE:-metabolite,protein,rna,gene}}"
EVAL_INPUT_TYPE="${EVAL_INPUT_TYPE:-disease}"
EVAL_INPUT_NAME="${EVAL_INPUT_NAME:-psoriasis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/target_full_local}"
LOG_DIR="${LOG_DIR:-logs}"
# Production BioGT defaults are aligned with the main-branch scripts and
# nc26biogt manuscript. Override via environment variables for smoke/debug runs.
EPOCHS="${EPOCHS:-10000}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"
DEFAULT_EMBED_SIZE="${DEFAULT_EMBED_SIZE:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-16}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-3}"
NUM_GNN_LAYERS="${NUM_GNN_LAYERS:-3}"
TRANSFORMER_FF="${TRANSFORMER_FF:-512}"
NUM_LATENTS="${NUM_LATENTS:-32}"
DROPOUT="${DROPOUT:-0.1}"
K_NEG_SAMPLES="${K_NEG_SAMPLES:-1}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-100}"
EARLY_STOPPING_METRICS="${EARLY_STOPPING_METRICS:-train_loss,val_loss}"
FUSION_MODE="${FUSION_MODE:-weighted}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random}"
SPLIT_HOLDOUT_SCOPE="${SPLIT_HOLDOUT_SCOPE:-target_edges}"
TARGET_EDGE_SCOPE="${TARGET_EDGE_SCOPE:-}"
USE_DEFAULT_NODE_FEATURES_ONLY="${USE_DEFAULT_NODE_FEATURES_ONLY:-0}"
DISABLE_SEQUENCE_FEATURES="${DISABLE_SEQUENCE_FEATURES:-1}"
DISABLE_ISOLATED_NODE_FILTERING="${DISABLE_ISOLATED_NODE_FILTERING:-0}"
SYNC_MODEL_TARGET_ETYPES="${SYNC_MODEL_TARGET_ETYPES:-0}"
REGENERATE_FEATURES_AFTER_FILTER="${REGENERATE_FEATURES_AFTER_FILTER:-0}"
USE_TRAIN_GRAPH_FOR_EVAL_MESSAGE_PASSING="${USE_TRAIN_GRAPH_FOR_EVAL_MESSAGE_PASSING:-0}"
# Default runs BioGT only. BioGT is implemented as model type "original".
# Optional reviewer baselines, enabled only by explicitly setting MODELS:
#   gcn, gat, transformer, kg_embedding, complex, rotate, rgcn, network_propagation, ppr
# Example: MODELS=original,gcn,gat,transformer,kg_embedding,complex,rotate,rgcn,network_propagation,ppr
MODELS_CSV="${MODELS:-original}"
GPU="${GPU:--1}"
GRAPH_NODE_TYPES="${GRAPH_NODE_TYPES:-protein,disease,gene,metabolite,rna}"
GRAPH_RELATION_TYPES="${GRAPH_RELATION_TYPES:-protein_disease,gene_disease,metabolite_disease,rna_disease}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
export DGLDEFAULTDIR="${DGLDEFAULTDIR:-$PWD/.dgl}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence_transformers}"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/.."
mkdir -p "$DGLDEFAULTDIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$SENTENCE_TRANSFORMERS_HOME"

IFS=',' read -r -a MODEL_TYPES <<< "$MODELS_CSV"
IFS=',' read -r -a RAW_TARGET_TYPES <<< "$TARGET_TYPES_CSV"
declare -a TARGET_TYPES_ARRAY=()
for target_type in "${RAW_TARGET_TYPES[@]}"; do
  target_type="$(printf '%s' "$target_type" | xargs)"
  [[ -z "$target_type" ]] && continue
  TARGET_TYPES_ARRAY+=("$target_type")
done
if [[ ${#TARGET_TYPES_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: TARGET_TYPES did not contain any target types." >&2
  exit 1
fi
declare -a EXTRA_ARG_ARRAY=()
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARG_ARRAY <<< "$EXTRA_ARGS"
fi
declare -a GRAPH_FILTER_ARGS=()
if [[ -n "$GRAPH_NODE_TYPES" ]]; then
  GRAPH_FILTER_ARGS+=(--graph-node-types "$GRAPH_NODE_TYPES")
fi
if [[ -n "$GRAPH_RELATION_TYPES" ]]; then
  GRAPH_FILTER_ARGS+=(--graph-relation-types "$GRAPH_RELATION_TYPES")
fi
is_truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}
declare -a OPTIONAL_ARGS=()
if is_truthy "$USE_DEFAULT_NODE_FEATURES_ONLY"; then
  OPTIONAL_ARGS+=(--default-node-features-only)
fi
if is_truthy "$DISABLE_SEQUENCE_FEATURES"; then
  OPTIONAL_ARGS+=(--disable-sequence-features)
fi
if is_truthy "$DISABLE_ISOLATED_NODE_FILTERING"; then
  OPTIONAL_ARGS+=(--disable-isolated-node-filtering)
fi
if is_truthy "$SYNC_MODEL_TARGET_ETYPES"; then
  OPTIONAL_ARGS+=(--sync-model-target-etypes)
fi
if is_truthy "$REGENERATE_FEATURES_AFTER_FILTER"; then
  OPTIONAL_ARGS+=(--regenerate-features-after-filter)
fi
if is_truthy "$USE_TRAIN_GRAPH_FOR_EVAL_MESSAGE_PASSING"; then
  OPTIONAL_ARGS+=(--use-train-graph-for-eval-message-passing)
fi

echo "==== Target full local run ===="
echo "models=${MODELS_CSV}"
echo "target_types=${TARGET_TYPES_CSV}"
echo "epochs=${EPOCHS}"
echo "lr=${LR}"
echo "default_embed_size=${DEFAULT_EMBED_SIZE}"
echo "hgcn_hidden=${HIDDEN_SIZE}"
echo "transformer_heads=${TRANSFORMER_HEADS}"
echo "num_transformer_layers=${NUM_TRANSFORMER_LAYERS}"
echo "num_gnn_layers=${NUM_GNN_LAYERS}"
echo "split_strategy=${SPLIT_STRATEGY}"
echo "val_split=${VAL_SPLIT}"
echo "test_split=${TEST_SPLIT}"
echo "fusion_mode=${FUSION_MODE}"
echo "data_path=${DATA_PATH}"
echo "nodes=${NODES_CSV}"
echo "edges=${EDGES_CSV}"
echo "local_model_path=${LOCAL_MODEL_PATH}"
echo "gpu=${GPU}"
echo "graph_node_types=${GRAPH_NODE_TYPES:-ALL}"
echo "graph_relation_types=${GRAPH_RELATION_TYPES:-ALL}"

for target_type in "${TARGET_TYPES_ARRAY[@]}"; do
  train_target_types="$target_type"
  eval_output_types="$target_type"
  target_relation_types="${target_type}_disease"
  effective_target_relation_types="${TARGET_RELATION_TYPES:-$target_relation_types}"
  declare -a SPLIT_ARGS=(
    --split-strategy "$SPLIT_STRATEGY"
    --val_split "$VAL_SPLIT"
    --test_split "$TEST_SPLIT"
  )
  if [[ "$SPLIT_STRATEGY" == "independent_cohort" || "$SPLIT_STRATEGY" == "cohort" || "$SPLIT_STRATEGY" == "cohort_validation" ]]; then
    SPLIT_ARGS+=(
      --split-holdout-scope "$SPLIT_HOLDOUT_SCOPE"
      --target-edge-scope "${TARGET_EDGE_SCOPE:-input_output}"
      --target-relation-types "$effective_target_relation_types"
    )
  else
    if [[ -n "$TARGET_EDGE_SCOPE" ]]; then
      SPLIT_ARGS+=(--target-edge-scope "$TARGET_EDGE_SCOPE")
    fi
    if [[ -n "${TARGET_RELATION_TYPES:-}" ]]; then
      SPLIT_ARGS+=(--target-relation-types "$effective_target_relation_types")
    fi
  fi

  for model_type in "${MODEL_TYPES[@]}"; do
    run_dir="$OUTPUT_ROOT/$target_type/$model_type"
    mkdir -p "$run_dir/cache"
    log_file="$LOG_DIR/${target_type}-full-local-${model_type}-$(date +%Y%m%d_%H%M%S).log"

    echo "==== Running full local target test: target=${target_type}, model=${model_type} ===="
    echo "train_target_types=${train_target_types}"
    echo "eval_output_types=${eval_output_types}"
    echo "target_relation_types=${target_relation_types}"
    echo "run_dir=${run_dir}"
    "$PYTHON_BIN" train_test_single.py \
      --nodes_csv "$NODES_CSV" \
      --edges_csv "$EDGES_CSV" \
      --gpu "$GPU" \
      --model-type "$model_type" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --default_embed_size "$DEFAULT_EMBED_SIZE" \
      --hgcn_hidden "$HIDDEN_SIZE" \
      --hgcn_out "$HIDDEN_SIZE" \
      --transformer_heads "$TRANSFORMER_HEADS" \
      --num_transformer_layers "$NUM_TRANSFORMER_LAYERS" \
      --num_gnn_layers "$NUM_GNN_LAYERS" \
      --transformer_ff "$TRANSFORMER_FF" \
      --num_latents "$NUM_LATENTS" \
      --dropout "$DROPOUT" \
      --k_neg_samples "$K_NEG_SAMPLES" \
      --local_model_path "$LOCAL_MODEL_PATH" \
      --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA" \
      --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
      --early_stopping_metrics "$EARLY_STOPPING_METRICS" \
      --train_target_types "$train_target_types" \
      --eval_input_name "$EVAL_INPUT_NAME" \
      --eval_input_type "$EVAL_INPUT_TYPE" \
      --eval_output_types "$eval_output_types" \
      --fusion-mode "$FUSION_MODE" \
      --save_model_path "$run_dir/model_final.pt" \
      --graph_cache_path "$run_dir/cache" \
      "${SPLIT_ARGS[@]}" \
      ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"} \
      ${GRAPH_FILTER_ARGS[@]+"${GRAPH_FILTER_ARGS[@]}"} \
      ${EXTRA_ARG_ARRAY[@]+"${EXTRA_ARG_ARRAY[@]}"} \
      2>&1 | tee "$log_file"
  done
done
