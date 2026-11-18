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

RUNNER="${RUNNER:-run_target_full_local.sh}"
# Default runs BioGT only. BioGT is implemented as model type "original".
# Optional reviewer baselines, enabled only by explicitly setting MODELS:
#   gcn, gat, transformer, kg_embedding, complex, rotate, rgcn, network_propagation, ppr
# Example: MODELS=original,gcn,gat,transformer,kg_embedding,complex,rotate,rgcn,network_propagation,ppr
MODELS="${MODELS:-original}"
TARGET_TYPES_CSV="${TARGET_TYPES:-${TARGET_TYPE:-metabolite,protein,rna,gene}}"
EPOCHS="${EPOCHS:-10000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/evidence_weight_ablation_all_targets}"
LOG_DIR="${LOG_DIR:-logs}"
EXTRA_ARGS_BASE="${EXTRA_ARGS:-}"

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

run_condition() {
  local condition="$1"
  local condition_extra_args="$2"
  local target_type="$3"
  local target_relation_types="${target_type}_disease"
  local output_dir="$OUTPUT_ROOT/$condition"
  local merged_extra_args="$EXTRA_ARGS_BASE"
  if [[ -n "$condition_extra_args" ]]; then
    if [[ -n "$merged_extra_args" ]]; then
      merged_extra_args="$merged_extra_args $condition_extra_args"
    else
      merged_extra_args="$condition_extra_args"
    fi
  fi

  echo "==== Evidence-weight ablation: condition=${condition}, target=${target_type} ===="
  echo "runner=${RUNNER}"
  echo "models=${MODELS}"
  echo "target_type=${target_type}"
  echo "target_relation_types=${target_relation_types}"
  echo "output=${output_dir}"
  echo "data_path=${DATA_PATH:-unset}"
  echo "nodes=${NODES_CSV:-unset}"
  echo "edges=${EDGES_CSV:-unset}"
  PYTHON_BIN="$PYTHON_BIN" \
  MODELS="$MODELS" \
  TARGET_TYPES="$target_type" \
  EPOCHS="$EPOCHS" \
  OUTPUT_ROOT="$output_dir" \
  LOG_DIR="$LOG_DIR" \
  EXTRA_ARGS="$merged_extra_args" \
  bash "$RUNNER"
}

echo "==== Evidence-weight ablation target set ===="
echo "target_types=${TARGET_TYPES_CSV}"

for target_type in "${TARGET_TYPES_ARRAY[@]}"; do
  run_condition "evidence_weighted" "" "$target_type"
  run_condition "evidence_uniform" "--disable-evidence-weights" "$target_type"
done
