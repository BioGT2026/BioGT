#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv312/bin/python" ]]; then
    PYTHON_BIN=".venv312/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
EPOCHS="${EPOCHS:-2}"
# Default runs BioGT only. Optional reviewer baselines:
#   gcn, gat, transformer, kg_embedding, complex, rotate, rgcn, network_propagation, ppr
# Example: MODELS=original,gcn,gat,transformer,kg_embedding,complex,rotate,rgcn,network_propagation,ppr
MODELS="${MODELS:-original}"
TARGET_TYPES="${TARGET_TYPES:-${TARGET_TYPE:-metabolite,protein,rna,gene}}"
ABLATION_OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/target_smoke_ablations}"

echo "==== Target smoke ablation: evidence weights disabled ===="
echo "target_types=${TARGET_TYPES}"
PYTHON_BIN="$PYTHON_BIN" \
EPOCHS="$EPOCHS" \
MODELS="$MODELS" \
TARGET_TYPES="$TARGET_TYPES" \
OUTPUT_ROOT="$ABLATION_OUTPUT_ROOT/evidence_uniform" \
EXTRA_ARGS="--disable-evidence-weights" \
bash run_target_smoke.sh

echo "==== Target smoke ablation: source HMDB excluded ===="
echo "target_types=${TARGET_TYPES}"
PYTHON_BIN="$PYTHON_BIN" \
EPOCHS="$EPOCHS" \
MODELS="$MODELS" \
TARGET_TYPES="$TARGET_TYPES" \
OUTPUT_ROOT="$ABLATION_OUTPUT_ROOT/source_exclude_hmdb" \
EXTRA_ARGS="--source-ablation-mode exclude --source-ablation-sources HMDB --source-ablation-match any_endpoint" \
bash run_target_smoke.sh
