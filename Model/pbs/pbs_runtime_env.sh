#!/usr/bin/env bash
set -euo pipefail

export BIOMARKER_RUNTIME_ENV_READY=1
MODEL_DIR="$PWD"

mkdir -p logs .dgl .hf_cache .cache/matplotlib .cache/torch

export DGLDEFAULTDIR="${DGLDEFAULTDIR:-$PWD/.dgl}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence_transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/.."

resolve_biomarker_data_path() {
  if [[ -n "${DATA_PATH:-}" ]]; then
    return 0
  fi

  local -a candidates=()
  local custom_candidates="${BIOMARKER_DATA_PATH_CANDIDATES:-}"
  if [[ -n "$custom_candidates" ]]; then
    IFS=':' read -r -a candidates <<< "$custom_candidates"
  fi

  candidates+=(
    "$HOME/LN/dataset/graphWithNodeType/merge"
  )

  if [[ -n "${USER:-}" ]]; then
    candidates+=(
      "/home/$USER/dataset/graphWithNodeType/mergeNew"
      "/home/$USER/dataset/graphWithNodeType/merge"
    )
  fi

  candidates+=(
    "$MODEL_DIR/../Data"
    "$MODEL_DIR/Data"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate/merged_node.csv" && -f "$candidate/merged_edge.csv" ]]; then
      export DATA_PATH="$candidate"
      return 0
    fi
  done

  export DATA_PATH="$HOME/LN/dataset/graphWithNodeType/merge"
}

if [[ -z "${DATA_PATH:-}" && -n "${NODES_CSV:-}" ]]; then
  export DATA_PATH="$(cd "$(dirname "$NODES_CSV")" 2>/dev/null && pwd || dirname "$NODES_CSV")"
fi

if [[ -z "${NODES_CSV:-}" || -z "${EDGES_CSV:-}" ]]; then
  resolve_biomarker_data_path
fi

export NODES_CSV="${NODES_CSV:-$DATA_PATH/merged_node.csv}"
export EDGES_CSV="${EDGES_CSV:-$DATA_PATH/merged_edge.csv}"

if [[ -z "${LOCAL_MODEL_PATH:-}" ]]; then
  if [[ -n "${USER:-}" && -d "/home/$USER/LN/models_cache" ]]; then
    export LOCAL_MODEL_PATH="/home/$USER/LN/models_cache"
  elif [[ -d "$HOME/LN/models_cache" ]]; then
    export LOCAL_MODEL_PATH="$HOME/LN/models_cache"
  elif [[ -d "/home/20212008/LN/models_cache" ]]; then
    export LOCAL_MODEL_PATH="/home/20212008/LN/models_cache"
  elif [[ -n "${USER:-}" && -d "/home/$USER/models_cache" ]]; then
    export LOCAL_MODEL_PATH="/home/$USER/models_cache"
  elif [[ -d "$HOME/models_cache" ]]; then
    export LOCAL_MODEL_PATH="$HOME/models_cache"
  else
    export LOCAL_MODEL_PATH="$MODEL_DIR/models_cache"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv312/bin/python" ]]; then
    export PYTHON_BIN=".venv312/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    export PYTHON_BIN=".venv/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${PBS_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$PBS_GPUS"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${PBS_GPUFILE:-}" && -f "$PBS_GPUFILE" ]]; then
  first_gpu="$(head -n 1 "$PBS_GPUFILE" | awk -F'[-:]' '{print $NF}')"
  if [[ -n "$first_gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$first_gpu"
  fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v pbsnodes >/dev/null 2>&1; then
  node_gpu="$(pbsnodes "$(hostname)" 2>/dev/null | awk -F'= *' '/resources_available.ngpus/ {print $2; exit}' || true)"
  if [[ -n "${node_gpu:-}" && "$node_gpu" != "0" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  fi
fi

if [[ -z "${GPU:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export GPU=0
  else
    export GPU=-1
  fi
fi

echo "==== PBS runtime environment ===="
echo "cwd=$PWD"
echo "python=${PYTHON_BIN}"
echo "gpu=${GPU}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "pbs_jobid=${PBS_JOBID:-manual}"
echo "data_path=${DATA_PATH}"
echo "nodes_csv=${NODES_CSV}"
echo "edges_csv=${EDGES_CSV}"
echo "local_model_path=${LOCAL_MODEL_PATH}"
