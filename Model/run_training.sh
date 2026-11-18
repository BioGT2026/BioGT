#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

data_path="${DATA_PATH:-$HOME/LN/dataset/graphWithNodeType/merge}"
output_root="${OUTPUT_ROOT:-$script_dir/experiments/unified_sparse}"
logs_dir="${LOGS_DIR:-$script_dir/logs}"
seed="${SEED:-42}"
epochs="${EPOCHS:-1000}"
batch_size="${BATCH_SIZE:-128}"
steps_per_epoch="${STEPS_PER_EPOCH:-0}"
eval_every="${EVAL_EVERY:-5}"
neighbor_fanouts="${NEIGHBOR_FANOUTS:-15,10,5}"
context_topk="${CONTEXT_TOPK:-256}"
targets="${TRAIN_TARGET_TYPES:-gene,rna,protein,metabolite}"
fusion_modes="${MODALITY_FUSION_MODES:-gene:weighted,dna:weighted,rna:concat,protein:concat,metabolite:hierarchical}"

mkdir -p "$output_root" "$logs_dir"

allocated_gpu="${PBS_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -n "$allocated_gpu" ]]; then
    allocated_gpu="${allocated_gpu%%,*}"
    export CUDA_VISIBLE_DEVICES="$allocated_gpu"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    training_gpu=0
else
    training_gpu=-1
fi

run_id="unified-four-output-seed${seed}"
run_dir="$output_root/$run_id"
cache_dir="${GRAPH_CACHE_PATH:-$output_root/shared_joint_feature_cache_v2}"
checkpoint_path="$run_dir/model_final.pt"
metrics_path="$run_dir/metrics.json"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$logs_dir/${run_id}-${timestamp}.log"
mkdir -p "$run_dir" "$cache_dir"

sequence_args=(--merged-sequence-features)
if [[ "${USE_SEQUENCE_FEATURES:-1}" == "0" ]]; then
    sequence_args=(--disable-sequence-features)
fi

python -m Model.train_test_single \
    --training-engine unified \
    --nodes_csv "$data_path/merged_node.csv" \
    --edges_csv "$data_path/merged_edge.csv" \
    --gpu "$training_gpu" \
    --model-type original \
    --epochs "$epochs" \
    --seed "$seed" \
    --lr 0.0001 \
    --embedding-lr 0.0001 \
    --weight_decay 0.00001 \
    --default_embed_size 64 \
    --hgcn_hidden 64 \
    --hgcn_out 64 \
    --num_gnn_layers 3 \
    --num_transformer_layers 3 \
    --transformer_heads 8 \
    --transformer_ff 256 \
    --dropout 0.1 \
    --batch-size "$batch_size" \
    --steps-per-epoch "$steps_per_epoch" \
    --neighbor-fanouts "$neighbor_fanouts" \
    --sparse-transformer-topk "$context_topk" \
    --sparse-transformer-max-hops 4 \
    --k_neg_samples 4 \
    --ranking-loss-weight 0.3 \
    --ranking-temperature 1.0 \
    --hard-negative-warmup 5 \
    --hard-negative-interval 5 \
    --hard-negative-candidates 16 \
    --hard-negative-ratio 0.5 \
    --hard-negative-max-edges 20000 \
    --score-chunk-size 4096 \
    --eval-every "$eval_every" \
    --validation-batch-size 256 \
    --fixed-validation-seed 1042 \
    --checkpoint-metric mrr \
    --early_stopping_patience 20 \
    --early_stopping_min_delta 0.0001 \
    --moe-num-experts 4 \
    --moe-top-k 2 \
    --moe-aux-loss-weight 0.01 \
    --modality-fusion-modes "$fusion_modes" \
    --train_target_types "$targets" \
    --eval_input_type disease \
    --eval_output_types "$targets" \
    --split-strategy random \
    --val_split 0.1 \
    --test_split 0.1 \
    --disable-isolated-node-filter \
    --save_model_path "$checkpoint_path" \
    --metrics-output "$metrics_path" \
    --graph_cache_path "$cache_dir" \
    --local_model_path "${LOCAL_MODEL_PATH:-$HOME/models_cache}" \
    "${sequence_args[@]}" \
    "$@" 2>&1 | tee "$log_path"
