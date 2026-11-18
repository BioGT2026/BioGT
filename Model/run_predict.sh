#!/bin/bash

# ==============================================================================
# Shell script to run predictions using a trained Disease-Biomarker model.
# ==============================================================================
#
# Usage:
#   ./predict.sh [MODEL_TYPE] [OUTPUT_TYPE] [DISEASE_NAMES] [MODEL_PATH_PREFIX] [TOP_K] [DATA_PATH] [LOCAL_MODEL_PATH] [MONDO_DISEASE_INPUT] [ENABLE_TOP1_PROTEIN_DRUG_MODE] [ENABLE_MARKER_EDGE_MODE] [MARKER_EDGE_CSV] [FORCE_TARGET_IDS]
#
# Arguments (all optional with defaults):
#   $1: MODEL_TYPE          - Model architecture (default: "original")
#                             Options: original, gcn, gat, transformer
#   $2: OUTPUT_TYPE         - Target biomarker type (default: "protein")
#                             Options: protein, metabolite, rna, gene
#   $3: DISEASE_NAMES       - Comma-separated disease names (default: "metabolic disease,diabetes")
#   $4: MODEL_PATH_PREFIX   - Experiments directory (default: "./experiments")
#   $5: TOP_K               - Number of top predictions (default: 30)
#   $6: DATA_PATH           - Path to dataset directory (default: "~/LN/dataset/graphWithNodeType/merge")
#   $7: LOCAL_MODEL_PATH    - Path to local Hugging Face cache (default: "/home/$USER/models_cache")
#   $8: MONDO_DISEASE_INPUT - Either a path to MONDO disease node CSV for batch prediction,
#                             or a comma/space/newline-separated MONDO ID list (default: empty)
#                             CSV expected schema (header): ID,Type,Name,Sequence,Comments
#                             Example CSV row: MONDO:0000004,Disease,adrenocortical insufficiency,...
#                             Example ID list: MONDO:0010796,MONDO:0005086
#   $9: ENABLE_TOP1_PROTEIN_DRUG_MODE - Enable new search mode (default: false)
#                             true: enable "top1 target result with path containing drug + hops>=4" mode
#                             false: keep normal prediction mode
#   ${10}: ENABLE_MARKER_EDGE_MODE - Enable marker_edge-driven batch mode (default: false)
#                             true: parse marker_edge CSV, extract unique MONDO disease IDs,
#                                   split by marker type, and run one prediction batch per type.
#                                   In this mode, diseases are resolved by their original node ID
#                                   from marker_edge / pair CSV disease_id, not by disease name
#                             false: keep standard input mode
#   ${11}: MARKER_EDGE_CSV   - Path to marker_edge CSV (default: "./marker_edge.csv")
#   ${12}: FORCE_TARGET_IDS  - Comma/space/newline-separated original target node IDs to
#                             force-score and print for every input disease, regardless of top-k rank.
#                             Example: NCBIGene:673,NCBIGene:675
#
# Examples:
#   # Use all defaults
#   ./predict.sh
#
#   # Predict for specific disease with GCN model
#   ./predict.sh gcn protein "Alzheimer's disease"
#
#   # Full specification with custom model cache
#   ./predict.sh original metabolite "diabetes,psoriasis" ./experiments 50 ~/data /path/to/models
#
#   # Batch prediction for all diseases listed in a mondo CSV
#   ./predict.sh original protein "" ./experiments 30 ~/data /path/to/models /path/to/mondo_disease_nodes.csv
#
#   # Enable dedicated new search mode
#   ./predict.sh original protein "" ./experiments 30 ~/data /path/to/models /path/to/mondo_disease_nodes.csv true
#
#   # Enable marker_edge batch mode
#   ./predict.sh original gene "" ./experiments 30 ~/data /path/to/models "" false true ./marker_edge.csv
#
#   # Force output specific gene IDs for every input disease
#   ./predict.sh original gene "" ./experiments 30 ~/data /path/to/models "MONDO:0010796" false false ./marker_edge.csv "NCBIGene:673,NCBIGene:675"
#
# ==============================================================================

# --- User Configuration ---

# 0. Model type and target
MODEL_TYPE="${1:-"original"}" # e.g., original, gcn, gat, transformer
OUTPUT_TYPE="${2:-"gene"}" # e.g., protein, metabolite, rna, gene
FUSION_MODE="weighted" # e.g., sequential, concat, weighted, hierarchical, routing, attention, film
SPLIT_METHOD="random" # e.g., random, kfold, specific_ids_by_disease_category
ABLATION_LABEL="none" # e.g., none, hgcn, transformer, moe

# 1. Comma-separated list of diseases to predict biomarkers for.
#    Can be overridden by the third command-line argument.
#    Example: "Alzheimer's disease,Parkinson's disease"
DISEASE_NAMES="${3:-""}"

# 1.1 Optional MONDO disease input for batch prediction.
#     If arg8 is an existing file, it is treated as a MONDO CSV.
#     Otherwise, values that look like MONDO IDs are treated as a direct ID list.
MONDO_DISEASE_INPUT="${8:-""}"
MONDO_DISEASE_CSV=""
MONDO_DISEASE_IDS="MONDO:0010796"
if [ -n "$MONDO_DISEASE_INPUT" ]; then
    if [ -f "$MONDO_DISEASE_INPUT" ]; then
        MONDO_DISEASE_CSV="$MONDO_DISEASE_INPUT"
    elif printf "%s" "$MONDO_DISEASE_INPUT" | grep -Eq 'MONDO:[0-9]+'; then
        MONDO_DISEASE_IDS="$MONDO_DISEASE_INPUT"
    else
        MONDO_DISEASE_CSV="$MONDO_DISEASE_INPUT"
    fi
fi
# "/home/$USER/dataset/graphWithNodeType/mondo_node.csv"}"

# 2. Path to the trained model directory.
#    Can be overridden by the fourth command-line argument.
MODEL_PATH_PREFIX="${4:-"./experiments"}"

build_model_layout() {
    local output_type="$1"
    CURRENT_EXP_DIR_NAME="target-${output_type}_split-${SPLIT_METHOD}_fusion-${FUSION_MODE}_hops-10"
    CURRENT_MODEL_DIR="${MODEL_PATH_PREFIX}/${CURRENT_EXP_DIR_NAME}"
    CURRENT_MODEL_PATH="${CURRENT_MODEL_DIR}/model_final_best.pt"
    CURRENT_GRAPH_CACHE_PATH="${CURRENT_MODEL_DIR}/cache"
}

build_model_layout "$OUTPUT_TYPE"
EXP_DIR_NAME="$CURRENT_EXP_DIR_NAME"
MODEL_DIR="$CURRENT_MODEL_DIR"
MODEL_PATH="$CURRENT_MODEL_PATH"

# 3. Number of top results to return for each biomarker type.
#    Can be overridden by the fifth command-line argument.
TOP_K="${5:-5}"

# 4. Paths to the dataset files.
#    IMPORTANT: These should be the same files used for training the model.
DATA_PATH="${6:-"/home/$USER/dataset/graphWithNodeType/mergeNew"}"
NODES_CSV="${DATA_PATH}/merged_node.csv"
EDGES_CSV="${DATA_PATH}/merged_edge.csv"
GRAPH_CACHE_PATH="$CURRENT_GRAPH_CACHE_PATH" # <-- should match training cache path

# 5. Model Architecture Parameters
#    IMPORTANT: These parameters MUST EXACTLY MATCH the ones used to train the model specified in MODEL_PATH.
#    Check your training logs or scripts if you are unsure.
DEFAULT_EMBED_SIZE=64
HGCN_HIDDEN=64
HGCN_OUT=64
NUM_TRANSFORMER_LAYERS=3
NUM_HGCN_LAYERS=3
TRANSFORMER_HEADS=16
TRANSFORMER_FF=512
NUM_LATENTS=32
DROPOUT=0.1

# 6. Explainability Settings
#    Configure explainability output options
ENABLE_EXPLAINABILITY=true # Set to false to disable explainability outputs
EXPLAINABILITY_TOP_K=1    # Top-K nodes/edges to include in explainability summaries
EXPLAINABILITY_MAX_HOPS=20  # Maximum hops to trace when constructing multi-hop explanation paths
EXPLAINABILITY_HOP_PENALTY=0.18  # Conservative default for hub-heavy graphs: discourages fragmented multi-step paths
EXPLAINABILITY_VIRTUAL_EDGE_PENALTY=0.60  # Conservative default: virtual edges should be a rare fallback, not a shortcut
EXPLAINABILITY_VIRTUAL_SIM_THRESHOLD=0.45  # Conservative default: only stronger embedding matches create virtual edges
EXPLAINABILITY_ATTRIBUTION_METHOD="input_x_gradient"  # input_x_gradient or gradient_norm

# 6.1 Isolated-node filtering
#    Set to true to keep isolated nodes in the prediction graph
DISABLE_ISOLATED_FILTER=true

# 6.2 Sequence Features
#    Set to true to disable sequence features (must match training configuration)
DISABLE_SEQUENCE_FEATURES=true

# 6.3 Local Model Path
#    Path to local Hugging Face model cache for offline execution
#    Used when generating node features (e.g., sentence transformers)
LOCAL_MODEL_PATH="${7:-"/home/$USER/models_cache"}"

# 6.4 Dedicated switch for new prediction mode
#    When enabled, script runs the targeted search:
#    top1 target result + explain path contains drug + path hops >= 4.
ENABLE_TOP1_PROTEIN_DRUG_MODE="${9:-"false"}"
NEW_MODE_MIN_EXPLAIN_HOPS=0
NEW_MODE_PREFILTER_DRUG_HOPS=0
NEW_MODE_PREFILTER_DRUG_TO_TARGET_HOPS=0
NEW_MODE_MAX_SEARCH_MATCHES=0
ENABLE_MARKER_EDGE_MODE="${10:-"false"}"
MARKER_EDGE_CSV="${11:-"./marker_edge.csv"}"
FORCE_TARGET_IDS="${12:-"NCBIGene:5979,NCBIGene:23095,NCBIGene:6389,NCBIGene:6390,NCBIGene:6392,NCBIGene:55654,NCBIGene:7428,NCBIGene:4763,NCBIGene:4149,NCBIGene:6391,NCBIGene:2271"}"
RERANK_HOP_BONUS_LAMBDA="1.0"
RERANK_MAX_HOPS="1"

# 7. Ablation Flags
#    Uncomment any flags that were used during the training of the loaded model.
ABLATION_FLAGS=""
# ABLATION_FLAGS="--disable-hgcn"
# ABLATION_FLAGS="$ABLATION_FLAGS --disable-transformer"
# ABLATION_FLAGS="$ABLATION_FLAGS --disable-moe"
# ABLATION_FLAGS="$ABLATION_FLAGS --disable-feature-gating"

# 8. Log Settings
#    Configure logging output
LOG_DIR="./prediction_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Build a compact, filesystem-safe disease token to avoid overlong log filenames.
sanitize_for_filename() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_//' | sed 's/_$//'
}

build_disease_log_token() {
    local raw_names="$1"

    if [ "$ENABLE_MARKER_EDGE_MODE" = "true" ]; then
        echo "marker_edge_batch"
        return
    fi

    # Use a stable token for batch mode from MONDO CSV or ID list.
    if [ -n "$MONDO_DISEASE_CSV" ]; then
        echo "mondo_batch"
        return
    fi

    if [ -n "$MONDO_DISEASE_IDS" ]; then
        echo "mondo_id_batch"
        return
    fi

    local cleaned first_name first_clean disease_count short_hash
    cleaned=$(sanitize_for_filename "$raw_names")

    if [ -z "$cleaned" ]; then
        echo "no_disease"
        return
    fi

    first_name=$(echo "$raw_names" | awk -F',' '{print $1}')
    first_clean=$(sanitize_for_filename "$first_name")
    disease_count=$(echo "$raw_names" | awk -F',' '{print NF}')
    short_hash=$(printf "%s" "$raw_names" | shasum | awk '{print $1}' | cut -c1-10)

    if [ ${#first_clean} -gt 24 ]; then
        first_clean=$(echo "$first_clean" | cut -c1-24)
    fi

    echo "${first_clean}_n${disease_count}_${short_hash}"
}

DISEASE_TOKEN=$(build_disease_log_token "$DISEASE_NAMES")
LOG_OUTPUT_LABEL="$OUTPUT_TYPE"
if [ "$ENABLE_MARKER_EDGE_MODE" = "true" ]; then
    LOG_OUTPUT_LABEL="marker_edge_multi"
fi
LOG_FILE="${LOG_DIR}/predict_${LOG_OUTPUT_LABEL}_${DISEASE_TOKEN}_${TIMESTAMP}.log"
CURRENT_BATCH_LOG_FILE=""

# --- Script Execution ---

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at '$MODEL_PATH'"
    exit 1
fi

# Function to log with timestamp
log_info() {
    if [ -n "$CURRENT_BATCH_LOG_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE" "$CURRENT_BATCH_LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
    fi
}

run_prediction_for_type() {
    local current_output_type="$1"
    local input_mode="$2"
    local pair_csv="$3"
    local current_top_k="$4"
    local current_new_mode_flags="$5"
    local safe_output_type

    build_model_layout "$current_output_type"

    safe_output_type=$(sanitize_for_filename "$current_output_type")
    if [ "$ENABLE_MARKER_EDGE_MODE" = "true" ] && [ "$input_mode" = "pair_csv" ]; then
        CURRENT_BATCH_LOG_FILE="${LOG_DIR}/predict_${safe_output_type}_${DISEASE_TOKEN}_${TIMESTAMP}.log"
        {
            echo "========================================================================================================="
            echo "                      Disease-Biomarker Prediction Batch Log"
            echo "========================================================================================================="
            echo "Batch Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Parent Log File: $LOG_FILE"
            echo "Batch Log File: $CURRENT_BATCH_LOG_FILE"
            echo "Marker Type: $current_output_type"
            echo "Pair CSV: $pair_csv"
            echo "Model Path: $CURRENT_MODEL_PATH"
            echo "Graph Cache Path: $CURRENT_GRAPH_CACHE_PATH"
            echo "========================================================================================================="
            echo ""
        } > "$CURRENT_BATCH_LOG_FILE"
    else
        CURRENT_BATCH_LOG_FILE=""
    fi

    if [ ! -f "$CURRENT_MODEL_PATH" ]; then
        log_info "✗ Model file not found for output type '$current_output_type': $CURRENT_MODEL_PATH"
        CURRENT_BATCH_LOG_FILE=""
        return 1
    fi

    local input_args=()
    if [ "$input_mode" = "pair_csv" ]; then
        input_args=(--eval_marker_pairs_csv "$pair_csv")
        log_info "Input mode: marker pair CSV batch prediction ($pair_csv)"
    elif [ -n "$MONDO_DISEASE_CSV" ]; then
        input_args=(--eval_input_mondo_csv "$MONDO_DISEASE_CSV")
        log_info "Input mode: mondo CSV batch prediction"
    elif [ -n "$MONDO_DISEASE_IDS" ]; then
        input_args=(--eval_input_ids "$MONDO_DISEASE_IDS")
        log_info "Input mode: MONDO ID list batch prediction"
    else
        input_args=(--eval_input_names "$DISEASE_NAMES")
        log_info "Input mode: manual disease names"
    fi

    if [ -n "$FORCE_TARGET_IDS" ]; then
        input_args+=(--force_target_ids "$FORCE_TARGET_IDS")
        log_info "Forced target IDs active for every input disease"
    fi

    log_info "Executing prediction for output type '$current_output_type' using model: $CURRENT_MODEL_PATH"
    if [ -n "$CURRENT_BATCH_LOG_FILE" ]; then
        echo "" | tee -a "$LOG_FILE" "$CURRENT_BATCH_LOG_FILE"
    else
        echo "" | tee -a "$LOG_FILE"
    fi

    if [ -n "$CURRENT_BATCH_LOG_FILE" ]; then
        python -u predict.py \
            --load_model_path "$CURRENT_MODEL_PATH" \
            --nodes_csv "$NODES_CSV" \
            --edges_csv "$EDGES_CSV" \
            --graph_cache_path "$CURRENT_GRAPH_CACHE_PATH" \
            --local_model_path "$LOCAL_MODEL_PATH" \
            "${input_args[@]}" \
            --eval_top_k "$current_top_k" \
            --eval_output_types "$current_output_type" \
            --gpu 0 \
            --default_embed_size "$DEFAULT_EMBED_SIZE" \
            --hgcn_hidden "$HGCN_HIDDEN" \
            --hgcn_out "$HGCN_OUT" \
            --num_transformer_layers "$NUM_TRANSFORMER_LAYERS" \
            --num_gnn_layers "$NUM_HGCN_LAYERS" \
            --transformer_heads "$TRANSFORMER_HEADS" \
            --transformer_ff "$TRANSFORMER_FF" \
            --num_latents "$NUM_LATENTS" \
            --dropout "$DROPOUT" \
            --fusion-mode "$FUSION_MODE" \
            --rerank-hop-bonus-lambda "$RERANK_HOP_BONUS_LAMBDA" \
            --rerank-max-hops "$RERANK_MAX_HOPS" \
            $ISOLATED_FILTER_FLAGS \
            $SEQUENCE_FEATURES_FLAGS \
            $current_new_mode_flags \
            $EXPLAINABILITY_FLAGS \
            $ABLATION_FLAGS 2>&1 | tee -a "$LOG_FILE" "$CURRENT_BATCH_LOG_FILE"
    else
        python -u predict.py \
            --load_model_path "$CURRENT_MODEL_PATH" \
            --nodes_csv "$NODES_CSV" \
            --edges_csv "$EDGES_CSV" \
            --graph_cache_path "$CURRENT_GRAPH_CACHE_PATH" \
            --local_model_path "$LOCAL_MODEL_PATH" \
            "${input_args[@]}" \
            --eval_top_k "$current_top_k" \
            --eval_output_types "$current_output_type" \
            --gpu 0 \
            --default_embed_size "$DEFAULT_EMBED_SIZE" \
            --hgcn_hidden "$HGCN_HIDDEN" \
            --hgcn_out "$HGCN_OUT" \
            --num_transformer_layers "$NUM_TRANSFORMER_LAYERS" \
            --num_gnn_layers "$NUM_HGCN_LAYERS" \
            --transformer_heads "$TRANSFORMER_HEADS" \
            --transformer_ff "$TRANSFORMER_FF" \
            --num_latents "$NUM_LATENTS" \
            --dropout "$DROPOUT" \
            --fusion-mode "$FUSION_MODE" \
            --rerank-hop-bonus-lambda "$RERANK_HOP_BONUS_LAMBDA" \
            --rerank-max-hops "$RERANK_MAX_HOPS" \
            $ISOLATED_FILTER_FLAGS \
            $SEQUENCE_FEATURES_FLAGS \
            $current_new_mode_flags \
            $EXPLAINABILITY_FLAGS \
            $ABLATION_FLAGS 2>&1 | tee -a "$LOG_FILE"
    fi

    local batch_exit_code=${PIPESTATUS[0]}

    if [ -n "$CURRENT_BATCH_LOG_FILE" ]; then
        {
            echo ""
            echo "========================================================================================================="
            echo "Batch End Time: $(date '+%Y-%m-%d %H:%M:%S')"
            if [ $batch_exit_code -eq 0 ]; then
                echo "Batch Status: SUCCESS"
            else
                echo "Batch Status: FAILED (Exit Code: $batch_exit_code)"
            fi
            echo "========================================================================================================="
        } | tee -a "$LOG_FILE" "$CURRENT_BATCH_LOG_FILE"
    fi

    CURRENT_BATCH_LOG_FILE=""
    return $batch_exit_code
}

# Write header to log file
{
    echo "========================================================================================================="
    echo "                           Disease-Biomarker Prediction Log"
    echo "========================================================================================================="
    echo "Prediction Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log File: $LOG_FILE"
    echo ""
    echo "=== Prediction Configuration ==="
    echo "  Model Type:               $MODEL_TYPE"
    echo "  Requested Output Type:    $OUTPUT_TYPE"
    if [ "$ENABLE_MARKER_EDGE_MODE" = "true" ]; then
        echo "  Effective Output Type:    dynamic per marker_type batch from marker_edge manifest"
    else
        echo "  Effective Output Type:    $OUTPUT_TYPE"
    fi
    echo "  Disease Names:            $DISEASE_NAMES"
    echo "  Mondo Disease CSV:        ${MONDO_DISEASE_CSV:-<not set>}"
    echo "  Mondo Disease IDs:        ${MONDO_DISEASE_IDS:-<not set>}"
    echo "  Marker Edge Mode:         $ENABLE_MARKER_EDGE_MODE"
    echo "  Marker Edge CSV:          $MARKER_EDGE_CSV"
    echo "  Force Target IDs:         ${FORCE_TARGET_IDS:-<not set>}"
    echo "  Top K:                    $TOP_K"
    echo "  Fusion Mode:              $FUSION_MODE"
    echo "  Split Method:             $SPLIT_METHOD"
    echo "  Ablation Label:           $ABLATION_LABEL"
    echo ""
    echo "=== Model Configuration ==="
    echo "  Experiment Directory:     $EXP_DIR_NAME"
    echo "  Model Directory:          $MODEL_DIR"
    echo "  Model Path:               $MODEL_PATH"
    echo "  Default Embed Size:       $DEFAULT_EMBED_SIZE"
    echo "  HGCN Hidden:              $HGCN_HIDDEN"
    echo "  HGCN Out:                 $HGCN_OUT"
    echo "  Transformer Layers:       $NUM_TRANSFORMER_LAYERS"
    echo "  GNN Layers:               $NUM_HGCN_LAYERS"
    echo "  Transformer Heads:        $TRANSFORMER_HEADS"
    echo "  Transformer FF:           $TRANSFORMER_FF"
    echo "  Num Latents:              $NUM_LATENTS"
    echo "  Dropout:                  $DROPOUT"
    echo ""
    echo "=== Explainability Configuration ==="
    echo "  Enabled:                  $ENABLE_EXPLAINABILITY"
    echo "  Top-K:                    $EXPLAINABILITY_TOP_K"
    echo "  Max Hops:                 $EXPLAINABILITY_MAX_HOPS"
    echo "  Hop Penalty:              $EXPLAINABILITY_HOP_PENALTY"
    echo "  Virtual Edge Penalty:     $EXPLAINABILITY_VIRTUAL_EDGE_PENALTY"
    echo "  Virtual Sim Threshold:    $EXPLAINABILITY_VIRTUAL_SIM_THRESHOLD"
    echo "  Attribution Method:       $EXPLAINABILITY_ATTRIBUTION_METHOD"
    echo "  Disable Isolated Filter:  $DISABLE_ISOLATED_FILTER"
    echo "  Disable Sequence Features: $DISABLE_SEQUENCE_FEATURES"
    echo "  Local Model Path:         $LOCAL_MODEL_PATH"
    echo "  New Mode Enabled:         $ENABLE_TOP1_PROTEIN_DRUG_MODE"
    echo "  New Mode Min Hops:        $NEW_MODE_MIN_EXPLAIN_HOPS"
    echo "  New Mode Prefilter Hops:  $NEW_MODE_PREFILTER_DRUG_HOPS"
    echo "  New Mode Drug->Target Prefilter Hops: $NEW_MODE_PREFILTER_DRUG_TO_TARGET_HOPS"
    echo "  New Mode Max Matches:     $NEW_MODE_MAX_SEARCH_MATCHES"
    echo "  Soft Rerank Structure λ:  $RERANK_HOP_BONUS_LAMBDA"
    echo "  Soft Rerank Max Hops:     $RERANK_MAX_HOPS"
    echo ""
    echo "=== Data Paths ==="
    echo "  Nodes CSV:                $NODES_CSV"
    echo "  Edges CSV:                $EDGES_CSV"
    echo "  Graph Cache Path:         $GRAPH_CACHE_PATH"
    echo ""
    echo "========================================================================================================="
    echo ""
} > "$LOG_FILE"

log_info "Starting prediction process..."

# Build explainability flags
EXPLAINABILITY_FLAGS=""
if [ "$ENABLE_EXPLAINABILITY" = "false" ]; then
    EXPLAINABILITY_FLAGS="--no-explainability"
else
    EXPLAINABILITY_FLAGS="--explainability-top-k $EXPLAINABILITY_TOP_K --explainability-max-hops $EXPLAINABILITY_MAX_HOPS --explainability-hop-penalty $EXPLAINABILITY_HOP_PENALTY --explainability-virtual-edge-penalty $EXPLAINABILITY_VIRTUAL_EDGE_PENALTY --explainability-virtual-sim-threshold $EXPLAINABILITY_VIRTUAL_SIM_THRESHOLD --explainability-attribution-method $EXPLAINABILITY_ATTRIBUTION_METHOD"
fi

# Build dedicated new-mode flags
NEW_MODE_FLAGS=""
if [ "$ENABLE_TOP1_PROTEIN_DRUG_MODE" = "true" ] && { [ -n "$MONDO_DISEASE_CSV" ] || [ -n "$MONDO_DISEASE_IDS" ]; }; then
    NEW_MODE_FLAGS="--find_top1_protein_with_drug_path --min_explain_hops $NEW_MODE_MIN_EXPLAIN_HOPS --prefilter-by-drug-hops $NEW_MODE_PREFILTER_DRUG_HOPS --prefilter-by-drug-to-target-hops $NEW_MODE_PREFILTER_DRUG_TO_TARGET_HOPS --max_search_matches $NEW_MODE_MAX_SEARCH_MATCHES"
    log_info "Dedicated mode enabled: top1 target result with drug-in-path + hops>=$NEW_MODE_MIN_EXPLAIN_HOPS"
    log_info "For dedicated mode, keeping output type as '$OUTPUT_TYPE' and overriding top_k to 1 for strict top1 checking."
    TOP_K=1
elif [ "$ENABLE_TOP1_PROTEIN_DRUG_MODE" = "true" ]; then
    log_info "Dedicated mode skipped: non-batch input (MONDO CSV / ID list not provided), keeping backward compatibility."
fi

# Build isolated filter flags
ISOLATED_FILTER_FLAGS=""
if [ "$DISABLE_ISOLATED_FILTER" = "true" ]; then
    ISOLATED_FILTER_FLAGS="--disable-isolated-filter"
fi

# Build sequence features flags
SEQUENCE_FEATURES_FLAGS=""
if [ "$DISABLE_SEQUENCE_FEATURES" = "true" ]; then
    SEQUENCE_FEATURES_FLAGS="--disable-sequence-features"
fi

PYTHON_EXIT_CODE=0

if [ "$ENABLE_MARKER_EDGE_MODE" = "true" ]; then
    MARKER_EDGE_CACHE_ROOT="${LOG_DIR}/marker_edge_cache"
    mkdir -p "$MARKER_EDGE_CACHE_ROOT"
    MARKER_EDGE_SOURCE_KEY=$(printf "%s" "$MARKER_EDGE_CSV" | shasum | awk '{print $1}' | cut -c1-12)
    MARKER_EDGE_BATCH_DIR="${MARKER_EDGE_CACHE_ROOT}/${MARKER_EDGE_SOURCE_KEY}"
    MARKER_EDGE_MANIFEST="${MARKER_EDGE_BATCH_DIR}/batch_manifest.tsv"

    if [ "$ENABLE_TOP1_PROTEIN_DRUG_MODE" = "true" ]; then
        log_info "Marker-edge mode does not combine with dedicated top1 drug-path mode. Dedicated mode flags will be ignored."
    fi

    if [ ! -f "$MARKER_EDGE_CSV" ]; then
        log_info "✗ marker_edge CSV not found: $MARKER_EDGE_CSV"
        PYTHON_EXIT_CODE=1
    else
        log_info "Preparing or reusing cached marker-edge batches from: $MARKER_EDGE_CSV"
        log_info "Marker-edge cache directory: $MARKER_EDGE_BATCH_DIR"
        python -u prepare_marker_edge_batches.py \
            --marker_edge_csv "$MARKER_EDGE_CSV" \
            --output_dir "$MARKER_EDGE_BATCH_DIR" 2>&1 | tee -a "$LOG_FILE"
        PREP_EXIT_CODE=${PIPESTATUS[0]}

        if [ $PREP_EXIT_CODE -ne 0 ]; then
            PYTHON_EXIT_CODE=$PREP_EXIT_CODE
        elif [ ! -f "$MARKER_EDGE_MANIFEST" ]; then
            log_info "✗ Marker-edge manifest was not created: $MARKER_EDGE_MANIFEST"
            PYTHON_EXIT_CODE=1
        else
            while IFS=$'\t' read -r marker_type pair_csv pair_count disease_count disease_ids_csv; do
                if [ "$marker_type" = "marker_type" ]; then
                    continue
                fi
                if [ -z "$marker_type" ] || [ -z "$pair_csv" ]; then
                    continue
                fi

                log_info "Running marker-edge batch: output_type=$marker_type disease_count=$disease_count pair_count=$pair_count disease_queue=${disease_ids_csv:-N/A}"
                run_prediction_for_type "$marker_type" "pair_csv" "$pair_csv" "$TOP_K" ""
                BATCH_EXIT_CODE=$?
                if [ $BATCH_EXIT_CODE -ne 0 ]; then
                    PYTHON_EXIT_CODE=$BATCH_EXIT_CODE
                    log_info "✗ Marker-edge batch failed for output type '$marker_type'"
                    break
                fi
            done < "$MARKER_EDGE_MANIFEST"
        fi
    fi
else
    if [ -n "$MONDO_DISEASE_CSV" ] && [ ! -f "$MONDO_DISEASE_CSV" ]; then
        log_info "✗ Mondo disease CSV not found: $MONDO_DISEASE_CSV"
        PYTHON_EXIT_CODE=1
    else
        run_prediction_for_type "$OUTPUT_TYPE" "standard" "" "$TOP_K" "$NEW_MODE_FLAGS"
        PYTHON_EXIT_CODE=$?
    fi
fi

echo "" | tee -a "$LOG_FILE"
{
    echo "========================================================================================================="
    echo "Prediction End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED (Exit Code: $PYTHON_EXIT_CODE)"
    fi
    echo "========================================================================================================="
} | tee -a "$LOG_FILE"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    log_info "✓ Prediction completed successfully!"
    log_info "Results saved to log file: $LOG_FILE"
else
    log_info "✗ Prediction failed with exit code: $PYTHON_EXIT_CODE"
    log_info "Check log file for details: $LOG_FILE"
    exit $PYTHON_EXIT_CODE
fi