#!/bin/bash
# Chain of Reward (CoR) Complete Training Pipeline
#
# This script runs the full CoR training pipeline:
# 1. Prepare data with self-ratings (cold-start)
# 2. SFT on rated data (reference model)
# 3. GRPO with CoR rewards
# 4. Evaluate on benchmarks
#
# Usage:
#   bash train/run_cor_pipeline.sh [--skip-data] [--skip-sft] [--eval-only]

set -e

# Parse arguments
SKIP_DATA=false
SKIP_SFT=false
EVAL_ONLY=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-data) SKIP_DATA=true ;;
        --skip-sft) SKIP_SFT=true ;;
        --eval-only) EVAL_ONLY=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Configuration
uid="$(date +%Y%m%d_%H%M%S)"
BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
INPUT_DATA="simplescaling/s1K_tokenized"
RATED_DATA="local_data/s1K_rated"
SFT_OUTPUT="ckpts/cor-sft-${uid}"
GRPO_OUTPUT="ckpts/cor-grpo-${uid}"

echo "=============================================="
echo "Chain of Reward (CoR) Training Pipeline"
echo "=============================================="
echo "Timestamp: ${uid}"
echo "Base model: ${BASE_MODEL}"
echo ""

# Step 1: Prepare data with self-ratings
if [ "$SKIP_DATA" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "Step 1: Preparing data with self-ratings..."
    echo "----------------------------------------------"
    
    python data/add_self_ratings.py \
        --input_path ${INPUT_DATA} \
        --output_path ${RATED_DATA} \
        --method rule \
        --num_workers 4
    
    echo "✅ Data preparation complete"
    echo ""
else
    echo "⏭️ Skipping data preparation"
    RATED_DATA=${INPUT_DATA}  # Use original data
fi

# Step 2: SFT on rated data
if [ "$SKIP_SFT" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "Step 2: Running SFT on rated data..."
    echo "----------------------------------------------"
    
    gpu_count=$(nvidia-smi -L | wc -l)
    
    torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
        train/sft.py \
        --block_size=32768 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --num_train_epochs=3 \
        --train_file_path="${RATED_DATA}" \
        --model_name=${BASE_MODEL} \
        --warmup_ratio=0.05 \
        --fsdp="full_shard auto_wrap" \
        --fsdp_config="train/fsdp_config_qwen.json" \
        --bf16=True \
        --eval_strategy="no" \
        --logging_steps=1 \
        --save_strategy="epoch" \
        --lr_scheduler_type="cosine" \
        --learning_rate=1e-5 \
        --weight_decay=1e-4 \
        --output_dir=${SFT_OUTPUT}
    
    echo "✅ SFT complete: ${SFT_OUTPUT}"
    echo ""
else
    echo "⏭️ Skipping SFT"
    if [ -z "${SFT_OUTPUT}" ]; then
        SFT_OUTPUT="ckpts/cor-sft"  # Use existing checkpoint
    fi
fi

# Step 3: GRPO with CoR rewards
if [ "$EVAL_ONLY" = false ]; then
    echo "Step 3: Running GRPO with CoR rewards..."
    echo "----------------------------------------------"
    
    gpu_count=$(nvidia-smi -L | wc -l)
    
    torchrun --nproc-per-node ${gpu_count} --master_port 12346 \
        train/grpo.py \
        --model_name=${BASE_MODEL} \
        --ref_model_name=${SFT_OUTPUT} \
        --train_file_path="${RATED_DATA}" \
        --block_size=32768 \
        --num_generations=4 \
        --lambda_intrinsic=1.0 \
        --self_rating_weight=0.2 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=2 \
        --learning_rate=1e-6 \
        --warmup_ratio=0.1 \
        --epsilon=0.2 \
        --beta=0.01 \
        --fsdp="full_shard auto_wrap" \
        --fsdp_config="train/fsdp_config_qwen.json" \
        --bf16=True \
        --logging_steps=1 \
        --save_strategy="steps" \
        --save_steps=100 \
        --output_dir=${GRPO_OUTPUT} \
        --report_to="wandb" \
        --wandb_project="cor-grpo"
    
    echo "✅ GRPO complete: ${GRPO_OUTPUT}"
    echo ""
fi

# Step 4: Evaluation
echo "Step 4: Evaluating on benchmarks..."
echo "----------------------------------------------"

MODEL_TO_EVAL=${GRPO_OUTPUT}
if [ "$EVAL_ONLY" = true ]; then
    MODEL_TO_EVAL=${1:-"ckpts/cor-grpo"}
fi

# AIME24
echo "Evaluating on AIME24..."
python eval/generate.py \
    --model_path ${MODEL_TO_EVAL} \
    --benchmark aime24 \
    --output_dir results/eval_${uid}

# MATH500
echo "Evaluating on MATH500..."
python eval/generate.py \
    --model_path ${MODEL_TO_EVAL} \
    --benchmark math500 \
    --output_dir results/eval_${uid}

# GPQA Diamond
echo "Evaluating on GPQA Diamond..."
python eval/generate.py \
    --model_path ${MODEL_TO_EVAL} \
    --benchmark gpqa_diamond \
    --output_dir results/eval_${uid}

echo ""
echo "=============================================="
echo "✅ CoR Training Pipeline Complete!"
echo "=============================================="
echo "Results saved to: results/eval_${uid}"
echo "Model checkpoint: ${GRPO_OUTPUT}"
