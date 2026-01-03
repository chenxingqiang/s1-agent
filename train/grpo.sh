#!/bin/bash
# GRPO Training Script for Chain of Reward (CoR)
#
# This script trains a model using GRPO with CoR rewards.
# Based on s1's sft.sh with GRPO-specific modifications.
#
# Usage:
#   bash train/grpo.sh
#
# Prerequisites:
#   - Run SFT first to get reference model: bash train/sft.sh
#   - Adjust ref_model_path to point to SFT checkpoint

uid="$(date +%Y%m%d_%H%M%S)"

# Model configuration
base_model="Qwen/Qwen2.5-32B-Instruct"
ref_model_path="ckpts/s1-sft"  # Path to SFT checkpoint (reference model)

# Training hyperparameters
lr=1e-6
epochs=3
micro_batch_size=1
gradient_accumulation_steps=4

# GRPO specific
num_generations=4  # N candidates per input
epsilon=0.2        # Clipping range (δ)
beta=0.01          # KL penalty coefficient (β)

# CoR specific  
lambda_intrinsic=1.0
self_rating_weight=0.2

# Hardware
gpu_count=$(nvidia-smi -L | wc -l)

# Output
output_dir="ckpts/cor-grpo-${uid}"

echo "Starting CoR + GRPO training..."
echo "Base model: ${base_model}"
echo "Reference model: ${ref_model_path}"
echo "Output: ${output_dir}"
echo "GPUs: ${gpu_count}"

torchrun --nproc-per-node ${gpu_count} --master_port 12346 \
    train/grpo.py \
    --model_name=${base_model} \
    --ref_model_name=${ref_model_path} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --block_size=32768 \
    --num_generations=${num_generations} \
    --lambda_intrinsic=${lambda_intrinsic} \
    --self_rating_weight=${self_rating_weight} \
    --per_device_train_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --learning_rate=${lr} \
    --warmup_ratio=0.1 \
    --epsilon=${epsilon} \
    --beta=${beta} \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --bf16=True \
    --logging_steps=1 \
    --save_strategy="steps" \
    --save_steps=100 \
    --output_dir=${output_dir} \
    --report_to="wandb" \
    --wandb_project="cor-grpo"

echo "Training complete! Model saved to ${output_dir}"
