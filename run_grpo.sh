#!/usr/bin/env bash
# GRPO training (gsm/math) — base policy = Qwen3-4B-Instruct-2507, single GPU.
#
# Reward = gsm_format (Final Answer: <x> well-formed) + gsm_accuracy (matches
# gold, same judge as scripts/06_eval.py). Plugin: scripts/grpo_rewards.py.
# vLLM colocate mode: generation + training share one GPU.
#
# Usage:
#   bash run_grpo.sh smoke      # tiny: 256 prompts, ~30 steps, quick sanity
#   bash run_grpo.sh full       # full GSM8K train split
#
# GPU is selected via CUDA_VISIBLE_DEVICES (default 1, since GPU0 runs mllm).
set -euo pipefail

MODE="${1:-smoke}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
cd "$LZL_ROOT"

MODEL="/data5/lzl/models/Qwen3-4B-Instruct-2507"
PLUGIN="$LZL_ROOT/scripts/grpo_rewards.py"
CKPT_ROOT="/data5/lzl/checkpoints"
LOG_DIR="$LZL_ROOT/logs/grpo"
mkdir -p "$LOG_DIR"

PY="/data2/anaconda3/envs/vcts/bin/python"
SWIFT="/data2/anaconda3/envs/vcts/bin/swift"

# ---- mode-specific knobs ---------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
  DATA_LIMIT_ARGS="--limit 256"
  DATA_FILE="$LZL_ROOT/data/grpo/gsm8k_train_limit256.jsonl"
  MAX_STEPS=30
  SAVE_STEPS=30
  OUTPUT_DIR="$CKPT_ROOT/grpo_smoke"
  NUM_GEN=8
else
  DATA_LIMIT_ARGS=""
  DATA_FILE="$LZL_ROOT/data/grpo/gsm8k_train.jsonl"
  MAX_STEPS=-1            # use num_train_epochs instead
  SAVE_STEPS=200
  OUTPUT_DIR="$CKPT_ROOT/grpo"
  NUM_GEN=8
fi

# ---- build the GRPO prompt dataset (idempotent) ----------------------------
if [[ ! -f "$DATA_FILE" ]]; then
  echo "[grpo] building dataset → $DATA_FILE"
  $PY "$LZL_ROOT/scripts/07_build_grpo_dataset.py" --split train $DATA_LIMIT_ARGS
fi

# generation_batch_size must be a multiple of num_generations.
GEN_BATCH=$NUM_GEN

echo "[grpo] mode=$MODE gpu=$GPU model=$MODEL data=$DATA_FILE out=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" \
$SWIFT rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --dataset "$DATA_FILE" \
  --external_plugins "$PLUGIN" \
  --reward_funcs gsm_format gsm_accuracy \
  --reward_weights 0.5 1.0 \
  --num_generations $NUM_GEN \
  --generation_batch_size $GEN_BATCH \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_max_model_len 2048 \
  --sleep_level 1 \
  --max_completion_length 1024 \
  --temperature 1.0 \
  --top_p 0.9 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --max_steps $MAX_STEPS \
  --save_steps $SAVE_STEPS \
  --save_total_limit 2 \
  --logging_steps 1 \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --beta 0.04 \
  --seed 42 \
  2>&1 | tee "$LOG_DIR/${MODE}.out"
