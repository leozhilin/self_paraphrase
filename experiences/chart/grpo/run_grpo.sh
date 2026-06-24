#!/usr/bin/env bash
# GRPO training (chart) — base policy from chart_config.yaml (Qwen3-4B).
#
# Usage:
#   bash experiences/chart/grpo/run_grpo.sh smoke
#   bash experiences/chart/grpo/run_grpo.sh full
set -euo pipefail

MODE="${1:-smoke}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LZL_ROOT="$(cd "$GRPO_DIR/../../.." && pwd)"
cd "$LZL_ROOT"

CONFIG="${LZL_CONFIG:-$LZL_ROOT/chart_config.yaml}"
PLUGIN="$GRPO_DIR/rewards.py"
LOG_DIR="$LZL_ROOT/logs/chart_grpo"
mkdir -p "$LOG_DIR"

PY="/data2/anaconda3/envs/vcts/bin/python"
SWIFT="/data2/anaconda3/envs/vcts/bin/swift"

read_cfg() {
  local key="$1" cfg="$2"
  $PY -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c$key)"
}

MODEL="$(read_cfg "['model']['path']" "$CONFIG")"
CKPT_ROOT="$(read_cfg "['paths']['checkpoint_root']" "$CONFIG")"
export HF_HOME="$(read_cfg "['datasets']['gsm8k_cache']" "$CONFIG")"
export HF_DATASETS_CACHE="$HF_HOME"
GRPO_ROOT="$(read_cfg "['paths']['grpo_data_root']" "$CONFIG")"

if [[ "$MODE" == "smoke" ]]; then
  DATA_LIMIT_ARGS="--limit 256 --config $CONFIG"
  DATA_FILE="$GRPO_ROOT/chart_train_limit256.jsonl"
  MAX_STEPS=30
  SAVE_STEPS=30
  OUTPUT_DIR="$CKPT_ROOT/chart_grpo_smoke"
  NUM_GEN=8
else
  DATA_LIMIT_ARGS="--config $CONFIG"
  DATA_FILE="$GRPO_ROOT/chart_train.jsonl"
  MAX_STEPS=-1
  SAVE_STEPS=200
  OUTPUT_DIR="$CKPT_ROOT/chart_grpo"
  NUM_GEN=8
fi

if [[ ! -f "$DATA_FILE" ]]; then
  echo "[chart-grpo] building dataset -> $DATA_FILE"
  $PY "$GRPO_DIR/07_build_grpo_dataset.py" $DATA_LIMIT_ARGS --output "$DATA_FILE"
fi

GEN_BATCH=$NUM_GEN

echo "[chart-grpo] mode=$MODE gpu=$GPU model=$MODEL data=$DATA_FILE out=$OUTPUT_DIR"

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
  --reward_funcs chart_format chart_accuracy \
  --reward_weights 0.5 1.0 \
  --num_generations $NUM_GEN \
  --generation_batch_size $GEN_BATCH \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_max_model_len 4096 \
  --sleep_level 1 \
  --max_completion_length 2048 \
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
