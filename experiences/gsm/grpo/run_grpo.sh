#!/usr/bin/env bash
# GRPO training (gsm8k) — base policy = Qwen3-4B-Instruct-2507, LoRA.
#
# Usage:
#   bash experiences/gsm/grpo/run_grpo.sh smoke
#   bash experiences/gsm/grpo/run_grpo.sh full
set -euo pipefail

MODE="${1:-smoke}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NPROC="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
MASTER_PORT="${MASTER_PORT:-29520}"

GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LZL_ROOT="$(cd "$GRPO_DIR/../../.." && pwd)"
cd "$LZL_ROOT"

CONFIG="${LZL_CONFIG:-$LZL_ROOT/experiences/gsm/configs/config.yaml}"
PLUGIN="$GRPO_DIR/rewards.py"
LOG_DIR="$LZL_ROOT/logs/grpo"
mkdir -p "$LOG_DIR"

PY="/data2/anaconda3/envs/vcts/bin/python"
SWIFT="/data2/anaconda3/envs/vcts/bin/swift"

read_cfg() {
  local key="$1" cfg="$2"
  $PY -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c$key)"
}

MODEL="$(read_cfg "['model']['path']" "$CONFIG")"
CKPT_ROOT="/data5/lzl/checkpoints"
export HF_HOME="$(read_cfg "['datasets']['gsm8k_cache']" "$CONFIG")"
export HF_DATASETS_CACHE="$HF_HOME"
GRPO_ROOT="$(read_cfg "['paths']['grpo_data_root']" "$CONFIG")"

if [[ "$MODE" == "smoke" ]]; then
  DATA_LIMIT_ARGS="--limit 256"
  DATA_FILE="$GRPO_ROOT/gsm8k_train_limit256.jsonl"
  MAX_STEPS=30
  SAVE_STEPS=30
  OUTPUT_DIR="$CKPT_ROOT/grpo_smoke"
  NUM_GEN=8
else
  DATA_LIMIT_ARGS=""
  DATA_FILE="$GRPO_ROOT/gsm8k_train.jsonl"
  MAX_STEPS=-1
  SAVE_STEPS=200
  OUTPUT_DIR="$CKPT_ROOT/grpo"
  NUM_GEN=8
fi

if [[ ! -f "$DATA_FILE" ]]; then
  echo "[grpo] building dataset -> $DATA_FILE"
  $PY "$GRPO_DIR/07_build_grpo_dataset.py" --config "$CONFIG" $DATA_LIMIT_ARGS \
      --output "$DATA_FILE"
fi

GEN_BATCH=$NUM_GEN

echo "[grpo] mode=$MODE gpus=$GPUS model=$MODEL data=$DATA_FILE out=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" \
NPROC_PER_NODE="$NPROC" \
MASTER_PORT="$MASTER_PORT" \
$SWIFT rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --enable_thinking false \
  --dataset "$DATA_FILE" \
  --external_plugins "$PLUGIN" \
  --reward_funcs gsm_format gsm_accuracy \
  --reward_weights 0.5 1.0 \
  --num_generations $NUM_GEN \
  --generation_batch_size $GEN_BATCH \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.18 \
  --vllm_max_model_len 8192 \
  --sleep_level 1 \
  --max_completion_length 2048 \
  --temperature 1.0 \
  --top_p 0.9 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
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
