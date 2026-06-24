#!/usr/bin/env bash
# GRPO training (mllm) — base policy = Qwen3.5-4B, enable_thinking=false, single GPU.
#
# Usage:
#   bash experiences/mllm/grpo/run_grpo.sh smoke
#   bash experiences/mllm/grpo/run_grpo.sh full
set -euo pipefail

MODE="${1:-smoke}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LZL_ROOT="$(cd "$GRPO_DIR/../../.." && pwd)"
cd "$LZL_ROOT"

CONFIG="${LZL_CONFIG:-$LZL_ROOT/experiences/mllm/configs/config.yaml}"
SMOKE_CONFIG="$LZL_ROOT/experiences/mllm/smoke/configs/config_smoke.yaml"
PLUGIN="$GRPO_DIR/rewards.py"
LOG_DIR="$LZL_ROOT/logs/mllm_grpo"
mkdir -p "$LOG_DIR"

PY="/data2/anaconda3/envs/vcts/bin/python"
SWIFT="/data2/anaconda3/envs/vcts/bin/swift"

read_cfg() {
  local key="$1" cfg="$2"
  $PY -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c$key)"
}

MODEL="$(read_cfg "['model']['path']" "$CONFIG")"
CKPT_ROOT="/data5/lzl/checkpoints"
export HF_HOME="$(read_cfg "['datasets']['hf_cache']" "$CONFIG")"
export HF_DATASETS_CACHE="$HF_HOME"

if [[ "$MODE" == "smoke" ]]; then
  DATA_LIMIT_ARGS="--limit 64"
  SRC_JSONL="$(read_cfg "['datasets']['train_jsonl']" "$SMOKE_CONFIG")"
  DATA_FILE="/data4/FTSO/datasets/mllm/grpo/mllm_train_limit64.jsonl"
  MAX_STEPS=16
  SAVE_STEPS=16
  OUTPUT_DIR="$CKPT_ROOT/mllm_grpo_smoke"
  NUM_GEN=4
else
  DATA_LIMIT_ARGS=""
  SRC_JSONL="$(read_cfg "['datasets']['train_jsonl']" "$CONFIG")"
  DATA_FILE="/data4/FTSO/datasets/mllm/grpo/mllm_train_full.jsonl"
  MAX_STEPS=-1
  SAVE_STEPS=200
  OUTPUT_DIR="$CKPT_ROOT/mllm_grpo"
  NUM_GEN=4
fi

if [[ ! -f "$DATA_FILE" ]] || [[ "$(wc -l < "$DATA_FILE")" -lt 8000 && "$MODE" == "full" ]]; then
  echo "[mllm-grpo] building dataset -> $DATA_FILE (from $SRC_JSONL)"
  $PY "$GRPO_DIR/07_build_grpo_dataset.py" \
      --config "$CONFIG" --src "$SRC_JSONL" $DATA_LIMIT_ARGS \
      --output "$DATA_FILE" --require_image
fi

GEN_BATCH=$NUM_GEN

echo "[mllm-grpo] mode=$MODE gpu=$GPU model=$MODEL data=$DATA_FILE out=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" \
$SWIFT rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --model_type qwen3_5 \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --enable_thinking false \
  --dataset "$DATA_FILE" \
  --external_plugins "$PLUGIN" \
  --reward_funcs mllm_format mllm_accuracy \
  --reward_weights 0.5 1.0 \
  --num_generations $NUM_GEN \
  --generation_batch_size $GEN_BATCH \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.40 \
  --vllm_max_model_len 8192 \
  --vllm_limit_mm_per_prompt '{"image": 1}' \
  --sleep_level 0 \
  --vllm_enable_prefix_caching false \
  --vllm_mm_processor_cache_gb 0 \
  --max_completion_length 4096 \
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
