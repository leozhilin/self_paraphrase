#!/usr/bin/env bash
# GRPO training (chart) — base policy = Qwen3-4B-Instruct-2507 (text-only,
# table given as text in the question), multi-GPU full fine-tuning.
#
# Reward = chart_format (Final Answer: <x> well-formed) + chart_accuracy
# (matches gold via chart_answers_match, same judge as 06_eval_chart).
# Plugin: experiences/chart/grpo/rewards.py.
# vLLM colocate mode: generation + training share each visible GPU.
#
# Usage:
#   bash experiences/chart/grpo/run_grpo.sh smoke  # 256 prompts, ~30 steps
#   bash experiences/chart/grpo/run_grpo.sh full   # full ChartQA train split
#
# GPUs are selected via CUDA_VISIBLE_DEVICES (default 0,1).  ms-swift turns
# NPROC_PER_NODE into torch.distributed.run automatically.
set -euo pipefail

MODE="${1:-smoke}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NPROC="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
MASTER_PORT="${MASTER_PORT:-29521}"

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
cd "$LZL_ROOT"

MODEL="/data5/lzl/models/Qwen3-4B-Instruct-2507"
PLUGIN="$LZL_ROOT/experiences/chart/grpo/rewards.py"
CKPT_ROOT="/data5/lzl/checkpoints"
LOG_DIR="$LZL_ROOT/logs/chart_grpo"
mkdir -p "$LOG_DIR"

PY="/data2/anaconda3/envs/vcts/bin/python"
SWIFT="/data2/anaconda3/envs/vcts/bin/swift"

# ---- mode-specific knobs ---------------------------------------------------
# Defaults: conservative full-parameter GRPO with colocated vLLM.
VLLM_GMU=0.18
VLLM_MAXLEN=8192
MAX_COMPLETION=2048
GRAD_ACCUM=4
SLEEP_LEVEL=1
OFFLOAD_MODEL=true
OFFLOAD_OPTIMIZER=true

if [[ "$MODE" == "smoke" ]]; then
  DATA_LIMIT_ARGS="--limit 256"
  DATA_FILE="$LZL_ROOT/data/grpo/chart_train_limit256.jsonl"
  MAX_STEPS=30
  SAVE_STEPS=30
  OUTPUT_DIR="$CKPT_ROOT/chart_grpo_smoke"
  NUM_GEN=8
else
  DATA_LIMIT_ARGS=""
  DATA_FILE="$LZL_ROOT/data/grpo/chart_train.jsonl"
  MAX_STEPS=-1
  SAVE_STEPS=200
  OUTPUT_DIR="$CKPT_ROOT/chart_grpo_full"
  NUM_GEN=8
  VLLM_GMU=0.18
  VLLM_MAXLEN=8192
  MAX_COMPLETION=2048
  GRAD_ACCUM=4
fi

# ---- build the GRPO prompt dataset (idempotent) ----------------------------
if [[ ! -f "$DATA_FILE" ]]; then
  echo "[chart-grpo] building dataset -> $DATA_FILE"
  $PY "$LZL_ROOT/experiences/chart/grpo/07_build_grpo_dataset.py" $DATA_LIMIT_ARGS \
      --output "$DATA_FILE"
fi

# generation_batch_size must be a multiple of num_generations.
GEN_BATCH=$NUM_GEN

echo "[chart-grpo] mode=$MODE gpus=$GPUS nproc=$NPROC master_port=$MASTER_PORT"
echo "[chart-grpo] model=$MODEL data=$DATA_FILE out=$OUTPUT_DIR"
echo "[chart-grpo] tuner=full num_gen=$NUM_GEN gen_batch=$GEN_BATCH"
echo "[chart-grpo] gmu=$VLLM_GMU maxlen=$VLLM_MAXLEN completion=$MAX_COMPLETION grad_accum=$GRAD_ACCUM sleep=$SLEEP_LEVEL"
echo "[chart-grpo] offload_model=$OFFLOAD_MODEL offload_optimizer=$OFFLOAD_OPTIMIZER"

CUDA_VISIBLE_DEVICES="$GPUS" \
NPROC_PER_NODE="$NPROC" \
MASTER_PORT="$MASTER_PORT" \
$SWIFT rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --dataset "$DATA_FILE" \
  --external_plugins "$PLUGIN" \
  --reward_funcs chart_format chart_accuracy \
  --reward_weights 0.5 1.0 \
  --num_generations $NUM_GEN \
  --generation_batch_size $GEN_BATCH \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization $VLLM_GMU \
  --vllm_max_model_len $VLLM_MAXLEN \
  --vllm_enable_prefix_caching false \
  --sleep_level $SLEEP_LEVEL \
  --offload_model $OFFLOAD_MODEL \
  --offload_optimizer $OFFLOAD_OPTIMIZER \
  --max_completion_length $MAX_COMPLETION \
  --temperature 1.0 \
  --top_p 0.9 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps $GRAD_ACCUM \
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
