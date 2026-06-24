#!/usr/bin/env bash
# Continue chart pipeline from existing raw/paraphrase/vanilla JSONL files:
# convert to ms-swift messages format, run full-parameter SFT, then vLLM eval.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="${CONFIG:-$HERE/configs/config.yaml}"

PY="${PY:-/data2/anaconda3/envs/vcts/bin/python}"
SWIFT="${SWIFT:-/data2/anaconda3/envs/vcts/bin/swift}"

GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NPROC="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
MASTER_PORT="${MASTER_PORT:-29541}"

MODEL="${MODEL:-$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['path'])")}"
CKPT_ROOT="${CKPT_ROOT:-$($PY -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['paths']['checkpoint_root'])")}"
DATA_ROOT="${DATA_ROOT:-$($PY -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['paths']['raw_jsonl'].replace('/sft/raw/raw.jsonl','/sft'))")}"
LOG_DIR="${LOG_DIR:-$LZL_ROOT/logs/chart_qwen35}"

CONDITIONS="${CONDITIONS:-raw paraphrase vanilla}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
MAX_STEPS="${MAX_STEPS:--1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"

mkdir -p "$LOG_DIR" "$CKPT_ROOT"

cd "$HERE"
export LZL_CONFIG="$CONFIG"

echo "============================================================"
echo "=== Chart ms-swift SFT + eval  $(date)"
echo "=== GPU: $GPUS nproc=$NPROC port=$MASTER_PORT"
echo "=== config: $CONFIG"
echo "============================================================"

echo; echo "[0/3] convert SFT JSONL -> ms-swift messages"
$PY 05a_prepare_swift_sft.py --config "$CONFIG" --condition all

for cond in $CONDITIONS; do
  case "$cond" in
    raw) DATA_FILE="$DATA_ROOT/raw/swift_messages.jsonl" ;;
    paraphrase) DATA_FILE="$DATA_ROOT/paraphrase/swift_messages.jsonl" ;;
    vanilla) DATA_FILE="$DATA_ROOT/vanilla/swift_messages.jsonl" ;;
    *) echo "unknown condition: $cond" >&2; exit 2 ;;
  esac
  OUT_DIR="$CKPT_ROOT/$cond"

  echo; echo "[1/3] swift full SFT condition=$cond data=$DATA_FILE out=$OUT_DIR"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  NPROC_PER_NODE="$NPROC" \
  MASTER_PORT="$MASTER_PORT" \
  $SWIFT sft \
    --model "$MODEL" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --enable_thinking false \
    --dataset "$DATA_FILE" \
    --max_length "$MAX_LENGTH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --output_dir "$OUT_DIR" \
    --add_version false \
    --report_to none \
    --seed 42
done

if [[ "$RUN_EVAL" == "1" ]]; then
  echo; echo "[2/3] eval base + raw + paraphrase + vanilla"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  $PY 06_eval_vllm.py --config "$CONFIG" \
      --conditions base raw paraphrase vanilla \
      --max_model_len "$EVAL_MAX_MODEL_LEN"
else
  echo; echo "[2/3] skip eval (RUN_EVAL=$RUN_EVAL)"
fi

echo; echo "============================================================"
echo "=== Chart ms-swift SFT + eval done  $(date)"
echo "============================================================"
