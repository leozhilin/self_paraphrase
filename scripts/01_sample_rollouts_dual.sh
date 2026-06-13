#!/usr/bin/env bash
# 双卡并行 GSM8K train rollout 采样，完成后 merge 并可选续跑 pipeline
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data5/lzl/hf_datasets}"
export HF_HOME="${HF_HOME:-/data5/lzl/hf_cache}"

SCRIPTS="$VCTS/lzl/scripts"
ROLLOUTS="$VCTS/lzl/data/rollouts/gsm8k_train_g32.jsonl"
SHARD_DIR="$VCTS/lzl/data/rollouts/shards"
TOTAL=7473
mkdir -p "$SHARD_DIR" "$VCTS/lzl/logs"

EXISTING=0
PREFIX=""
if [ -f "$ROLLOUTS" ]; then
  EXISTING=$(wc -l < "$ROLLOUTS")
fi

if [ "$EXISTING" -ge "$TOTAL" ]; then
  echo "Rollouts already complete ($EXISTING/$TOTAL), skip sampling."
  exit 0
fi

if [ "$EXISTING" -gt 0 ]; then
  PREFIX="$SHARD_DIR/prefix_${EXISTING}.jsonl"
  head -n "$EXISTING" "$ROLLOUTS" > "$PREFIX"
  echo "Reuse existing prefix: $EXISTING questions → $PREFIX"
fi

START=$EXISTING
REMAIN=$((TOTAL - START))
HALF=$((REMAIN / 2))
SHARD0="$SHARD_DIR/gpu0_s${START}_n${HALF}.jsonl"
SHARD1="$SHARD_DIR/gpu1_s$((START + HALF))_n$((REMAIN - HALF)).jsonl"

echo "=== dual-GPU sampling $(date) ==="
echo "  total=$TOTAL  done=$EXISTING  remain=$REMAIN"
echo "  GPU0: start=$START limit=$HALF → $SHARD0"
echo "  GPU1: start=$((START + HALF)) limit=$((REMAIN - HALF)) → $SHARD1"

run_shard() {
  local gpu=$1 start=$2 limit=$3 out=$4
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCRIPTS/01_sample_rollouts.py" \
    --start "$start" --limit "$limit" --output "$out"
}

run_shard 0 "$START" "$HALF" "$SHARD0" &
PID0=$!
run_shard 1 $((START + HALF)) $((REMAIN - HALF)) "$SHARD1" &
PID1=$!

echo "  GPU0 pid=$PID0  GPU1 pid=$PID1"
wait "$PID0" "$PID1"

echo "=== merge $(date) ==="
: > "$ROLLOUTS"
if [ -n "$PREFIX" ]; then
  cat "$PREFIX" >> "$ROLLOUTS"
fi
cat "$SHARD0" >> "$ROLLOUTS"
cat "$SHARD1" >> "$ROLLOUTS"

FINAL=$(wc -l < "$ROLLOUTS")
echo "Merged → $ROLLOUTS  ($FINAL lines)"
if [ "$FINAL" -ne "$TOTAL" ]; then
  echo "ERROR: expected $TOTAL lines, got $FINAL" >&2
  exit 1
fi

echo "=== dual sampling done $(date) ==="
