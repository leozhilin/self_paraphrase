#!/usr/bin/env bash
# Dual-GPU HF rollout for chart: split train questions, sample on two GPUs, merge.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="${CONFIG:-$HERE/configs/config.yaml}"
PY="${PY:-/data2/anaconda3/envs/vcts/bin/python}"

export PYTHONPATH="$LZL_ROOT:$LZL_ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
export LZL_CONFIG="$CONFIG"

readarray -t CFG_LINES < <("$PY" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from paths import get_paths, load_config
cfg = load_config(sys.argv[1])
paths = get_paths(cfg)
print(cfg["datasets"]["train_jsonl"])
print(paths.rollouts)
PY
)

TRAIN_JSONL="${CFG_LINES[0]}"
ROLLOUTS="${CFG_LINES[1]}"
SHARD_DIR="$(dirname "$ROLLOUTS")/shards_hf_dual"
mkdir -p "$SHARD_DIR" "$(dirname "$ROLLOUTS")"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Train JSONL not found: $TRAIN_JSONL" >&2
  exit 1
fi

TOTAL=$(wc -l < "$TRAIN_JSONL")
EXISTING=0
PREFIX=""
if [[ -f "$ROLLOUTS" ]]; then
  EXISTING=$(wc -l < "$ROLLOUTS")
fi

if [[ "$EXISTING" -ge "$TOTAL" ]]; then
  echo "Rollouts already complete ($EXISTING/$TOTAL), skip sampling."
  exit 0
fi

if [[ "$EXISTING" -gt 0 ]]; then
  PREFIX="$SHARD_DIR/prefix_${EXISTING}.jsonl"
  head -n "$EXISTING" "$ROLLOUTS" > "$PREFIX"
  echo "Reuse existing prefix: $EXISTING questions -> $PREFIX"
fi

START=$EXISTING
REMAIN=$((TOTAL - START))
HALF=$((REMAIN / 2))
SHARD0="$SHARD_DIR/gpu0_s${START}_n${HALF}.jsonl"
SHARD1="$SHARD_DIR/gpu1_s$((START + HALF))_n$((REMAIN - HALF)).jsonl"

echo "=== chart HF dual rollout $(date) ==="
echo "  config=$CONFIG"
echo "  total=$TOTAL done=$EXISTING remain=$REMAIN"
echo "  GPU0: start=$START limit=$HALF -> $SHARD0"
echo "  GPU1: start=$((START + HALF)) limit=$((REMAIN - HALF)) -> $SHARD1"

run_shard() {
  local gpu=$1 start=$2 limit=$3 out=$4
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$HERE/01_sample_rollouts_hf.py" \
    --config "$CONFIG" \
    --start "$start" \
    --limit "$limit" \
    --output "$out"
}

run_shard 0 "$START" "$HALF" "$SHARD0" &
PID0=$!
run_shard 1 $((START + HALF)) $((REMAIN - HALF)) "$SHARD1" &
PID1=$!
echo "  GPU0 pid=$PID0 GPU1 pid=$PID1"
wait "$PID0" "$PID1"

echo "=== merge $(date) ==="
: > "$ROLLOUTS"
if [[ -n "$PREFIX" ]]; then
  cat "$PREFIX" >> "$ROLLOUTS"
fi
cat "$SHARD0" >> "$ROLLOUTS"
cat "$SHARD1" >> "$ROLLOUTS"

FINAL=$(wc -l < "$ROLLOUTS")
echo "Merged -> $ROLLOUTS ($FINAL lines)"
if [[ "$FINAL" -ne "$TOTAL" ]]; then
  echo "ERROR: expected $TOTAL lines, got $FINAL" >&2
  exit 1
fi
echo "=== chart HF dual rollout done $(date) ==="
