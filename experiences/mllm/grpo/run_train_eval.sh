#!/usr/bin/env bash
# MLLM GRPO train → full eval（experiences/mllm/grpo 入口）
#
# Usage:
#   nohup bash experiences/mllm/grpo/run_train_eval.sh \
#       > logs/mllm_grpo/pipeline.out 2>&1 &
#   echo $! > /tmp/mllm_grpo_pipeline.pid
set -euo pipefail

GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LZL_ROOT="$(cd "$GRPO_DIR/../../.." && pwd)"
cd "$LZL_ROOT"

PY=/data2/anaconda3/envs/vcts/bin/python
LOG="$LZL_ROOT/logs/mllm_grpo"
CKPT_ROOT=/data5/lzl/checkpoints/mllm_grpo
CONFIG="$LZL_ROOT/experiences/mllm/configs/config.yaml"
GRPO_MODEL=$($PY -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['path'])")
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LZL_CONFIG="$CONFIG"
export PYTHONPATH="$LZL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$($PY -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['datasets']['hf_cache'])")
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1

TRAIN_LOG="$LOG/full.out"
GMU=0.92
SEQS=256

echo "============================================================"
echo "=== MLLM GRPO train→eval (Qwen3.5-4B)  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES  tag: $TS"
echo "============================================================"

find_mllm_grpo_pid() {
  pgrep -f "swift/cli/rlhf.py.*grpo.*Qwen3.5-4B" | head -1 || true
}

resolve_grpo_adapter() {
  local ckpt=""
  if [[ -f "$TRAIN_LOG" ]]; then
    ckpt=$(grep -oE 'last_model_checkpoint: [^ ]+' "$TRAIN_LOG" | tail -1 | awk '{print $2}' || true)
  fi
  if [[ -n "$ckpt" && -d "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi
  local run_dir
  run_dir=$(ls -td "$CKPT_ROOT"/v*/ 2>/dev/null | head -1 || true)
  if [[ -z "$run_dir" ]]; then
    echo "[pipeline] ERROR: no GRPO run dir under $CKPT_ROOT" >&2
    exit 1
  fi
  ckpt=$(ls -d "$run_dir"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
    echo "[pipeline] ERROR: no checkpoint-* in $run_dir" >&2
    exit 1
  fi
  echo "$ckpt"
}

RUNNING_PID=$(find_mllm_grpo_pid)
if [[ -n "$RUNNING_PID" ]]; then
  echo "[stage 1] MLLM GRPO already running (PID=$RUNNING_PID), waiting..."
  while kill -0 "$RUNNING_PID" 2>/dev/null; do sleep 120; done
else
  echo "[stage 1] launching MLLM GRPO full train"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$GRPO_DIR/run_grpo.sh" full \
    2>&1 | tee "$LOG/train_${TS}.log"
  cp -f "$LOG/train_${TS}.log" "$TRAIN_LOG" 2>/dev/null || true
fi

GRPO_ADAPTER=$(resolve_grpo_adapter)
echo "[stage 1] GRPO adapter → $GRPO_ADAPTER"

echo
echo "[stage 2] vLLM eval base + grpo (all MLLM benchmarks)"
EVAL_ROOT="$LZL_ROOT/results/mllm/eval_grpo"
$PY "$LZL_ROOT/experiences/mllm/06_eval_vllm.py" \
  --config "$LZL_CONFIG" \
  --conditions base grpo \
  --grpo_adapter "$GRPO_ADAPTER" \
  --model "$GRPO_MODEL" \
  --output_dir "$EVAL_ROOT" \
  --gpu_memory_utilization "$GMU" \
  --max_num_seqs "$SEQS" \
  2>&1 | tee "$LOG/eval_${TS}.log"

SUMMARY="$EVAL_ROOT/summary_${TS}.txt"
{
  echo "MLLM GRPO train→eval finished: $(date)"
  echo "adapter: $GRPO_ADAPTER"
  echo "base model: $GRPO_MODEL (enable_thinking=false)"
  echo "results: $EVAL_ROOT"
  echo
  for f in "$EVAL_ROOT"/*.json; do
    [[ -f "$f" ]] || continue
    echo "=== $(basename "$f") ==="
    $PY - <<PY
import json
from pathlib import Path
data = json.loads(Path("$f").read_text())
for k, v in sorted(data.items()):
    if isinstance(v, dict) and "accuracy" in v:
        print(f"  {k}: {v['correct']}/{v['total']} = {v['accuracy']:.1%}")
PY
  done
} | tee "$SUMMARY"

echo
echo "============================================================"
echo "=== MLLM GRPO pipeline done  $(date)"
echo "=== adapter:  $GRPO_ADAPTER"
echo "=== results:  $EVAL_ROOT"
echo "=== summary:  $SUMMARY"
echo "============================================================"
