#!/usr/bin/env bash
# GSM GRPO train → eval on GPU1 (Qwen3-4B-Instruct-2507, SFT-aligned).
#
# Usage:
#   nohup bash experiences/gsm/grpo/run_train_eval.sh > logs/grpo/pipeline_gpu1.out 2>&1 &
#   echo $! > /tmp/gsm_grpo_pipeline.pid
set -euo pipefail

GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LZL_ROOT="$(cd "$GRPO_DIR/../../.." && pwd)"
cd "$LZL_ROOT"

PY=/data2/anaconda3/envs/vcts/bin/python
LOG="$LZL_ROOT/logs/grpo"
CKPT_ROOT=/data5/lzl/checkpoints/grpo
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export LZL_CONFIG="$LZL_ROOT/experiences/gsm/configs/config.yaml"
export PYTHONPATH="$LZL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1

TRAIN_LOG="$LOG/full_gpu1.out"
GMU=0.90
SEQS=256

echo "============================================================"
echo "=== GSM GRPO train→eval (Qwen3-4B)  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES  tag: $TS"
echo "============================================================"

find_gsm_grpo_pid() {
  pgrep -f "swift/cli/rlhf.py.*grpo.*Qwen3-4B-Instruct-2507" | head -1 || true
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

RUNNING_PID=$(find_gsm_grpo_pid)
if [[ -n "$RUNNING_PID" ]]; then
  echo "[stage 1] GSM GRPO already running (PID=$RUNNING_PID), waiting..."
  while kill -0 "$RUNNING_PID" 2>/dev/null; do sleep 120; done
else
  echo "[stage 1] launching GSM GRPO full train on GPU $CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$GRPO_DIR/run_grpo.sh" full \
    2>&1 | tee "$LOG/train_${TS}.log"
  cp -f "$LOG/train_${TS}.log" "$TRAIN_LOG" 2>/dev/null || true
fi

GRPO_ADAPTER=$(resolve_grpo_adapter)
echo "[stage 1] GRPO adapter -> $GRPO_ADAPTER"

echo
echo "[stage 2] build eval caches"
$PY "$LZL_ROOT/scripts/06b_robust.py" build --datasets main gsm8k_test \
  2>&1 | tee "$LOG/eval_build_${TS}.log"

echo
echo "[stage 3] vLLM eval base + grpo (all GSM test sets)"
$PY "$LZL_ROOT/experiences/gsm/06_eval.py" \
  --datasets all \
  --conditions base grpo \
  --grpo_adapter "$GRPO_ADAPTER" \
  --output_subdir eval_grpo \
  --max_new_tokens 4096 \
  --max_model_len 12288 \
  --max_num_seqs "$SEQS" \
  --gpu_memory_utilization "$GMU" \
  --max_lora_rank 16 \
  2>&1 | tee "$LOG/eval_grpo_${TS}.log"

SUMMARY="$LZL_ROOT/results/eval_grpo/summary_${TS}.txt"
{
  echo "GSM GRPO train→eval finished: $(date)"
  echo "adapter: $GRPO_ADAPTER"
  echo "results: $LZL_ROOT/results/eval_grpo"
  echo
  for f in "$LZL_ROOT/results/eval_grpo"/*.json; do
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
echo "=== GSM GRPO pipeline done  $(date)"
echo "=== adapter:  $GRPO_ADAPTER"
echo "=== summary:  $SUMMARY"
echo "============================================================"
