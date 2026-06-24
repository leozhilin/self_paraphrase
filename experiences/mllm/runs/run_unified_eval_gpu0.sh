#!/usr/bin/env bash
# Unified MLLM eval on GPU0: re-run ALL models (base/raw/paraphrase/vanilla/grpo)
# in ONE vLLM instance with ONE eval code path, so every number shares the same
# decoding config, answer extractor and base. Writes to results/mllm/eval_unified/.
#
# SFT adapters : /data5/lzl/checkpoints/mllm_paraphrase_pgps9k/{raw,paraphrase,vanilla}
# GRPO adapter : /data5/lzl/checkpoints/mllm_grpo/v3-20260620-021753/checkpoint-8021
# Base model   : /data5/lzl/models/Qwen3.5-4B
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/mllm"
CONFIG="$HERE/configs/config.yaml"
PY="/data2/anaconda3/envs/vcts/bin/python"
LOG_DIR="$LZL_ROOT/logs/mllm"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/unified_eval_gpu0_${TS}.log"

read_cfg() {
  local key="$1"
  $PY -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c$key)"
}

GRPO_ADAPTER="/data5/lzl/checkpoints/mllm_grpo/v3-20260620-021753/checkpoint-8021"
BASE_MODEL="$(read_cfg "['model']['path']")"
EVAL_ROOT="$LZL_ROOT/results/mllm/eval_unified"
DATASETS="pgps9k_test mathverse mathvision ai2d_test mmmu_pro"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LZL_CONFIG="$CONFIG"
export PYTHONPATH="$LZL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$(read_cfg "['datasets']['hf_cache']")"
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$EVAL_ROOT"
cd "$HERE"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "=== Unified MLLM eval (base+raw+paraphrase+vanilla+grpo)  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   base: $BASE_MODEL"
echo "=== GRPO adapter: $GRPO_ADAPTER"
echo "=== datasets: $DATASETS"
echo "=== output: $EVAL_ROOT"
echo "=== LOG: $LOG"
echo "============================================================"

if [[ ! -d "$GRPO_ADAPTER" ]]; then
  echo "ERROR: GRPO adapter not found: $GRPO_ADAPTER" >&2
  exit 1
fi

$PY 06_eval_vllm.py \
    --config "$CONFIG" \
    --model "$BASE_MODEL" \
    --conditions base raw paraphrase vanilla grpo \
    --grpo_adapter "$GRPO_ADAPTER" \
    --datasets $DATASETS \
    --output_dir "$EVAL_ROOT" \
    --gpu_memory_utilization 0.90 \
    --max_num_seqs 64

echo
echo "=== SUMMARY ==="
SUMMARY="$EVAL_ROOT/summary_${TS}.txt"
{
  echo "Unified MLLM eval finished: $(date)"
  echo "base model: $BASE_MODEL"
  echo "SFT adapters: /data5/lzl/checkpoints/mllm_paraphrase_pgps9k/{raw,paraphrase,vanilla}"
  echo "GRPO adapter: $GRPO_ADAPTER"
  echo "results: $EVAL_ROOT"
  echo
  for f in "$EVAL_ROOT"/*.json; do
    [[ -f "$f" ]] || continue
    echo "=== $(basename "$f") ==="
    $PY - "$f" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
order = ["base", "raw_sft", "paraphrase_sft", "vanilla_sft", "grpo_rl"]
for k in order:
    v = data.get(k)
    if isinstance(v, dict) and "accuracy" in v:
        print(f"  {k:16s}: {v['correct']}/{v['total']} = {v['accuracy']:.1%}")
PY
  done
} | tee "$SUMMARY"

echo
echo "============================================================"
echo "=== Done  $(date)"
echo "=== Results: $EVAL_ROOT"
echo "=== Summary: $SUMMARY"
echo "=== LOG: $LOG"
echo "============================================================"
