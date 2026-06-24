#!/usr/bin/env bash
# Train (full SFT) + eval for chart Qwen3.5 — skips data prep.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="${CONFIG:-$HERE/configs/config.yaml}"
PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"
export PYTHONUNBUFFERED=1

mkdir -p /data4/models/Qwen3.5-4B/checkpoints/chart
mkdir -p "$LZL_ROOT/logs/chart_qwen35"

LOG="$LZL_ROOT/logs/chart_qwen35/train_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

cd "$HERE"

echo "============================================================"
echo "=== Chart Qwen3.5 train→eval  $(date)"
echo "=== GPU=$CUDA_VISIBLE_DEVICES  CONFIG=$CONFIG"
echo "=== LOG=$LOG"
echo "============================================================"

RAW="$LZL_ROOT/data/chart_qwen35/sft/raw/raw.jsonl"
PARA="$LZL_ROOT/data/chart_qwen35/sft/paraphrase/paraphrase.jsonl"
VAN="$LZL_ROOT/data/chart_qwen35/sft/vanilla/vanilla.jsonl"
for p in "$RAW" "$PARA" "$VAN"; do
  test -f "$p" || { echo "Missing $p"; exit 1; }
  echo "  data OK: $p ($(wc -l < "$p") lines)"
done

echo; echo "[train] full SFT — raw + paraphrase + vanilla"
SFT_ARGS=(--config "$CONFIG" --condition all)
if [[ -n "${SFT_TUNING:-}" ]]; then
  SFT_ARGS+=(--tuning "$SFT_TUNING")
fi
$PY 05_sft_train.py "${SFT_ARGS[@]}"

echo; echo "[eval] base + raw + paraphrase + vanilla (vLLM)"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase vanilla \
    --max_model_len "${EVAL_MAX_MODEL_LEN:-8192}"

echo; echo "============================================================"
echo "=== DONE  $(date)"
echo "============================================================"
