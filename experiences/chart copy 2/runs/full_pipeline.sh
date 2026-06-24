#!/usr/bin/env bash
# Chart Qwen3.5-4B full pipeline: rollout → paraphrase → full SFT (3 cases) → eval.
#
# Usage:
#   bash experiences/chart/runs/full_pipeline.sh
#   CUDA_VISIBLE_DEVICES=1 bash experiences/chart/runs/full_pipeline.sh
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
mkdir -p /home/liuyu/Projects/GRPO_research/VCTS/lzl/logs/chart_qwen35

LOG="/home/liuyu/Projects/GRPO_research/VCTS/lzl/logs/chart_qwen35/full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

cd "$HERE"

echo "============================================================"
echo "=== Chart Qwen3.5 FULL pipeline  $(date)"
echo "=== GPU=$CUDA_VISIBLE_DEVICES  CONFIG=$CONFIG"
echo "============================================================"

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  echo; echo "[0/7] prepare datasets"
  $PY 00_prepare_datasets.py
fi

echo; echo "[1/7] sample rollouts"
$PY 01_sample_rollouts.py --config "$CONFIG"

echo; echo "[2/7] build raw + vanilla manifests"
$PY 02_build_raw_manifest.py --config "$CONFIG"
$PY 02_build_vanilla_manifest.py --config "$CONFIG"

echo; echo "[3/7] generate paraphrases"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/7] build paraphrase manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/7] SFT all conditions"
SFT_ARGS=(--config "$CONFIG" --condition all)
if [[ -n "${SFT_TUNING:-}" ]]; then
  SFT_ARGS+=(--tuning "$SFT_TUNING")
fi
$PY 05_sft_train.py "${SFT_ARGS[@]}"

echo; echo "[6/7] eval"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase vanilla \
    --max_model_len "${EVAL_MAX_MODEL_LEN:-8192}"

echo; echo "============================================================"
echo "=== FULL pipeline done  $(date)  log=$LOG"
echo "============================================================"
