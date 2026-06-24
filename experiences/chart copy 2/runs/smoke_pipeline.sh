#!/usr/bin/env bash
# Smoke: Qwen3.5-4B chart pipeline on raw + paraphrase (2 cases), tiny limits.
#
# Usage:
#   bash experiences/chart/runs/smoke_pipeline.sh
#   CUDA_VISIBLE_DEVICES=1 bash experiences/chart/runs/smoke_pipeline.sh
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="${CONFIG:-$HERE/configs/config_smoke.yaml}"

PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
ROLLout_LIMIT="${SMOKE_ROLLOUT_LIMIT:-8}"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"
export PYTHONUNBUFFERED=1

mkdir -p /data4/models/Qwen3.5-4B/checkpoints/chart_smoke
mkdir -p /home/liuyu/Projects/GRPO_research/VCTS/lzl/logs/chart_qwen35_smoke

LOG="/home/liuyu/Projects/GRPO_research/VCTS/lzl/logs/chart_qwen35_smoke/smoke_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

cd "$HERE"

echo "============================================================"
echo "=== Chart Qwen3.5 SMOKE  $(date)"
echo "=== GPU=$CUDA_VISIBLE_DEVICES  CONFIG=$CONFIG"
echo "=== rollout_limit=$ROLLout_LIMIT  cases=raw+paraphrase"
echo "============================================================"

echo; echo "[1/6] sample rollouts (limit=$ROLLout_LIMIT)"
$PY 01_sample_rollouts.py --config "$CONFIG" --limit "$ROLLout_LIMIT"

echo; echo "[2/6] build raw manifest"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[3/6] generate paraphrases"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/6] build paraphrase manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/6] SFT raw + paraphrase (tuning from config)"
$PY 05_sft_train.py --config "$CONFIG" --condition both

echo; echo "[6/6] eval raw + paraphrase"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --datasets chartqa_test \
    --conditions raw paraphrase

echo; echo "============================================================"
echo "=== SMOKE OK  $(date)  log=$LOG"
echo "============================================================"
