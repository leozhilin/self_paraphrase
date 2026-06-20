#!/usr/bin/env bash
# Chart/Table-QA standard full pipeline: prepare -> rollout -> paraphrase ->
# SFT -> eval. All paths come from experiences/chart/configs/config.yaml.
#
# Usage:
#   bash experiences/chart/runs/full_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash experiences/chart/runs/full_pipeline.sh
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="$HERE/configs/config.yaml"

PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"

cd "$HERE"

echo "============================================================"
echo "=== Chart full pipeline  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   config: $CONFIG"
echo "============================================================"

echo; echo "[0/7] prepare ChartQA train/eval JSONL"
$PY 00_prepare_datasets.py --config "$CONFIG"

echo; echo "[1/7] sample rollouts (G samples per question)"
$PY 01_sample_rollouts.py --config "$CONFIG"

echo; echo "[2/7] build raw manifest"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[2/7] build vanilla manifest"
$PY 02_build_vanilla_manifest.py --config "$CONFIG"

echo; echo "[3/7] generate paraphrase candidates (vLLM)"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/7] filter paraphrase candidates -> manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/7] SFT three conditions (raw / vanilla / paraphrase)"
$PY 05_sft_train.py --config "$CONFIG" --condition all

echo; echo "[6/7] eval base + three SFT adapters (vLLM)"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase vanilla

echo; echo "============================================================"
echo "=== Chart pipeline done  $(date)"
echo "============================================================"
