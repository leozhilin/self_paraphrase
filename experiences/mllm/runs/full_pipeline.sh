#!/usr/bin/env bash
# MLLM (vision) standard full pipeline: prepare -> rollout -> paraphrase ->
# SFT -> eval. Vision path: image+question conditioning throughout.
# All paths come from experiences/mllm/configs/config.yaml.
#
# Usage:
#   bash experiences/mllm/runs/full_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash experiences/mllm/runs/full_pipeline.sh
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/mllm"
CONFIG="$HERE/configs/config.yaml"

PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"

cd "$HERE"

echo "============================================================"
echo "=== MLLM full pipeline  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   config: $CONFIG"
echo "============================================================"

echo; echo "[0/7] prepare MLLM datasets (PGPS9K + eval benchmarks)"
$PY 00_prepare_datasets.py --config "$CONFIG"

echo; echo "[1/7] sample rollouts (vLLM, vision)"
$PY 01_sample_rollouts.py --config "$CONFIG"

echo; echo "[2/7] build raw manifest"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[2/7] build vanilla manifest"
$PY 02_build_vanilla_manifest.py --config "$CONFIG"

echo; echo "[3/7] generate paraphrase candidates (vLLM, vision)"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/7] filter paraphrase candidates -> manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/7] SFT three conditions (raw / vanilla / paraphrase)"
$PY 05_sft_train.py --config "$CONFIG" --condition all

echo; echo "[6/7] eval base + three SFT adapters (vLLM, vision)"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase vanilla

echo; echo "============================================================"
echo "=== MLLM pipeline done  $(date)"
echo "============================================================"
