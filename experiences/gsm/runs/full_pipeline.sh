#!/usr/bin/env bash
# GSM/Math standard full pipeline: rollout -> paraphrase -> SFT -> eval.
#
# Each step writes to paths defined in experiences/gsm/configs/config.yaml.
# Re-running is mostly idempotent (rollout/paraphrase resume from existing
# JSONL; manifest builders skip if outputs exist; SFT trains all conditions).
#
# Usage:
#   bash experiences/gsm/runs/full_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash experiences/gsm/runs/full_pipeline.sh
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/gsm"
CONFIG="$HERE/configs/config.yaml"

PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"

cd "$HERE"

echo "============================================================"
echo "=== GSM full pipeline  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   config: $CONFIG"
echo "============================================================"

echo; echo "[1/6] sample rollouts (G samples per question)"
$PY 01_sample_rollouts.py --config "$CONFIG"

echo; echo "[2/6] build raw manifest (token-matched, NAC-balanced)"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[2/6] build vanilla manifest (compute-matched control)"
$PY 02_build_vanilla_manifest.py --config "$CONFIG"

echo; echo "[3/6] generate paraphrase candidates (vLLM)"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/6] filter paraphrase candidates -> manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/6] SFT three conditions (raw / vanilla / paraphrase)"
$PY 05_sft_train.py --config "$CONFIG" --condition all

echo; echo "[6/6] eval base + three SFT adapters on all benchmarks"
$PY 06_eval.py --config "$CONFIG" \
    --datasets all \
    --conditions base raw paraphrase vanilla

echo; echo "============================================================"
echo "=== GSM pipeline done  $(date)"
echo "============================================================"
