#!/usr/bin/env bash
# Rebuild raw + paraphrase at 1.44M tokens, then full SFT + eval on GPU1.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="$HERE/configs/config.yaml"
PY="/data2/anaconda3/envs/vcts/bin/python"
LOG_DIR="$LZL_ROOT/logs/chart"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/rebuild_1440k_sft_eval_gpu1_${TS}.log"

export CUDA_VISIBLE_DEVICES=1
export LZL_CONFIG="$CONFIG"

mkdir -p "$LOG_DIR" /data5/lzl/checkpoints/chart_full
cd "$HERE"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "=== Chart 1.44M rebuild + full SFT + eval  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   config: $CONFIG"
echo "=== LOG: $LOG"
echo "============================================================"

echo
echo "[1/5] Rebuild raw manifest (target_tokens=1.44M)"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo
echo "[2/5] Generate paraphrase candidates (vLLM, new raw)"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo
echo "[3/5] Build paraphrase manifest (target_tokens=1.44M)"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo
echo "[4/5] Full fine-tuning: raw + paraphrase"
$PY 05_sft_train.py --config "$CONFIG" --condition both --tuning full

echo
echo "[5/5] Full eval: base + raw + paraphrase"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase \
    --datasets chartqa_test plotqa tabmwp finqa

echo
echo "============================================================"
echo "=== Done  $(date)"
echo "=== Data: .../data/chart/sft/{raw,paraphrase}/"
echo "=== Checkpoints: /data5/lzl/checkpoints/chart_full/{raw,paraphrase}"
echo "=== Results: $LZL_ROOT/results/chart/eval/"
echo "=== LOG: $LOG"
echo "============================================================"
