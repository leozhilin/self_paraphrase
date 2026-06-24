#!/usr/bin/env bash
# Full SFT (raw + paraphrase) then complete vLLM eval on GPU 1.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="$HERE/configs/config.yaml"
PY="/data2/anaconda3/envs/vcts/bin/python"
LOG_DIR="$LZL_ROOT/logs/chart"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/full_sft_eval_gpu1_${TS}.log"

export CUDA_VISIBLE_DEVICES=1
export LZL_CONFIG="$CONFIG"

CKPT_ROOT=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['checkpoint_root'])")
mkdir -p "$LOG_DIR" "$CKPT_ROOT"
cd "$HERE"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "=== Chart full SFT + eval on GPU1  $(date)"
echo "=== LOG: $LOG"
echo "============================================================"

echo
echo "[1/2] Full fine-tuning: raw + paraphrase"
$PY 05_sft_train.py --config "$CONFIG" --condition both --tuning full

echo
echo "[2/2] Full eval: base + raw + paraphrase (4 datasets)"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase \
    --datasets chartqa_test plotqa tabmwp finqa

echo
echo "============================================================"
echo "=== Done  $(date)"
echo "=== Checkpoints: $CKPT_ROOT/{raw,paraphrase}"
echo "=== Results: $LZL_ROOT/results/chart/eval/"
echo "=== LOG: $LOG"
echo "============================================================"
