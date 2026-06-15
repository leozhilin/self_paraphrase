#!/usr/bin/env bash
# GSM answer-only vanilla ablation (chart-style: Final Answer: <gold> only).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash lzl/run_gsm_vanilla_answer_only.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # VCTS/

PY=/data2/anaconda3/envs/vcts/bin/python
LZL=/home/liuyu/Projects/GRPO_research/VCTS/lzl
LOG=$LZL/logs/vanilla_answer_only
mkdir -p "$LOG"

GPU=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "=== GSM vanilla answer-only  GPU=$GPU  $(date)"
echo "============================================================"

echo
echo "[stage 1] build manifest"
$PY $LZL/scripts/02f_build_vanilla_answer_only_manifest.py \
    2>&1 | tee "$LOG/01_manifest.log"

echo
echo "[stage 2] SFT train → /data5/lzl/checkpoints/vanilla_answer_only"
$PY $LZL/scripts/05_sft_train.py --condition vanilla_answer_only \
    2>&1 | tee "$LOG/02_sft.log"

echo
echo "[stage 3] eval all text benchmarks"
$PY $LZL/scripts/06_eval.py \
    --datasets gsm8k_test gsm svamp asdiv multiarith \
    --conditions vanilla_answer_only \
    2>&1 | tee "$LOG/03_eval.log"

echo
echo "============================================================"
echo "=== DONE  $(date)"
echo "============================================================"
