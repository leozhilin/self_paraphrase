#!/usr/bin/env bash
# Finish the interrupted 1.44M full-SFT run.
#
# State at creation time (2026-06-22):
#   - raw/paraphrase/vanilla manifests already rebuilt at 1.44M tokens.
#   - chart_full/raw     : full SFT DONE (top-level config.json + model.safetensors).
#   - chart_full/paraphrase : empty (not trained).
#   - chart_full/vanilla : missing (not trained).
#
# This script skips rollout + paraphrase generation + manifest build (already done)
# and the already-finished raw training, then:
#   1) full fine-tune paraphrase
#   2) full fine-tune vanilla
#   3) full eval: base + raw + paraphrase + vanilla on all 4 datasets
#
# To force-retrain raw too, set RETRAIN_RAW=1.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="$HERE/configs/config.yaml"
PY="/data2/anaconda3/envs/vcts/bin/python"
LOG_DIR="$LZL_ROOT/logs/chart"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/finish_rebuild_1440k_full_${TS}.log"
CKPT_ROOT=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['checkpoint_root'])")
RETRAIN_RAW="${RETRAIN_RAW:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LZL_CONFIG="$CONFIG"

mkdir -p "$LOG_DIR" "$CKPT_ROOT"
cd "$HERE"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "=== Finish 1.44M full SFT + eval  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   config: $CONFIG"
echo "=== RETRAIN_RAW=$RETRAIN_RAW"
echo "=== LOG: $LOG"
echo "============================================================"

raw_done() { [[ -f "$CKPT_ROOT/raw/config.json" && -f "$CKPT_ROOT/raw/model.safetensors" ]]; }

if [[ "$RETRAIN_RAW" == "1" ]] || ! raw_done; then
  echo
  echo "[train] full fine-tune raw"
  $PY 05_sft_train.py --config "$CONFIG" --condition raw --tuning full
else
  echo
  echo "[skip] raw full SFT already complete → $CKPT_ROOT/raw"
fi

echo
echo "[train] full fine-tune paraphrase"
$PY 05_sft_train.py --config "$CONFIG" --condition paraphrase --tuning full

echo
echo "[train] full fine-tune vanilla"
$PY 05_sft_train.py --config "$CONFIG" --condition vanilla --tuning full

echo
echo "[eval] full eval: base + raw + paraphrase + vanilla"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase vanilla \
    --datasets chartqa_test plotqa tabmwp finqa

echo
echo "============================================================"
echo "=== Done  $(date)"
echo "=== Checkpoints: $CKPT_ROOT/{raw,paraphrase,vanilla}"
echo "=== Results: $LZL_ROOT/results/chart/eval/"
echo "=== LOG: $LOG"
echo "============================================================"
