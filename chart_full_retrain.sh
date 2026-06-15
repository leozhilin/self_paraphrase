#!/usr/bin/env bash
# Chart 完整重训 + 全量评测（SFT label-mask 修复后重跑）
#
# 复用已有 rollout / paraphrase 数据（28k 题 rollout 不重采样）。
# 备份整目录 chart_paraphrase → chart_paraphrase_bak_<TS> 后重训 raw/paraphrase/vanilla。
#
# Usage:
#   nohup bash lzl/chart_full_retrain.sh > lzl/logs/chart/chart_full_retrain.out 2>&1 &
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export LZL_CONFIG="$VCTS/lzl/chart_config.yaml"
export PYTHONPATH="$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LZL="$VCTS/lzl"
SCRIPTS="$LZL/scripts"
LOG="$LZL/logs/chart"
CKPT=/data5/lzl/checkpoints/chart_paraphrase
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

echo "============================================================"
echo "=== chart full retrain start  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES"
echo "=== tag: $TS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0. Backup existing adapters (full tree, incl. *_v0)
# ---------------------------------------------------------------------------
echo
echo "[stage 0] backup chart_paraphrase → chart_paraphrase_bak_${TS}"
if [[ -d "$CKPT" ]]; then
  BK="/data5/lzl/checkpoints/chart_paraphrase_bak_${TS}"
  cp -a "$CKPT" "$BK"
  echo "  $CKPT → $BK"
else
  echo "  no $CKPT — skip backup"
fi

# ---------------------------------------------------------------------------
# Stage 1. Rebuild manifests (CPU, fast)
# ---------------------------------------------------------------------------
echo
echo "[stage 1a] build raw manifest (idempotent)"
$PY "$SCRIPTS/02_build_raw_manifest.py" 2>&1 | tee "$LOG/raw_manifest_${TS}.log"

echo
echo "[stage 1b] build paraphrase manifest (idempotent)"
$PY "$SCRIPTS/04_build_paraphrase_manifest.py" 2>&1 | tee "$LOG/paraphrase_manifest_${TS}.log"

echo
echo "[stage 1c] build vanilla manifest"
$PY "$SCRIPTS/02d_build_chart_vanilla_manifest.py" 2>&1 | tee "$LOG/vanilla_manifest_${TS}.log"

echo
echo "[stage 1d] build ChartQA test eval JSONL"
$PY "$SCRIPTS/00_prepare_chart_datasets.py" --only chartqa_test 2>&1 \
  | tee "$LOG/chartqa_test_build_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 2. SFT all conditions (assistant-only label mask)
# ---------------------------------------------------------------------------
echo
echo "[stage 2] SFT train raw + paraphrase + vanilla"
$PY "$SCRIPTS/05_sft_train.py" --condition all 2>&1 | tee "$LOG/sft_all_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 3. Eval base + raw + paraphrase + vanilla (vLLM)
# ---------------------------------------------------------------------------
echo
echo "[stage 3] eval ChartQA test + PlotQA + TabMWP + FinQA (vLLM)"
$PY "$SCRIPTS/06_eval_chart_vllm.py" \
  --datasets chartqa_test plotqa tabmwp finqa \
  --conditions base raw paraphrase vanilla \
  --gpu_memory_utilization 0.85 \
  --max_num_seqs 128 \
  2>&1 | tee "$LOG/eval_vllm_${TS}.log"

echo
echo "============================================================"
echo "=== chart full retrain done  $(date)"
echo "=== backup: /data5/lzl/checkpoints/chart_paraphrase_bak_${TS}"
echo "=== checkpoints: $CKPT"
echo "=== results: $LZL/results/chart/eval/"
echo "============================================================"
