#!/usr/bin/env bash
# MLLM smoke pipeline (PGPS9K 1k subset + special manifest builders).
# Steps: prepare-subset -> rollout -> raw_one_trace -> paraphrase ->
# para_all_valid -> SFT -> eval. Uses smoke config.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
SMOKE="$LZL_ROOT/experiences/mllm/smoke"
PARENT="$LZL_ROOT/experiences/mllm"           # standard 03/05/06 reused
CONFIG="$SMOKE/configs/config_smoke.yaml"

PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"

echo "============================================================"
echo "=== MLLM smoke pipeline  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES   config: $CONFIG"
echo "============================================================"

echo; echo "[0/7] make 1k smoke subset of PGPS9K"
$PY "$SMOKE/00_make_pgps9k_smoke_subset.py" --config "$CONFIG"

echo; echo "[1/7] sample rollouts (vLLM, vision)"
$PY "$PARENT/01_sample_rollouts.py" --config "$CONFIG"

echo; echo "[2/7] build raw manifest (one trace per sample, smoke-special)"
$PY "$SMOKE/02_build_raw_one_trace_per_sample.py" --config "$CONFIG"

echo; echo "[2/7] build vanilla manifest (standard)"
$PY "$PARENT/02_build_vanilla_manifest.py" --config "$CONFIG"

echo; echo "[3/7] generate paraphrase candidates (vLLM, vision)"
$PY "$PARENT/03_generate_paraphrases.py" --config "$CONFIG"

echo; echo "[4/7] filter paraphrase candidates (all-valid, smoke-special)"
$PY "$SMOKE/04_build_paraphrase_all_valid.py" --config "$CONFIG"

echo; echo "[5/7] SFT three conditions"
$PY "$PARENT/05_sft_train.py" --config "$CONFIG" --condition all

echo; echo "[6/7] eval base + three SFT adapters (vLLM, vision)"
$PY "$PARENT/06_eval_vllm.py" --config "$CONFIG" \
    --conditions base raw paraphrase vanilla

echo; echo "============================================================"
echo "=== MLLM smoke pipeline done  $(date)"
echo "============================================================"
