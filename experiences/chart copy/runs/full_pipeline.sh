#!/usr/bin/env bash
# Chart/Table-QA standard full pipeline: prepare -> rollout -> paraphrase ->
# SFT -> eval. All paths come from experiences/chart/configs/config.yaml.
#
# Usage:
#   bash experiences/chart/runs/full_pipeline.sh
#   CONFIG=/path/to/config.yaml bash experiences/chart/runs/full_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash experiences/chart/runs/full_pipeline.sh
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="${CONFIG:-$HERE/configs/config.yaml}"

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
if [[ "${SKIP_PREPARE:-0}" == "1" ]]; then
  echo "skip prepare (SKIP_PREPARE=1)"
else
  $PY 00_prepare_datasets.py
fi

echo; echo "[1/7] sample rollouts (G samples per question)"
if [[ "${SKIP_ROLLOUT:-0}" == "1" ]]; then
  echo "skip rollout (SKIP_ROLLOUT=1)"
elif [[ "${ROLLOUT_BACKEND:-vllm}" == "hf_dual" ]]; then
  CONFIG="$CONFIG" PY="$PY" bash "$HERE/runs/rollout_hf_dual.sh"
else
  $PY 01_sample_rollouts.py --config "$CONFIG" \
      --gpu_memory_utilization "${ROLLOUT_GMU:-0.92}" \
      --max_model_len "${ROLLOUT_MAX_MODEL_LEN:-8192}" \
      --max_num_seqs "${ROLLOUT_MAX_NUM_SEQS:-128}" \
      --chunk_size "${ROLLOUT_CHUNK_SIZE:-4}" \
      --tensor_parallel_size "${ROLLOUT_TP:-1}"
fi

echo; echo "[2/7] build raw manifest"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[2/7] build vanilla manifest"
$PY 02_build_vanilla_manifest.py --config "$CONFIG"

echo; echo "[3/7] generate paraphrase candidates (vLLM)"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/7] filter paraphrase candidates -> manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/7] SFT raw + paraphrase + vanilla (sft.tuning from config, override with SFT_TUNING=lora|full)"
SFT_ARGS=(--config "$CONFIG" --condition all)
if [[ -n "${SFT_TUNING:-}" ]]; then
  SFT_ARGS+=(--tuning "$SFT_TUNING")
fi
$PY 05_sft_train.py "${SFT_ARGS[@]}"

echo; echo "[6/7] eval base + raw + paraphrase + vanilla SFT (vLLM)"
$PY 06_eval_vllm.py --config "$CONFIG" \
    --conditions base raw paraphrase vanilla \
    --max_model_len "${EVAL_MAX_MODEL_LEN:-8192}"

echo; echo "============================================================"
echo "=== Chart pipeline done  $(date)"
echo "============================================================"
