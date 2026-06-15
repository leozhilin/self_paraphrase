#!/usr/bin/env bash
# Eval legacy vcts_sft_pilot adapters with new vLLM stack (isolated output dir).
#   nohup bash lzl/run_tmp_eval_legacy_pilot.sh > lzl/logs/eval_legacy_pilot.out 2>&1 &
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export LZL_CONFIG="$VCTS/lzl/config.yaml"
export PYTHONPATH="$VCTS:$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MNT=4096
MML=12288
SEQS=128
GMU=0.80

echo "=== legacy pilot eval (old #### prompts) start $(date) GPU=$CUDA_VISIBLE_DEVICES ==="

$PY "$VCTS/lzl/scripts/tmp_eval_legacy_pilot_vllm.py" \
  --datasets all \
  --conditions base raw paraphrase mixed control \
  --max_new_tokens "$MNT" \
  --max_model_len "$MML" \
  --max_num_seqs "$SEQS" \
  --gpu_memory_utilization "$GMU"

echo "=== legacy pilot eval done $(date) ==="
echo "Results: $VCTS/lzl/results/eval_legacy_pilot/"
