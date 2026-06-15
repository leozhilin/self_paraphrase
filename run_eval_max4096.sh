#!/usr/bin/env bash
# Sequential GSM + chart eval with max_new_tokens=4096 (no truncation study).
# vLLM max_model_len bumped to 12288 to fit long prompts (finqa/tabmwp) + 4096 gen.
#   nohup bash lzl/run_eval_max4096.sh > lzl/logs/eval_max4096.out 2>&1 &
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export PYTHONPATH="$VCTS:$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG="$VCTS/lzl/logs"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

MNT=4096          # max_new_tokens
MML=12288         # max_model_len (prompt + generation)
SEQS=128
GMU=0.80          # leave headroom for ReplayRAG on the same GPU

echo "############################################################"
echo "### GSM eval  max_new_tokens=$MNT  start $(date)"
echo "############################################################"
LZL_CONFIG="$VCTS/lzl/config.yaml" $PY "$VCTS/lzl/scripts/06_eval.py" \
  --datasets all \
  --conditions base raw paraphrase vanilla \
  --max_new_tokens "$MNT" \
  --max_model_len "$MML" \
  --max_num_seqs "$SEQS" \
  --gpu_memory_utilization "$GMU" \
  2>&1 | tee "$LOG/eval_gsm_4096_${TS}.log"

echo "############################################################"
echo "### chart eval  max_new_tokens=$MNT  start $(date)"
echo "############################################################"
LZL_CONFIG="$VCTS/lzl/chart_config.yaml" $PY "$VCTS/lzl/scripts/06_eval_chart_vllm.py" \
  --datasets chartqa_test plotqa tabmwp finqa \
  --conditions base raw paraphrase vanilla \
  --max_new_tokens "$MNT" \
  --max_model_len "$MML" \
  --max_num_seqs "$SEQS" \
  --gpu_memory_utilization "$GMU" \
  2>&1 | tee "$LOG/eval_chart_4096_${TS}.log"

echo "############################################################"
echo "### all eval done $(date)"
echo "###   GSM   results: $VCTS/lzl/results/eval/"
echo "###   chart results: $VCTS/lzl/results/chart/eval/"
echo "############################################################"
