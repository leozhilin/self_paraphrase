#!/usr/bin/env bash
# 统一评测（纯 eval，无 robust 阶段）。
# 一次性在所有公认测试集上用 vLLM 评 base/raw/paraphrase/vanilla：
#   gsm(GSM-Symbolic) gsm8k_test svamp asdiv multiarith
#   aqua_rat math500 mawps gsm_hard   <- 原 robust 里的公认集，已并入统一脚本
#
# 说明：
#   - 答案匹配与原 06b 完全一致（共用 extract_model_answer + answers_match，
#     answers_match 内部用 math_verify，支持 LaTeX / 分数 / 字母选项）。
#   - 06b_robust.py 仅保留用于构建 gsm_symbolic / gsm8k_test 的 JSONL，不再做评测。
#   - 大 batch 提速：vLLM 走 --max_num_seqs 256（连续批处理，越大吞吐越高）。
#
# 用法：
#   nohup bash lzl/run_eval_unified.sh > lzl/logs/eval_unified.out 2>&1 &
#   echo $! > /tmp/eval_unified.pid
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export LZL_CONFIG="$VCTS/lzl/config.yaml"
export PYTHONPATH="$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1
# 用空闲的 GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LZL="$VCTS/lzl"
SCRIPTS="$LZL/scripts"
LOG="$LZL/logs"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

echo "============================================================"
echo "=== Unified eval start  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage A. 构建 gsm_symbolic / gsm8k_test 的 JSONL（06b 仅作 builder，幂等）
# ---------------------------------------------------------------------------
echo
echo "[stage A] build base eval JSONLs (gsm_symbolic main + gsm8k_test)"
$PY "$SCRIPTS/06b_robust.py" build --datasets main gsm8k_test \
  2>&1 | tee "$LOG/eval_build_${TS}.log"

# ---------------------------------------------------------------------------
# Stage B. 统一 vLLM 评测（全部公认集，大 batch 提速）
# ---------------------------------------------------------------------------
echo
echo "[stage B] unified vLLM eval on all public test sets"
$PY "$SCRIPTS/06_eval.py" \
  --datasets all \
  --conditions base raw paraphrase vanilla \
  --max_num_seqs 256 \
  --gpu_memory_utilization 0.85 \
  2>&1 | tee "$LOG/eval_unified_${TS}.log"

echo
echo "============================================================"
echo "=== Unified eval done  $(date)"
echo "=== results: $LZL/results/eval/"
echo "============================================================"
