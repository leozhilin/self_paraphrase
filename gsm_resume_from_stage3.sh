#!/usr/bin/env bash
# 从 stage 3 续跑 GSM 全量重训实验（上次在 stage3 paraphrase 生成处 OOM 崩溃）。
# stage 0/1/2 的产物（rollouts + raw.jsonl + manifest）已存在，无需重跑。
#
# 关键修复：
#   1. 固定用空闲的 GPU 0（上次崩溃是显存被占）
#   2. stage 3 paraphrase 生成显存比例降到 0.6（4B 模型够用，避免再 OOM）
#
# 用法：
#   nohup bash lzl/gsm_resume_from_stage3.sh > lzl/logs/gsm_resume.out 2>&1 &
#   echo $! > /tmp/gsm_resume.pid
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export LZL_CONFIG="$VCTS/lzl/config.yaml"
export PYTHONPATH="$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1
# 用空闲的物理 GPU 0 续跑（GPU1 正被评估占用）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LZL="$VCTS/lzl"
SCRIPTS="$LZL/scripts"
LOG="$LZL/logs"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

echo "============================================================"
echo "=== GSM resume from stage 3  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES"
echo "=== log tag: $TS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 3. Generate paraphrases (降显存比例，避免 OOM)
# ---------------------------------------------------------------------------
echo
echo "[stage 3] generate paraphrases"
$PY "$SCRIPTS/03_generate_paraphrases.py" \
  --gpu_memory_utilization 0.6 \
  2>&1 | tee "$LOG/paraphrase_gen_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 4 / 4b. Build paraphrase + vanilla manifests
# ---------------------------------------------------------------------------
echo
echo "[stage 4] build paraphrase manifest"
$PY "$SCRIPTS/04_build_paraphrase_manifest.py" 2>&1 | tee "$LOG/paraphrase_manifest_${TS}.log"

echo
echo "[stage 4b] build vanilla manifest"
$PY "$SCRIPTS/02c_build_vanilla_manifest.py" 2>&1 | tee "$LOG/vanilla_manifest_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 5. Build eval caches (idempotent)
# ---------------------------------------------------------------------------
echo
echo "[stage 5] build eval caches"
$PY "$SCRIPTS/06b_robust.py" build 2>&1 | tee "$LOG/robust_build_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 6. SFT all conditions
# ---------------------------------------------------------------------------
echo
echo "[stage 6] SFT train (raw + paraphrase + vanilla)"
$PY "$SCRIPTS/05_sft_train.py" --condition all 2>&1 | tee "$LOG/sft_all_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 7. Evaluate
# ---------------------------------------------------------------------------
echo
echo "[stage 7a] eval gsm_symbolic + gsm8k_test + svamp/asdiv/multiarith"
$PY "$SCRIPTS/06_eval.py" \
  --datasets gsm gsm8k_test svamp asdiv multiarith \
  --conditions base raw paraphrase vanilla \
  2>&1 | tee "$LOG/eval_main_${TS}.log"

echo
echo "[stage 7b] eval robustness benchmarks (06b)"
$PY "$SCRIPTS/06b_robust.py" eval \
  --datasets all \
  --conditions base raw paraphrase vanilla \
  --batch_size 16 \
  --max_new_tokens 2048 \
  2>&1 | tee "$LOG/eval_robust_${TS}.log"

echo
echo "============================================================"
echo "=== GSM resume done  $(date)"
echo "=== results: $LZL/results/eval/"
echo "============================================================"
