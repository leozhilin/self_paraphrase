#!/usr/bin/env bash
# GSM 完整重训 + 全量评测（从 rollout 到 eval）
#
# 使用 GPU 1（避免与 MLLM eval 争用 GPU 0）
#   nohup bash lzl/gsm_full_retrain.sh > lzl/logs/gsm_full_retrain.out 2>&1 &
#   echo $! > /tmp/gsm_full_retrain.pid
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export LZL_CONFIG="$VCTS/lzl/config.yaml"
export PYTHONPATH="$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_datasets
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

LZL="$VCTS/lzl"
SCRIPTS="$LZL/scripts"
LOG="$LZL/logs"
CKPT=/data5/lzl/checkpoints
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

echo "============================================================"
echo "=== GSM full retrain start  $(date)"
echo "=== GPU: $CUDA_VISIBLE_DEVICES"
echo "=== log tag: $TS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 0. Backup checkpoints + intermediate data
# ---------------------------------------------------------------------------
echo
echo "[stage 0] backup checkpoints and data"
for cond in raw paraphrase vanilla; do
  src="$CKPT/$cond"
  if [[ -d "$src" ]]; then
    dst="$CKPT/${cond}_bak_${TS}"
    cp -a "$src" "$dst"
    echo "  $src → $dst"
  fi
done

DATA_BAK="$LZL/data/backup_${TS}"
mkdir -p "$DATA_BAK"
for f in \
  "$LZL/data/rollouts/gsm8k_train_g32.jsonl" \
  "$LZL/data/sft/raw/raw.jsonl" \
  "$LZL/data/sft/paraphrase/paraphrase.jsonl" \
  "$LZL/data/sft/vanilla/vanilla.jsonl" \
  "$LZL/data/cache/paraphrase_candidates.jsonl" \
  "$LZL/data/cache/paraphrase_tokens.jsonl"; do
  if [[ -f "$f" ]]; then
    cp -a "$f" "$DATA_BAK/$(basename "$f")"
    rm -f "$f"
    echo "  archived $(basename "$f")"
  fi
done

# ---------------------------------------------------------------------------
# Stage 1. Rollout sampling (single GPU)
# ---------------------------------------------------------------------------
echo
echo "[stage 1] rollout sampling (G=32, ~7473 questions)"
$PY "$SCRIPTS/01_sample_rollouts.py" 2>&1 | tee "$LOG/rollout_${TS}.log"

# ---------------------------------------------------------------------------
# Stage 2–4. Manifests + paraphrase
# ---------------------------------------------------------------------------
echo
echo "[stage 2] build raw manifest"
$PY "$SCRIPTS/02_build_raw_manifest.py" 2>&1 | tee "$LOG/raw_manifest_${TS}.log"

echo
echo "[stage 3] generate paraphrases"
$PY "$SCRIPTS/03_generate_paraphrases.py" 2>&1 | tee "$LOG/paraphrase_gen_${TS}.log"

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
echo "=== GSM full retrain done  $(date)"
echo "=== backups: $DATA_BAK, $CKPT/*_bak_${TS}"
echo "=== results: $LZL/results/eval/"
echo "============================================================"
