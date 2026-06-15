#!/usr/bin/env bash
# =============================================================================
# MLLM (vision) paraphrase pipeline — reproducible end-to-end runner.
#
# Stages (use --from / --to to resume a subset):
#   0  00_prepare_mllm_datasets.py     download + materialise AI2D train + evals
#   1  01_sample_mllm_rollouts.py      vLLM vision rollout (G per question, w/ image)
#   2  02_build_raw_manifest.py        raw_correct pool  (carries image_path)
#      02e_build_mllm_vanilla_manifest.py  vanilla (all rollouts) pool
#   3  03_generate_paraphrases.py      vLLM paraphrase candidates (carries image_path)
#   4  04_build_paraphrase_manifest.py paraphrase pool   (carries image_path)
#   5  05_sft_train.py --condition all VISION LoRA SFT (raw / paraphrase / vanilla)
#   6  06_eval_mllm_vllm.py            vLLM vision eval, all conditions
#
# Usage:
#   bash lzl/run_mllm_pipeline.sh                  # full 0..6
#   bash lzl/run_mllm_pipeline.sh --from 3         # resume from paraphrase
#   CUDA_VISIBLE_DEVICES=0 bash lzl/run_mllm_pipeline.sh --from 5
#
# Key config lives in lzl/mllm_config.yaml (G, max_new_tokens, max_length, ...).
# This script does NOT hard-code those — change them there for reproducibility.
# =============================================================================
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
LZL="$VCTS/lzl"
cd "$VCTS"

# ---- args ----
FROM=0
TO=6
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="${2:-0}"; shift 2 ;;
    --from=*) FROM="${1#*=}"; shift ;;
    --to) TO="${2:-6}"; shift 2 ;;
    --to=*) TO="${1#*=}"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

# ---- environment (fixed for reproducibility) ----
PY="${PY:-/data2/anaconda3/envs/vcts/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LZL_CONFIG="$LZL/mllm_config.yaml"
export HF_HOME="${HF_HOME:-/data2/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data2/huggingface/datasets}"
[ -f /data2/huggingface/token ] && export HF_TOKEN="$(cat /data2/huggingface/token)"
export PYTHONUNBUFFERED=1

SCRIPTS="$LZL/scripts"
LOGDIR="$LZL/logs/mllm"
mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).out"
exec > >(tee -a "$RUNLOG") 2>&1

echo "=== MLLM pipeline start $(date)  FROM=$FROM TO=$TO  GPU=$CUDA_VISIBLE_DEVICES ==="
echo "config=$LZL_CONFIG  HF_HOME=$HF_HOME"
echo "runlog=$RUNLOG"

# ---- GPU preflight: refuse to start on a busy GPU (avoids OOM cascades) ----
GID="$CUDA_VISIBLE_DEVICES"
if [[ "$GID" =~ ^[0-9]+$ ]]; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GID" 2>/dev/null | tr -d ' ' || echo 0)
  if [ "${USED:-0}" -gt 2000 ]; then
    echo "ABORT: GPU $GID already using ${USED} MiB — free it first (stale vLLM worker?)."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GID" 2>/dev/null || true
    exit 1
  fi
  echo "GPU $GID free (${USED:-0} MiB used). Proceeding."
fi

# ---- stage runner --------------------------------------------------------
# IMPORTANT: pipe to tee but propagate the *python* exit code (PIPESTATUS[0]),
# not tee's. Combined with `set -o pipefail` this guarantees a failed stage
# aborts the whole run instead of silently cascading (the bug that bit us:
# `python ... | tee log` always returned 0, so `&&` never stopped).
run() {
  local n="$1"; shift
  local desc="$1"; shift
  if [ "$n" -lt "$FROM" ] || [ "$n" -gt "$TO" ]; then
    echo "[skip] step $n ($desc)"
    return 0
  fi
  echo ""
  echo "========== Step $n: $desc  ($(date +%H:%M:%S)) =========="
  local log="$LOGDIR/$(printf '%02d' "$n")_${desc}.log"
  "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  if [ "$rc" -ne 0 ]; then
    echo "FAILED: step $n ($desc) exit=$rc — aborting."
    exit "$rc"
  fi
}

run 0 prep      "$PY" "$SCRIPTS/00_prepare_mllm_datasets.py"
run 1 rollout   "$PY" "$SCRIPTS/01_sample_mllm_rollouts.py"
run 2 raw       "$PY" "$SCRIPTS/02_build_raw_manifest.py"
run 2 vanilla   "$PY" "$SCRIPTS/02e_build_mllm_vanilla_manifest.py"
run 3 para      "$PY" "$SCRIPTS/03_generate_paraphrases.py"
run 4 paramanif "$PY" "$SCRIPTS/04_build_paraphrase_manifest.py"
run 5 sft       "$PY" "$SCRIPTS/05_sft_train.py" --condition all
run 6 eval      "$PY" "$SCRIPTS/06_eval_mllm_vllm.py" \
                      --conditions base raw paraphrase vanilla \
                      --max_num_seqs 64 \
                      --gpu_memory_utilization 0.92

echo ""
echo "=== MLLM pipeline DONE $(date) ==="
echo "Checkpoints: /data5/lzl/checkpoints/mllm_paraphrase/"
echo "Results:     $LZL/results/mllm/eval/"
