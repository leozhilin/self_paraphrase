#!/usr/bin/env bash
# PGPS9K smoke pipeline: 1k train subset, G=16, full eval.
#
#   nohup bash lzl/run_mllm_smoke_gpu0.sh > lzl/logs/mllm/smoke/pipeline.out 2>&1 &
#   bash lzl/run_mllm_smoke_gpu0.sh --from 1
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
LZL="$VCTS/lzl"
cd "$VCTS"

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

PY="${PY:-/data2/anaconda3/envs/vcts/bin/python}"
export CUDA_VISIBLE_DEVICES=0
export LZL_CONFIG="$LZL/mllm_config_smoke.yaml"
export HF_HOME="${HF_HOME:-/data5/lzl/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data5/lzl/hf_cache}"
[ -f /data2/huggingface/token ] && export HF_TOKEN="$(cat /data2/huggingface/token)"
export PYTHONUNBUFFERED=1

SCRIPTS="$LZL/scripts"
LOGDIR="$LZL/logs/mllm/smoke"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

GMU=0.92
SEQS=256
# Rollout: chunk_size*G = 64*16 = 1024 seqs/chunk; max_num_seqs=512 avoids half-batch queueing.
ROLLOUT_SEQS=512
ROLLOUT_CHUNK=64

echo "============================================================"
echo "=== MLLM SMOKE pipeline (1k subset, G=16)  $(date)"
echo "=== config=$LZL_CONFIG"
echo "=== gpu_mem=$GMU  rollout_seqs=$ROLLOUT_SEQS chunk=$ROLLOUT_CHUNK"
echo "=== para/eval_seqs=$SEQS  log_tag=$TS  FROM=$FROM TO=$TO"
echo "============================================================"

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
if [ "${USED:-0}" -gt 8000 ]; then
  echo "ABORT: GPU0 already using ${USED} MiB — free it first."
  exit 1
fi

run() {
  local n="$1"; shift; local desc="$1"; shift
  if [ "$n" -lt "$FROM" ] || [ "$n" -gt "$TO" ]; then
    echo "[skip] step $n ($desc)"
    return 0
  fi
  echo ""
  echo "========== Step $n: $desc  ($(date +%H:%M:%S)) =========="
  "$@" 2>&1 | tee "$LOGDIR/${TS}_$(printf '%02d' "$n")_${desc}.log"
  local rc="${PIPESTATUS[0]}"
  if [ "$rc" -ne 0 ]; then
    echo "FAILED step $n ($desc) exit=$rc — aborting."
    exit "$rc"
  fi
}

# Stage 0: eval jsonls (idempotent) + 1k train subset
run 0 prep      "$PY" "$SCRIPTS/00_prepare_mllm_datasets.py"
run 0 subset    "$PY" "$SCRIPTS/00b_make_pgps9k_smoke_subset.py" --n 1000 --seed 2026

run 1 rollout   "$PY" "$SCRIPTS/01_sample_mllm_rollouts.py" \
                      --gpu_memory_utilization "$GMU" \
                      --max_num_seqs "$ROLLOUT_SEQS" \
                      --chunk_size "$ROLLOUT_CHUNK"

run 2 raw       "$PY" "$SCRIPTS/smoke/02_build_mllm_raw_one_trace_per_sample.py"
run 2 vanilla   "$PY" "$SCRIPTS/02e_build_mllm_vanilla_manifest.py"

run 3 para      "$PY" "$SCRIPTS/03_generate_paraphrases.py" \
                      --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

run 4 paramanif "$PY" "$SCRIPTS/smoke/04_build_mllm_paraphrase_all_valid.py"

run 5 sft       "$PY" "$SCRIPTS/05_sft_train.py" --condition all

run 6 eval      "$PY" "$SCRIPTS/06_eval_mllm_vllm.py" \
                      --conditions base raw paraphrase vanilla \
                      --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

echo ""
echo "=== SMOKE pipeline DONE $(date) ==="
echo "=== checkpoints: /data5/lzl/checkpoints/mllm_paraphrase_pgps9k_smoke/"
echo "=== results:     $LZL/results/mllm/smoke/eval/"
