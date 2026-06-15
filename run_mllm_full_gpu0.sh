#!/usr/bin/env bash
# MLLM 完整 pipeline (stage 0..6) — 固定 GPU0，高吞吐跑满单卡。
#
# 与 run_mllm_pipeline.sh 的区别：
#   - 强制 CUDA_VISIBLE_DEVICES=0（GPU1 上有他人常驻 swift 服务，勿碰）
#   - vLLM 阶段(1/3/6)提高吞吐：gpu_memory_utilization=0.92, max_num_seqs=256
#   - rollout G 等采样配置不改，沿用 mllm_config.yaml（G=32，已与 chart/gsm 对齐）
#   - eval 阶段已包含 MMMU-Pro 多图修复 + MCQ 双向匹配修复
#
# 用法：
#   nohup bash lzl/run_mllm_full_gpu0.sh > lzl/logs/mllm/full_gpu0.out 2>&1 &
#   bash lzl/run_mllm_full_gpu0.sh --from 1          # skip prep, start at rollout
#   echo $! > /tmp/mllm_full.pid
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
export LZL_CONFIG="$LZL/mllm_config.yaml"
export HF_HOME="${HF_HOME:-/data5/lzl/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data5/lzl/hf_cache}"
[ -f /data2/huggingface/token ] && export HF_TOKEN="$(cat /data2/huggingface/token)"
export PYTHONUNBUFFERED=1

SCRIPTS="$LZL/scripts"
LOGDIR="$LZL/logs/mllm"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

# 高吞吐参数（GPU0 93GB 空闲，可激进）
GMU=0.92
SEQS=256

echo "============================================================"
echo "=== MLLM full pipeline (GPU0, high-throughput)  $(date)"
echo "=== gpu_mem=$GMU  max_num_seqs=$SEQS  log_tag=$TS  FROM=$FROM TO=$TO"
echo "============================================================"

# GPU0 预检（阈值放宽到 8000 MiB：~4GB 是 Xorg/显示器基线占用，非实验进程）
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
if [ "${USED:-0}" -gt 8000 ]; then
  echo "ABORT: GPU0 already using ${USED} MiB — likely a running job, free it first."
  exit 1
fi
echo "GPU0 free enough (${USED} MiB used, display baseline). Proceeding."

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

# Stage 0: 数据准备（幂等，已存在则快速跳过）
run 0 prep      "$PY" "$SCRIPTS/00_prepare_mllm_datasets.py"

# Stage 1: rollout（G=32，高吞吐）
run 1 rollout   "$PY" "$SCRIPTS/01_sample_mllm_rollouts.py" \
                      --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

# Stage 2: raw + vanilla manifest
run 2 raw       "$PY" "$SCRIPTS/02_build_raw_manifest.py"
run 2 vanilla   "$PY" "$SCRIPTS/02e_build_mllm_vanilla_manifest.py"

# Stage 3: paraphrase 生成（高吞吐）
run 3 para      "$PY" "$SCRIPTS/03_generate_paraphrases.py" \
                      --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

# Stage 4: paraphrase manifest
run 4 paramanif "$PY" "$SCRIPTS/04_build_paraphrase_manifest.py"

# Stage 5: vision LoRA SFT (raw / paraphrase / vanilla)
run 5 sft       "$PY" "$SCRIPTS/05_sft_train.py" --condition all

# Stage 6: vision eval（含多图修复 + MCQ 双向匹配修复，高吞吐）
run 6 eval      "$PY" "$SCRIPTS/06_eval_mllm_vllm.py" \
                      --conditions base raw paraphrase vanilla \
                      --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

echo ""
echo "============================================================"
echo "=== MLLM full pipeline DONE  $(date)"
echo "=== checkpoints: /data5/lzl/checkpoints/mllm_paraphrase_pgps9k/"
echo "=== results:     $LZL/results/mllm/eval/"
echo "============================================================"
