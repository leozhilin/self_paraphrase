#!/usr/bin/env bash
# lzl paraphrase 实验全流程
# Usage:
#   bash lzl/run_pipeline.sh              # 全流程
#   bash lzl/run_pipeline.sh --from 3     # 从 step 3 开始（已有 raw.jsonl）
#   LIMIT=20 bash lzl/run_pipeline.sh     # 调试：只采样 20 题
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

FROM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="${2:-1}"; shift 2 ;;
    --from=*) FROM="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data5/lzl/hf_datasets}"
export HF_HOME="${HF_HOME:-/data5/lzl/hf_cache}"

SCRIPTS="$VCTS/lzl/scripts"
LOG="$VCTS/lzl/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$VCTS/lzl/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== lzl pipeline start $(date) FROM=$FROM ==="
echo "Log: $LOG"

run_step() {
  local n=$1
  local script=$2
  shift 2
  if [ "$n" -lt "$FROM" ]; then
    echo "[skip] step $n"
    return
  fi
  echo ""
  echo "========== Step $n: $script =========="
  $PYTHON "$SCRIPTS/$script" "$@"
}

# Step 1: rollout 采样（全量默认双卡并行 + batch_size=32）
if [ "$FROM" -le 1 ]; then
  echo ""
  if [ -n "${LIMIT:-}" ]; then
    echo "========== Step 1: 01_sample_rollouts.py (debug) =========="
    $PYTHON "$SCRIPTS/01_sample_rollouts.py" --limit "$LIMIT"
  else
    echo "========== Step 1: 01_sample_rollouts_dual.sh =========="
    bash "$SCRIPTS/01_sample_rollouts_dual.sh"
  fi
fi

# Step 2: raw manifest
run_step 2 02_build_raw_manifest.py

# Step 3: paraphrase 生成
run_step 3 03_generate_paraphrases.py

# Step 4: paraphrase manifest
run_step 4 04_build_paraphrase_manifest.py

# Step 5: SFT
run_step 5 05_sft_train.py --condition paraphrase

# Step 6: eval
run_step 6 06_eval.py --datasets all --conditions base paraphrase

echo ""
echo "=== pipeline done $(date) ==="
echo "Checkpoints: /data5/lzl/checkpoints/"
echo "Results:     $VCTS/lzl/results/eval/"
