#!/usr/bin/env bash
# Chart/table paraphrase pipeline (ChartQA train → PlotQA/TabMWP/FinQA eval)
# Usage:
#   bash lzl/run_chart_pipeline.sh
#   LIMIT=20 EVAL_LIMIT=50 bash lzl/run_chart_pipeline.sh
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

FROM="${FROM:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="${2:-1}"; shift 2 ;;
    --from=*) FROM="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

PYTHON="${PYTHON:-/data2/anaconda3/envs/vcts/bin/python}"
export PYTHONPATH="$VCTS:$VCTS/lzl${PYTHONPATH:+:$PYTHONPATH}"
export LZL_CONFIG="$VCTS/lzl/chart_config.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data5/lzl/hf_datasets}"
export HF_HOME="${HF_HOME:-/data5/lzl/hf_cache}"

SCRIPTS="$VCTS/lzl/scripts"
LOG="$VCTS/lzl/logs/chart/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$VCTS/lzl/logs/chart"
exec > >(tee -a "$LOG") 2>&1

echo "=== chart pipeline start $(date) FROM=$FROM LIMIT=${LIMIT:-all} ==="
echo "Log: $LOG"

run_step() {
  local n=$1 script=$2
  shift 2
  if [ "$n" -lt "$FROM" ]; then
    echo "[skip] step $n"
    return
  fi
  echo ""
  echo "========== Step $n: $script =========="
  $PYTHON "$SCRIPTS/$script" "$@"
}

if [ "$FROM" -le 0 ]; then
  echo "========== Step 0: prepare datasets =========="
  $PYTHON "$SCRIPTS/00_prepare_chart_datasets.py"
fi

if [ "$FROM" -le 1 ]; then
  if [ -n "${LIMIT:-}" ]; then
    echo "========== Step 1: 01_sample_chart_rollouts.py (debug) =========="
    $PYTHON "$SCRIPTS/01_sample_chart_rollouts.py" --limit "$LIMIT"
  else
    echo "========== Step 1: 01_sample_chart_rollouts_dual.sh =========="
    bash "$SCRIPTS/01_sample_chart_rollouts_dual.sh"
  fi
fi

run_step 2 02_build_raw_manifest.py
run_step 3 03_generate_paraphrases.py
run_step 4 04_build_paraphrase_manifest.py
run_step 5 05_sft_train.py --condition paraphrase

EVAL_ARGS=(--datasets plotqa tabmwp finqa --conditions base paraphrase)
[ -n "${EVAL_LIMIT:-}" ] && EVAL_ARGS+=(--limit "$EVAL_LIMIT")
run_step 6 06_eval_chart.py "${EVAL_ARGS[@]}"

echo ""
echo "=== chart pipeline done $(date) ==="
echo "Checkpoints: /data5/lzl/checkpoints/chart_paraphrase/"
echo "Results:     $VCTS/lzl/results/chart/eval/"
