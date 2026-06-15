#!/usr/bin/env bash
# MLLM pipeline 从 stage 1 续跑（stage 0 数据已就绪）。GPU0 高吞吐。
# 结构刻意简化：直接顺序执行 + 每步检查退出码，避免复杂 tee/exec 嵌套。
#
#   nohup bash lzl/run_mllm_from_stage1.sh > lzl/logs/mllm/from_stage1.out 2>&1 &
#   echo $! > /tmp/mllm_full.pid
set -uo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
LZL="$VCTS/lzl"
cd "$VCTS"

PY=/data2/anaconda3/envs/vcts/bin/python
export CUDA_VISIBLE_DEVICES=0
export LZL_CONFIG="$LZL/mllm_config.yaml"
export HF_HOME=/data5/lzl/hf_cache
export HF_DATASETS_CACHE=/data5/lzl/hf_cache
[ -f /data2/huggingface/token ] && export HF_TOKEN="$(cat /data2/huggingface/token)"
export PYTHONUNBUFFERED=1

SCRIPTS="$LZL/scripts"
GMU=0.92
SEQS=256

step() {
  local name="$1"; shift
  echo ""
  echo "########## $name  $(date '+%H:%M:%S') ##########"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!!!! STEP FAILED: $name (exit=$rc). Aborting. !!!!!"
    exit $rc
  fi
  echo "########## $name DONE  $(date '+%H:%M:%S') ##########"
}

echo "=== MLLM from stage1 start $(date)  GPU=$CUDA_VISIBLE_DEVICES gmu=$GMU seqs=$SEQS ==="

step "stage1_rollout"   "$PY" "$SCRIPTS/01_sample_mllm_rollouts.py" \
                              --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"
step "stage2_raw"       "$PY" "$SCRIPTS/02_build_raw_manifest.py"
step "stage2_vanilla"   "$PY" "$SCRIPTS/02e_build_mllm_vanilla_manifest.py"
step "stage3_para"      "$PY" "$SCRIPTS/03_generate_paraphrases.py" \
                              --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"
step "stage4_paramanif" "$PY" "$SCRIPTS/04_build_paraphrase_manifest.py"
step "stage5_sft"       "$PY" "$SCRIPTS/05_sft_train.py" --condition all
step "stage6_eval"      "$PY" "$SCRIPTS/06_eval_mllm_vllm.py" \
                              --conditions base raw paraphrase vanilla \
                              --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

echo ""
echo "=== MLLM pipeline ALL DONE $(date) ==="
echo "=== results: $LZL/results/mllm/eval/ ==="
