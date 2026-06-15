#!/usr/bin/env bash
# MLLM pipeline 从 stage3 续跑（n_per=4 重新改写）。GPU0 高吞吐。
# 复用：stage0 数据、stage1 rollout(G=32)、stage2 raw/vanilla manifest（均已就绪）。
# 重做：stage3 改写(n_per=4) → stage4 manifest → stage5 训练 → stage6 评估。
#
#   nohup bash lzl/run_mllm_from_stage3.sh > lzl/logs/mllm/from_stage3.out 2>&1 &
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

echo "=== MLLM from stage3 (n_per=4) start $(date)  GPU=$CUDA_VISIBLE_DEVICES gmu=$GMU seqs=$SEQS ==="

# stage3: 重新改写（n_per 由 config 控制 = 4）。删旧候选确保重生成。
rm -f "$LZL/data/mllm/cache/paraphrase_candidates.jsonl" \
      "$LZL/data/mllm/cache/paraphrase_tokens.jsonl"
step "stage3_para"      "$PY" "$SCRIPTS/03_generate_paraphrases.py" \
                              --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"
step "stage4_paramanif" "$PY" "$SCRIPTS/04_build_paraphrase_manifest.py"
step "stage5_sft"       "$PY" "$SCRIPTS/05_sft_train.py" --condition all
step "stage6_eval"      "$PY" "$SCRIPTS/06_eval_mllm_vllm.py" \
                              --conditions base raw paraphrase vanilla \
                              --gpu_memory_utilization "$GMU" --max_num_seqs "$SEQS"

echo ""
echo "=== MLLM pipeline (n_per=4) ALL DONE $(date) ==="
echo "=== results: $LZL/results/mllm/eval/ ==="
