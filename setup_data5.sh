#!/usr/bin/env bash
# 初始化 /data5/lzl 目录，并链接 base 模型到本地 cache
set -euo pipefail

DATA5=/data5/lzl
VCTS=/home/liuyu/Projects/GRPO_research/VCTS
SRC_MODEL=/home/liuyu/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507

mkdir -p "$DATA5"/{models,checkpoints,hf_datasets,hf_cache}

if [ ! -e "$DATA5/models/Qwen3-4B-Instruct-2507" ]; then
  if [ -d "$SRC_MODEL" ]; then
    ln -sf "$SRC_MODEL" "$DATA5/models/Qwen3-4B-Instruct-2507"
    echo "Linked model → $DATA5/models/Qwen3-4B-Instruct-2507"
  else
    echo "WARN: source model not found at $SRC_MODEL"
    echo "      Download Qwen3-4B-Instruct to $DATA5/models/ manually"
  fi
fi

# lzl 本地数据目录
mkdir -p "$VCTS/lzl"/{data/{rollouts,cache/eval,sft/{raw,paraphrase}},results/eval,logs,scripts}

# 评测集软链（OOD eval 题面；rollout 数据独立在 lzl/data/rollouts）
EVAL_SRC="$VCTS/data/cache"
EVAL_DST="$VCTS/lzl/data/cache/eval"
for f in gsm_symbolic_eval_500.jsonl gsm_symbolic_eval_500_B.jsonl \
         gsm_symbolic_eval_extra1000.jsonl svamp_eval.jsonl; do
  if [ -f "$EVAL_SRC/$f" ] && [ ! -e "$EVAL_DST/$f" ]; then
    ln -sf "$EVAL_SRC/$f" "$EVAL_DST/$f"
  fi
done

echo ""
echo "=== /data5/lzl layout ==="
ls -la "$DATA5"
echo ""
echo "=== model ==="
ls -la "$DATA5/models/" || true
echo ""
echo "Done. Edit lzl/config.yaml if paths need adjustment."
