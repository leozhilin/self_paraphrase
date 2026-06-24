#!/usr/bin/env bash
# Full MLLM pipeline smoke: 10 train samples, G=4, data4 FTSO only.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/mllm"
CONFIG="$HERE/configs/config_pipeline10.yaml"
PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
N=10
LOG="$LZL_ROOT/logs/mllm/pipeline10/e2e_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"
export PYTHONPATH="$LZL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['datasets']['hf_cache'])")
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$LOG")" /data4/FTSO/datasets/mllm/pipeline10
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "=== MLLM pipeline10 e2e  $(date)"
echo "=== CONFIG: $CONFIG  GPU: $GPU  N=$N"
echo "=== LOG: $LOG"
echo "============================================================"

SRC_TRAIN=/data4/FTSO/datasets/mllm/processed/pgps9k_train.jsonl
SUB_TRAIN=/data4/FTSO/datasets/mllm/pipeline10/pgps9k_train_10.jsonl
head -"$N" "$SRC_TRAIN" > "$SUB_TRAIN"
echo "[prep] train subset: $SUB_TRAIN ($(wc -l < "$SUB_TRAIN") lines)"

cd "$HERE"

echo; echo "[1/6] rollout (vLLM vision, G=4)"
$PY 01_sample_rollouts.py --config "$CONFIG" --limit "$N"

echo; echo "[2/6] raw manifest"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[2/6] vanilla manifest"
$PY 02_build_vanilla_manifest.py --config "$CONFIG"

echo; echo "[3/6] paraphrase candidates"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/6] paraphrase manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/6] SFT (raw + paraphrase + vanilla, 1 epoch)"
$PY 05_sft_train.py --config "$CONFIG" --condition all

echo; echo "[6/6] eval (base + 3 adapters, pgps9k_test limit=$N)"
$PY 06_eval_vllm.py --config "$CONFIG" \
  --datasets pgps9k_test \
  --conditions base raw paraphrase vanilla \
  --limit "$N" \
  --max_new_tokens 128 \
  --max_num_seqs 8

echo; echo "=== SUMMARY ==="
RESULTS=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
for f in "$RESULTS/eval"/pgps9k_test.json "$RESULTS"/eval/pgps9k_test.json; do
  [[ -f "$f" ]] && EVAL_JSON="$f" && break
done
EVAL_JSON="${EVAL_JSON:-$RESULTS/eval/pgps9k_test.json}"
$PY - "$EVAL_JSON" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(f"MISSING {p}")
    sys.exit(1)
data = json.loads(p.read_text())
for k, v in sorted(data.items()):
    if isinstance(v, dict) and "accuracy" in v:
        print(f"  {k}: {v['correct']}/{v['total']} = {v['accuracy']:.1%}")
PY

echo; echo "============================================================"
echo "=== pipeline10 e2e DONE  $(date)"
echo "=== LOG: $LOG"
echo "============================================================"
