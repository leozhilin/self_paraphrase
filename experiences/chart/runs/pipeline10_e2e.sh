#!/usr/bin/env bash
# ChartQA full pipeline smoke: 10 train questions, G=4, data4 FTSO only.
set -euo pipefail

LZL_ROOT="/home/liuyu/Projects/GRPO_research/VCTS/lzl"
HERE="$LZL_ROOT/experiences/chart"
CONFIG="$HERE/configs/config_pipeline10.yaml"
PY="/data2/anaconda3/envs/vcts/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
N=10
LOG="$LZL_ROOT/logs/chart/pipeline10/e2e_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="$GPU"
export LZL_CONFIG="$CONFIG"
export PYTHONPATH="$LZL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['datasets']['gsm8k_cache'])")
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$LOG")" /data4/FTSO/datasets/chart/pipeline10
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "=== Chart pipeline10 e2e  $(date)"
echo "=== CONFIG: $CONFIG  GPU: $GPU  N=$N"
echo "=== LOG: $LOG"
echo "============================================================"

cd "$HERE"

echo; echo "[1/6] rollout (vLLM, G=4, limit=$N)"
$PY 01_sample_rollouts.py --config "$CONFIG" --limit "$N"

echo; echo "[2/6] raw manifest"
$PY 02_build_raw_manifest.py --config "$CONFIG"

echo; echo "[2/6] vanilla manifest (limit=$N)"
$PY 02_build_vanilla_manifest.py --config "$CONFIG" --limit "$N"

echo; echo "[3/6] paraphrase candidates"
$PY 03_generate_paraphrases.py --config "$CONFIG"

echo; echo "[4/6] paraphrase manifest"
$PY 04_build_paraphrase_manifest.py --config "$CONFIG"

echo; echo "[5/6] SFT (raw + paraphrase + vanilla, 1 epoch, LoRA)"
$PY 05_sft_train.py --config "$CONFIG" --condition all

echo; echo "[6/6] eval (chartqa_test limit=$N, base + 3 adapters, vLLM)"
$PY 06_eval_vllm.py --config "$CONFIG" \
  --datasets chartqa_test \
  --conditions base raw paraphrase vanilla \
  --limit "$N" \
  --max_new_tokens 128 \
  --max_model_len 4096

echo; echo "=== SUMMARY ==="
RESULTS=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['results_root'])")
EVAL_JSON="$RESULTS/eval/chartqa_test.json"
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
echo "=== Chart pipeline10 e2e DONE  $(date)"
echo "=== LOG: $LOG"
echo "============================================================"
