#!/usr/bin/env bash
# Reset GPU0 (Unknown Error poisons all new CUDA contexts) then run chart smoke test.
# Usage (in your terminal — needs sudo password once):
#   bash lzl/run_chart_after_gpu_reset.sh
set -euo pipefail

VCTS=/home/liuyu/Projects/GRPO_research/VCTS
cd "$VCTS"

echo "=== Reset GPU0 (required: GPU0 Unknown Error blocks new CUDA init) ==="
sudo nvidia-smi --gpu-reset -i 0

echo "=== Verify CUDA on GPU1 ==="
CUDA_VISIBLE_DEVICES=1 /data2/anaconda3/envs/vcts/bin/python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA still broken'; print('OK', torch.cuda.get_device_name(0))"

echo "=== Chart pipeline LIMIT=20 (GPU1) ==="
FROM="${FROM:-1}"
CUDA_VISIBLE_DEVICES=1 LIMIT=20 EVAL_LIMIT=50 FROM="$FROM" bash lzl/run_chart_pipeline.sh

echo "=== Resume GSM dual-GPU sampling (optional) ==="
echo "  nohup bash lzl/run_pipeline.sh >> lzl/logs/full_pipeline.log 2>&1 &"
