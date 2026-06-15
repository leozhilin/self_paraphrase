# lzl — 独立 paraphrase 复现实验

与主 `VCTS/scripts/` 隔离；采样/训练数据在 `lzl/data/`，模型与 checkpoint 在 `/data5/lzl/`。

## 目录

```
lzl/
  config.yaml          # 全部路径与超参
  paths.py             # 路径加载
  setup_data5.sh       # 初始化 /data5/lzl
  run_pipeline.sh      # 一键全流程
  scripts/
    01_sample_rollouts.py
    02_build_raw_manifest.py
    03_generate_paraphrases.py
    04_build_paraphrase_manifest.py
    05_sft_train.py
    06_eval.py
  data/
    rollouts/          # GSM8K train 采样（独立）
    sft/raw/           # raw.jsonl
    sft/paraphrase/    # paraphrase.jsonl
    cache/             # paraphrase 候选等
  results/eval/        # 评测 JSON
/data5/lzl/
  models/              # Qwen3-4B（软链）
  checkpoints/         # LoRA adapter
  hf_datasets/         # GSM8K 缓存
```

## 快速开始

```bash
cd /home/liuyu/Projects/GRPO_research/VCTS
conda activate vcts

# 1. 初始化 data5 目录 + 模型软链
bash lzl/setup_data5.sh

# 2. 调试（20 题）
LIMIT=20 bash lzl/run_pipeline.sh

# 3. 正式全流程（GSM8K train 全量 ~7473 题，数小时）
bash lzl/run_pipeline.sh

# 4. 已有 raw.jsonl，从 paraphrase 开始
bash lzl/run_pipeline.sh --from 3
```

## 分步运行

```bash
cd VCTS
python lzl/scripts/01_sample_rollouts.py [--limit N]
python lzl/scripts/02_build_raw_manifest.py
python lzl/scripts/03_generate_paraphrases.py
python lzl/scripts/04_build_paraphrase_manifest.py
python lzl/scripts/05_sft_train.py --condition paraphrase
python lzl/scripts/06_eval.py --datasets all --conditions base paraphrase
```

## 说明

- **采样**：G=32, temp=0.7，GSM8K **train** split，输出 `lzl/data/rollouts/gsm8k_train_g32.jsonl`
- **评测集**：`lzl/data/cache/eval/` 软链到主项目 cache（题面固定）；rollout 数据完全独立
- **修改路径**：编辑 `lzl/config.yaml`
