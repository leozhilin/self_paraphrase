#!/usr/bin/env python3
"""Step 2d (chart): Build a *vanilla SFT* manifest for the chart task.

Mirror of `02c_build_vanilla_manifest.py` but for chart datasets. ChartQA
training data only ships (question, gold_answer) pairs without reasoning
traces, so for "vanilla SFT" we use the most standard (and most honest)
definition: ``trace_text = "Final Answer: <gold>"`` — the model is supervised
to map (table+question) directly to the short answer with no intermediate
reasoning. This is the canonical short-answer SFT baseline reported by the
ChartQA paper and follow-ups.

Source: `data/chart/chartqa_train.jsonl` (built by 00_prepare_chart_datasets.py
                                           — full ChartQA train: human +
                                           augmented).

Output: `data/chart/sft/vanilla/vanilla.jsonl` with the same schema raw /
        paraphrase use, so 05_sft_train.py reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data5/lzl/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/data5/lzl/hf_cache")

from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from utils import CHART_SYSTEM_PROMPT  # noqa: E402
from paths import ensure_dirs, get_paths, load_config  # noqa: E402


def count_chat_tokens(tokenizer, system_prompt: str, question: str,
                      trace_text: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": trace_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False,
    )
    return len(tokenizer(rendered)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="/home/liuyu/Projects/GRPO_research/VCTS/lzl/chart_config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))

    train_jsonl = Path(cfg["datasets"]["train_jsonl"])
    if not train_jsonl.exists():
        sys.exit(f"ChartQA train manifest missing at {train_jsonl}. "
                 "Run scripts/00_prepare_chart_datasets.py first.")

    out_path = paths.vanilla_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_path}")

    print("\n[1/2] Loading ChartQA train manifest...")
    train_rows = [json.loads(l) for l in train_jsonl.open()]
    print(f"  loaded {len(train_rows)} questions (full ChartQA train)")

    print("\n[2/2] Tokenizing chat-rendered examples (for bookkeeping)...")
    tokenizer = AutoTokenizer.from_pretrained(str(paths.model_path),
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_tokens = 0
    with out_path.open("w") as f:
        for ex in train_rows:
            q = str(ex["question"]).strip()
            gold = str(ex["answer"]).strip()
            trace = f"Final Answer: {gold}"
            n_tok = count_chat_tokens(tokenizer, CHART_SYSTEM_PROMPT, q, trace)
            total_tokens += n_tok
            f.write(json.dumps({
                "question_id": ex["question_id"],
                "question": q,
                "answer": gold,
                "trace_text": trace,
                "source": "vanilla_gold",
                "tokens": n_tok,
                "bin": "vanilla",
            }, ensure_ascii=False) + "\n")

    print(f"  wrote {len(train_rows)} examples, total ~{total_tokens} tokens "
          f"(avg {total_tokens/len(train_rows):.1f} tokens/example)")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
