#!/usr/bin/env python3
"""Step 2f: Build answer-only vanilla SFT manifest for GSM8K (chart-style).

Chart vanilla (`02d`) supervises only ``Final Answer: <gold>`` because ChartQA
ships no reasoning traces. This script applies the same definition to GSM8K:
no gold CoT, just the numeric (or letter) final answer extracted from the
official ``####`` line.

Output: ``data/sft/vanilla_answer_only/vanilla.jsonl`` — same schema as raw /
paraphrase / full-trace vanilla so ``05_sft_train.py`` can consume it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data5/lzl/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/data5/lzl/hf_cache")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config  # noqa: E402

add_vcts_to_syspath()
from sampling.sample_rollouts import extract_gsm8k_gold_answer  # noqa: E402
from scripts.eval_gsm_symbolic import ANSWER_FORMAT_HINT  # noqa: E402

SYSTEM_PROMPT_MATH = (
    "You are a helpful assistant. Solve math problems step by step. "
    "Show all your work clearly."
)

OUT_REL = Path("data/sft/vanilla_answer_only/vanilla.jsonl")


def count_chat_tokens(tokenizer, question: str, trace_text: str) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MATH},
        {"role": "user", "content": question + ANSWER_FORMAT_HINT},
        {"role": "assistant", "content": trace_text},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(rendered)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--output", type=str, default=None,
                    help=f"Override output path (default: lzl/{OUT_REL})")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    out_path = Path(args.output) if args.output else paths.lzl_root / OUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_path}")

    print("\n[1/2] Loading GSM8K train split (gsm8k/main)...")
    ds = load_dataset("gsm8k", "main", split="train", cache_dir=str(paths.hf_datasets))
    print(f"  loaded {len(ds)} questions (full training set)")

    print("\n[2/2] Building answer-only traces (Final Answer: <gold>)...")
    tokenizer = AutoTokenizer.from_pretrained(str(paths.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_tokens = 0
    with out_path.open("w") as f:
        for i, ex in enumerate(ds):
            q = str(ex["question"]).strip()
            gold = extract_gsm8k_gold_answer(str(ex["answer"]).strip())
            trace = f"Final Answer: {gold}"
            n_tok = count_chat_tokens(tokenizer, q, trace)
            total_tokens += n_tok
            f.write(json.dumps({
                "question_id": i,
                "question": q,
                "answer": gold,
                "trace_text": trace,
                "source": "vanilla_gold_answer_only",
                "tokens": n_tok,
                "bin": "vanilla",
            }, ensure_ascii=False) + "\n")

    print(f"  wrote {len(ds)} examples, total ~{total_tokens} tokens "
          f"(avg {total_tokens/len(ds):.1f} tokens/example)")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
