#!/usr/bin/env python3
"""Step 2c: Build a *vanilla SFT* manifest from official GSM8K train answers.

This is the standard "instruction-tune on gold solutions" baseline that most math
SFT papers compare against. Unlike `02_build_raw_manifest.py` (which uses the
*model's own correct rollouts*) and the paraphrase manifest (which rewrites those
rollouts), this manifest uses the **human-written solution** shipped with each
GSM8K training question, without any rollout sampling, filtering, or rewriting,
and uses the FULL gsm8k/main train split (7473 examples) — no token budget.

Construction:
  1. Load gsm8k/main train (7473 questions) from HF.
  2. For each question, the trace_text is the official `answer` field — i.e. the
     full step-by-step reasoning followed by `#### N` (already in GSM8K format).
  3. Tokenize each (system + user + assistant) chat string with the same chat
     template the trainer uses, just for bookkeeping (token-count column).
  4. Write all 7473 examples to `data/sft/vanilla/vanilla.jsonl` with the same
     schema as raw / paraphrase so `05_sft_train.py` can consume it unchanged.

Output schema per line:
  {question_id, question, answer, trace_text, source="vanilla_gold",
   tokens, bin="vanilla"}

Note on fairness: this manifest is intentionally *not* token-matched with raw /
paraphrase. raw / paraphrase share a ~480k-token budget; vanilla uses the full
GSM8K train. This mirrors how "vanilla SFT" is typically reported in the
literature (full training set, fixed epochs).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from paths import add_vcts_to_syspath, apply_hf_env, ensure_dirs, get_paths, load_config  # noqa: E402

add_vcts_to_syspath()
apply_hf_env(load_config())

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# GSM8K's gold `answer` field embeds OpenAI's calculator-API placeholders of the
# form `<<expr=value>>value` (e.g. `<<48-3-4=41>>41`). These were never meant to
# be predicted by an LM — they were filled in by an external Python evaluator at
# data-collection time. If we feed them in as SFT labels, the model learns a
# brittle "shorthand CoT" that skips natural-language intermediate steps and
# hallucinates `=value`, which empirically wrecks GSM8K test accuracy
# (vanilla_sft 81.3% vs base 93.3% in our v0 run). The standard fix used by
# MetaMath / WizardMath / PRM800K etc. is simply to strip these markers before
# training and let the model learn natural-language step-by-step reasoning.
_CALC_MARKUP_RE = re.compile(r"<<[^>]*>>")


def _strip_calc_markup(answer: str) -> str:
    """Remove GSM8K calculator placeholders `<<expr=value>>` from a gold answer.

    Examples:
        `Janet eats <<3+4=7>>7 eggs ...`  →  `Janet eats 7 eggs ...`
        `... = $<<17*2=34>>34\n#### 34`   →  `... = $34\n#### 34`
    """
    cleaned = _CALC_MARKUP_RE.sub("", answer)
    # Collapse any double-spaces or `$ ` artifacts left behind by the substitution.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# Unified Final-Answer template (shared with 01 rollout / 03 paraphrase /
# 05 SFT / 06 eval, all sourced from gsm_utils).
from utils import ANSWER_FORMAT_HINT, GSM_SYSTEM_PROMPT  # noqa: E402

SYSTEM_PROMPT_MATH = GSM_SYSTEM_PROMPT


def count_chat_tokens(tokenizer, system_prompt: str, question: str,
                      user_suffix: str, trace_text: str) -> int:
    """Count tokens of the full SFT example (for bookkeeping only)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question + user_suffix},
        {"role": "assistant", "content": trace_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False,
    )
    return len(tokenizer(rendered)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows from HF train split (smoke tests).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_hf_env(cfg)
    paths = ensure_dirs(get_paths(cfg))

    out_path = paths.vanilla_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_path}")

    print("\n[1/2] Loading GSM8K train split (gsm8k/main)...")
    ds = load_dataset("gsm8k", "main", split="train", cache_dir=str(paths.hf_datasets))
    n_rows = min(len(ds), args.limit) if args.limit else len(ds)
    print(f"  loaded {len(ds)} questions, writing {n_rows}")

    print("\n[2/2] Tokenizing chat-rendered examples (for bookkeeping)...")
    tokenizer = AutoTokenizer.from_pretrained(str(paths.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_tokens = 0
    n_stripped = 0
    written = 0
    with out_path.open("w") as f:
        for i, ex in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            q = str(ex["question"]).strip()
            a_raw = str(ex["answer"]).strip()  # original "<reasoning with <<>> markup>\n#### N"
            a = _strip_calc_markup(a_raw)      # natural-language reasoning + "#### N"
            if a != a_raw:
                n_stripped += 1
            # Convert the trailing GSM8K '#### N' line into the unified
            # 'Final Answer: N' format so vanilla matches raw / paraphrase / eval.
            a = re.sub(r"\n?####\s*(.+?)\s*$", r"\nFinal Answer: \1", a).strip()
            n_tok = count_chat_tokens(tokenizer, SYSTEM_PROMPT_MATH, q,
                                      ANSWER_FORMAT_HINT, a)
            total_tokens += n_tok
            f.write(json.dumps({
                "question_id": i,
                "question": q,
                "answer": a,
                "trace_text": a,
                "source": "vanilla_gold_nocalc",
                "tokens": n_tok,
                "bin": "vanilla",
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  wrote {written} examples, total ~{total_tokens} tokens "
          f"(avg {total_tokens/written:.1f} tokens/example)")
    print(f"  stripped <<expr=value>> calculator markup from {n_stripped}/{written} examples")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
