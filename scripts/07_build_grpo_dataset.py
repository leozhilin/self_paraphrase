#!/usr/bin/env python3
"""Step 7 (gsm/math, GRPO): build the GRPO prompt dataset from GSM8K train.

Output JSONL — swift GRPO format. Each row carries:
    messages : [{"role": "system", ...}, {"role": "user", ...}]   (prompt only)
    solution : raw GSM8K answer string (contains '#### N')

The ``messages`` mirror exactly the eval/rollout prompt
(``GSM_SYSTEM_PROMPT`` + question + ``ANSWER_FORMAT_HINT``) so the policy is
trained on the same distribution it is evaluated on. ``solution`` is consumed
by the ``gsm_accuracy`` reward (scripts/grpo_rewards.py), which runs
``extract_gsm8k_gold_answer`` on it — identical to scripts/06_eval.py.

Usage:
    python scripts/07_build_grpo_dataset.py                 # full train split
    python scripts/07_build_grpo_dataset.py --limit 256     # smoke subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsm_utils import ANSWER_FORMAT_HINT, GSM_SYSTEM_PROMPT
from paths import ensure_dirs, get_paths, load_config


def build_rows(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        user = q["question"] + ANSWER_FORMAT_HINT
        rows.append({
            "messages": [
                {"role": "system", "content": GSM_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "solution": q["answer"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "test"])
    p.add_argument("--limit", type=int, default=None,
                   help="Keep only the first N questions (smoke).")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. Default: data/grpo/gsm8k_<split>.jsonl")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))

    ds = load_dataset(
        "openai/gsm8k", "main", split=args.split,
        cache_dir=str(paths.hf_datasets),
    )
    questions = [
        {"question_id": i, "question": ex["question"], "answer": ex["answer"]}
        for i, ex in enumerate(ds)
    ]
    if args.limit:
        questions = questions[:args.limit]

    rows = build_rows(questions)

    if args.output:
        out = Path(args.output)
    else:
        suffix = f"_limit{args.limit}" if args.limit else ""
        out = paths.lzl_root / "data" / "grpo" / f"gsm8k_{args.split}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[grpo-data] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
