#!/usr/bin/env python3
"""Build MLLM smoke raw.jsonl with one random correct trace per solved sample."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mllm_utils import MLLM_SYSTEM_PROMPT
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()


def correctness_bin(num_correct: int, G: int) -> str:
    if num_correct == 0:
        return "all-wrong"
    pct = num_correct / G
    if pct <= 0.25:
        return "low"
    if pct <= 0.5:
        return "mid-low"
    if pct <= 0.75:
        return "mid-high"
    if pct < 1.0:
        return "high"
    return "all-correct"


def is_non_ac(bin_name: str) -> bool:
    return bin_name != "all-correct"


def count_tokens(tokenizer, text: str) -> int:
    messages = [
        {"role": "system", "content": MLLM_SYSTEM_PROMPT},
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False,
    )
    return len(tokenizer(rendered)["input_ids"])


def compute_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    total = sum(r["tokens"] for r in rows)
    nac_tokens = sum(r["tokens"] for r in rows if is_non_ac(r["bin"]))
    return {
        "n": len(rows),
        "q": len({r["question_id"] for r in rows}),
        "tokens": total,
        "nac_pct": round(nac_tokens / total * 100, 1) if total else 0,
        "bins": dict(Counter(r["bin"] for r in rows)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.get("task") != "mllm":
        sys.exit("This smoke script is only for task: mllm")

    paths = ensure_dirs(get_paths(cfg))
    rollouts_path = Path(args.rollouts) if args.rollouts else paths.rollouts
    out_path = Path(args.output) if args.output else paths.raw_jsonl

    if not rollouts_path.exists():
        sys.exit(f"Rollouts not found: {rollouts_path}\nRun 01_sample_mllm_rollouts.py first.")

    with open(rollouts_path) as f:
        items = [json.loads(line) for line in f]
    print(f"Loaded {len(items)} questions from {rollouts_path}")

    rng = random.Random(paths.seed)
    tok = AutoTokenizer.from_pretrained(str(paths.model_path), trust_remote_code=True)

    rows = []
    total_correct_traces = 0
    G = len(items[0]["rollouts"]) if items else 0
    for item in items:
        correct = [r for r in item["rollouts"] if r.get("is_correct")]
        total_correct_traces += len(correct)
        if not correct:
            continue

        chosen = rng.choice(correct)
        text = chosen["completion"]
        num_correct = item.get("num_correct", len(correct))
        rows.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "answer": item["gold_answer"],
            "image_path": item.get("image_path"),
            "trace_text": text,
            "source": "raw_correct_one_per_sample",
            "rollout_id": chosen["rollout_id"],
            "num_correct": num_correct,
            "bin": correctness_bin(num_correct, G),
            "tokens": count_tokens(tok, text),
        })

    rng.shuffle(rows)
    stats = compute_stats(rows)
    print(f"Correct trace pool: {total_correct_traces} traces")
    print(f"Selected one trace per solved sample: {stats}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "version": "lzl_mllm_smoke_raw_one_correct_v1",
        "selection_mode": "one_correct_trace_per_solved_sample",
        "source_rollouts": str(rollouts_path),
        "input_questions": len(items),
        "questions_with_correct": len(rows),
        "correct_trace_pool": total_correct_traces,
        "seed": paths.seed,
        "stats": stats,
    }
    with open(paths.raw_manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved {out_path}")
    print(f"Manifest -> {paths.raw_manifest}")


if __name__ == "__main__":
    main()
