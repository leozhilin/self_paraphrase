#!/usr/bin/env python3
"""Step 2: Build token-matched raw.jsonl from rollouts (paraphrase 输入源)."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()

SYSTEM_PROMPT_MATH = (
    "You are a helpful assistant. Solve math problems step by step. "
    "Show all your work clearly."
)


def get_system_prompt(cfg: dict) -> str:
    task = cfg.get("task")
    if task == "chart":
        from chart_utils import CHART_SYSTEM_PROMPT
        return CHART_SYSTEM_PROMPT
    if task == "mllm":
        from utils import MLLM_SYSTEM_PROMPT
        return MLLM_SYSTEM_PROMPT
    return SYSTEM_PROMPT_MATH


def is_non_ac(b: str) -> bool:
    return b != "all-correct"


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


def count_tokens(tokenizer, text: str, system_prompt: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False,
    )
    return len(tokenizer(rendered)["input_ids"])


def build_raw_pool(items, tokenizer, system_prompt):
    pool = []
    G = len(items[0]["rollouts"]) if items else 0
    for item in items:
        for r in item["rollouts"]:
            if not r["is_correct"]:
                continue
            text = r["completion"]
            pool.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["gold_answer"],
                "image_path": item.get("image_path"),  # vision tasks; None for text
                "trace_text": text,
                "source": "raw_correct",
                "rollout_id": r["rollout_id"],
                "num_correct": item["num_correct"],
                "bin": correctness_bin(item["num_correct"], G),
                "tokens": count_tokens(tokenizer, text, system_prompt),
            })
    return pool


def bin_aware_sample(pool: list, target_tokens: int, ac_max_pct: float, seed: int) -> list:
    rng = random.Random(seed)
    ac = [ex for ex in pool if not is_non_ac(ex.get("bin", "all-correct"))]
    nac = [ex for ex in pool if is_non_ac(ex.get("bin", "all-correct"))]
    rng.shuffle(ac)
    rng.shuffle(nac)

    ac_budget = int(target_tokens * ac_max_pct)
    nac_budget = target_tokens - ac_budget

    selected_nac, nac_total = [], 0
    nac_idx = 0
    while nac_total < nac_budget and nac:
        ex = nac[nac_idx % len(nac)]
        selected_nac.append(ex)
        nac_total += ex["tokens"]
        nac_idx += 1
        if nac_idx >= len(nac) * 5:
            break

    selected_ac, ac_total = [], 0
    for ex in ac:
        if ac_total >= ac_budget:
            break
        selected_ac.append(ex)
        ac_total += ex["tokens"]

    result = selected_nac + selected_ac
    rng.shuffle(result)
    return result


def compute_stats(ds: list) -> dict:
    if not ds:
        return {"n": 0}
    nac_tok = sum(ex["tokens"] for ex in ds if is_non_ac(ex.get("bin", "all-correct")))
    total = sum(ex["tokens"] for ex in ds)
    return {
        "n": len(ds),
        "q": len({ex["question_id"] for ex in ds}),
        "tokens": total,
        "nac_pct": round(nac_tok / total * 100, 1) if total else 0,
        "bins": dict(Counter(ex["bin"] for ex in ds)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    rollouts_path = Path(args.rollouts) if args.rollouts else paths.rollouts
    out_path = Path(args.output) if args.output else paths.raw_jsonl

    if not rollouts_path.exists():
        sys.exit(f"Rollouts not found: {rollouts_path}\nRun 01_sample_rollouts.py first.")

    with open(rollouts_path) as f:
        items = [json.loads(line) for line in f]
    print(f"Loaded {len(items)} questions from {rollouts_path}")

    tok = AutoTokenizer.from_pretrained(str(paths.model_path), trust_remote_code=True)
    system_prompt = get_system_prompt(cfg)
    pool = build_raw_pool(items, tok, system_prompt)
    print(f"Raw correct pool: {len(pool)} traces, {sum(x['tokens'] for x in pool)} tokens")

    selected = bin_aware_sample(
        pool,
        paths.target_tokens,
        paths.ac_max_pct,
        paths.seed,
    )
    stats = compute_stats(selected)
    print(f"Selected: {stats}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    manifest = {
        "version": "lzl_raw_v1",
        "source_rollouts": str(rollouts_path),
        "target_tokens": paths.target_tokens,
        "target_nac_pct": paths.target_nac_pct,
        "stats": stats,
    }
    with open(paths.raw_manifest, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved {out_path}")
    print(f"Manifest → {paths.raw_manifest}")


if __name__ == "__main__":
    main()
