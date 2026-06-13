#!/usr/bin/env python3
"""Sample a reproducible random subset from PGPS9K train jsonl for smoke tests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--src", default=None, help="Full train jsonl (default: pgps9k_train.jsonl)")
    p.add_argument("--out", default=None, help="Output subset jsonl")
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = cfg["datasets"]
    src = Path(args.src or ds.get("pgps9k_full_train_jsonl",
                                 str(Path(ds["train_jsonl"]).parent / "pgps9k_train.jsonl")))
    out = Path(args.out or ds["train_jsonl"])

    if not src.exists():
        sys.exit(f"Source train jsonl not found: {src}\nRun 00_prepare_mllm_datasets.py first.")
    rows = [json.loads(line) for line in src.open()]
    if args.n > len(rows):
        sys.exit(f"Requested n={args.n} but only {len(rows)} rows in {src}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    subset = rows[:args.n]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "n": len(subset),
        "seed": args.seed,
        "source": str(src),
        "question_ids": [r["question_id"] for r in subset],
    }, indent=2, ensure_ascii=False))
    print(f"[smoke] Wrote {len(subset)} questions → {out}")
    print(f"[smoke] meta → {meta}")


if __name__ == "__main__":
    main()
