#!/usr/bin/env python3
"""Step 7 (chart, GRPO): build the GRPO prompt dataset from ChartQA train.

Output JSONL — swift GRPO format. Each row carries:
    messages : [{"role": "system", ...}, {"role": "user", ...}]   (prompt only)
    solution : raw gold answer string (e.g. "Yes" / "47806" / "China")

The ``messages`` mirror exactly the eval/rollout prompt
(``CHART_SYSTEM_PROMPT`` + question; question already contains the Table block
in chartqa_train.jsonl). ``solution`` is consumed by ``chart_accuracy`` in
``scripts/grpo_chart_rewards.py``, which runs ``chart_answers_match`` —
identical to ``scripts/06_eval_chart{,_vllm}.py``.

Usage:
    python scripts/07_build_chart_grpo_dataset.py                  # full
    python scripts/07_build_chart_grpo_dataset.py --limit 256      # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # task dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # lzl/ for paths
from utils import CHART_SYSTEM_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str,
                   default="data/chart/chartqa_train.jsonl",
                   help="Source ChartQA train JSONL (question/answer rows).")
    p.add_argument("--limit", type=int, default=None,
                   help="Keep only the first N questions (smoke).")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. Default: data/grpo/chart_train{,_limitN}.jsonl")
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"Source not found: {src}")

    rows_in = []
    with src.open() as f:
        for ln in f:
            rows_in.append(json.loads(ln))
    if args.limit:
        rows_in = rows_in[: args.limit]

    rows_out = []
    for ex in rows_in:
        rows_out.append({
            "messages": [
                {"role": "system", "content": CHART_SYSTEM_PROMPT},
                {"role": "user",   "content": ex["question"]},
            ],
            "solution": str(ex.get("answer", "")),
        })

    if args.output:
        out = Path(args.output)
    else:
        suffix = f"_limit{args.limit}" if args.limit else ""
        out = Path(f"data/grpo/chart_train{suffix}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[chart-grpo-data] wrote {len(rows_out)} rows -> {out}")


if __name__ == "__main__":
    main()
