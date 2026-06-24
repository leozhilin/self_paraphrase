#!/usr/bin/env python3
"""Step 7 (chart, GRPO): build the GRPO prompt dataset from ChartQA train.

Output JSONL — swift GRPO format. Each row carries:
    messages : [{"role": "system", ...}, {"role": "user", ...}]   (prompt only)
    solution : raw gold answer string (e.g. "Yes" / "47806" / "China")

Usage:
    python experiences/chart/grpo/07_build_grpo_dataset.py --config chart_config.yaml
    python experiences/chart/grpo/07_build_grpo_dataset.py --config chart_config.yaml --limit 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # task dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # lzl/ for paths
from utils import CHART_SYSTEM_PROMPT
from paths import ensure_dirs, get_paths, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--src", type=str, default=None,
                   help="Source ChartQA train JSONL (overrides config train_jsonl).")
    p.add_argument("--limit", type=int, default=None,
                   help="Keep only the first N questions (smoke).")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))

    src = Path(args.src or cfg["datasets"]["train_jsonl"])
    if not src.exists():
        sys.exit(f"Source not found: {src}")

    rows_in = []
    with src.open() as f:
        for ln in f:
            rows_in.append(json.loads(ln))
    if args.limit:
        rows_in = rows_in[:args.limit]

    rows_out = []
    for ex in rows_in:
        rows_out.append({
            "messages": [
                {"role": "system", "content": CHART_SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "solution": str(ex.get("answer", "")),
        })

    if args.output:
        out = Path(args.output)
    else:
        grpo_root = Path(cfg.get("paths", {}).get(
            "grpo_data_root", "/data4/FTSO/datasets/chart/grpo"))
        suffix = f"_limit{args.limit}" if args.limit else ""
        out = grpo_root / f"chart_train{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[chart-grpo-data] wrote {len(rows_out)} rows → {out}")


if __name__ == "__main__":
    main()
