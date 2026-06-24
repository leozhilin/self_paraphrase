#!/usr/bin/env python3
"""Convert chart SFT JSONL files to ms-swift messages JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from paths import ensure_dirs, get_paths, load_config  # noqa: E402
from utils import CHART_SYSTEM_PROMPT  # noqa: E402


def _condition_paths(paths) -> dict[str, Path]:
    return {
        "raw": paths.raw_jsonl,
        "paraphrase": paths.paraphrase_jsonl,
        "vanilla": paths.vanilla_jsonl,
    }


def _targets(condition: str) -> list[str]:
    if condition == "all":
        return ["raw", "paraphrase", "vanilla"]
    return [condition]


def convert_one(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            trace = row.get("trace_text")
            if not question or not trace:
                continue
            out = {
                "messages": [
                    {"role": "system", "content": CHART_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": trace},
                ],
                "question_id": row.get("question_id"),
                "source": row.get("source"),
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--condition",
        choices=["raw", "paraphrase", "vanilla", "all"],
        default="all",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    data_map = _condition_paths(paths)

    for cond in _targets(args.condition):
        src = data_map[cond]
        if not src.exists():
            sys.exit(f"missing {cond} SFT data: {src}")
        dst = src.with_name("swift_messages.jsonl")
        n = convert_one(src, dst)
        print(f"[swift-sft] {cond}: wrote {n} rows -> {dst}")


if __name__ == "__main__":
    main()
