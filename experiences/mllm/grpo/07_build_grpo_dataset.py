#!/usr/bin/env python3
"""Step 7 (mllm, GRPO): build the GRPO prompt dataset for Qwen3.5-4B.

Output JSONL — swift multimodal GRPO format. Each row carries:
    messages      : [{role: system, content: ...},
                      {role: user, content: "<image>{question}"}]
    images        : [absolute_image_path]            (single image per row)
    solution      : raw gold answer string
    question_text : the rendered question text (used by mllm_accuracy reward
                    to re-parse MCQ options when matching letter↔fulltext)

The ``messages`` mirror exactly ``06_eval_mllm_vllm.build_request`` (system =
``MLLM_SYSTEM_PROMPT``; user has an image followed by the question text). At
training time swift composes the chat template with the actual image tokens
inserted in place of ``<image>``; at eval time the same prompt is regenerated.

``solution`` is consumed by ``mllm_accuracy`` (scripts/grpo_mllm_rewards.py),
running ``mllm_answers_match`` — identical to ``06_eval_mllm_vllm.py``.

Usage:
    python scripts/07_build_mllm_grpo_dataset.py                   # full
    python scripts/07_build_mllm_grpo_dataset.py --limit 64        # smoke
    python scripts/07_build_mllm_grpo_dataset.py \
        --src data/mllm/pgps9k_train_1k_smoke.jsonl --limit 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # task dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # lzl/ for paths
from utils import MLLM_SYSTEM_PROMPT
from paths import apply_hf_env, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--src", type=str, default=None,
                   help="Source MLLM train JSONL (default: config datasets.train_jsonl).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--require_image", action="store_true",
                   help="Drop rows whose image_path is missing on disk.")
    args = p.parse_args()

    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    cfg = load_config(args.config or str(default_cfg))
    apply_hf_env(cfg)

    src = Path(args.src or cfg["datasets"]["train_jsonl"])
    if not src.exists():
        sys.exit(f"Source not found: {src}")

    rows_in = []
    with src.open() as f:
        for ln in f:
            rows_in.append(json.loads(ln))
    if args.limit:
        rows_in = rows_in[: args.limit]

    rows_out = []
    skipped_no_img = 0
    for ex in rows_in:
        img_path = ex.get("image_path")
        # Some sources store image_path as list (e.g. multi-image MMMU); keep
        # only the first to stay single-image like rollout/eval.
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0] if img_path else None
        if not img_path:
            if args.require_image:
                skipped_no_img += 1
                continue
            # Pure text fallback (no image marker).
            user_content = ex["question"]
            images_field = []
        else:
            ip = Path(img_path)
            if args.require_image and not ip.exists():
                skipped_no_img += 1
                continue
            user_content = "<image>" + ex["question"]
            images_field = [str(ip)]

        rows_out.append({
            "messages": [
                {"role": "system", "content": MLLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "images": images_field,
            "solution": str(ex.get("answer", "")),
            # Carry the rendered question text (no <image> marker) so the
            # accuracy reward can re-parse MCQ option lines for robust matching.
            "question_text": ex["question"],
        })

    if args.output:
        out = Path(args.output)
    else:
        grpo_root = Path(cfg.get("paths", {}).get(
            "grpo_data_root", "/data4/FTSO/datasets/mllm/grpo"))
        suffix = f"_limit{args.limit}" if args.limit else "_full"
        out = grpo_root / f"mllm_train{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mllm-grpo-data] wrote {len(rows_out)} rows -> {out}"
          + (f" (skipped {skipped_no_img} rows with missing image)"
             if skipped_no_img else ""))


if __name__ == "__main__":
    main()
