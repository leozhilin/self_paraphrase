#!/usr/bin/env python3
"""Step 1 (mllm, vllm): vLLM-based vision rollout sampler.

Drop-in replacement for the vision branch of ``01_sample_mllm_rollouts.py``.
Same output schema, same resume behaviour. The HF transformers backend
sequentially generates G rollouts per question; vLLM uses
``SamplingParams(n=G)`` + continuous batching to interleave many questions,
and prefix-caches the (large) image+prompt KV between the G samples.

Output JSONL (same as HF version):
  {question_id, question, gold_answer, image_path, rollouts:[...G...],
   num_correct, all_wrong, no_valid_trace}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mllm_utils import (
    MLLM_SYSTEM_PROMPT,
    extract_mllm_answer,
    mllm_answers_match,
    normalize_gold_answer,
    _strip_thinking,
)
from paths import ensure_dirs, get_paths, load_config


def build_request(processor, question: str, image_path: str | None) -> dict:
    content = []
    img = None
    if image_path:
        try:
            img = Image.open(image_path).convert("RGB")
            content.append({"type": "image"})
        except Exception:
            img = None
    content.append({"type": "text", "text": question})
    messages = [
        {"role": "system", "content": MLLM_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    req: dict = {"prompt": prompt}
    if img is not None:
        req["multi_modal_data"] = {"image": img}
    return req


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output", type=str, default=None,
                   help="Override output jsonl path.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--mm_max_pixels", type=int, default=1280 * 1280)
    p.add_argument("--chunk_size", type=int, default=32,
                   help="How many questions to feed to one llm.generate() "
                        "call (each yields chunk_size*G sequences).")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    samp = cfg["sampling"]
    train_path = Path(cfg["datasets"]["train_jsonl"])
    out_path = Path(args.output) if args.output else paths.rollouts
    model_path = Path(args.model_path) if args.model_path else paths.model_path

    if not train_path.exists():
        sys.exit(f"Train JSONL not found: {train_path}")
    if not Path(model_path).exists():
        sys.exit(f"Model not found: {model_path}")

    # Load question list (with resume).
    questions = []
    with train_path.open() as f:
        for line in f:
            ex = json.loads(line)
            questions.append({
                "question_id": ex["question_id"],
                "question":    ex["question"],
                "answer":      ex["answer"],
                "image_path":  ex.get("image_path"),
            })
    if args.start:
        questions = questions[args.start:]
    if args.limit:
        questions = questions[:args.limit]

    existing = 0
    if out_path.exists():
        with out_path.open() as f:
            existing = sum(1 for _ in f)
        if existing:
            print(f"Resume from {existing}")
            questions = questions[existing:]
    if not questions:
        print("Nothing to sample.")
        return

    G = int(samp["G"])
    print(f"[mllm rollout vllm] model={model_path} G={G} n={len(questions)} "
          f"chunk={args.chunk_size} max_seqs={args.max_num_seqs}")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    t_load = time.time()
    llm = LLM(
        model=str(model_path), trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        mm_processor_kwargs={"max_pixels": args.mm_max_pixels},
    )
    print(f"[vllm] LLM loaded in {time.time()-t_load:.1f}s")

    sp = SamplingParams(
        n=G,
        temperature=float(samp["temperature"]),
        top_p=float(samp["top_p"]),
        top_k=int(samp["top_k"]),
        max_tokens=int(samp["max_new_tokens"]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.time()
    with out_path.open("a") as f_out:
        for chunk_start in tqdm(range(0, len(questions), args.chunk_size),
                                desc="rollout chunks"):
            chunk = questions[chunk_start:chunk_start + args.chunk_size]
            requests = [build_request(processor, q["question"], q.get("image_path"))
                        for q in chunk]
            outs = llm.generate(requests, sampling_params=sp)
            assert len(outs) == len(chunk)
            for q, o in zip(chunk, outs):
                gold = normalize_gold_answer(q["answer"])
                rollouts = []
                for i, completion in enumerate(o.outputs):
                    text = _strip_thinking(completion.text)
                    extracted = extract_mllm_answer(text)
                    rollouts.append({
                        "rollout_id":       i,
                        "completion":       text,
                        "extracted_answer": extracted,
                        "is_correct":       mllm_answers_match(extracted, gold),
                    })
                num_correct = sum(r["is_correct"] for r in rollouts)
                rec = {
                    "question_id":    q["question_id"],
                    "question":       q["question"],
                    "gold_answer":    gold,
                    "image_path":     q.get("image_path"),
                    "rollouts":       rollouts,
                    "num_correct":    num_correct,
                    "all_wrong":      num_correct == 0,
                    "no_valid_trace": False,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f_out.flush()

    elapsed = time.time() - t0
    print(f"Done → {out_path}")
    print(f"[stats] {written} questions, {written*G} rollouts, "
          f"{elapsed:.1f}s, {elapsed/max(written,1):.2f}s/q")


if __name__ == "__main__":
    main()
