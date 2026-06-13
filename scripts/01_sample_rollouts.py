#!/usr/bin/env python3
"""Step 1 (gsm8k/math, vllm): vLLM-based text rollout sampler.

Drop-in replacement for ``01_sample_rollouts.py`` (which wraps the HF-based
``vcts/scripts/run_sampling.py``). Same output schema and resume behaviour.

Speed-up: ``SamplingParams(n=G)`` returns G rollouts in one prefix-cached
generate() call, and continuous-batching across questions interleaves
short-completing samples with long ones — so the long-tail no longer
stalls the whole batch as it does with HF transformers.

Output JSONL (identical to the HF version):
  {question_id, question, gold_answer, rollouts:[...G...],
   num_correct, all_wrong, no_valid_trace}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsm_utils import ANSWER_FORMAT_HINT, apply_gsm_template
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()
from sampling.sample_rollouts import (
    answers_match,
    extract_gsm8k_gold_answer,
    extract_model_answer,
    _strip_thinking,
)


def load_questions(dataset: str, split: str, gsm8k_cache: str) -> list[dict]:
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=gsm8k_cache)
        return [
            {"question_id": i, "question": ex["question"], "answer": ex["answer"]}
            for i, ex in enumerate(ds)
        ]
    if dataset in ("math", "math_mixed"):
        root = Path(__file__).resolve().parent.parent.parent
        if dataset == "math_mixed":
            jp = root / "data" / "cache" / "math_mixed_gate_50.jsonl"
        else:
            jp = root / "data" / "cache" / "math_subset_200.jsonl"
        items = [json.loads(l) for l in open(jp)]

        def _boxed(s: str) -> str:
            for prefix in ("\\boxed{", "\\\\boxed{"):
                idx = s.rfind(prefix)
                if idx != -1:
                    start = idx + len(prefix) - 1
                    break
            else:
                return ""
            depth = 0
            for i in range(start, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start + 1:i].strip()
            return ""

        out = []
        for i, item in enumerate(items):
            gold = _boxed(item["solution"])
            out.append({"question_id": i, "question": item["problem"],
                        "answer": f"#### {gold}"})
        return out
    raise ValueError(f"Unknown dataset: {dataset}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math", "math_mixed"])
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "test"])
    p.add_argument("--output", type=str, default=None,
                   help="Override rollouts output path.")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--chunk_size", type=int, default=64,
                   help="Questions per llm.generate() call "
                        "(each emits chunk_size*G sequences).")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    samp = cfg["sampling"]
    out_path = Path(args.output) if args.output else paths.rollouts
    model_path = Path(args.model_path) if args.model_path else paths.model_path
    gsm8k_cache = cfg["datasets"].get("gsm8k_cache", "")

    if not Path(model_path).exists():
        sys.exit(f"Model not found: {model_path}")

    # Load question list (with resume).
    questions = load_questions(args.dataset, args.split, gsm8k_cache)
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
    print(f"[gsm rollout vllm] model={model_path} dataset={args.dataset} "
          f"split={args.split} G={G} n={len(questions)} "
          f"chunk={args.chunk_size} max_seqs={args.max_num_seqs}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t_load = time.time()
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
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
            prompts = [
                apply_gsm_template(
                    tok, q["question"] + ANSWER_FORMAT_HINT, enable_thinking=False
                )
                for q in chunk
            ]
            outs = llm.generate(prompts, sampling_params=sp)
            assert len(outs) == len(chunk)
            for q, o in zip(chunk, outs):
                gold = extract_gsm8k_gold_answer(q["answer"])
                rollouts = []
                for i, completion in enumerate(o.outputs):
                    text = _strip_thinking(completion.text)
                    extracted = extract_model_answer(text)
                    rollouts.append({
                        "rollout_id":       i,
                        "completion":       text,
                        "extracted_answer": extracted,
                        "is_correct":       answers_match(extracted, gold),
                    })
                num_correct = sum(r["is_correct"] for r in rollouts)
                rec = {
                    "question_id":    q["question_id"],
                    "question":       q["question"],
                    "gold_answer":    gold,
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
