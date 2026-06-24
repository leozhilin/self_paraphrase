#!/usr/bin/env python3
"""Step 1 (chart): Sample rollouts from ChartQA train JSONL (vLLM)."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()
from utils import (
    apply_chart_template,
    chart_answers_match,
    extract_chart_answer,
    is_qwen35_model,
    normalize_gold_answer,
    strip_thinking,
    vllm_text_only_kwargs,
)


def _strip_thinking(completion: str) -> str:
    return strip_thinking(completion)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--chunk_size", type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    samp = cfg["sampling"]
    train_path = Path(cfg["datasets"]["train_jsonl"])
    out_path = Path(args.output) if args.output else paths.rollouts
    model_path = str(paths.model_path)

    if not train_path.exists():
        sys.exit(f"Train JSONL not found: {train_path}")
    if not paths.model_path.exists():
        sys.exit(f"Model not found: {paths.model_path}")

    questions = []
    with train_path.open() as f:
        for line in f:
            ex = json.loads(line)
            questions.append({
                "question_id": ex["question_id"],
                "question": ex["question"],
                "answer": ex["answer"],
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

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    G = int(samp["G"])
    print(f"[rollout] model={model_path} G={G} n={len(questions)} qwen35={is_qwen35_model(model_path)}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=str(cfg["model"].get("dtype", "bfloat16")),
    )
    if is_qwen35_model(model_path):
        llm_kwargs.update(vllm_text_only_kwargs())

    t_load = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[vllm] loaded in {time.time()-t_load:.1f}s")

    stop_token_ids = {
        tid for tid in (
            tok.eos_token_id,
            tok.pad_token_id,
            tok.convert_tokens_to_ids("<|im_end|>"),
            tok.convert_tokens_to_ids("<|endoftext|>"),
        )
        if isinstance(tid, int) and tid >= 0
    }
    sp = SamplingParams(
        n=G,
        temperature=float(samp["temperature"]),
        top_p=float(samp["top_p"]),
        top_k=int(samp["top_k"]),
        max_tokens=int(samp["max_new_tokens"]),
        stop_token_ids=sorted(stop_token_ids),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.time()
    with out_path.open("a") as f_out:
        for chunk_start in tqdm(range(0, len(questions), args.chunk_size), desc="rollouts"):
            chunk = questions[chunk_start:chunk_start + args.chunk_size]
            prompts = [
                apply_chart_template(tok, q["question"], enable_thinking=False)
                for q in chunk
            ]
            outs = llm.generate(prompts, sampling_params=sp)
            for q_item, o in zip(chunk, outs):
                gold = normalize_gold_answer(q_item["answer"])
                rollouts = []
                for i, completion in enumerate(o.outputs):
                    text = _strip_thinking(completion.text)
                    extracted = extract_chart_answer(text)
                    rollouts.append({
                        "rollout_id": i,
                        "completion": text,
                        "extracted_answer": extracted,
                        "is_correct": chart_answers_match(extracted, gold),
                    })
                num_correct = sum(r["is_correct"] for r in rollouts)
                rec = {
                    "question_id": q_item["question_id"],
                    "question": q_item["question"],
                    "gold_answer": gold,
                    "rollouts": rollouts,
                    "num_correct": num_correct,
                    "all_wrong": num_correct == 0,
                    "no_valid_trace": False,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f_out.flush()

    elapsed = time.time() - t0
    print(f"Done → {out_path}  ({written} q, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
