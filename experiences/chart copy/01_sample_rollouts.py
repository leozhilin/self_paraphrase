#!/usr/bin/env python3
"""Step 1 (chart, vllm): Sample rollouts from ChartQA train JSONL.

Same output schema as the earlier HF backend, but uses vLLM
``SamplingParams(n=G)`` and continuous batching across questions.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()
from utils import (
    apply_chart_template,
    chart_answers_match_4dp,
    extract_chart_answer,
    normalize_gold_answer,
)


def _strip_thinking(completion: str) -> str:
    completion = re.sub(r"<think>.*?</think>", "", completion,
                        flags=re.DOTALL).strip()
    completion = re.sub(r"<redacted_thinking>.*?</redacted_thinking>", "",
                        completion, flags=re.DOTALL).strip()
    return completion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Override sampling.max_new_tokens, mainly for smoke tests.")
    p.add_argument("--chunk_size", type=int, default=4,
                   help="Questions per llm.generate() call. Each question "
                        "emits G sequences, so active sequences are roughly "
                        "chunk_size * G.")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    samp = cfg["sampling"]
    train_path = Path(cfg["datasets"]["train_jsonl"])
    out_path = Path(args.output) if args.output else paths.rollouts
    model_path = Path(args.model_path) if args.model_path else paths.model_path

    if not train_path.exists():
        sys.exit(f"Train JSONL not found: {train_path}\nRun 00_prepare_chart_datasets.py first.")
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

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

    G = int(samp["G"])
    print(f"[chart rollout vllm] model={model_path} G={G} n={len(questions)} "
          f"chunk={args.chunk_size} max_seqs={args.max_num_seqs} "
          f"tp={args.tensor_parallel_size} gmu={args.gpu_memory_utilization}")

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
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=str(cfg["model"].get("dtype", "bfloat16")),
        limit_mm_per_prompt={"image": 0},
    )
    print(f"[vllm] LLM loaded in {time.time()-t_load:.1f}s")

    stop_token_ids = {
        tid for tid in (
            tok.eos_token_id,
            tok.pad_token_id,
            tok.convert_tokens_to_ids("<|im_end|>"),
            tok.convert_tokens_to_ids("<|endoftext|>"),
        )
        if isinstance(tid, int) and tid >= 0
    }
    print(f"[vllm] stop_token_ids={sorted(stop_token_ids)}")

    sp = SamplingParams(
        n=G,
        temperature=float(samp["temperature"]),
        top_p=float(samp["top_p"]),
        top_k=int(samp["top_k"]),
        max_tokens=int(args.max_new_tokens or samp["max_new_tokens"]),
        stop_token_ids=sorted(stop_token_ids),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with out_path.open("a") as f_out:
        for chunk_start in tqdm(range(0, len(questions), args.chunk_size),
                                desc="rollout chunks"):
            chunk = questions[chunk_start:chunk_start + args.chunk_size]
            prompts = [
                apply_chart_template(tok, q["question"], enable_thinking=False)
                for q in chunk
            ]
            outs = llm.generate(prompts, sampling_params=sp)
            assert len(outs) == len(chunk)

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
                        "is_correct": chart_answers_match_4dp(extracted, gold),
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

    print(f"Done → {out_path}")
    elapsed = time.time() - t0
    print(f"[stats] {written} questions, {written*G} rollouts, "
          f"{elapsed:.1f}s, {elapsed/max(written,1):.2f}s/q")


if __name__ == "__main__":
    main()
