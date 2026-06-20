#!/usr/bin/env python3
"""Step 6 (chart, vllm): vLLM-based eval for base / raw / paraphrase / vanilla.

Drop-in replacement for ``06_eval_chart.py``. Same outputs:
    results/chart/eval/<dataset>.json with keys
        base / raw_sft / paraphrase_sft / vanilla_sft

Speed-up: one ``vllm.LLM`` instance loads the base model once, then iterates
all conditions via ``LoRARequest`` (base = no adapter). Greedy decoding
(temperature=0, top_p=1) matches the original HF eval semantics.

By default runs the *full* benchmarks (no 1500-cap), unlike the previous
HF eval. Use ``--limit N`` to override.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from utils import (
    apply_chart_template,
    chart_answers_match,
    extract_chart_answer,
    normalize_gold_answer,
)
from paths import ensure_dirs, get_paths, load_config


def load_questions(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    if limit:
        rows = rows[:limit]
    return rows


def evaluate_one_condition(llm, tok, sp, lora_request, questions: list[dict],
                            cond_name: str, save_completion: bool = True) -> dict:
    """Run greedy eval for one condition. lora_request=None means base model."""
    prompts = [
        apply_chart_template(tok, q["question"], enable_thinking=False)
        for q in questions
    ]
    t0 = time.time()
    if lora_request is None:
        outs = llm.generate(prompts, sampling_params=sp)
    else:
        outs = llm.generate(prompts, sampling_params=sp, lora_request=lora_request)
    elapsed = time.time() - t0

    correct = 0
    per_q = []
    for q, o in zip(questions, outs):
        completion = o.outputs[0].text
        gold = normalize_gold_answer(q["answer"])
        extracted = extract_chart_answer(completion)
        is_correct = chart_answers_match(extracted, gold)
        if is_correct:
            correct += 1
        row = {
            "question_id": q.get("question_id"),
            "gold": gold,
            "extracted": extracted,
            "is_correct": is_correct,
        }
        if save_completion:
            row["response"] = completion
        per_q.append(row)
    n = len(questions)
    print(f"[{cond_name}] {correct}/{n} = {correct/n:.1%}  ({elapsed:.1f}s)")
    return {
        "name": cond_name,
        "total": n,
        "correct": correct,
        "accuracy": correct / n,
        "eval_mode": "vllm",
        "elapsed": round(elapsed, 1),
        "per_question": per_q,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--datasets", nargs="+",
                   default=["chartqa_test", "plotqa", "tabmwp", "finqa"])
    p.add_argument("--conditions", nargs="+",
                   default=["base", "raw", "paraphrase", "vanilla"])
    p.add_argument("--limit", type=int, default=None,
                   help="Per-dataset cap. Default: full benchmark.")
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Override eval.max_new_tokens.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--max_lora_rank", type=int, default=64,
                   help="Must be ≥ the LoRA r of any adapter.")
    p.add_argument("--no_save_completion", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    eval_cfg = cfg["eval"]
    max_new = args.max_new_tokens or int(eval_cfg["max_new_tokens"])
    limit = args.limit or eval_cfg.get("limit")
    model_path = str(paths.model_path)
    eval_root = paths.results_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    eval_paths = cfg["paths"].get("chart_eval", {})

    adapter_map = {
        "raw":        Path(paths.checkpoint_root) / "raw",
        "paraphrase": Path(paths.checkpoint_root) / "paraphrase",
        "vanilla":    Path(paths.checkpoint_root) / "vanilla",
    }

    # ----- assemble dataset spec list (name, jsonl_path, out_json) ----------
    dataset_specs: list[tuple[str, Path, Path]] = []
    for ds_name in args.datasets:
        ds_path = Path(eval_paths.get(ds_name, paths.eval_cache / f"{ds_name}_eval.jsonl"))
        if not ds_path.exists():
            print(f"[skip] {ds_name}: missing {ds_path}")
            continue
        dataset_specs.append((ds_name, ds_path, eval_root / f"{ds_name}.json"))

    if not dataset_specs:
        sys.exit("Nothing to evaluate.")

    # ----- collect adapter LoRARequests up-front ----------------------------
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_reqs: dict[str, LoRARequest | None] = {}
    if "base" in args.conditions:
        lora_reqs["base"] = None
    next_id = 1
    for cond in args.conditions:
        if cond == "base":
            continue
        ad = adapter_map.get(cond)
        if ad is None or not ad.exists():
            print(f"[skip] adapter not found for {cond}: {ad}")
            continue
        lora_reqs[cond] = LoRARequest(cond, next_id, str(ad))
        next_id += 1

    if not lora_reqs:
        sys.exit("No conditions left to run.")

    # ----- load LLM once ----------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enable_lora = any(v is not None for v in lora_reqs.values())
    t_load = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank if enable_lora else None,
    )
    print(f"[vllm] LLM loaded in {time.time()-t_load:.1f}s "
          f"(enable_lora={enable_lora})")

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new)

    # ----- iterate datasets × conditions ------------------------------------
    save_completion = not args.no_save_completion
    for ds_name, ds_path, out_json in dataset_specs:
        questions = load_questions(ds_path, limit)
        print(f"\n=== [{ds_name}] n={len(questions)} → {out_json.name} ===")
        results = json.load(out_json.open()) if out_json.exists() else {}
        for cond, lr in lora_reqs.items():
            key = "base" if cond == "base" else f"{cond}_sft"
            r = evaluate_one_condition(
                llm, tok, sp, lr, questions,
                cond_name=f"{ds_name}/{cond}", save_completion=save_completion,
            )
            results[key] = r
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)
        print(f"  saved → {out_json}")

    print("\n[done] all chart eval finished.")


if __name__ == "__main__":
    main()
