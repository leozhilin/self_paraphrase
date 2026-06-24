#!/usr/bin/env python3
"""Step 6 (chart, vllm): eval for base / raw / paraphrase / vanilla.

Drop-in replacement for ``06_eval_chart.py``. Same outputs:
    results/chart/eval/<dataset>.json with keys
        base / raw_sft / paraphrase_sft / vanilla_sft

LoRA adapters run through one shared vLLM base model. Full SFT checkpoints
are delegated to ``06_eval.py`` (HF generate), because merged full weights
are not loaded as LoRA adapters here.

Greedy decoding (temperature=0, top_p=1) matches the original HF eval semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from utils import (
    apply_chart_template,
    chart_answers_match,
    checkpoint_kind,
    extract_chart_answer,
    latest_usable_checkpoint,
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
    print(f"[{cond_name}] {correct}/{n} = {correct/n:.1%}  ({elapsed:.1f}s)", flush=True)
    return {
        "name": cond_name,
        "total": n,
        "correct": correct,
        "accuracy": correct / n,
        "eval_mode": "vllm",
        "elapsed": round(elapsed, 1),
        "per_question": per_q,
    }


def _load_hf_eval_module():
    path = Path(__file__).resolve().parent / "06_eval.py"
    spec = importlib.util.spec_from_file_location("_chart_eval_hf", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import HF eval module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_full_checkpoint(hf_mod, base_model_path: str, ckpt_dir: Path,
                             questions: list[dict], max_new_tokens: int, batch_size: int,
                             cond_name: str, save_completion: bool = True) -> dict:
    t0 = time.time()
    raw = hf_mod.evaluate_model(
        base_model_path, str(ckpt_dir), questions, max_new_tokens, batch_size,
    )
    elapsed = time.time() - t0
    per_q = []
    for row in raw["per_question"]:
        item = {
            "question_id": row.get("question_id"),
            "gold": row["gold"],
            "extracted": row["extracted"],
            "is_correct": row["is_correct"],
        }
        if save_completion:
            item["response"] = row.get("completion") or row.get("response")
        per_q.append(item)
    n = raw["total"]
    return {
        "name": cond_name,
        "total": n,
        "correct": raw["correct"],
        "accuracy": raw["accuracy"],
        "eval_mode": "hf",
        "elapsed": round(elapsed, 1),
        "per_question": per_q,
    }


def _split_conditions(conditions, checkpoint_map):
    """Return (vllm_conds, hf_conds) where values are checkpoint dirs for hf."""
    vllm_conds: list[str] = []
    hf_conds: dict[str, Path] = {}
    for cond in conditions:
        if cond == "base":
            vllm_conds.append("base")
            continue
        ckpt_dir = latest_usable_checkpoint(checkpoint_map.get(cond))
        if ckpt_dir is None or not ckpt_dir.exists():
            print(f"[skip] checkpoint not found for {cond}: {ckpt_dir}")
            continue
        kind = checkpoint_kind(ckpt_dir)
        if kind == "lora":
            vllm_conds.append(cond)
        elif kind == "full":
            hf_conds[cond] = ckpt_dir
            print(f"[plan] {cond}: full checkpoint → HF eval ({ckpt_dir})")
        else:
            print(f"[skip] unrecognized checkpoint for {cond}: {ckpt_dir}")
    return vllm_conds, hf_conds


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
    batch_size = int(eval_cfg.get("batch_size", 4))
    limit = args.limit or eval_cfg.get("limit")
    model_path = str(paths.model_path)
    eval_root = paths.results_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    eval_paths = cfg["paths"].get("chart_eval", {})
    checkpoint_map = {
        "raw":        Path(paths.checkpoint_root) / "raw",
        "paraphrase": Path(paths.checkpoint_root) / "paraphrase",
        "vanilla":    Path(paths.checkpoint_root) / "vanilla",
    }

    dataset_specs: list[tuple[str, Path, Path]] = []
    for ds_name in args.datasets:
        ds_path = Path(eval_paths.get(ds_name, paths.eval_cache / f"{ds_name}_eval.jsonl"))
        if not ds_path.exists():
            print(f"[skip] {ds_name}: missing {ds_path}")
            continue
        dataset_specs.append((ds_name, ds_path, eval_root / f"{ds_name}.json"))

    if not dataset_specs:
        sys.exit("Nothing to evaluate.")

    vllm_conds, hf_conds = _split_conditions(args.conditions, checkpoint_map)
    if not vllm_conds and not hf_conds:
        sys.exit("No conditions left to run.")

    save_completion = not args.no_save_completion
    hf_mod = _load_hf_eval_module() if hf_conds else None

    llm = None
    tok = None
    sp = None
    lora_reqs: dict[str, object] = {}
    if vllm_conds:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        next_id = 1
        for cond in vllm_conds:
            if cond == "base":
                lora_reqs["base"] = None
                continue
            ckpt_dir = latest_usable_checkpoint(checkpoint_map[cond])
            lora_reqs[cond] = LoRARequest(cond, next_id, str(ckpt_dir))
            next_id += 1

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
              f"(enable_lora={enable_lora})", flush=True)

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
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new,
            stop_token_ids=sorted(stop_token_ids),
        )

    for ds_name, ds_path, out_json in dataset_specs:
        questions = load_questions(ds_path, limit)
        print(f"\n=== [{ds_name}] n={len(questions)} → {out_json.name} ===", flush=True)
        results = json.load(out_json.open()) if out_json.exists() else {}

        for cond, lr in lora_reqs.items():
            key = "base" if cond == "base" else f"{cond}_sft"
            results[key] = evaluate_one_condition(
                llm, tok, sp, lr, questions,
                cond_name=f"{ds_name}/{cond}", save_completion=save_completion,
            )
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)

        for cond, ckpt_dir in hf_conds.items():
            key = f"{cond}_sft"
            results[key] = evaluate_full_checkpoint(
                hf_mod, model_path, ckpt_dir, questions, max_new, batch_size,
                cond_name=f"{ds_name}/{cond} (full-hf)",
                save_completion=save_completion,
            )
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)

        print(f"  saved → {out_json}", flush=True)

    print("\n[done] all chart eval finished.", flush=True)


if __name__ == "__main__":
    main()
