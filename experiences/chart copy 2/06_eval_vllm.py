#!/usr/bin/env python3
"""Step 6 (chart, vllm): eval for base / raw / paraphrase / vanilla.

LoRA adapters share one base vLLM instance. Full SFT checkpoints are loaded
directly as ``model=`` (preserving Qwen3.5 multimodal config, text-only eval).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import (
    apply_chart_template,
    chart_answers_match,
    checkpoint_kind,
    ensure_renamed_lora_adapter,
    extract_chart_answer,
    is_qwen35_model,
    latest_usable_checkpoint,
    normalize_gold_answer,
    sync_vllm_multimodal_assets,
    vllm_text_only_kwargs,
)
from paths import ensure_dirs, get_paths, load_config


def load_questions(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    if limit:
        rows = rows[:limit]
    return rows


def evaluate_one_condition(llm, tok, sp, lora_request, questions: list[dict],
                            cond_name: str, save_completion: bool = True) -> dict:
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
        "accuracy": correct / n if n else 0.0,
        "eval_mode": "vllm",
        "elapsed": round(elapsed, 1),
        "per_question": per_q,
    }


def _gpu_mem_free_gb() -> float:
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def _release_llm(llm) -> None:
    """Tear down a vLLM instance and wait until GPU memory is actually free."""
    if llm is None:
        return
    import torch

    model_path = getattr(getattr(llm, "llm_engine", None), "model_config", None)
    model_path = getattr(model_path, "model", "?")
    print(f"[vllm] releasing {model_path} ...", flush=True)

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:
        print(f"[vllm] engine_core.shutdown warning: {exc}", flush=True)

    try:
        del llm.llm_engine
    except Exception:
        pass
    del llm

    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # vLLM multiprocess workers need a moment to exit after shutdown.
    deadline = time.time() + 120
    while time.time() < deadline:
        free_gb = _gpu_mem_free_gb()
        if free_gb >= 80.0:
            print(f"[vllm] released, GPU free {free_gb:.1f} GiB", flush=True)
            return
        time.sleep(2)

    print(f"[vllm] released (GPU free {_gpu_mem_free_gb():.1f} GiB after wait)", flush=True)


def _make_llm(model_path: str, args, *, enable_lora: bool):
    from vllm import LLM

    kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
    )
    if is_qwen35_model(model_path):
        kwargs.update(vllm_text_only_kwargs())
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = args.max_lora_rank
    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"[vllm] loaded {model_path} in {time.time()-t0:.1f}s "
          f"(lora={enable_lora})", flush=True)
    return llm


def _split_conditions(conditions, checkpoint_map, base_model_path):
    lora_conds: dict[str, Path] = {}
    full_conds: dict[str, Path] = {}
    want_base = "base" in conditions

    for cond in conditions:
        if cond == "base":
            continue
        ckpt = latest_usable_checkpoint(checkpoint_map.get(cond))
        if ckpt is None or not ckpt.exists():
            print(f"[skip] checkpoint not found for {cond}: {ckpt}")
            continue
        kind = checkpoint_kind(ckpt)
        if kind == "lora":
            lora_conds[cond] = ckpt
        elif kind == "full":
            full_conds[cond] = ckpt
            print(f"[plan] {cond}: full checkpoint → vLLM ({ckpt})")
        else:
            print(f"[skip] unrecognized checkpoint for {cond}: {ckpt}")

    if not want_base and not lora_conds and not full_conds:
        return False, lora_conds, full_conds
    return want_base, lora_conds, full_conds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--datasets", nargs="+",
                   default=["chartqa_test", "plotqa", "tabmwp", "finqa"])
    p.add_argument("--conditions", nargs="+",
                   default=["base", "raw", "paraphrase", "vanilla"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--max_lora_rank", type=int, default=64)
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

    want_base, lora_conds, full_conds = _split_conditions(
        args.conditions, checkpoint_map, model_path,
    )
    if not want_base and not lora_conds and not full_conds:
        sys.exit("No conditions left to run.")

    save_completion = not args.no_save_completion
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
        temperature=0.0, top_p=1.0, max_tokens=max_new,
        stop_token_ids=sorted(stop_token_ids),
    )

    lora_requests: dict[str, LoRARequest] = {}
    for i, (cond, ckpt) in enumerate(lora_conds.items(), start=1):
        ad = ensure_renamed_lora_adapter(ckpt)
        lora_requests[cond] = LoRARequest(cond, i, ad)

    base_llm = None
    if want_base or lora_conds:
        base_llm = _make_llm(model_path, args, enable_lora=bool(lora_conds))

    for ds_name, ds_path, out_json in dataset_specs:
        questions = load_questions(ds_path, limit)
        print(f"\n=== [{ds_name}] n={len(questions)} → {out_json.name} ===", flush=True)
        results = json.load(out_json.open()) if out_json.exists() else {}

        if want_base and base_llm is not None:
            results["base"] = evaluate_one_condition(
                base_llm, tok, sp, None, questions,
                cond_name=f"{ds_name}/base", save_completion=save_completion,
            )
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)

        for cond, lr in lora_requests.items():
            results[f"{cond}_sft"] = evaluate_one_condition(
                base_llm, tok, sp, lr, questions,
                cond_name=f"{ds_name}/{cond}", save_completion=save_completion,
            )
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)

        for cond, ckpt in full_conds.items():
            sync_vllm_multimodal_assets(model_path, ckpt)
            full_llm = _make_llm(str(ckpt), args, enable_lora=False)
            results[f"{cond}_sft"] = evaluate_one_condition(
                full_llm, tok, sp, None, questions,
                cond_name=f"{ds_name}/{cond} (full)", save_completion=save_completion,
            )
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)
            _release_llm(full_llm)

        print(f"  saved → {out_json}", flush=True)

    _release_llm(base_llm)
    print("\n[done] all chart eval finished.", flush=True)


if __name__ == "__main__":
    main()
