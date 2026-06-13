#!/usr/bin/env python3
"""One-off eval: legacy vcts_sft_pilot LoRA adapters via vLLM + *old* pilot prompts.

Does NOT modify lzl/results/eval/*.json — outputs go to
lzl/results/eval_legacy_pilot/ by default.

Prompt / extraction match VCTS/scripts/eval_gsm_symbolic.py (pilot era):
  - system: sampling.prompt_templates.SYSTEM_PROMPT (no Final Answer in system)
  - user:   question + ANSWER_FORMAT_HINT  → conclude with ``#### <answer>``
  - extract: sampling.sample_rollouts.extract_model_answer + answers_match

Example:
  LZL_CONFIG=lzl/config.yaml python lzl/scripts/tmp_eval_legacy_pilot_vllm.py \\
    --datasets gsm8k_test svamp aqua_rat math500 gsm_hard \\
    --conditions base raw paraphrase \\
    --max_new_tokens 4096 --max_model_len 12288
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LZL_ROOT = SCRIPT_DIR.parent
VCTS_ROOT = LZL_ROOT.parent

sys.path.insert(0, str(LZL_ROOT))
sys.path.insert(0, str(VCTS_ROOT))

from paths import ensure_dirs, get_paths, load_config  # noqa: E402

# Dataset builders + load_questions from 06_eval (no prompt logic).
_EVAL_SPEC = importlib.util.spec_from_file_location(
    "_gsm_eval_vllm", SCRIPT_DIR / "06_eval.py",
)
eval_mod = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(eval_mod)

from sampling.prompt_templates import apply_template  # noqa: E402
from sampling.sample_rollouts import (  # noqa: E402
    answers_match,
    extract_gsm8k_gold_answer,
    extract_model_answer,
    _strip_thinking,
)
from scripts.eval_gsm_symbolic import ANSWER_FORMAT_HINT  # noqa: E402

DEFAULT_LEGACY_ROOT = Path(
    "/home/liuyu/Projects/MM2026_checkpoints/vcts_sft_pilot",
)
DEFAULT_LEGACY_MODEL = Path(
    "/home/liuyu/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507",
)
DEFAULT_OUT = LZL_ROOT / "results" / "eval_legacy_pilot"

LEGACY_CONDITION_DIRS = {
    "raw": "raw",
    "paraphrase": "paraphrase",
    "mixed": "mixed",
    "control": "control",
    "vcts": "vcts",
    "multi_correct": "multi_correct",
}


def legacy_build_prompt(tokenizer, question: str) -> str:
    """Same as eval_gsm_symbolic.evaluate_model."""
    return apply_template(
        tokenizer, question + ANSWER_FORMAT_HINT, enable_thinking=False,
    )


def legacy_evaluate_one_condition(
    llm, tok, sp, lora_request, questions: list[dict],
    cond_name: str, save_completion: bool = True,
) -> dict:
    prompts = [legacy_build_prompt(tok, q["question"]) for q in questions]
    t0 = time.time()
    if lora_request is None:
        outs = llm.generate(prompts, sampling_params=sp)
    else:
        outs = llm.generate(prompts, sampling_params=sp, lora_request=lora_request)
    elapsed = time.time() - t0

    correct = 0
    per_q = []
    for q, o in zip(questions, outs):
        completion = _strip_thinking(o.outputs[0].text)
        gold = extract_gsm8k_gold_answer(q["answer"])
        extracted = extract_model_answer(completion)
        is_correct = answers_match(extracted, gold)
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
        "eval_mode": "vllm_legacy_prompt",
        "prompt_style": "eval_gsm_symbolic (#### hint)",
        "elapsed": round(elapsed, 1),
        "per_question": per_q,
    }


def build_dataset_specs(ds_list: list[str], paths, eval_root: Path) -> list[tuple[str, Path, Path]]:
    specs: list[tuple[str, Path, Path]] = []
    if "gsm" in ds_list:
        gsm_jsonl = paths.eval_cache / "gsm_symbolic_main.jsonl"
        if gsm_jsonl.exists():
            specs.append(("gsm", gsm_jsonl, eval_root / "gsm_symbolic_main.json"))
        else:
            print(f"[skip] GSM main missing: {gsm_jsonl}")
    if "gsm8k_test" in ds_list:
        p = paths.eval_cache / "gsm8k_test.jsonl"
        if p.exists():
            specs.append(("gsm8k_test", p, eval_root / "gsm8k_test.json"))
    if "svamp" in ds_list:
        p = paths.eval_cache / "svamp_eval.jsonl"
        if p.exists():
            specs.append(("svamp", p, eval_root / "svamp.json"))
    if "asdiv" in ds_list:
        specs.append(("asdiv", eval_mod.build_asdiv(paths.eval_cache),
                      eval_root / "asdiv.json"))
    if "multiarith" in ds_list:
        specs.append(("multiarith", eval_mod.build_multiarith(paths.eval_cache),
                      eval_root / "multiarith.json"))
    if "aqua_rat" in ds_list:
        specs.append(("aqua_rat", eval_mod.build_aqua_rat(paths.eval_cache),
                      eval_root / "aqua_rat.json"))
    if "math500" in ds_list:
        specs.append(("math500", eval_mod.build_math500(paths.eval_cache),
                      eval_root / "math500.json"))
    if "mawps" in ds_list:
        specs.append(("mawps", eval_mod.build_mawps(paths.eval_cache),
                      eval_root / "mawps.json"))
    if "gsm_hard" in ds_list:
        specs.append(("gsm_hard", eval_mod.build_gsm_hard(paths.eval_cache),
                      eval_root / "gsm_hard.json"))
    return specs


def main():
    p = argparse.ArgumentParser(description="Legacy pilot eval (old #### prompts, isolated output)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--adapter_root", type=Path, default=DEFAULT_LEGACY_ROOT)
    p.add_argument("--model_path", type=Path, default=DEFAULT_LEGACY_MODEL)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=["gsm", "gsm8k_test", "svamp", "asdiv", "multiarith",
                            "aqua_rat", "math500", "mawps", "gsm_hard", "all"])
    p.add_argument("--conditions", nargs="+",
                   default=["base", "raw", "paraphrase"],
                   choices=["base", "raw", "paraphrase", "mixed", "control",
                            "vcts", "multi_correct"])
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no_save_completion", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))

    if not args.model_path.exists():
        sys.exit(f"Base model not found: {args.model_path}")
    if not args.adapter_root.exists():
        sys.exit(f"Legacy adapter root not found: {args.adapter_root}")

    ds_list = (["gsm", "gsm8k_test", "svamp", "asdiv", "multiarith",
                "aqua_rat", "math500", "mawps", "gsm_hard"]
               if "all" in args.datasets else args.datasets)

    eval_root = args.output_dir
    eval_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "tag": "legacy_vcts_sft_pilot",
        "adapter_root": str(args.adapter_root),
        "model_path": str(args.model_path),
        "eval_stack": "vLLM + eval_gsm_symbolic prompts (#### ANSWER_FORMAT_HINT)",
        "extraction": "sample_rollouts.extract_model_answer",
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": args.conditions,
        "datasets": ds_list,
    }
    json.dump(meta, (eval_root / "run_meta.json").open("w"), indent=2)
    print(f"[legacy eval] adapter_root={args.adapter_root}")
    print(f"[legacy eval] model={args.model_path}")
    print(f"[legacy eval] prompt=eval_gsm_symbolic (####)")
    print(f"[legacy eval] output={eval_root}")

    dataset_specs = build_dataset_specs(ds_list, paths, eval_root)
    if not dataset_specs:
        sys.exit("Nothing to evaluate.")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    lora_reqs: dict[str, LoRARequest | None] = {}
    if "base" in args.conditions:
        lora_reqs["base"] = None
    next_id = 1
    for cond in args.conditions:
        if cond == "base":
            continue
        sub = LEGACY_CONDITION_DIRS.get(cond, cond)
        ad = args.adapter_root / sub
        if not ad.exists():
            print(f"[skip] adapter missing: {cond} → {ad}")
            continue
        lora_reqs[cond] = LoRARequest(f"legacy_{cond}", next_id, str(ad))
        next_id += 1

    if not lora_reqs:
        sys.exit("No conditions to run.")

    model_path = str(args.model_path)
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
    print(f"[vllm] loaded in {time.time()-t_load:.1f}s (enable_lora={enable_lora})")

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    save_completion = not args.no_save_completion

    for ds_name, ds_path, out_json in dataset_specs:
        questions = eval_mod.load_questions(ds_path)
        if args.limit:
            questions = questions[:args.limit]
        print(f"\n=== [legacy/{ds_name}] n={len(questions)} → {out_json} ===")
        results = {}
        for cond, lr in lora_reqs.items():
            key = "base" if cond == "base" else f"legacy_{cond}_sft"
            r = legacy_evaluate_one_condition(
                llm, tok, sp, lr, questions,
                cond_name=f"legacy/{ds_name}/{cond}",
                save_completion=save_completion,
            )
            results[key] = r
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)
        print(f"  saved → {out_json}")

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(meta, (eval_root / "run_meta.json").open("w"), indent=2)
    print("\n[done] legacy pilot eval finished.")


if __name__ == "__main__":
    main()
