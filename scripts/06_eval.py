#!/usr/bin/env python3
"""Step 6 (gsm/math, vllm): vLLM-based eval for base / raw / paraphrase / vanilla.

Drop-in replacement for ``06_eval.py``. Same outputs:
    results/eval/<dataset>.json with keys
        base / raw_sft / paraphrase_sft / vanilla_sft
    each containing {name, total, correct, accuracy, eval_mode, elapsed,
                     per_question:[{question_id, gold, extracted,
                                    is_correct, response}]}

Speed-up: one ``vllm.LLM`` instance loads the base model once, then iterates
all conditions via ``LoRARequest`` (base = no adapter). Greedy decoding
(temperature=0, top_p=1) matches the original HF eval semantics.

Datasets supported (mirrors 06_eval.py):
    gsm        — GSM-Symbolic main set (built by 06b_robust.py first)
    gsm8k_test — official GSM8K test split (1319, eval_cache/gsm8k_test.jsonl)
    svamp      — eval_cache/svamp_eval.jsonl
    asdiv — built on the fly (yimingzhang/asdiv test split)
    multiarith — built on the fly (ChilleD/MultiArith test split)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset

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


# ----- unified final-answer extraction --------------------------------------

# Prompt template (gsm_utils.ANSWER_FORMAT_HINT) instructs the model to end with
#   "Final Answer: <bare answer>"
# We extract that trailing content VERBATIM so it works uniformly for numbers,
# MCQ letters (AQuA-RAT) and LaTeX (MATH-500). Falls back to the legacy numeric
# extractor only when no explicit Final Answer / #### marker is present.
_FINAL_ANSWER_RE = re.compile(
    r"(?:####|\*{0,2}\s*[Ff]inal\s+[Aa]nswer\s*\*{0,2})\s*[:：]?\s*([^\n]+)"
)


def _clean_answer_span(val: str) -> str:
    val = val.strip()
    # If the span itself still contains a 'Final Answer:' / '####' marker
    # (e.g. model wrote it twice), keep only the part after the LAST marker.
    inner = list(re.finditer(
        r"(?:####|\*{0,2}\s*[Ff]inal\s+[Aa]nswer\s*\*{0,2})\s*[:：]?\s*", val))
    if inner:
        val = val[inner[-1].end():].strip()
    val = val.strip().rstrip(" .。,，;；")
    val = re.sub(r"^\*+|\*+$", "", val).strip()          # strip ** bold
    if val.startswith("$"):
        val = val[1:].strip()
    if val.endswith("$"):
        val = val[:-1].strip()
    bm = re.match(r"^\\boxed\{(.+)\}$", val)              # unwrap \boxed{X}
    if bm and bm.group(1).count("{") == bm.group(1).count("}"):
        val = bm.group(1).strip()
    return val


def extract_final_answer(completion: str) -> str | None:
    """Extract the content after the LAST 'Final Answer:' / '####' marker, verbatim.

    Works for numeric, letter (MCQ) and LaTeX answers alike. If no marker is
    found, fall back to the legacy numeric extractor (extract_model_answer).
    """
    matches = list(_FINAL_ANSWER_RE.finditer(completion))
    if matches:
        val = _clean_answer_span(matches[-1].group(1))
        if val:
            return val
    return extract_model_answer(completion)


# ----- ad-hoc dataset builders (mirrors 06_eval.py) -------------------------

def _clean_asdiv(text: str) -> str:
    text = re.sub(r"^Question:\s*", "", text.strip())
    return re.sub(r"\s*Answer:\s*$", "", text).strip()


def _normalize(x) -> str:
    s = str(x).strip()
    return s.split(">>")[-1].strip() if ">>" in s else s


def build_asdiv(cache_dir: Path) -> Path:
    out = cache_dir / "asdiv_eval.jsonl"
    if out.exists():
        return out
    ds = load_dataset("yimingzhang/asdiv", split="test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(ds):
            f.write(json.dumps({
                "question_id": i,
                "question": _clean_asdiv(ex["text"]),
                "answer": _normalize(ex.get("label", ex.get("target", ""))),
                "source": "ASDiv",
            }, ensure_ascii=False) + "\n")
    return out


def build_multiarith(cache_dir: Path) -> Path:
    out = cache_dir / "multiarith_eval.jsonl"
    if out.exists():
        return out
    ds = load_dataset("ChilleD/MultiArith", split="test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(ds):
            f.write(json.dumps({
                "question_id": i,
                "question": str(ex["question"]).strip(),
                "answer": _normalize(ex["final_ans"]),
                "source": "MultiArith",
            }, ensure_ascii=False) + "\n")
    return out


def _fmt_number(x):
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else repr(x)
    return str(x).strip()


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize_math_answer(ans: str) -> str:
    s = str(ans).strip().strip("$")
    m = _BOXED_RE.search(s)
    return (m.group(1) if m else s).strip()


# ----- additional public benchmarks (mirrors 06b_robust.py builders) --------

def build_aqua_rat(cache_dir: Path) -> Path:
    out = cache_dir / "aqua_rat_eval.jsonl"
    if out.exists():
        return out
    ds = load_dataset("deepmind/aqua_rat", split="test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(ds):
            opts = ex["options"]
            q = ex["question"].strip() + "\n\nOptions:\n" + "\n".join(opts)
            gold = ex["correct"].strip().upper()
            f.write(json.dumps({
                "question_id": f"aqua_{i:04d}",
                "question": q,
                "answer": f"#### {gold}",
                "source": "aqua_rat",
            }, ensure_ascii=False) + "\n")
    return out


def build_math500(cache_dir: Path) -> Path:
    out = cache_dir / "math500_eval.jsonl"
    if out.exists():
        return out
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(ds):
            gold = _normalize_math_answer(ex["answer"])
            f.write(json.dumps({
                "question_id": ex.get("unique_id") or f"math500_{i:04d}",
                "question": ex["problem"].strip(),
                "answer": f"#### {gold}",
                "source": "math500",
            }, ensure_ascii=False) + "\n")
    return out


def build_mawps(cache_dir: Path) -> Path:
    out = cache_dir / "mawps_eval.jsonl"
    if out.exists():
        return out
    ds = load_dataset("MU-NLPC/Calc-mawps", split="test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(ds):
            f.write(json.dumps({
                "question_id": ex.get("id") or f"mawps_{i:04d}",
                "question": ex["question"].strip(),
                "answer": f"#### {_fmt_number(ex['result_float'])}",
                "source": "mawps",
            }, ensure_ascii=False) + "\n")
    return out


def build_gsm_hard(cache_dir: Path) -> Path:
    out = cache_dir / "gsm_hard_eval.jsonl"
    if out.exists():
        return out
    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, ex in enumerate(ds):
            f.write(json.dumps({
                "question_id": f"gsmhard_{i:05d}",
                "question": ex["input"].strip(),
                "answer": f"#### {_fmt_number(ex['target'])}",
                "source": "gsm_hard",
            }, ensure_ascii=False) + "\n")
    return out


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


# ---------------------------------------------------------------------------
# vLLM eval core
# ---------------------------------------------------------------------------

def evaluate_one_condition(llm, tok, sp, lora_request, questions: list[dict],
                            cond_name: str, save_completion: bool = True) -> dict:
    """Run greedy eval for one condition. lora_request=None means base model."""
    prompts = [
        apply_gsm_template(tok, q["question"] + ANSWER_FORMAT_HINT, enable_thinking=False)
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
        completion = _strip_thinking(o.outputs[0].text)
        gold = extract_gsm8k_gold_answer(q["answer"])
        extracted = extract_final_answer(completion)
        is_correct = answers_match(extracted, gold)
        if is_correct:
            correct += 1
        row = {
            "question_id": q["question_id"],
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
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=["gsm", "gsm8k_test", "svamp", "asdiv", "multiarith",
                            "aqua_rat", "math500", "mawps", "gsm_hard", "all"])
    p.add_argument("--conditions", nargs="+",
                   default=["base", "raw", "paraphrase", "vanilla"],
                   choices=["base", "raw", "paraphrase", "vanilla",
                            "vanilla_answer_only"])
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Override eval.max_new_tokens.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--max_lora_rank", type=int, default=64,
                   help="Must be ≥ the LoRA r of any adapter.")
    p.add_argument("--no_save_completion", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N questions per dataset (debug).")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    eval_cfg = cfg["eval"]
    max_new = args.max_new_tokens or int(eval_cfg["max_new_tokens"])
    model_path = str(paths.model_path)
    eval_root = paths.results_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    ds_list = (["gsm", "gsm8k_test", "svamp", "asdiv", "multiarith",
                "aqua_rat", "math500", "mawps", "gsm_hard"]
               if "all" in args.datasets else args.datasets)

    adapter_map = {
        "raw":                  Path(paths.checkpoint_root) / "raw",
        "paraphrase":           Path(paths.checkpoint_root) / "paraphrase",
        "vanilla":              Path(paths.checkpoint_root) / "vanilla",
        "vanilla_answer_only":  Path(paths.checkpoint_root) / "vanilla_answer_only",
    }

    # ----- assemble dataset spec list (name, jsonl_path, out_json) ----------
    dataset_specs: list[tuple[str, Path, Path]] = []
    if "gsm" in ds_list:
        gsm_jsonl = paths.eval_cache / "gsm_symbolic_main.jsonl"
        if not gsm_jsonl.exists():
            print(f"[skip] GSM main set missing: {gsm_jsonl} "
                  f"(run scripts/06b_robust.py build first)")
        else:
            dataset_specs.append(("gsm", gsm_jsonl,
                                  eval_root / "gsm_symbolic_main.json"))
    if "gsm8k_test" in ds_list:
        gsm8k_jsonl = paths.eval_cache / "gsm8k_test.jsonl"
        if gsm8k_jsonl.exists():
            dataset_specs.append(("gsm8k_test", gsm8k_jsonl,
                                  eval_root / "gsm8k_test.json"))
        else:
            print(f"[skip] gsm8k_test missing: {gsm8k_jsonl} "
                  f"(run scripts/06b_robust.py build gsm8k_test)")
    if "svamp" in ds_list:
        sp_path = paths.eval_cache / "svamp_eval.jsonl"
        if sp_path.exists():
            dataset_specs.append(("svamp", sp_path, eval_root / "svamp.json"))
        else:
            print(f"[skip] svamp missing: {sp_path}")
    if "asdiv" in ds_list:
        dataset_specs.append(("asdiv", build_asdiv(paths.eval_cache),
                              eval_root / "asdiv.json"))
    if "multiarith" in ds_list:
        dataset_specs.append(("multiarith", build_multiarith(paths.eval_cache),
                              eval_root / "multiarith.json"))
    if "aqua_rat" in ds_list:
        dataset_specs.append(("aqua_rat", build_aqua_rat(paths.eval_cache),
                              eval_root / "aqua_rat.json"))
    if "math500" in ds_list:
        dataset_specs.append(("math500", build_math500(paths.eval_cache),
                              eval_root / "math500.json"))
    if "mawps" in ds_list:
        dataset_specs.append(("mawps", build_mawps(paths.eval_cache),
                              eval_root / "mawps.json"))
    if "gsm_hard" in ds_list:
        dataset_specs.append(("gsm_hard", build_gsm_hard(paths.eval_cache),
                              eval_root / "gsm_hard.json"))

    if not dataset_specs:
        sys.exit("Nothing to evaluate.")

    # ----- collect adapter LoRARequests up-front ----------------------------
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_reqs: dict[str, LoRARequest | None] = {"base": None}
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

    if not any(lora_reqs.values()) and "base" not in args.conditions:
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
        questions = load_questions(ds_path)
        if args.limit:
            questions = questions[:args.limit]
        print(f"\n=== [{ds_name}] n={len(questions)} → {out_json.name} ===")
        results = json.load(out_json.open()) if out_json.exists() else {}
        for cond in args.conditions:
            if cond not in lora_reqs:
                continue
            key = "base" if cond == "base" else (
                "vanilla_answer_only_sft" if cond == "vanilla_answer_only"
                else f"{cond}_sft"
            )
            r = evaluate_one_condition(
                llm, tok, sp, lora_reqs[cond], questions,
                cond_name=f"{ds_name}/{cond}", save_completion=save_completion,
            )
            results[key] = r
            json.dump(results, out_json.open("w"), indent=2, ensure_ascii=False)
        print(f"  saved → {out_json}")

    print("\n[done] all eval finished.")


if __name__ == "__main__":
    main()
