#!/usr/bin/env python3
"""Step 6 (chart): Evaluate base / paraphrase SFT on PlotQA, TabMWP, FinQA."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from utils import apply_chart_template, chart_answers_match, extract_chart_answer, normalize_gold_answer
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

# chart_utils.extract_chart_answer falls back to sampling.sample_rollouts.extract_model_answer
# whenever a model output doesn't contain "Final Answer:". sampling/ lives at the VCTS repo
# root (one level above lzl/), so we need to put VCTS/ on sys.path explicitly.
add_vcts_to_syspath()


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def evaluate_model(base_model_path, adapter_path, questions, max_new_tokens=512, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    name = Path(adapter_path).name if adapter_path else "base"
    correct = 0
    results = []

    for i in tqdm(range(0, len(questions), batch_size), desc=f"Eval {name}"):
        batch = questions[i:i + batch_size]
        prompts = [apply_chart_template(tokenizer, q["question"], enable_thinking=False) for q in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j, q in enumerate(batch):
            gold = normalize_gold_answer(q["answer"])
            completion = tokenizer.decode(out[j][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            completion = _strip_thinking(completion)
            extracted = extract_chart_answer(completion)
            is_correct = chart_answers_match(extracted, gold)
            if is_correct:
                correct += 1
            results.append({
                "question_id": q["question_id"],
                "question": q.get("question"),
                "gold": gold,
                "extracted": extracted,
                "is_correct": is_correct,
                "completion": completion,
            })

    del model
    torch.cuda.empty_cache()
    acc = correct / len(questions) if questions else 0.0
    print(f"[{name}] {correct}/{len(questions)} = {acc:.1%}")
    return {"name": name, "total": len(questions), "correct": correct, "accuracy": acc, "per_question": results}


def load_questions(path: Path, limit: int | None):
    rows = [json.loads(line) for line in path.open()]
    if limit:
        rows = rows[:limit]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["chartqa_test", "plotqa", "tabmwp", "finqa"])
    p.add_argument("--conditions", nargs="+",
                   default=["base", "raw", "paraphrase", "vanilla"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    eval_cfg = cfg["eval"]
    batch_size = args.batch_size or eval_cfg["batch_size"]
    max_new_tokens = eval_cfg["max_new_tokens"]
    limit = args.limit or eval_cfg.get("limit")

    eval_paths = cfg["paths"].get("chart_eval", {})
    eval_root = paths.results_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    model_path = str(paths.model_path)
    adapter_map = {
        "paraphrase": str(paths.checkpoint_root / "paraphrase"),
        "raw":        str(paths.checkpoint_root / "raw"),
        "vanilla":    str(paths.checkpoint_root / "vanilla"),
    }

    for ds_name in args.datasets:
        ds_path = Path(eval_paths.get(ds_name, paths.eval_cache / f"{ds_name}_eval.jsonl"))
        if not ds_path.exists():
            print(f"Skip {ds_name}: missing {ds_path}")
            continue
        questions = load_questions(ds_path, limit)
        out_file = eval_root / f"{ds_name}.json"
        print(f"\n[{ds_name.upper()}] n={len(questions)}")
        results = json.load(out_file.open()) if out_file.exists() else {}
        if "base" in args.conditions:
            results["base"] = evaluate_model(model_path, None, questions, max_new_tokens, batch_size)
        for cond in args.conditions:
            if cond == "base":
                continue
            ad = adapter_map.get(cond)
            if ad and Path(ad).exists():
                results[f"{cond}_sft"] = evaluate_model(model_path, ad, questions, max_new_tokens, batch_size)
            else:
                print(f"  skip {cond}: adapter missing at {ad}")
        json.dump(results, out_file.open("w"), indent=2, ensure_ascii=False)
        print(f"  saved → {out_file}")


if __name__ == "__main__":
    main()
