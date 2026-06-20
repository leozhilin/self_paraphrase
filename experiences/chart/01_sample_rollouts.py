#!/usr/bin/env python3
"""Step 1 (chart): Sample rollouts from ChartQA train JSONL."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()
from utils import apply_chart_template, chart_answers_match, extract_chart_answer, normalize_gold_answer


def _strip_thinking(completion: str) -> str:
    tag = "redacted_thinking"
    pat = rf"<{tag}>.*?</{tag}>"
    return re.sub(pat, "", completion, flags=re.DOTALL).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    samp = cfg["sampling"]
    train_path = Path(cfg["datasets"]["train_jsonl"])
    out_path = Path(args.output) if args.output else paths.rollouts

    if not train_path.exists():
        sys.exit(f"Train JSONL not found: {train_path}\nRun 00_prepare_chart_datasets.py first.")
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

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg["model"].get("dtype", "bfloat16"), torch.bfloat16)

    print(f"Loading model → {device}, G={samp['G']}, batch={samp['batch_size']}, n={len(questions)}")
    tok = AutoTokenizer.from_pretrained(str(paths.model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(paths.model_path), torch_dtype=torch_dtype, device_map={"": device}, trust_remote_code=True,
    )
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    G = int(samp["G"])
    batch_size = int(samp["batch_size"])

    with out_path.open("a") as f_out:
        for q_item in tqdm(questions, desc="chart rollouts"):
            gold = normalize_gold_answer(q_item["answer"])
            prompt_text = apply_chart_template(tok, q_item["question"], enable_thinking=False)
            encoded = tok(prompt_text, return_tensors="pt")
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            prompt_len = input_ids.shape[1]

            rollouts = []
            remaining = G
            while remaining > 0:
                cur_batch = min(remaining, batch_size)
                batch_input = input_ids.expand(cur_batch, -1)
                batch_mask = attention_mask.expand(cur_batch, -1)
                with torch.no_grad():
                    outputs = model.generate(
                        batch_input,
                        attention_mask=batch_mask,
                        max_new_tokens=int(samp["max_new_tokens"]),
                        temperature=float(samp["temperature"]),
                        top_p=float(samp["top_p"]),
                        top_k=int(samp["top_k"]),
                        do_sample=True,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    )
                for i in range(cur_batch):
                    completion = tok.decode(outputs[i][prompt_len:], skip_special_tokens=True)
                    completion = _strip_thinking(completion)
                    extracted = extract_chart_answer(completion)
                    rollouts.append({
                        "rollout_id": len(rollouts),
                        "completion": completion,
                        "extracted_answer": extracted,
                        "is_correct": chart_answers_match(extracted, gold),
                    })
                remaining -= cur_batch

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
            f_out.flush()

    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
