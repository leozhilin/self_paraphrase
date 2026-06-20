#!/usr/bin/env python3
"""Step 6 (mllm, vllm): vLLM-based vision evaluation with LoRARequest.

The SFT script saves LoRA adapters with keys ``base_model.model.model.layers.X``,
but the multimodal Qwen3.5-4B model exposes its language layers at
``model.language_model.layers.X``. The two differ by a ``language_model.``
segment. We patch keys in-memory to a sibling ``{adapter}_renamed`` directory
the first time we see one, so vllm's PEFT loader (and PunicaWrapperGPU) can
inject the adapter correctly.

Output is JSON-merged with any previous run's content:
  results/mllm/eval/{ds}.json
  ├── base / raw_sft / paraphrase_sft / vanilla_sft
  │   └── name, total, correct, accuracy, eval_mode="vllm",
  │       elapsed, per_question
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from utils import (
    MLLM_SYSTEM_PROMPT,
    extract_mllm_answer,
    mllm_answers_match,
    normalize_gold_answer,
)
from paths import ensure_dirs, get_paths, load_config


_OLD_KEY = "base_model.model.model.layers."
_NEW_KEY = "base_model.model.model.language_model.layers."


def ensure_renamed_adapter(adapter_dir: str) -> str:
    """Return a path to a LoRA whose keys match the multimodal model.

    If the original adapter already uses the multimodal key prefix, return it
    unchanged. Otherwise produce a sibling ``{dir}_renamed`` (idempotent) and
    return that.
    """
    from safetensors.torch import load_file, save_file

    src = Path(adapter_dir)
    safetensors = src / "adapter_model.safetensors"
    if not safetensors.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {src}")

    sd = load_file(str(safetensors))
    if not any(k.startswith(_OLD_KEY) for k in sd):
        # Already correctly named (e.g. trained against multimodal model).
        return str(src)

    dst = src.parent / f"{src.name}_renamed"
    dst.mkdir(parents=True, exist_ok=True)
    # Mirror config / tokenizer files (skip checkpoint-* subdirs).
    for f in os.listdir(src):
        if f.startswith("checkpoint-") or f == "adapter_model.safetensors":
            continue
        sp = src / f
        if sp.is_file():
            shutil.copy2(sp, dst / f)
    new_sd = {}
    n = 0
    for k, v in sd.items():
        if k.startswith(_OLD_KEY):
            new_sd[k.replace(_OLD_KEY, _NEW_KEY, 1)] = v
            n += 1
        else:
            new_sd[k] = v
    save_file(new_sd, str(dst / "adapter_model.safetensors"))
    print(f"  [rename] {n}/{len(sd)} keys → {dst}")
    return str(dst)


def load_questions(path: Path, limit: int | None):
    rows = [json.loads(l) for l in path.open()]
    if limit:
        rows = rows[:limit]
    return rows


def build_request(processor, question: str, image_path: str | list | None) -> dict:
    content = []
    img = None
    # Multi-image samples (e.g. MMMU-Pro) store image_path as a list. The whole
    # pipeline (rollout / paraphrase / SFT) is single-image, so for consistency
    # we take the first image here instead of crashing on Image.open(list).
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0] if image_path else None
    if image_path:
        try:
            img = Image.open(image_path).convert("RGB")
            content.append({"type": "image"})
        except Exception:
            img = None
    content.append({"type": "text", "text": question})
    messages = [
        {"role": "system", "content": MLLM_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    req: dict = {"prompt": prompt}
    if img is not None:
        req["multi_modal_data"] = {"image": img}
    return req


def run_condition(llm, processor, sp, questions, name, lora_request=None,
                  save_completion=True):
    requests = [build_request(processor, q["question"], q.get("image_path"))
                for q in questions]
    t0 = time.time()
    if lora_request is not None:
        outs = llm.generate(requests, sampling_params=sp,
                            lora_request=lora_request)
    else:
        outs = llm.generate(requests, sampling_params=sp)
    elapsed = time.time() - t0

    correct = 0
    rows = []
    for q, o in zip(questions, outs):
        gold = normalize_gold_answer(q["answer"])
        completion = o.outputs[0].text
        extracted = extract_mllm_answer(completion)
        is_correct = mllm_answers_match(extracted, gold, q.get("question"))
        if is_correct:
            correct += 1
        row = {
            "question_id": q["question_id"],
            "gold":        gold,
            "extracted":   extracted,
            "is_correct":  is_correct,
        }
        if save_completion:
            row["response"] = completion
        rows.append(row)

    n = len(questions)
    acc = correct / n if n else 0.0
    print(f"  [{name}] {correct}/{n} = {acc:.1%}  ({elapsed:.1f}s, "
          f"{elapsed/max(n,1):.2f}s/q)")
    return {
        "name":         name,
        "total":        n,
        "correct":      correct,
        "accuracy":     acc,
        "eval_mode":    "vllm",
        "elapsed":      round(elapsed, 1),
        "per_question": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["pgps9k_test", "mathverse", "mathvision", "hle",
                            "geoqa", "geometry3k",
                            "ai2d_test", "mmmu_pro"])
    p.add_argument("--conditions", nargs="+",
                   default=["base", "raw", "paraphrase", "vanilla"])
    p.add_argument("--grpo_adapter", type=Path, default=None,
                   help="LoRA checkpoint for --conditions grpo (GRPO RL adapter).")
    p.add_argument("--model", type=str, default=None,
                   help="Override base model path (default: config model.path).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768,
                   help="Must cover both image tokens and the prompt.")
    p.add_argument("--max_num_seqs", type=int, default=32)
    p.add_argument("--mm_max_pixels", type=int, default=1280 * 1280,
                   help="Cap each image to this many pixels via the "
                        "Qwen2VL/Qwen3VL preprocessor; protects against "
                        "extreme high-res scans pushing prompts past "
                        "max_model_len.")
    p.add_argument("--no_save_completion", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    eval_cfg = cfg["eval"]
    max_new_tokens = args.max_new_tokens or eval_cfg.get("max_new_tokens", 4096)
    limit = args.limit or eval_cfg.get("limit")
    save_completion = not args.no_save_completion
    if args.max_num_seqs == 32 and eval_cfg.get("max_num_seqs"):
        args.max_num_seqs = int(eval_cfg["max_num_seqs"])
    if args.gpu_memory_utilization == 0.85 and eval_cfg.get("gpu_memory_utilization"):
        args.gpu_memory_utilization = float(eval_cfg["gpu_memory_utilization"])

    eval_paths = cfg["paths"].get("mllm_eval", {})
    eval_root = (Path(args.output_dir) if args.output_dir
                 else paths.results_root / "eval")
    eval_root.mkdir(parents=True, exist_ok=True)
    model_path = args.model or str(paths.model_path)

    cond_keymap = {"base": "base", "raw": "raw_sft",
                   "paraphrase": "paraphrase_sft", "vanilla": "vanilla_sft",
                   "para_clear": "para_clear_sft", "grpo": "grpo_rl"}

    # Resolve LoRA paths (rename keys when needed) for non-base conditions.
    cond_adapter = {}
    for c in args.conditions:
        if c == "base":
            continue
        if c == "grpo":
            if args.grpo_adapter is None:
                print("[skip grpo] --grpo_adapter not set")
                continue
            if not args.grpo_adapter.exists():
                print(f"[skip grpo] adapter not found at {args.grpo_adapter}")
                continue
            cond_adapter[c] = ensure_renamed_adapter(str(args.grpo_adapter))
            continue
        ad = paths.checkpoint_root / c
        if not ad.exists():
            print(f"[skip {c}] adapter not found at {ad}")
            continue
        cond_adapter[c] = ensure_renamed_adapter(str(ad))

    needs_lora = bool(cond_adapter)

    # Pre-load datasets once.
    ds_questions = {}
    for ds_name in args.datasets:
        ds_path = Path(eval_paths.get(ds_name,
                                      paths.eval_cache / f"{ds_name}_eval.jsonl"))
        if not ds_path.exists():
            print(f"[skip dataset] {ds_name}: missing {ds_path}")
            continue
        ds_questions[ds_name] = load_questions(ds_path, limit)
        print(f"  {ds_name}: n={len(ds_questions[ds_name])}")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[vllm] loading {model_path}  max_seqs={args.max_num_seqs}  "
          f"gpu_mem={args.gpu_memory_utilization}  lora={needs_lora}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    t0 = time.time()
    llm_kwargs = dict(
        model=model_path, trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        max_num_seqs=args.max_num_seqs, dtype="bfloat16",
        mm_processor_kwargs={"max_pixels": args.mm_max_pixels},
    )
    if needs_lora:
        llm_kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=16)
    llm = LLM(**llm_kwargs)
    print(f"[vllm] LLM loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=int(max_new_tokens))

    # Build LoRARequest objects with stable int ids.
    lora_requests = {}
    for i, (c, path) in enumerate(cond_adapter.items(), start=1):
        lora_requests[c] = LoRARequest(
            lora_name=c, lora_int_id=i, lora_path=path,
        )

    # Outer loop: dataset; inner: condition (so each ds.json is written once).
    for ds_name, questions in ds_questions.items():
        out_file = eval_root / f"{ds_name}.json"
        existing = (json.load(out_file.open())
                    if out_file.exists() else {})
        print(f"\n[{ds_name.upper()}] n={len(questions)}")
        for c in args.conditions:
            if c != "base" and c not in lora_requests:
                continue
            lora_req = lora_requests.get(c)
            existing[cond_keymap[c]] = run_condition(
                llm, processor, sp, questions, c,
                lora_request=lora_req, save_completion=save_completion,
            )
            json.dump(existing, out_file.open("w"), indent=2,
                      ensure_ascii=False)
        print(f"  saved → {out_file}")


if __name__ == "__main__":
    main()
