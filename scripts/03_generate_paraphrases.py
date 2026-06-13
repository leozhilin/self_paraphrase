#!/usr/bin/env python3
"""Step 3 (mllm/chart, vllm): vLLM-based paraphrase generator.

Drop-in replacement for ``03_generate_paraphrases.py``. Same input/output
schema (paraphrase_candidates.jsonl with paraphrase_text /
paraphrase_extracted / paraphrase_match), same prompts.

Notes:
  * GSM / chart: pure text (paraphrase rewrites trace text only).
  * MLLM: vision paraphrase — image + question + trace text, matching rollout.
  * ``enable_thinking=False`` is honored (no Qwen3 chain-of-thought leak).
  * vLLM ``SamplingParams(n=n_per)`` produces n_per paraphrases per source
    trace in one go; cross-trace batching gives continuous-batching speedup.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsm_utils import PARAPHRASE_GSM_PROMPT
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config

add_vcts_to_syspath()
from sampling.sample_rollouts import (
    answers_match as gsm_answers_match,
    extract_model_answer as gsm_extract,
)


def get_paraphrase_helpers(cfg: dict):
    task = cfg.get("task")
    if task == "chart":
        from chart_utils import (
            PARAPHRASE_CHART_PROMPT,
            chart_answers_match,
            extract_chart_answer,
        )
        return PARAPHRASE_CHART_PROMPT, extract_chart_answer, chart_answers_match
    if task == "mllm":
        from mllm_utils import (
            PARAPHRASE_MLLM_PROMPT,
            extract_mllm_answer,
            mllm_answers_match,
        )
        return PARAPHRASE_MLLM_PROMPT, extract_mllm_answer, mllm_answers_match
    return PARAPHRASE_GSM_PROMPT, gsm_extract, gsm_answers_match


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_text_paraphrase_request(tok, user_msg: str) -> dict:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return {"prompt": prompt}


def build_mllm_paraphrase_request(processor, user_msg: str, image_path: str | None) -> dict:
    """Vision paraphrase request — same image wiring as rollout / eval."""
    content = []
    img = None
    if image_path:
        try:
            img = Image.open(image_path).convert("RGB")
            content.append({"type": "image"})
        except OSError:
            img = None
    content.append({"type": "text", "text": user_msg})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    req: dict = {"prompt": prompt}
    if img is not None:
        req["multi_modal_data"] = {"image": img}
    return req


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n_per", type=int, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192,
                   help="Some raw traces are >4k tokens, leave headroom.")
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--mm_max_pixels", type=int, default=1280 * 1280)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    task = cfg.get("task")
    para_cfg = cfg["paraphrase"]
    paraphrase_prompt, extract_answer, answers_match = get_paraphrase_helpers(cfg)

    raw_path = Path(args.raw) if args.raw else paths.raw_jsonl
    out_path = Path(args.out) if args.out else paths.paraphrase_candidates
    n_per = args.n_per or para_cfg["n_per"]

    if not raw_path.exists():
        sys.exit(f"raw.jsonl not found: {raw_path}\nRun 02_build_raw_manifest.py first.")

    raw = []
    with open(raw_path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r["src_idx"] = i
            raw.append(r)
    print(f"Loaded {len(raw)} traces → {len(raw) * n_per} paraphrase outputs "
          f"(n_per={n_per})")

    from vllm import LLM, SamplingParams

    use_vision = task == "mllm"
    if use_vision:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            str(paths.model_path), trust_remote_code=True,
        )
        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        llm_kwargs = {
            "limit_mm_per_prompt": {"image": 1},
            "mm_processor_kwargs": {"max_pixels": args.mm_max_pixels},
        }
        print(f"[mllm] vision paraphrase (image + text), max_pixels={args.mm_max_pixels}")
    else:
        from transformers import AutoTokenizer
        processor = None
        tok = AutoTokenizer.from_pretrained(
            str(paths.model_path), trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        llm_kwargs = {"limit_mm_per_prompt": {"image": 0}}
        print(f"[{task}] text-only paraphrase")

    t_load = time.time()
    llm = LLM(
        model=str(paths.model_path), trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        **llm_kwargs,
    )
    print(f"[vllm] LLM loaded in {time.time()-t_load:.1f}s")

    sp = SamplingParams(
        n=n_per,
        temperature=float(para_cfg["temperature"]),
        top_p=float(para_cfg["top_p"]),
        max_tokens=int(para_cfg["max_new_tokens"]),
    )

    requests = []
    for r in raw:
        user_msg = paraphrase_prompt.format(
            question=r["question"], trace=r["trace_text"],
        )
        if use_vision:
            requests.append(
                build_mllm_paraphrase_request(processor, user_msg, r.get("image_path")),
            )
        else:
            requests.append(build_text_paraphrase_request(tok, user_msg))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    outs = llm.generate(requests, sampling_params=sp)
    elapsed = time.time() - t0

    written = 0
    with open(out_path, "w") as f_out:
        for src, o in zip(raw, outs):
            for k, sample in enumerate(o.outputs):
                gen = _strip_thinking(sample.text.strip())
                extracted = extract_answer(gen)
                if task == "mllm":
                    matched = answers_match(
                        extracted, str(src["answer"]), src.get("question"),
                    )
                else:
                    matched = answers_match(extracted, str(src["answer"]))
                rec = {
                    "question_id":          src["question_id"],
                    "question":             src["question"],
                    "answer":               src["answer"],
                    "image_path":           src.get("image_path"),
                    "src_rollout_id":       src.get("rollout_id"),
                    "src_tokens":           src.get("tokens"),
                    "src_bin":              src.get("bin"),
                    "src_trace_text":       src["trace_text"],
                    "para_idx":             k,
                    "paraphrase_text":      gen,
                    "paraphrase_extracted": extracted,
                    "paraphrase_match":     bool(matched),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} candidates → {out_path}")
    print(f"[stats] {len(raw)} traces × {n_per} = {written}, "
          f"{elapsed:.1f}s, {elapsed/max(written,1):.3f}s/sample")


if __name__ == "__main__":
    main()
