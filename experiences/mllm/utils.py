"""Multimodal-task helpers (visual reasoning across AI2D / MMMU / DocVQA / ...).

Mirror of chart_utils.py but for the broader MLLM benchmark family. We expose
both a text-only path (for sanity-checking the pipeline with a text LLM that
ignores images) and a vision path (used at eval time with Qwen3-VL or similar).

Output format convention (all conditions / models):
    ``Final Answer: <answer>``
matching the chart pipeline so downstream extraction is unified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MLLM_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question carefully using "
    "step-by-step reasoning. Use only the provided image (if any) and text "
    "context — do not rely on outside knowledge that is not grounded in them. "
    "End your response with a single line in this exact format:\n"
    "Final Answer: <your final answer>\n"
    "For multiple-choice questions output only the option letter (e.g. "
    "'Final Answer: A'); for free-form questions output the bare answer with "
    "no extra words, units or LaTeX delimiters."
)


PARAPHRASE_MLLM_PROMPT = """You are given a question (which may reference an image) and a step-by-step reasoning trace that arrives at the correct answer. Rewrite the reasoning trace, preserving:
1. The final answer (must be identical).
2. The same intermediate observations and conclusions.
3. The final line must remain "Final Answer: <answer>" with the same answer value.

Vary sentence structure and phrasing only. Do not change observations about the image, numerical values, logic, or the final answer.
Output only the rewritten reasoning trace, ending with the Final Answer line.

---
Question:
{question}

Original reasoning trace:
{trace}

Rewritten reasoning trace:"""


# ---------------------------------------------------------------------------
# Question rendering — turns a (question + options) row into the user message.
# ---------------------------------------------------------------------------


def render_mc_question(question: str, options: list[str] | dict[str, str] | None) -> str:
    """Render a multi-choice question: question text + options block."""
    q = (question or "").strip()
    if not options:
        return q
    if isinstance(options, dict):
        # {"A": "...", "B": "..."} → ordered list
        keys = sorted(options.keys())
        opts = [f"{k}) {options[k]}" for k in keys]
    else:
        # list[str]; if items already start with "A)" keep, else add letters.
        opts = []
        for i, opt in enumerate(options):
            opt_s = str(opt).strip()
            if re.match(r"^[A-Z][\)\.]\s*", opt_s):
                opts.append(opt_s)
            else:
                opts.append(f"{chr(ord('A') + i)}) {opt_s}")
    return f"{q}\n\nOptions:\n" + "\n".join(opts)


def render_image_marker(has_image: bool) -> str:
    """Marker the text-only model sees when an image would otherwise be shown.

    For sanity-check runs with a pure-text LLM the actual pixels can't be
    consumed, but we still tell the model that an image exists so its prompt
    distribution stays close to what the vision model gets. This keeps the
    text-only smoke-test honest about *what context the model has access to*.
    """
    return "[Image: <image content omitted in text-only mode>]\n\n" if has_image else ""


# ---------------------------------------------------------------------------
# Chat-template application (text-only path).
# ---------------------------------------------------------------------------


def apply_mllm_template_text(tokenizer, question: str,
                             enable_thinking: bool = False) -> str:
    """Apply the tokenizer's chat template for a text-only model on an MLLM
    question. The image is *not* included; downstream callers should prepend
    ``render_image_marker(True)`` to the question if they want the model to know
    an image was originally part of the input.
    """
    messages = [
        {"role": "system", "content": MLLM_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    # Manual ChatML fallback
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction.
# ---------------------------------------------------------------------------

_FINAL_ANSWER_RES = [
    re.compile(r"Final\s*Answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<answer>\s*(.+?)\s*(?:</answer>)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\\boxed\{([^}]+)\}"),
    re.compile(r"^####\s*(.+?)\s*$", re.MULTILINE),
]


def _strip_thinking(text: str) -> str:
    """Drop any <think>...</think> blocks so they don't confuse extraction."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Some Qwen/VL completions omit the opening tag but still end reasoning with
    # </think>. In that case the answer is usually the short text after it.
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        text = parts[-1]
    return text.strip()


def _clean_mllm_answer(ans: str) -> str:
    ans = ans.strip().strip("`").strip()
    m = re.fullmatch(r"\*+\s*(.*?)\s*\*+", ans)
    if m:
        ans = m.group(1).strip()
    ans = re.sub(r"^[-•\s]+", "", ans).strip()
    m = re.search(
        r"(?:the\s+)?(?:correct\s+)?(?:answer|option)\s+(?:is|:)?\s*\(?([A-Z])\)?\s*\.?$",
        ans,
        re.IGNORECASE,
    )
    if m:
        ans = m.group(1).upper()
    return ans.rstrip(".").strip()


def extract_mllm_answer(text: str) -> str | None:
    """Pull the model's final answer out of the completion."""
    if text is None:
        return None
    text = _strip_thinking(text)
    for pat in _FINAL_ANSWER_RES:
        matches = pat.findall(text)
        if matches:
            return _clean_mllm_answer(matches[-1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        # Fallback for completions that obey the intent but not the literal
        # marker, e.g. "</think>\n\nC" or "</think>\n\nYes".
        cand = _clean_mllm_answer(lines[-1])
        if cand and len(cand) <= 80 and not cand.endswith(":"):
            return cand
    return None


def normalize_gold_answer(ans: Any) -> str:
    """Canonicalise a gold answer for matching."""
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    s = str(ans).strip()
    # strip surrounding $...$ / \boxed{}
    s = s.strip("$")
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        s = m.group(1)
    return s.strip().rstrip(".").strip()


def _normalize_for_match(s: str) -> str:
    """Lowercase + strip whitespace + drop trailing punctuation for compare."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:!?")
    return s


def parse_mc_options(question: str) -> dict[str, str]:
    """Parse a rendered MCQ's 'A) text' option lines into {letter: text}.

    Works on the 'Options:\n A) ... \n B) ...' block produced by
    render_mc_question. Returns {} when no option lines are found.
    """
    opts: dict[str, str] = {}
    for m in re.finditer(r"^\s*([A-Z])[\)\.]\s*(.+?)\s*$", question or "",
                         re.MULTILINE):
        opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def _to_choice_letter(value: str, options: dict[str, str]) -> str | None:
    """Map an answer (a letter OR an option's full text) to its letter."""
    v = _normalize_for_match(value)
    if not v:
        return None
    # bare letter like "a" / "a)" / "(a)"
    m = re.fullmatch(r"\(?([a-z])\)?", v)
    if m:
        return m.group(1).upper()
    # full option text exactly equals one of the choices
    for letter, text in options.items():
        if _normalize_for_match(text) == v:
            return letter.upper()
    # leading "a) ..." / "a. ..." / "(a) ..." prefix; verify the rest matches
    # the option text when we have the options map (avoids false positives).
    m = re.match(r"\(?([a-z])[\)\.]\s*(.*)$", v)
    if m:
        letter, rest = m.group(1).upper(), m.group(2).strip()
        if not options:
            return letter
        opt = options.get(letter)
        if opt is None or _normalize_for_match(opt) == rest or rest == "":
            return letter
    return None


def mllm_answers_match(extracted: str | None, gold: str,
                       question: str | None = None) -> bool:
    if extracted is None:
        return False
    e = _normalize_for_match(extracted)
    g = _normalize_for_match(gold)
    if e == g:
        return True
    # numeric tolerance
    try:
        return abs(float(e) - float(g)) < 1e-4
    except (ValueError, TypeError):
        pass
    # MCQ: map both sides to a canonical option letter and compare. This handles
    # letter-vs-letter, letter-vs-fulltext, and fulltext-vs-letter cases.
    options = parse_mc_options(question) if question else {}
    e_letter = _to_choice_letter(extracted, options)
    g_letter = _to_choice_letter(gold, options)
    if e_letter is not None and g_letter is not None:
        return e_letter == g_letter
    # last resort: leading-letter compare for short letter-only answers
    if re.fullmatch(r"[a-z]\)?", e):
        return e[0] == g[0] if g else False
    return False


# ---------------------------------------------------------------------------
# Image helpers (used by vision-mode evaluator only; text-only path skips this).
# ---------------------------------------------------------------------------


def resolve_image_path(image_path: str | Path | None) -> Path | None:
    if not image_path:
        return None
    p = Path(image_path)
    return p if p.exists() else None


def load_pil_image(image_path: str | Path | None):
    """Return a PIL RGB image, or None if path missing."""
    p = resolve_image_path(image_path)
    if not p:
        return None
    from PIL import Image
    return Image.open(p).convert("RGB")


def build_vision_user_content(question: str, image_path: str | Path | None):
    """Build Qwen3-VL user message content (image + text or text-only)."""
    image = load_pil_image(image_path)
    if image is None:
        return question
    return [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]


def build_vision_messages(question: str, image_path: str | Path | None,
                          system_prompt: str = MLLM_SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_vision_user_content(question, image_path)},
    ]


def tokenize_vision_sft_example(processor, question: str, trace_text: str,
                                image_path: str | Path | None, system_prompt: str,
                                max_length: int) -> dict:
    """Tokenize one vision SFT example for Qwen3-VL Trainer.

    Returns a per-example dict with:
      input_ids / attention_mask / labels  → 1-D [seq]   (collate stacks → [B, seq])
      pixel_values                          → 2-D [n_patch, dim]  (kept as-is)
      image_grid_thw                        → 2-D [n_img, 3]      (kept as-is)

    Only the *text* tensors (input_ids/attention_mask) are length-capped; the
    image placeholder tokens live at the start of the sequence so truncating
    the tail (the trace) never desyncs pixel_values ↔ image tokens. We cap
    conservatively to keep the final ``Final Answer:`` line when possible by
    truncating from the front of the trace, not the image block.
    """
    messages = build_vision_messages(question, image_path, system_prompt)
    prompt_inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])

    messages.append({"role": "assistant", "content": trace_text})
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
    # IMPORTANT: do NOT truncate input_ids for vision examples — the image
    # placeholder tokens must stay aligned with pixel_values / image_grid_thw.
    # Truncation desyncs them and crashes get_rope_index. ``max_length`` is used
    # only as a *skip* threshold by the caller (see ``vision_example_len``).

    out = {}
    for key, val in inputs.items():
        if key in ("input_ids", "attention_mask"):
            out[key] = val.squeeze(0)              # → 1-D [seq], full length
        else:
            # pixel_values [n_patch, dim] and image_grid_thw [n_img, 3]:
            # keep native shape (squeezing [1,3]→[3] breaks Qwen3-VL
            # fast_pos_embed_interpolate which indexes row[0]).
            out[key] = val
    labels = out["input_ids"].clone()
    labels[:prompt_len] = -100
    out["labels"] = labels
    return out


def use_vision_mode(cfg: dict) -> bool:
    return cfg.get("eval", {}).get("mode") == "vision"
