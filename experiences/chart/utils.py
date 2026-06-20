"""Chart/table-grounded text reasoning helpers (no images)."""

from __future__ import annotations

import re
from typing import Any

CHART_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions about charts and tables using "
    "step-by-step reasoning. Use only the provided table/context. "
    "End with a clear final answer in format 'Final Answer: <answer>'."
)

PARAPHRASE_CHART_PROMPT = """You are given a table/chart context, a question, and a step-by-step reasoning trace that arrives at the correct answer. Rewrite the reasoning trace, preserving:
1. The final answer (must be identical).
2. The same intermediate values and calculation order.
3. The final line must remain "Final Answer: <answer>" with the same answer value.

Vary sentence structure and phrasing only. Do not change numbers, logic, or the final answer.
Output only the rewritten reasoning trace, ending with the Final Answer line.

---
Context:
{question}

Original reasoning trace:
{trace}

Rewritten reasoning trace:"""


def format_chartqa(table: str, question: str) -> str:
    table = (table or "").strip()
    question = (question or "").strip()
    return f"Table:\n{table}\n\nQuestion: {question}"


def format_tabmwp(example: dict) -> str:
    title = example.get("table_title") or ""
    title_str = f' regarding "{title}"' if title else ""
    q = f"Read the following table{title_str} and answer the question:\n"
    q += f"{example['table']}\n\nQuestion: {example['question']}"
    choices = example.get("choices")
    if choices:
        q += f"\nOptions: {choices}"
    return q.strip()


def format_finqa(example: dict) -> str:
    qa = example.get("qa")
    if isinstance(qa, str):
        import ast
        try:
            qa = ast.literal_eval(qa)
        except (ValueError, SyntaxError):
            qa = {}
    elif not isinstance(qa, dict):
        qa = {}
    question = qa.get("question") or example.get("question") or ""
    parts = []
    for t in example.get("pre_text") or []:
        if str(t).strip():
            parts.append(str(t).strip())
    table = example.get("table") or example.get("table_ori") or []
    if table:
        header = " | ".join(str(c) for c in table[0])
        parts.append("Table:")
        parts.append(header)
        for row in table[1:]:
            parts.append(" | ".join(str(c) for c in row))
    for t in example.get("post_text") or []:
        if str(t).strip():
            parts.append(str(t).strip())
    parts.append(f"Question: {str(question).strip()}")
    return "\n".join(parts)


def format_finqa_hf(text: str, query: str) -> str:
    return f"{text.strip()}\n\nQuestion: {query.strip()}"


def apply_chart_template(tokenizer, question: str, enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": CHART_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def normalize_gold_answer(ans: Any) -> str:
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return str(ans).strip()


def normalize_chart_answer(ans: Any) -> str:
    if ans is None:
        return ""
    s = str(ans).strip().lower()
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"^(the answer is|answer:?)\s*", "", s)
    return s.strip()


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def chart_answers_match(pred: str | None, gold: Any) -> bool:
    """Relaxed ChartQA-style matching for text answers."""
    if pred is None:
        return False
    p = normalize_chart_answer(pred)
    g = normalize_chart_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # yes/no
    yes = {"yes", "true", "y"}
    no = {"no", "false", "n"}
    if p in yes and g in yes:
        return True
    if p in no and g in no:
        return True
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        if gf == 0:
            return abs(pf - gf) < 1e-6
        return abs(pf - gf) / max(abs(gf), 1e-9) <= 0.05
    return p.replace(" ", "") == g.replace(" ", "")


def _strip_answer_token(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\*+", "", s).strip()
    s = re.sub(r"^[\"\']|[\"\']$", "", s).strip()
    if s.endswith(".") and not re.search(r"\d\.\d$", s):
        s = s[:-1].strip()
    return s


def extract_chart_answer(completion: str) -> str | None:
    """Extract answer from the last 'Final Answer: ...' line."""
    matches = re.findall(
        r"(?:\*\*)?[Ff]inal\s+[Aa]nswer(?:\*\*)?\s*[:：]\s*(.+?)\s*(?:\n|$)",
        completion,
        re.I,
    )
    if matches:
        ans = _strip_answer_token(matches[-1])
        return ans or None

    # `sampling` package lives at the VCTS repo root (one level above lzl/).
    # Some launchers only add lzl/ to sys.path, so make sure VCTS/ is reachable
    # before doing the lazy fallback import. Falls back gracefully if `sampling`
    # really isn't importable (e.g. running outside the VCTS repo).
    import sys as _sys
    from pathlib import Path as _Path
    _vcts_root = str(_Path(__file__).resolve().parent.parent)
    if _vcts_root not in _sys.path:
        _sys.path.insert(0, _vcts_root)
    try:
        from sampling.sample_rollouts import extract_model_answer
        ans = extract_model_answer(completion)
        if ans is not None:
            return ans
    except (ModuleNotFoundError, ImportError):
        pass
    lines = [ln.strip() for ln in completion.strip().splitlines() if ln.strip()]
    if lines:
        return normalize_chart_answer(lines[-1]) or None
    return None
