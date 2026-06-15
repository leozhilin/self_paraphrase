"""GSM math reasoning helpers — unified ``Final Answer: <answer>`` format.

All three pipeline stages (rollout / paraphrase / eval) share the same
system prompt and final-line convention, mirroring ``chart_utils.py``.

Why we left the older free-form style behind:
  * The original VCTS pipeline let the model freely close with ``\\boxed{}`` /
    ``#### N`` / Markdown-bold "Final Answer", relying on
    ``sample_rollouts.extract_model_answer`` 8-rule fallback to recover the
    number. That worked for accuracy, but the SFT-data style was inconsistent
    and the chart pipeline (which enforces ``Final Answer: <answer>``)
    produced visibly cleaner outputs.
  * To unify the three pipelines (gsm / chart / mllm), we now ask gsm to
    emit the same ``Final Answer: <answer>`` final line.
"""

from __future__ import annotations


GSM_SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve math problems step by step. "
    "Show all your work clearly. "
    "End with a clear final answer in format 'Final Answer: <answer>'."
)


PARAPHRASE_GSM_PROMPT = """You are given a math word problem and a step-by-step reasoning trace that arrives at the correct answer. Rewrite the reasoning trace, preserving:
1. The final answer (must be identical).
2. The same intermediate values and calculation order.
3. The final line must remain "Final Answer: <answer>" with the same answer value.

Vary sentence structure and phrasing only. Do not change numbers, logic, or the final answer.
Output only the rewritten reasoning trace, ending with the Final Answer line.

---
Problem:
{question}

Original reasoning trace:
{trace}

Rewritten reasoning trace:"""


# Hint appended to the eval user message — reinforces the same final-line
# convention we taught at rollout/SFT time, so the eval prompt matches the
# training distribution.
ANSWER_FORMAT_HINT = (
    "\n\nAfter your reasoning, conclude your response on its own final line in exactly "
    "this format:\nFinal Answer: <your final answer>\n"
    "Use the bare answer with no extra words, units, or LaTeX delimiters around it. "
    "For multiple-choice questions, the answer is a single capital letter "
    "(e.g. `Final Answer: A`)."
)


def apply_gsm_template(tokenizer, question: str, enable_thinking: bool = False) -> str:
    """Build the gsm rollout/eval prompt (system prompt + user question)."""
    messages = [
        {"role": "system", "content": GSM_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
