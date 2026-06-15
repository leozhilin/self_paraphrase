"""GRPO reward plugin for the gsm/math pipeline.

Registered into swift's ``orms`` registry so they can be selected via
``--reward_funcs gsm_format gsm_accuracy`` together with
``--external_plugins scripts/grpo_rewards.py``.

Two rewards, mirroring exactly how ``scripts/06_eval.py`` judges correctness:

  * ``gsm_format``   — 1.0 if the completion ends with a well-formed
                       ``Final Answer: <x>`` line, else 0.0.
  * ``gsm_accuracy`` — 1.0 if the extracted answer matches the gold answer
                       (``answers_match``, with math-verify fallback), else 0.0.

The gold answer is carried per-example in the dataset column ``solution``
(swift passes every extra column to ``__call__`` as a same-named kwarg list).
``solution`` here is the raw GSM8K answer string (contains ``#### N``); we run
``extract_gsm8k_gold_answer`` on it so the comparison is identical to eval.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

# Make the lzl package importable so we reuse the SAME answer-matching logic as
# eval (sampling.sample_rollouts) and the same final-answer extractor.
_LZL_ROOT = Path(__file__).resolve().parent.parent
_VCTS_ROOT = _LZL_ROOT.parent
for _p in (str(_LZL_ROOT), str(_VCTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sampling.sample_rollouts import (  # noqa: E402
    answers_match,
    extract_gsm8k_gold_answer,
    extract_model_answer,
    _strip_thinking,
)
from swift.rewards.orm import ORM, orms  # noqa: E402


# ----- final-answer extraction (identical to scripts/06_eval.py) ------------

_FINAL_ANSWER_RE = re.compile(
    r"(?:####|\*{0,2}\s*[Ff]inal\s+[Aa]nswer\s*\*{0,2})\s*[:：]?\s*([^\n]+)"
)


def _clean_answer_span(val: str) -> str:
    val = val.strip()
    inner = list(re.finditer(
        r"(?:####|\*{0,2}\s*[Ff]inal\s+[Aa]nswer\s*\*{0,2})\s*[:：]?\s*", val))
    if inner:
        val = val[inner[-1].end():].strip()
    val = val.strip().rstrip(" .。,，;；")
    val = re.sub(r"^\*+|\*+$", "", val).strip()
    if val.startswith("$"):
        val = val[1:].strip()
    if val.endswith("$"):
        val = val[:-1].strip()
    bm = re.match(r"^\\boxed\{(.+)\}$", val)
    if bm and bm.group(1).count("{") == bm.group(1).count("}"):
        val = bm.group(1).strip()
    return val


def extract_final_answer(completion: str) -> str | None:
    """Mirror of scripts/06_eval.extract_final_answer."""
    matches = list(_FINAL_ANSWER_RE.finditer(completion))
    if matches:
        val = _clean_answer_span(matches[-1].group(1))
        if val:
            return val
    return extract_model_answer(completion)


# Strict format check: the completion must contain a final line of the exact
# shape  "Final Answer: <non-empty>"  (case-insensitive, optional bold/colon).
_STRICT_FORMAT_RE = re.compile(
    r"[Ff]inal\s+[Aa]nswer\s*[:：]\s*\S+"
)


def _gold_from_solution(sol: str) -> str:
    """Gold extraction identical to eval: pull the value after '#### '."""
    return extract_gsm8k_gold_answer(sol)


class GsmFormatReward(ORM):
    """1.0 if the completion presents a well-formed 'Final Answer: <x>' line."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for c in completions:
            text = _strip_thinking(c)
            rewards.append(1.0 if _STRICT_FORMAT_RE.search(text) else 0.0)
        return rewards


class GsmAccuracyReward(ORM):
    """1.0 if the extracted answer matches the gold answer (same as eval)."""

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        for c, sol in zip(completions, solution):
            text = _strip_thinking(c)
            gold = _gold_from_solution(sol)
            extracted = extract_final_answer(text)
            rewards.append(1.0 if answers_match(extracted, gold) else 0.0)
        return rewards


orms["gsm_format"] = GsmFormatReward
orms["gsm_accuracy"] = GsmAccuracyReward
