"""GRPO reward plugin for the multimodal (VL) pipeline.

Mirror of ``scripts/grpo_rewards.py`` but format/judge match
``scripts/06_eval_mllm_vllm.py`` exactly:

  * ``mllm_format``   — 1.0 if the completion ends with a well-formed
                        ``Final Answer: <x>`` line, else 0.0.
  * ``mllm_accuracy`` — 1.0 if ``mllm_answers_match(extracted, gold,
                        question)`` is true (handles MCQ letter↔text mapping +
                        numeric tolerance), else 0.0.

The gold answer is carried per-example in the dataset column ``solution`` and
the rendered question in ``question_text`` (so MCQ option-letter matching can
re-parse the rendered choices). swift forwards every extra column to
``__call__`` as a same-named kwarg list.

Plug into swift via:
    --external_plugins scripts/grpo_mllm_rewards.py
    --reward_funcs mllm_format mllm_accuracy
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

# experiences/<task>/grpo/rewards.py -> lzl is parent×3, VCTS is parent×4
_LZL_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VCTS_ROOT = _LZL_ROOT.parent
_TASK_DIR = Path(__file__).resolve().parent.parent  # experiences/<task>/
for _p in (str(_TASK_DIR), str(_LZL_ROOT), str(_VCTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # noqa: E402
    extract_mllm_answer,
    mllm_answers_match,
    normalize_gold_answer,
)
from swift.rewards.orm import ORM, orms  # noqa: E402


# Strict format check: completion must contain "Final Answer: <non-empty>".
_STRICT_FORMAT_RE = re.compile(
    r"[Ff]inal\s+[Aa]nswer\s*[:：]\s*\S+"
)


class MllmFormatReward(ORM):
    """1.0 if the completion presents a well-formed 'Final Answer: <x>' line."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [1.0 if _STRICT_FORMAT_RE.search(c or "") else 0.0
                for c in completions]


class MllmAccuracyReward(ORM):
    """1.0 if extracted answer matches gold (eval-equivalent)."""

    def __call__(self, completions: List[str], solution: List[str],
                 question_text: List[str] | None = None,
                 **kwargs) -> List[float]:
        # When question_text is missing, fall back to None per row (then MCQ
        # re-parsing simply skips). Length mismatch is treated defensively.
        qts = question_text if (question_text and
                                len(question_text) == len(completions)) \
            else [None] * len(completions)
        rewards = []
        for c, sol, qt in zip(completions, solution, qts):
            gold = normalize_gold_answer(sol)
            extracted = extract_mllm_answer(c or "")
            rewards.append(
                1.0 if mllm_answers_match(extracted, gold, qt) else 0.0
            )
        return rewards


orms["mllm_format"] = MllmFormatReward
orms["mllm_accuracy"] = MllmAccuracyReward
