"""GRPO reward plugin for the chart/table-QA pipeline.

Mirror of ``scripts/grpo_rewards.py`` (gsm/math), but the format/judge logic
matches ``scripts/06_eval_chart{,_vllm}.py`` exactly:

  * ``chart_format``   — 1.0 if the completion ends with a well-formed
                         ``Final Answer: <x>`` line, else 0.0.
  * ``chart_accuracy`` — 1.0 if ``chart_answers_match(extracted, gold)`` is
                         true (relaxed: numeric 5% tolerance + yes/no
                         normalisation), else 0.0.

The gold answer is carried per-example in the dataset column ``solution``
(swift forwards every extra column to ``__call__`` as a same-named kwarg list).
For chart, ``solution`` is the raw answer string from
``data/chart/chartqa_train.jsonl`` (e.g. ``"Yes"``, ``"47806"``, ``"China"``).

Plug into swift via:
    --external_plugins scripts/grpo_chart_rewards.py
    --reward_funcs chart_format chart_accuracy
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

# Make the lzl package importable so we reuse the SAME extractor/judge as eval.
# experiences/<task>/grpo/rewards.py -> lzl is parent×3, VCTS is parent×4
_LZL_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VCTS_ROOT = _LZL_ROOT.parent
_TASK_DIR = Path(__file__).resolve().parent.parent  # experiences/<task>/
for _p in (str(_TASK_DIR), str(_LZL_ROOT), str(_VCTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # noqa: E402
    chart_answers_match,
    extract_chart_answer,
    normalize_gold_answer,
)
from swift.rewards.orm import ORM, orms  # noqa: E402


# Strict format check: completion must contain a "Final Answer: <non-empty>" line.
_STRICT_FORMAT_RE = re.compile(
    r"(?:\*\*)?[Ff]inal\s+[Aa]nswer(?:\*\*)?\s*[:：]\s*\S+"
)


class ChartFormatReward(ORM):
    """1.0 if the completion presents a well-formed 'Final Answer: <x>' line."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [1.0 if _STRICT_FORMAT_RE.search(c or "") else 0.0
                for c in completions]


class ChartAccuracyReward(ORM):
    """1.0 if the extracted answer matches the gold answer (eval-equivalent)."""

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        for c, sol in zip(completions, solution):
            gold = normalize_gold_answer(sol)
            extracted = extract_chart_answer(c or "")
            rewards.append(1.0 if chart_answers_match(extracted, gold) else 0.0)
        return rewards


orms["chart_format"] = ChartFormatReward
orms["chart_accuracy"] = ChartAccuracyReward
