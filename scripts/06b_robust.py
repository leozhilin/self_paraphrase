#!/usr/bin/env python3
"""Unified robustness benchmark manager — build + evaluate, all in one place.

This script is the single entry point for building eval JSONLs and running
adapter evaluations on every robustness benchmark in the project. Adding a new
benchmark = registering one entry in ``BENCHMARKS`` (one source of truth, no
parallel lists in build vs. eval scripts).

Subcommands
-----------
  build  — Download / format every (or selected) benchmark into JSONL under
           ``data/cache/eval/``. Idempotent; existing JSONL files are skipped.
  eval   — For each (or selected) benchmark, load JSONL and run paired
           greedy-decoding eval for the requested ``--conditions``. Auto-builds
           any missing JSONL on first call so a clean clone "just works".
  all    — Equivalent to ``build`` then ``eval`` (with the same dataset filter).

Adapter / condition mapping (math task):
  base        = no LoRA
  paraphrase  = paths.checkpoint_root / "paraphrase"
  raw         = paths.checkpoint_root / "raw"
  vanilla     = paths.checkpoint_root / "vanilla"

Output JSONL schema (per line, common across all benchmarks):
  {question_id, question, answer, source, ...}    # answer ends with `#### gold`

Output evaluation JSON: ``results/eval/<benchmark>.json`` keyed by condition,
each value = {accuracy, correct, total, per_question[{id, gold, extracted,
is_correct, response (optional)}]}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Use the same HF cache as the rest of the project
os.environ.setdefault("HF_HOME", "/data5/lzl/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/data5/lzl/hf_cache")

from datasets import load_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_vcts_to_syspath, ensure_dirs, load_config  # noqa: E402

add_vcts_to_syspath()
from scripts.eval_gsm_symbolic import evaluate_model  # noqa: E402

HF_CACHE = "/data5/lzl/hf_cache"

# ---------------------------------------------------------------------------
# Helpers shared by builders
# ---------------------------------------------------------------------------


def _ensure_gsm_answer_format(answer: str, numeric: str | None = None) -> str:
    """Ensure the answer string ends with `#### NUMBER` (GSM8K convention)."""
    a = str(answer).strip()
    if "####" in a:
        return a
    if numeric is None:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", a)
        numeric = m.group(1) if m else ""
    return f"{a}\n#### {numeric}".strip()


def _fmt_number(x):
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else repr(x)
    return str(x).strip()


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize_math_answer(ans: str) -> str:
    s = str(ans).strip().strip("$")
    m = _BOXED_RE.search(s)
    return (m.group(1) if m else s).strip()


def _write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [done] {path}  ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Per-benchmark builders. Each takes (out_path: Path) and writes JSONL.
# ---------------------------------------------------------------------------


def build_gsm8k_test(out_path: Path):
    """Official GSM8K test split (gsm8k/main, n=1319). Used as the standard
    in-distribution sanity-check benchmark."""
    ds = load_dataset("gsm8k", "main", split="test", cache_dir=HF_CACHE)
    rows = [{
        "question_id": i,
        "question": str(ex["question"]).strip(),
        "answer": str(ex["answer"]).strip(),
        "source": "GSM8K-test",
    } for i, ex in enumerate(ds)]
    _write_jsonl(rows, out_path)


def _build_gsm_symbolic(config: str, source_name: str):
    def _do(out_path: Path):
        ds = load_dataset("apple/GSM-Symbolic", config, split="test", cache_dir=HF_CACHE)
        rows = [{
            "question_id": int(ex.get("id", i)) * 1000 + int(ex.get("instance", 0)),
            "question": str(ex["question"]).strip(),
            "answer": str(ex["answer"]).strip(),
            "original_id": int(ex.get("original_id", -1)),
            "instance": int(ex.get("instance", 0)),
            "source": source_name,
        } for i, ex in enumerate(ds)]
        _write_jsonl(rows, out_path)
    return _do


def _build_gsm_plus(split: str):
    def _do(out_path: Path):
        ds = load_dataset("qintongli/GSM-Plus", split=split, cache_dir=HF_CACHE)
        rows = []
        for i, ex in enumerate(ds):
            sol = str(ex.get("solution", "")).strip()
            ans = str(ex.get("answer", "")).strip()
            answer_field = _ensure_gsm_answer_format(sol, ans) if sol else f"#### {ans}"
            rows.append({
                "question_id": i,
                "question": str(ex["question"]).strip(),
                "answer": answer_field,
                "perturbation_type": ex.get("perturbation_type", ""),
                "source": f"GSM-Plus[{split}]",
            })
        _write_jsonl(rows, out_path)
    return _do


def build_gsm_ic(out_path: Path):
    ds = load_dataset("voidful/gsm-ic", split="validation", cache_dir=HF_CACHE)
    rows = []
    for i, ex in enumerate(ds):
        ans = str(ex["answer"]).strip()
        rows.append({
            "question_id": i,
            "question": str(ex["question"]).strip(),
            "answer": _ensure_gsm_answer_format("", ans),
            "n_steps": int(ex.get("n_steps", 0)) if ex.get("n_steps") is not None else 0,
            "source": "GSM-IC",
        })
    _write_jsonl(rows, out_path)


def build_gsm_hard(out_path: Path):
    ds = load_dataset("reasoning-machines/gsm-hard", split="train", cache_dir=HF_CACHE)
    rows = [{
        "question_id": f"gsmhard_{i:05d}",
        "original_id": f"gsmhard_{i:05d}",
        "question": ex["input"].strip(),
        "answer": f"#### {_fmt_number(ex['target'])}",
        "source": "gsm_hard",
    } for i, ex in enumerate(ds)]
    _write_jsonl(rows, out_path)


def build_aqua_rat(out_path: Path):
    ds = load_dataset("deepmind/aqua_rat", split="test", cache_dir=HF_CACHE)
    rows = []
    for i, ex in enumerate(ds):
        opts = ex["options"]
        q = ex["question"].strip() + "\n\nOptions:\n" + "\n".join(opts)
        gold = ex["correct"].strip().upper()
        rows.append({
            "question_id": f"aqua_{i:04d}",
            "original_id": f"aqua_{i:04d}",
            "question": q,
            "answer": f"#### {gold}",
            "source": "aqua_rat",
            "meta": {"options": opts, "rationale": ex.get("rationale", "")},
        })
    _write_jsonl(rows, out_path)


def build_mawps(out_path: Path):
    ds = load_dataset("MU-NLPC/Calc-mawps", split="test", cache_dir=HF_CACHE)
    rows = [{
        "question_id": ex.get("id") or f"mawps_{i:04d}",
        "original_id": ex.get("id") or f"mawps_{i:04d}",
        "question": ex["question"].strip(),
        "answer": f"#### {_fmt_number(ex['result_float'])}",
        "source": "mawps",
    } for i, ex in enumerate(ds)]
    _write_jsonl(rows, out_path)


def build_math500(out_path: Path):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=HF_CACHE)
    rows = []
    for i, ex in enumerate(ds):
        gold = _normalize_math_answer(ex["answer"])
        rows.append({
            "question_id": ex.get("unique_id") or f"math500_{i:04d}",
            "original_id": ex.get("unique_id") or f"math500_{i:04d}",
            "question": ex["problem"].strip(),
            "answer": f"#### {gold}",
            "source": "math500",
            "meta": {"subject": ex.get("subject"), "level": ex.get("level")},
        })
    _write_jsonl(rows, out_path)


# ---------------------------------------------------------------------------
# Single source of truth: benchmark registry.
# ---------------------------------------------------------------------------


@dataclass
class Benchmark:
    name: str                                   # cli flag
    jsonl_filename: str                         # under data/cache/eval/
    result_filename: str                        # under results/eval/
    builder: Callable[[Path], None]
    n: int                                      # rows (for display)
    note: str = ""                              # short description


BENCHMARKS: list[Benchmark] = [
    # === in-distribution sanity baseline ===
    Benchmark("gsm8k_test", "gsm8k_test.jsonl", "gsm8k_test.json",
              build_gsm8k_test, 1319,
              "GSM8K official test split (in-distribution sanity)"),
    # === statement-level robustness (non-numeric answers included) ===
    Benchmark("aqua_rat", "aqua_rat.jsonl", "aqua_rat.json",
              build_aqua_rat, 254, "AQuA-RAT MCQ (letters)"),
    Benchmark("math500",  "math500.jsonl",  "math500.json",
              build_math500, 500, "MATH-500 (LaTeX answers)"),
    Benchmark("mawps",    "mawps.jsonl",    "mawps.json",
              build_mawps, 520, "MAWPS test (numeric)"),
    Benchmark("gsm_hard", "gsm_hard.jsonl", "gsm_hard.json",
              build_gsm_hard, 1319, "GSM-Hard"),
    Benchmark("gsm_ic",   "gsm_ic.jsonl",   "gsm_ic.json",
              build_gsm_ic, 1000, "GSM-IC (irrelevant context)"),
    # === GSM-Plus + GSM-Symbolic (paraphrase / robustness family) ===
    Benchmark("gsm_plus", "gsm_plus_testmini.jsonl", "gsm_plus_testmini.json",
              _build_gsm_plus("testmini"), 2400,
              "GSM-Plus testmini (8 perturbation types)"),
    Benchmark("gsm_plus_full", "gsm_plus_full.jsonl", "gsm_plus_full.json",
              _build_gsm_plus("test"), 10552, "GSM-Plus full test (slow)"),
    Benchmark("main", "gsm_symbolic_main.jsonl", "gsm_symbolic_main.json",
              _build_gsm_symbolic("main", "GSM-Symbolic-main"), 5000,
              "GSM-Symbolic main (0 added clauses)"),
    Benchmark("p1", "gsm_symbolic_p1.jsonl", "gsm_symbolic_p1.json",
              _build_gsm_symbolic("p1", "GSM-Symbolic-P1"), 5000,
              "GSM-Symbolic +1 clause"),
    Benchmark("p2", "gsm_symbolic_p2.jsonl", "gsm_symbolic_p2.json",
              _build_gsm_symbolic("p2", "GSM-Symbolic-P2"), 2500,
              "GSM-Symbolic +2 clauses"),
]

DEFAULT_BENCHMARKS = [b.name for b in BENCHMARKS if b.name != "gsm_plus_full"]
ALL_NAMES = [b.name for b in BENCHMARKS]


def select(names: list[str]) -> list[Benchmark]:
    if "all" in names:
        chosen = DEFAULT_BENCHMARKS
    else:
        chosen = names
    by_name = {b.name: b for b in BENCHMARKS}
    return [by_name[n] for n in chosen if n in by_name]


# ---------------------------------------------------------------------------
# Build / eval drivers
# ---------------------------------------------------------------------------


def cmd_build(args, paths):
    cache = paths.eval_cache
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[output] {cache}\n")
    for b in select(args.datasets):
        out = cache / b.jsonl_filename
        if out.exists() and not args.force:
            print(f"[skip] {b.name}: {out} exists")
            continue
        print(f"=== {b.name} — {b.note} (n={b.n}) ===")
        b.builder(out)
    print("\n[DONE] build")


def cmd_eval(args, paths):
    cfg = load_config()
    eval_cfg = cfg["eval"]
    batch_size = args.batch_size or eval_cfg["batch_size"]
    if args.unlimited:
        max_new_tokens = 0
    else:
        max_new_tokens = (args.max_new_tokens
                          if args.max_new_tokens is not None
                          else eval_cfg["max_new_tokens"])
    save_completion = not args.no_save_completion
    model_path = str(paths.model_path)
    eval_root = paths.results_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    cache = paths.eval_cache

    adapter_map = {
        "paraphrase": str(paths.checkpoint_root / "paraphrase"),
        "raw":        str(paths.checkpoint_root / "raw"),
        "vanilla":    str(paths.checkpoint_root / "vanilla"),
    }

    chosen = select(args.datasets)
    for b in chosen:
        eval_set = cache / b.jsonl_filename
        if not eval_set.exists():
            print(f"[auto-build] {b.name}: missing {eval_set}, building...")
            b.builder(eval_set)
        questions = [json.loads(l) for l in open(eval_set)]
        if args.limit:
            questions = questions[:args.limit]
        out_file = eval_root / b.result_filename
        print(f"\n========== [{b.name}] n={len(questions)}  bs={batch_size} ==========",
              flush=True)

        def _run(adapter_path, key):
            r = evaluate_model(model_path, adapter_path, questions,
                               max_new_tokens=max_new_tokens,
                               batch_size=batch_size,
                               save_completion=save_completion)
            results = {}
            if out_file.exists():
                try:
                    results = json.load(out_file.open())
                except Exception:
                    results = {}
            results[key] = r
            json.dump(results, out_file.open("w"), indent=2, ensure_ascii=False)
            print(f"  {key}: {r['accuracy']:.1%}  → {out_file}", flush=True)

        if "base" in args.conditions:
            _run(None, "base")
        for cond in args.conditions:
            if cond == "base":
                continue
            ad = adapter_map.get(cond)
            if ad and Path(ad).exists():
                _run(ad, f"{cond}_sft")
            else:
                print(f"  skip {cond}: adapter not found at {ad}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_dataset_arg(p):
    p.add_argument("--datasets", nargs="+",
                   choices=ALL_NAMES + ["all"], default=["all"],
                   help=f"Subset of benchmarks. 'all' = {DEFAULT_BENCHMARKS} "
                        f"(excludes gsm_plus_full; pass it explicitly to include).")


def main():
    p = argparse.ArgumentParser(
        description="Unified robustness benchmark manager (build + eval).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build / download benchmark JSONLs.")
    _add_dataset_arg(pb)
    pb.add_argument("--force", action="store_true",
                    help="Rebuild even if JSONL already exists.")

    pe = sub.add_parser("eval",  help="Evaluate adapters on benchmarks.")
    _add_dataset_arg(pe)
    pe.add_argument("--conditions", nargs="+", default=["base", "paraphrase"],
                    help="Subset of: base, raw, paraphrase, vanilla.")
    pe.add_argument("--batch_size", type=int, default=None)
    pe.add_argument("--max_new_tokens", type=int, default=None,
                    help="Per-question generation cap. 0/--unlimited => 16384.")
    pe.add_argument("--unlimited", action="store_true")
    pe.add_argument("--no_save_completion", action="store_true",
                    help="Don't store raw completions (saves disk).")
    pe.add_argument("--limit", type=int, default=None,
                    help="Only evaluate first N questions per benchmark (debug).")

    pa = sub.add_parser("all", help="Build + Eval (same dataset filter).")
    _add_dataset_arg(pa)
    pa.add_argument("--conditions", nargs="+", default=["base", "paraphrase"])
    pa.add_argument("--batch_size", type=int, default=None)
    pa.add_argument("--max_new_tokens", type=int, default=None)
    pa.add_argument("--unlimited", action="store_true")
    pa.add_argument("--no_save_completion", action="store_true")
    pa.add_argument("--limit", type=int, default=None)
    pa.add_argument("--force", action="store_true",
                    help="Rebuild even if JSONL already exists.")

    args = p.parse_args()
    paths = ensure_dirs()

    if args.cmd == "build":
        cmd_build(args, paths)
    elif args.cmd == "eval":
        cmd_eval(args, paths)
    elif args.cmd == "all":
        cmd_build(args, paths)
        cmd_eval(args, paths)


if __name__ == "__main__":
    main()
