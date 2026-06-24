#!/usr/bin/env python3
"""Step 0 (mllm): Download + format every visual-reasoning eval dataset into a
unified JSONL schema, plus the PGPS9K (or other) training split that drives SFT.

Output JSONL schema (per line):
  {
    "question_id": str,                       # globally unique
    "image_path":  str | list[str] | null,    # absolute path(s); null if pure text
    "question":    str,                       # already rendered with options
    "answer":      str,                       # canonicalised gold (letter or text)
    "source":      str,                       # ai2d_train / ai2d_test / mmmu_pro / ...
    "meta":        dict (optional)
  }

Images are materialised under ``<image_root>/<dataset>/<question_id>.png`` so
that downstream pipeline (rollout / paraphrase / eval) can load them by path
without re-decoding from HF cache.

Raw HF caches and materialised images come from config (``datasets.hf_cache``,
``datasets.image_root``); eval JSONL under ``paths.eval_cache``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from paths import apply_hf_env, ensure_dirs, get_paths, load_config      # noqa: E402

from datasets import load_dataset                                     # noqa: E402
from PIL import Image                                                  # noqa: E402

from utils import render_mc_question                              # noqa: E402

LETTERS = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------
# Image saving helpers
# ---------------------------------------------------------------------------

def _save_image(img: Any, dest: Path) -> Path | None:
    """Save a PIL.Image / base64-string / path-like to ``dest`` as PNG.
    Returns the dest path on success, None otherwise."""
    if img is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    try:
        if isinstance(img, Image.Image):
            img.convert("RGB").save(dest, format="PNG")
        elif isinstance(img, str):
            s = img.strip()
            if not s:
                return None
            # try base64
            try:
                if s.startswith("data:image/"):
                    s = s.split(",", 1)[1]
                raw = base64.b64decode(s, validate=False)
                Image.open(io.BytesIO(raw)).convert("RGB").save(dest, format="PNG")
            except Exception:
                # maybe it's a filesystem path
                p = Path(s)
                if p.exists():
                    Image.open(p).convert("RGB").save(dest, format="PNG")
                else:
                    return None
        elif isinstance(img, dict) and img.get("bytes"):
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(dest, format="PNG")
        else:
            return None
        return dest
    except Exception as e:
        print(f"  [warn] failed to save image {dest}: {type(e).__name__}: {e}", flush=True)
        if dest.exists():
            try: dest.unlink()
            except Exception: pass
        return None


def _write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# AI2D — training + test split.  Schema: question / options(list[str]) / answer(int) / image
#       Single image per question.  Multi-choice with 4 options.
# ---------------------------------------------------------------------------

def _ai2d_row(ex, idx, split, img_dir) -> dict | None:
    options = ex["options"] or []
    ans_raw = ex["answer"]
    # AI2D stores answer as a string-formatted int ("0".."3"); convert.
    try:
        ans_idx = int(ans_raw)
    except (TypeError, ValueError):
        return None
    if not options or ans_idx < 0 or ans_idx >= len(options):
        return None
    qid = f"ai2d_{split}_{idx:05d}"
    img_p = _save_image(ex.get("image"), img_dir / f"{qid}.png")
    return {
        "question_id": qid,
        "image_path":  str(img_p) if img_p else None,
        "question":    render_mc_question(ex["question"], options),
        "answer":      LETTERS[ans_idx],
        "source":      f"ai2d_{split}",
        "meta":        {"options_text": options, "answer_text": options[ans_idx]},
    }


def export_ai2d(image_root: Path, eval_dir: Path, limit: int | None = None):
    """AI2D eval-only: full HF ``test`` split → ``ai2d_test_eval.jsonl``."""
    img_dir = image_root / "ai2d"
    print(f"\n[AI2D] downloading (eval split only)...", flush=True)
    ds = load_dataset("lmms-lab/ai2d", split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    test_rows = []
    for i in range(n):
        row = _ai2d_row(ds[i], i, "test", img_dir)
        if row is None:
            continue
        test_rows.append(row)
    _write_jsonl(test_rows, eval_dir / "ai2d_test_eval.jsonl")


# ---------------------------------------------------------------------------
# PGPS9K — training split (local CASIA/HF export).  MCQ with numeric options.
# ---------------------------------------------------------------------------

def _pgps9k_choice_index(answer: str, choices: list) -> int | None:
    """Map a numeric gold answer to the closest MCQ option index."""
    try:
        target = float(str(answer).strip())
    except (TypeError, ValueError):
        return None
    best_i, best_d = None, float("inf")
    for i, c in enumerate(choices):
        try:
            d = abs(float(c) - target)
        except (TypeError, ValueError):
            continue
        if d < best_d:
            best_d, best_i = d, i
    if best_i is None or best_d > 0.15:
        return None
    return best_i


def export_pgps9k(image_root: Path, train_jsonl: Path,
                  pgps9k_root: Path, limit: int | None = None):
    src_root = Path(pgps9k_root)
    train_json = src_root / "PGPS9K" / "train.json"
    src_img_dir = src_root / "Diagram_Visual"
    if not train_json.exists():
        sys.exit(f"PGPS9K train.json not found: {train_json}")
    if not src_img_dir.is_dir():
        sys.exit(f"PGPS9K image dir not found: {src_img_dir}")

    img_dir = image_root / "pgps9k"
    print(f"\n[PGPS9K] formatting train from {train_json}...", flush=True)
    raw = json.load(train_json.open())
    items = list(raw.items())
    if limit is not None:
        items = items[:limit]

    rows: list[dict] = []
    skipped = 0
    for idx, (prob_id, ex) in enumerate(items):
        choices = ex.get("choices") or []
        if len(choices) < 2:
            skipped += 1
            continue
        ans_idx = _pgps9k_choice_index(ex.get("answer", ""), choices)
        if ans_idx is None or ans_idx >= len(choices):
            skipped += 1
            continue
        diagram = ex.get("diagram") or ""
        src_img = src_img_dir / diagram
        if not src_img.exists():
            print(f"  [warn] missing image {src_img}", flush=True)
            skipped += 1
            continue

        qid = f"pgps9k_train_{idx:05d}"
        dest = img_dir / f"{qid}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dest)

        opts = [str(c) for c in choices]
        rows.append({
            "question_id": qid,
            "image_path":  str(dest),
            "question":    render_mc_question(str(ex.get("text", "")).strip(), opts),
            "answer":      LETTERS[ans_idx],
            "source":      "pgps9k_train",
            "meta":        {
                "prob_id":        prob_id,
                "options_text":   opts,
                "answer_text":    opts[ans_idx],
                "numeric_answer": str(ex.get("answer", "")).strip(),
                "problem_type":   ex.get("type"),
                "book":           ex.get("book"),
            },
        })
        if (idx + 1) % 500 == 0:
            print(f"  ... PGPS9K {idx + 1}/{len(items)}", flush=True)

    if skipped:
        print(f"  [PGPS9K] skipped {skipped} rows (bad choices/answer/image)", flush=True)
    _write_jsonl(rows, train_jsonl)


def _export_pgps9k_split(image_root: Path, eval_dir: Path, pgps9k_root: Path,
                         split: str, limit: int | None = None) -> None:
    src_root = Path(pgps9k_root)
    split_json = src_root / "PGPS9K" / f"{split}.json"
    src_img_dir = src_root / "Diagram_Visual"
    if not split_json.exists():
        sys.exit(f"PGPS9K {split}.json not found: {split_json}")

    img_dir = image_root / "pgps9k"
    print(f"\n[PGPS9K] formatting {split} from {split_json}...", flush=True)
    raw = json.load(split_json.open())
    items = list(raw.items())
    if limit is not None:
        items = items[:limit]

    rows: list[dict] = []
    skipped = 0
    for idx, (prob_id, ex) in enumerate(items):
        choices = ex.get("choices") or []
        if len(choices) < 2:
            skipped += 1
            continue
        ans_idx = _pgps9k_choice_index(ex.get("answer", ""), choices)
        if ans_idx is None or ans_idx >= len(choices):
            skipped += 1
            continue
        diagram = ex.get("diagram") or ""
        src_img = src_img_dir / diagram
        if not src_img.exists():
            skipped += 1
            continue

        qid = f"pgps9k_{split}_{idx:05d}"
        dest = img_dir / f"{qid}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dest)

        opts = [str(c) for c in choices]
        rows.append({
            "question_id": qid,
            "image_path":  str(dest),
            "question":    render_mc_question(str(ex.get("text", "")).strip(), opts),
            "answer":      LETTERS[ans_idx],
            "source":      f"pgps9k_{split}",
            "meta":        {
                "prob_id":        prob_id,
                "options_text":   opts,
                "answer_text":    opts[ans_idx],
                "numeric_answer": str(ex.get("answer", "")).strip(),
                "problem_type":   ex.get("type"),
                "book":           ex.get("book"),
            },
        })
    if skipped:
        print(f"  [PGPS9K {split}] skipped {skipped} rows", flush=True)
    out = eval_dir / f"pgps9k_{split}_eval.jsonl"
    _write_jsonl(rows, out)


def export_pgps9k_test(image_root: Path, eval_dir: Path, pgps9k_root: Path,
                       limit: int | None = None):
    _export_pgps9k_split(image_root, eval_dir, pgps9k_root, "test", limit)


# ---------------------------------------------------------------------------
# MMMU-Pro standard (4 options).  Multi-image: image_1..7. Use all non-None.
# ---------------------------------------------------------------------------

def export_mmmu_pro(image_root: Path, eval_dir: Path, limit: int | None = None):
    img_dir = image_root / "mmmu_pro"
    print(f"\n[MMMU-Pro] downloading...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (4 options)", split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for i in range(n):
        ex = ds[i]
        qid = ex.get("id") or f"mmmu_pro_{i:05d}"
        # collect images (up to 7)
        img_paths = []
        for k in range(1, 8):
            img = ex.get(f"image_{k}")
            if img is None:
                continue
            p = _save_image(img, img_dir / f"{qid}_img{k}.png")
            if p:
                img_paths.append(str(p))
        # parse options (sometimes stored as string list literal)
        opts = ex["options"]
        if isinstance(opts, str):
            try:
                import ast
                opts = ast.literal_eval(opts)
            except Exception:
                opts = [opts]
        rows.append({
            "question_id": qid,
            "image_path":  img_paths if len(img_paths) > 1 else (img_paths[0] if img_paths else None),
            "question":    render_mc_question(ex["question"], opts),
            "answer":      str(ex["answer"]).strip(),
            "source":      "mmmu_pro",
            "meta":        {"subject": ex.get("subject"), "img_type": ex.get("img_type"),
                            "topic_difficulty": ex.get("topic_difficulty")},
        })
    _write_jsonl(rows, eval_dir / "mmmu_pro_eval.jsonl")


# ---------------------------------------------------------------------------
# RealWorldQA. question already contains options inline; answer is a letter.
# ---------------------------------------------------------------------------

def export_realworldqa(image_root: Path, eval_dir: Path, limit: int | None = None):
    img_dir = image_root / "realworldqa"
    print(f"\n[RealWorldQA] downloading...", flush=True)
    ds = load_dataset("xai-org/RealworldQA", split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for i in range(n):
        ex = ds[i]
        qid = f"rwqa_{i:04d}"
        p = _save_image(ex.get("image"), img_dir / f"{qid}.png")
        rows.append({
            "question_id": qid,
            "image_path":  str(p) if p else None,
            "question":    str(ex["question"]).strip(),
            "answer":      str(ex["answer"]).strip(),
            "source":      "realworldqa",
        })
    _write_jsonl(rows, eval_dir / "realworldqa_eval.jsonl")


# ---------------------------------------------------------------------------
# MathVerse testmini.  question_for_eval / answer / image / question_type.
# ---------------------------------------------------------------------------

def export_mathverse(image_root: Path, eval_dir: Path, limit: int | None = None):
    img_dir = image_root / "mathverse"
    print(f"\n[MathVerse] downloading testmini...", flush=True)
    ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for i in range(n):
        ex = ds[i]
        qid = f"mathverse_{ex.get('sample_index', i)}_v{ex.get('problem_index', 0)}"
        p = _save_image(ex.get("image"), img_dir / f"{qid}.png")
        # question_for_eval has clean text (no leading instruction); fall back to question
        q = (ex.get("question_for_eval") or ex.get("question") or "").strip()
        rows.append({
            "question_id": qid,
            "image_path":  str(p) if p else None,
            "question":    q,
            "answer":      str(ex["answer"]).strip(),
            "source":      "mathverse",
            "meta":        {"problem_version": ex.get("problem_version"),
                            "question_type":   ex.get("question_type")},
        })
    _write_jsonl(rows, eval_dir / "mathverse_eval.jsonl")


def _mathvista_metadata(ex: dict) -> dict:
    md = ex.get("metadata") or {}
    if isinstance(md, str):
        try:
            md = ast.literal_eval(md)
        except Exception:
            md = {}
    return md if isinstance(md, dict) else {}


def _mathvista_question(ex: dict) -> str:
    q = (ex.get("query") or ex.get("question") or "").strip()
    choices = ex.get("choices")
    if choices and ex.get("question_type") == "multi_choice":
        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except Exception:
                choices = None
        if isinstance(choices, (list, tuple)) and choices:
            return render_mc_question(q, list(choices))
    return q


def export_mathvista_gps(image_root: Path, eval_dir: Path, limit: int | None = None):
    """MathVista test split filtered to GPS (geometry problem solving) rows."""
    img_dir = image_root / "mathvista_gps"
    print(f"\n[MathVista-GPS] downloading test + filtering task=GPS...", flush=True)
    ds = load_dataset("AI4Math/MathVista", split="test")
    rows = []
    for i, ex in enumerate(ds):
        md = _mathvista_metadata(ex)
        if md.get("task") != "geometry problem solving":
            continue
        qid = f"mathvista_gps_{ex.get('pid', i)}"
        p = _save_image(ex.get("decoded_image") or ex.get("image"),
                        img_dir / f"{qid}.png")
        if not p:
            continue
        rows.append({
            "question_id": qid,
            "image_path":  str(p),
            "question":    _mathvista_question(ex),
            "answer":      str(ex["answer"]).strip(),
            "source":      "mathvista_gps",
            "meta":        {
                "pid":           ex.get("pid"),
                "question_type": ex.get("question_type"),
                "answer_type":   ex.get("answer_type"),
                "metadata":      md,
            },
        })
        if limit is not None and len(rows) >= limit:
            break
    _write_jsonl(rows, eval_dir / "mathvista_gps_eval.jsonl")


def export_mathvision(image_root: Path, eval_dir: Path, limit: int | None = None):
    """MATH-V (MathVision) full test split — 3040 competition visual-math problems."""
    img_dir = image_root / "mathvision"
    print(f"\n[MATH-V] downloading test...", flush=True)
    ds = load_dataset("MathLLMs/MathVision", split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for i in range(n):
        ex = ds[i]
        qid = f"mathvision_{ex.get('id', i)}"
        img = ex.get("decoded_image") or ex.get("image")
        p = _save_image(img, img_dir / f"{qid}.png")
        if not p:
            continue
        q = re.sub(r"<image\d*>", "", str(ex.get("question") or "")).strip()
        opts = ex.get("options") or []
        if isinstance(opts, str):
            try:
                opts = ast.literal_eval(opts)
            except Exception:
                opts = []
        if opts:
            q = render_mc_question(q, list(opts))
        rows.append({
            "question_id": qid,
            "image_path":  str(p),
            "question":    q,
            "answer":      str(ex["answer"]).strip(),
            "source":      "mathvision",
            "meta":        {
                "subject": ex.get("subject"),
                "level":   ex.get("level"),
                "options": list(opts) if opts else [],
            },
        })
    _write_jsonl(rows, eval_dir / "mathvision_eval.jsonl")


_GEOQA_CHOICES_RE = re.compile(r"choices(\{.*?\})\.\s*", re.DOTALL)


def _parse_geoqa_problem(problem: str) -> tuple[str, dict[str, str]]:
    text = re.sub(r"^<image>\s*", "", (problem or "").strip())
    m = _GEOQA_CHOICES_RE.search(text)
    choices: dict[str, str] = {}
    q = text
    if m:
        try:
            raw = ast.literal_eval(m.group(1))
            if isinstance(raw, dict):
                choices = {str(k).upper(): str(v) for k, v in raw.items()}
        except Exception:
            pass
        q = text[:m.start()].strip()
    q = re.sub(r"请用 A、B、C、D 作答\.?\s*$", "", q).strip()
    return q, choices


def export_geoqa(image_root: Path, eval_dir: Path, limit: int | None = None):
    """GeoQA official test split (755 MCQ) via LoadingBFX/GeoQA-ABCD."""
    img_dir = image_root / "geoqa"
    print(f"\n[GeoQA] downloading test...", flush=True)
    ds = load_dataset("LoadingBFX/GeoQA-ABCD", split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for i in range(n):
        ex = ds[i]
        qid = f"geoqa_test_{i:05d}"
        img = ex.get("images")
        if isinstance(img, (list, tuple)):
            img = img[0] if img else None
        p = _save_image(img, img_dir / f"{qid}.png")
        if not p:
            continue
        q_text, choices = _parse_geoqa_problem(ex.get("problem", ""))
        question = render_mc_question(q_text, choices) if choices else q_text
        rows.append({
            "question_id": qid,
            "image_path":  str(p),
            "question":    question,
            "answer":      str(ex["answer"]).strip().upper(),
            "source":      "geoqa",
            "meta":        {"choices_text": choices},
        })
    _write_jsonl(rows, eval_dir / "geoqa_test_eval.jsonl")


def export_geometry3k(image_root: Path, eval_dir: Path, limit: int | None = None):
    """Geometry3K test split (601 free-form geometry problems)."""
    img_dir = image_root / "geometry3k"
    print(f"\n[Geometry3K] downloading test...", flush=True)
    ds = load_dataset("hiyouga/geometry3k", split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    for i in range(n):
        ex = ds[i]
        qid = f"geometry3k_test_{i:05d}"
        imgs = ex.get("images") or []
        img = imgs[0] if imgs else None
        p = _save_image(img, img_dir / f"{qid}.png")
        if not p:
            continue
        q = re.sub(r"^<image>\s*", "", str(ex.get("problem") or "")).strip()
        rows.append({
            "question_id": qid,
            "image_path":  str(p),
            "question":    q,
            "answer":      str(ex["answer"]).strip(),
            "source":      "geometry3k",
        })
    _write_jsonl(rows, eval_dir / "geometry3k_test_eval.jsonl")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

DATASETS = {
    "pgps9k":          export_pgps9k,
    "pgps9k_test":     export_pgps9k_test,
    "ai2d":            export_ai2d,
    "mmmu_pro":        export_mmmu_pro,
    "realworldqa":     export_realworldqa,
    "mathverse":       export_mathverse,
    "mathvision":      export_mathvision,
    "mathvista_gps":   export_mathvista_gps,
    "geoqa":           export_geoqa,
    "geometry3k":      export_geometry3k,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Config yaml (default: experiences/mllm/configs/config.yaml)")
    p.add_argument("--only", nargs="+", choices=list(DATASETS.keys()),
                   help="If set, only run these dataset builders.")
    p.add_argument("--limit", type=int, default=None,
                   help="Per-dataset row cap (smoke testing).")
    args = p.parse_args()

    default_cfg = Path(__file__).resolve().parent / "configs" / "config.yaml"
    cfg = load_config(args.config or str(default_cfg))
    apply_hf_env(cfg)
    paths = ensure_dirs(get_paths(cfg))
    image_root = Path(cfg["datasets"]["image_root"])
    eval_dir = paths.eval_cache
    eval_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    train_jsonl = Path(cfg["datasets"]["train_jsonl"])
    pgps9k_root = Path(
        cfg["datasets"].get(
            "pgps9k_root",
            "/data4/FTSO/datasets/mllm/raw/pgps9k/extracted/PGPS9K",
        )
    )

    targets = args.only or list(DATASETS.keys())
    for name in targets:
        fn = DATASETS[name]
        if name == "pgps9k":
            fn(image_root, train_jsonl, pgps9k_root, limit=args.limit)
        elif name == "pgps9k_test":
            fn(image_root, eval_dir, pgps9k_root, limit=args.limit)
        else:
            fn(image_root, eval_dir, limit=args.limit)
    print("\n[DONE] mllm dataset prep")


if __name__ == "__main__":
    main()
