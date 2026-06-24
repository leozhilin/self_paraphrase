#!/usr/bin/env python3
"""Download/prepare chart datasets → FTSO raw + processed chart JSONL."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir for utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # lzl/ for paths
from utils import format_chartqa, format_finqa, format_tabmwp
from paths import ensure_dirs, get_paths, load_config

PLOTQA_EVAL_N = 5000
CHARTQA_ZIP_CANDIDATES = [
    Path("/data4/FTSO/datasets/chart/raw/chartqa/raw/ChartQA Dataset.zip"),
    Path("/data5/lzl/datasets/chartqa/raw/ChartQA Dataset.zip"),
    Path("/home/liuyu/.cache/modelscope/hub/datasets/downloads/2d009a25aeefd22686b2f435b2abef91e293e84b23d2b4f1401f2f1454631361"),
]


def save_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {path} ({len(rows)} rows)")


def _ensure_chartqa_extracted(raw_root: Path) -> Path:
    extract_root = raw_root / "chartqa" / "raw" / "zip_extract"
    marker = extract_root / "ChartQA Dataset" / "train" / "train_human.json"
    if marker.exists():
        return extract_root / "ChartQA Dataset"

    zip_path = None
    for cand in CHARTQA_ZIP_CANDIDATES:
        if cand.exists() and cand.stat().st_size > 1_000_000:
            zip_path = cand
            break
    if not zip_path:
        raise FileNotFoundError(
            "ChartQA zip not found. Download OpenDataLab/ChartQA via ModelScope first."
        )

    dest_zip = raw_root / "chartqa" / "raw" / "ChartQA Dataset.zip"
    if not dest_zip.exists():
        shutil.copy2(zip_path, dest_zip)
        print(f"  Copied ChartQA zip → {dest_zip}")

    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {dest_zip} …")
    with zipfile.ZipFile(dest_zip) as zf:
        zf.extractall(extract_root)
    return extract_root / "ChartQA Dataset"


def _load_table_csv(split_dir: Path, imgname: str) -> str:
    stem = Path(imgname).stem
    csv_path = split_dir / "tables" / f"{stem}.csv"
    if not csv_path.exists():
        return ""
    return csv_path.read_text(encoding="utf-8", errors="replace").strip()


def export_chartqa_test(chart_root: Path, raw_root: Path, max_eval: int | None = None):
    """Export the official ChartQA *test* split (human + augmented merged)."""
    print("\n[ChartQA test] building eval JSONL from official zip …")
    root = _ensure_chartqa_extracted(raw_root)
    test_dir = root / "test"
    rows = []
    for fname in ("test_human.json", "test_augmented.json"):
        qa_path = test_dir / fname
        if not qa_path.exists():
            continue
        qa_list = json.loads(qa_path.read_text())
        for i, ex in enumerate(qa_list):
            table = _load_table_csv(test_dir, ex["imgname"])
            if not table:
                continue
            label = ex["label"]
            if isinstance(label, list):
                label = label[0] if label else ""
            rows.append({
                "question_id": f"chartqa_test_{fname}_{i}",
                "question": format_chartqa(table, ex["query"]),
                "answer": str(label).strip(),
                "source": "ChartQA-test",
                "meta": {"imgname": ex["imgname"], "subset": fname.replace(".json", "")},
            })
    if max_eval and len(rows) > max_eval:
        rows = rows[:max_eval]
    save_jsonl(rows, chart_root / "eval" / "chartqa_test_eval.jsonl")
    print(f"  ChartQA test={len(rows)}")


def export_chartqa(chart_root: Path, raw_root: Path):
    print("\n[ChartQA] building train JSONL from official zip …")
    root = _ensure_chartqa_extracted(raw_root)
    train_dir = root / "train"
    rows = []
    for fname in ("train_human.json", "train_augmented.json"):
        qa_path = train_dir / fname
        if not qa_path.exists():
            continue
        qa_list = json.loads(qa_path.read_text())
        for i, ex in enumerate(qa_list):
            table = _load_table_csv(train_dir, ex["imgname"])
            if not table:
                continue
            label = ex["label"]
            if isinstance(label, list):
                label = label[0] if label else ""
            rows.append({
                "question_id": f"chartqa_train_{fname}_{i}",
                "question": format_chartqa(table, ex["query"]),
                "answer": str(label).strip(),
                "source": "ChartQA",
                "meta": {"imgname": ex["imgname"], "subset": fname.replace(".json", "")},
            })

    raw_out = raw_root / "chartqa" / "raw" / "train_merged.json"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w") as f:
        json.dump(rows, f, ensure_ascii=False)
    save_jsonl(rows, chart_root / "chartqa_train.jsonl")
    print(f"  ChartQA train={len(rows)}")


def export_tabmwp(chart_root: Path, raw_root: Path, max_eval: int | None):
    print("\n[TabMWP] reading PromptPG JSON …")
    raw_dir = raw_root / "tabmwp" / "raw"
    test_path = raw_dir / "problems_test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path} — run GitHub download first.")

    data = json.loads(test_path.read_text())
    rows = []
    for k, ex in data.items():
        if ex.get("split") != "test":
            continue
        rows.append({
            "question_id": f"tabmwp_test_{k}",
            "question": format_tabmwp(ex),
            "answer": str(ex["answer"]).strip(),
            "source": "TabMWP",
            "meta": {"grade": ex.get("grade"), "ques_type": ex.get("ques_type")},
        })
    if max_eval and len(rows) > max_eval:
        rows = rows[:max_eval]
    save_jsonl(rows, chart_root / "eval" / "tabmwp_eval.jsonl")


def export_finqa(chart_root: Path, raw_root: Path, max_eval: int | None):
    print("\n[FinQA] reading official JSON …")
    test_path = raw_root / "finqa" / "raw" / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    test_data = json.loads(test_path.read_text())
    rows = []
    for i, ex in enumerate(test_data):
        qa = ex.get("qa")
        if isinstance(qa, str):
            import ast
            try:
                qa = ast.literal_eval(qa)
            except (ValueError, SyntaxError):
                qa = {}
        ans = (qa or {}).get("answer") or ex.get("answer") or ex.get("exe_ans")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        rows.append({
            "question_id": f"finqa_test_{ex.get('id', i)}",
            "question": format_finqa(ex),
            "answer": str(ans).strip(),
            "source": "FinQA",
            "meta": {"split": "test"},
        })
    if max_eval and len(rows) > max_eval:
        rows = rows[:max_eval]
    save_jsonl(rows, chart_root / "eval" / "finqa_eval.jsonl")


def _plot_table_from_annotation(ann: dict) -> str:
    lines = ["Chart data (extracted from plot metadata):"]
    models = ann.get("models") or []
    if not models:
        return ""
    for mi, model in enumerate(models):
        name = model.get("name") or model.get("label") or f"series_{mi}"
        xs = model.get("x") or []
        ys = model.get("y") or []
        if isinstance(xs, str):
            xs = [x.strip() for x in xs.split(",") if x.strip()]
        if isinstance(ys, str):
            ys = [y.strip() for y in ys.split(",") if y.strip()]
        lines.append(f"Series: {name}")
        for x, y in zip(xs, ys):
            lines.append(f"  {x}: {y}")
    return "\n".join(lines)


def export_plotqa(chart_root: Path, raw_root: Path, n_eval: int = PLOTQA_EVAL_N, seed: int = 42):
    print(f"\n[PlotQA] joining test annotations + QA (random n={n_eval}, seed={seed}) …")
    raw_dir = raw_root / "plotqa" / "raw"
    ann_path = raw_dir / "test_annotations.json"
    qa_path = raw_dir / "test_qa_v1.json"
    if not ann_path.exists() or not qa_path.exists():
        raise FileNotFoundError("PlotQA raw files missing — run gdown first.")

    annotations = json.loads(ann_path.read_text())
    ann_by_idx = {a["image_index"]: a for a in annotations}
    qa_raw = json.loads(qa_path.read_text())
    qa_list = qa_raw["qa_pairs"] if isinstance(qa_raw, dict) else qa_raw

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    for qa in qa_list:
        ann = ann_by_idx.get(qa.get("image_index"))
        if not ann:
            continue
        table = _plot_table_from_annotation(ann)
        if not table:
            continue
        row = {
            "question": format_chartqa(table, qa["question_string"]),
            "answer": str(qa["answer"]).strip(),
            "source": "PlotQA",
            "meta": {"image_index": qa.get("image_index"), "plot_type": qa.get("type")},
        }
        if len(reservoir) < n_eval:
            reservoir.append(row)
        else:
            j = rng.randrange(seen + 1)
            if j < n_eval:
                reservoir[j] = row
        seen += 1

    if not reservoir:
        raise RuntimeError("PlotQA export produced 0 rows")
    rows = [{**row, "question_id": f"plotqa_test_{i}"} for i, row in enumerate(reservoir)]
    print(f"  sampled {len(rows)} / {seen:,} joinable QA pairs")
    save_jsonl(rows, chart_root / "eval" / "plotqa_eval_5k.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--max-eval", type=int, default=0,
                   help="Cap ChartQA test / TabMWP / FinQA export (0 = no limit)")
    p.add_argument("--skip-plotqa", action="store_true")
    p.add_argument("--only", nargs="+",
                   choices=["chartqa", "chartqa_test", "tabmwp", "finqa", "plotqa"],
                   help="If set, only run these export steps (skip others).")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    ftso = Path(cfg.get("ftso_root", "/data4/FTSO"))
    raw_root = ftso / "datasets" / "chart" / "raw"
    chart_root = Path(cfg["datasets"]["train_jsonl"]).parent
    (chart_root / "eval").mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    def _do(name: str) -> bool:
        return only is None or name in only

    max_eval = args.max_eval if args.max_eval > 0 else None

    if _do("chartqa"):       export_chartqa(chart_root, raw_root)
    if _do("chartqa_test"):  export_chartqa_test(chart_root, raw_root, max_eval)
    if _do("tabmwp"):        export_tabmwp(chart_root, raw_root, max_eval)
    if _do("finqa"):         export_finqa(chart_root, raw_root, max_eval)
    if _do("plotqa") and not args.skip_plotqa:
        export_plotqa(chart_root, raw_root)

    print("\nDone →", chart_root)


if __name__ == "__main__":
    main()
