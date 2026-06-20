#!/usr/bin/env python3
"""Build MLLM smoke paraphrase.jsonl using the SAME token-matched, bin-aware
filtering/sampling as scripts/04_build_paraphrase_manifest.py.

Smoke runs on the 1k PGPS9K subset with a 1.44M-token budget
(set in mllm_config_smoke.yaml -> manifest.target_tokens)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mllm/ for utils
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # lzl/ for paths
from paths import ensure_dirs, get_paths, load_config

BIN_NAC = {"high", "mid-high", "mid-low", "low"}


def reject_paraphrase(p_text: str) -> str | None:
    t = p_text.strip()
    if len(t) < 50:
        return "too_short"
    if any(t.count(t[i:i + 30]) >= 4 for i in range(0, min(len(t) - 30, 200), 30)):
        return "repetition"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=str, default=None)
    p.add_argument("--raw", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.get("task") != "mllm":
        sys.exit("This smoke script is only for task: mllm")

    paths = ensure_dirs(get_paths(cfg))
    para_in = Path(args.candidates) if args.candidates else paths.paraphrase_candidates
    raw_in = Path(args.raw) if args.raw else paths.raw_jsonl
    out_dir = Path(args.out_dir) if args.out_dir else paths.paraphrase_jsonl.parent
    para_out = out_dir / "paraphrase.jsonl"
    manifest_path = out_dir / "manifest.json"
    token_file = paths.paraphrase_tokens

    raw_qids = set()
    with open(raw_in) as f:
        for line in f:
            raw_qids.add(json.loads(line)["question_id"])

    candidates = []
    rejected = Counter()
    raw_loaded = 0
    with open(para_in) as f:
        for line in f:
            raw_loaded += 1
            row = json.loads(line)
            if not row.get("paraphrase_match"):
                rejected["final_answer_mismatch"] += 1
                continue
            reason = reject_paraphrase(row["paraphrase_text"])
            if reason:
                rejected[reason] += 1
                continue
            candidates.append(row)

    print("=== FILTER REPORT ===")
    print(f"  raw candidates: {raw_loaded}")
    print(f"  after filter:   {len(candidates)}")
    for key, value in rejected.items():
        print(f"  rejected[{key}]: {value}")

    tok = AutoTokenizer.from_pretrained(str(paths.model_path), trust_remote_code=True)
    with open(token_file, "w") as f_tok:
        for cand in candidates:
            cand["tokens"] = len(tok(cand["paraphrase_text"], add_special_tokens=False)["input_ids"])
            f_tok.write(json.dumps({
                "qid": cand["question_id"],
                "para_idx": cand["para_idx"],
                "tokens": cand["tokens"],
            }) + "\n")

    rng = random.Random(paths.seed)
    target_ac = round(paths.target_tokens * (100 - paths.target_nac_pct) / 100)
    target_nac = paths.target_tokens - target_ac

    ac_pool, nac_pool = defaultdict(list), defaultdict(list)
    for cand in candidates:
        (nac_pool if cand["src_bin"] in BIN_NAC else ac_pool)[cand["question_id"]].append(cand)
    for d in (ac_pool, nac_pool):
        for q in d:
            rng.shuffle(d[q])

    def fill(pool, budget, label):
        chosen, used = [], 0
        qids = list(pool.keys())
        rng.shuffle(qids)
        idx = {q: 0 for q in qids}
        while True:
            progress = False
            for q in qids:
                if idx[q] < len(pool[q]):
                    cand = pool[q][idx[q]]
                    if used + cand["tokens"] > budget:
                        continue
                    chosen.append(cand)
                    used += cand["tokens"]
                    idx[q] += 1
                    progress = True
            if not progress or used >= budget * 0.999:
                break
        print(f"  [{label}] n={len(chosen)} tokens={used} (target {budget})")
        return chosen, used

    nac_chosen, nac_used = fill(nac_pool, target_nac, "non-AC")
    ac_chosen, ac_used = fill(ac_pool, target_ac, "AC")
    selected = nac_chosen + ac_chosen
    rng.shuffle(selected)
    total_used = nac_used + ac_used
    actual_nac_pct = nac_used / total_used * 100 if total_used else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(para_out, "w") as f:
        for cand in selected:
            f.write(json.dumps({
                "question_id": cand["question_id"],
                "question": cand["question"],
                "answer": cand["answer"],
                "image_path": cand.get("image_path"),
                "trace_text": cand["paraphrase_text"],
                "source": "paraphrase",
                "rollout_id": cand.get("src_rollout_id"),
                "para_idx": cand["para_idx"],
                "tokens": cand["tokens"],
                "bin": cand["src_bin"],
            }, ensure_ascii=False) + "\n")

    per_qid = Counter(c["question_id"] for c in selected)
    missing = sorted(raw_qids - set(per_qid))
    pq_n = sorted(per_qid.values()) if per_qid else [0]
    manifest = {
        "version": "lzl_mllm_smoke_paraphrase_token_matched_v1",
        "selection_mode": "token_matched_bin_aware",
        "covered_questions": len(per_qid),
        "missing_questions": missing,
        "target_tokens": paths.target_tokens,
        "actual_tokens": total_used,
        "target_nac_pct": paths.target_nac_pct,
        "actual_nac_pct": round(actual_nac_pct, 2),
        "n_traces": len(selected),
        "filter": {"raw_candidates": raw_loaded, "after_filter": len(candidates), "rejected": dict(rejected)},
        "per_qid_count": {"min": pq_n[0], "median": pq_n[len(pq_n) // 2], "max": pq_n[-1]},
        "seed": paths.seed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(selected)} traces -> {para_out}")
    if total_used >= paths.target_tokens * 0.98 and abs(actual_nac_pct - paths.target_nac_pct) <= 5:
        print("=== READY_FOR_SFT ===")
    else:
        print(f"=== NOT_READY: tokens={total_used}, nac_pct={actual_nac_pct:.1f}% ===")


if __name__ == "__main__":
    main()
