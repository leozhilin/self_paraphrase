"""Load lzl/config.yaml and expose resolved paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os
import yaml

LZL_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = LZL_ROOT / "config.yaml"
CHART_CONFIG_PATH = LZL_ROOT / "chart_config.yaml"


def resolve_config_path(path: Path | str | None = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get("LZL_CONFIG")
    return Path(env) if env else CONFIG_PATH


@dataclass
class LzlPaths:
    vcts_root: Path
    lzl_root: Path
    model_path: Path
    hf_datasets: Path
    data_cache: Path
    rollouts: Path
    raw_jsonl: Path
    vanilla_jsonl: Path
    raw_manifest: Path
    paraphrase_candidates: Path
    paraphrase_tokens: Path
    paraphrase_jsonl: Path
    paraphrase_manifest: Path
    checkpoint_root: Path
    results_root: Path
    eval_cache: Path
    logs: Path
    target_tokens: int
    target_nac_pct: float
    ac_max_pct: float
    seed: int


def load_config(path: Path | str | None = None) -> dict:
    cfg_path = resolve_config_path(path)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_chart_config() -> dict:
    return load_config(CHART_CONFIG_PATH)


def get_paths(cfg: dict | None = None) -> LzlPaths:
    cfg = cfg or load_config()
    p = cfg["paths"]
    m = cfg.get("manifest", {})
    return LzlPaths(
        vcts_root=Path(p["vcts_root"]),
        lzl_root=Path(p["lzl_root"]),
        model_path=Path(cfg["model"]["path"]),
        hf_datasets=Path(cfg["datasets"]["gsm8k_cache"]),
        data_cache=Path(cfg["datasets"]["data_cache"]),
        rollouts=Path(p["rollouts"]),
        raw_jsonl=Path(p["raw_jsonl"]),
        vanilla_jsonl=Path(p.get("vanilla_jsonl",
                                 str(Path(p["raw_jsonl"]).parent.parent / "vanilla" / "vanilla.jsonl"))),
        raw_manifest=Path(p["raw_manifest"]),
        paraphrase_candidates=Path(p["paraphrase_candidates"]),
        paraphrase_tokens=Path(p["paraphrase_tokens"]),
        paraphrase_jsonl=Path(p["paraphrase_jsonl"]),
        paraphrase_manifest=Path(p["paraphrase_manifest"]),
        checkpoint_root=Path(p["checkpoint_root"]),
        results_root=Path(p["results_root"]),
        eval_cache=Path(p["eval_cache"]),
        logs=Path(p["logs"]),
        target_tokens=int(m.get("target_tokens", 480_000)),
        target_nac_pct=float(m.get("target_nac_pct", 42.0)),
        ac_max_pct=float(m.get("ac_max_pct", 0.58)),
        seed=int(m.get("seed", 2026)),
    )


def ensure_dirs(paths: LzlPaths | None = None) -> LzlPaths:
    paths = paths or get_paths()
    for d in [
        paths.rollouts.parent,
        paths.raw_jsonl.parent,
        paths.vanilla_jsonl.parent,
        paths.paraphrase_jsonl.parent,
        paths.paraphrase_candidates.parent,
        paths.data_cache,
        paths.eval_cache,
        paths.checkpoint_root,
        paths.results_root,
        paths.results_root / "eval",
        paths.logs,
        paths.hf_datasets,
        paths.model_path.parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    return paths


def add_vcts_to_syspath():
    import sys
    root = str(get_paths().vcts_root)
    if root not in sys.path:
        sys.path.insert(0, root)
