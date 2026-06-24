#!/usr/bin/env python3
"""Step 5: SFT on chart training data — LoRA or full fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import add_vcts_to_syspath, ensure_dirs, get_paths, load_config
from utils import freeze_qwen35_vision, is_qwen35_model, sync_vllm_multimodal_assets

add_vcts_to_syspath()

SYSTEM_PROMPT_MATH = (
    "You are a helpful assistant. Solve math problems step by step. "
    "Show all your work clearly."
)
ANSWER_FORMAT_HINT = "\nPut your final answer within \\boxed{}."

LABEL_IGNORE = -100
MM_MAX_PIXELS = 1280 * 1280

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _resolve_torch_dtype(cfg: dict) -> torch.dtype:
    name = cfg.get("model", {}).get("dtype", "bfloat16")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _use_lora(sft_cfg: dict) -> bool:
    return sft_cfg.get("tuning", "lora").lower() != "full"


def _prepare_model(model, sft_cfg: dict, *, qwen35: bool):
    model.enable_input_require_grads()
    if qwen35:
        freeze_qwen35_vision(model)
    if _use_lora(sft_cfg):
        lora_cfg_dict = sft_cfg.get("lora", {})
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg_dict.get("r", 16),
            lora_alpha=lora_cfg_dict.get("alpha", 32),
            lora_dropout=lora_cfg_dict.get("dropout", 0.05),
            target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning: {trainable:,}/{total:,} trainable parameters")
    return model


def _load_chart_model(model_path: str, cfg: dict):
    torch_dtype = _resolve_torch_dtype(cfg)
    if is_qwen35_model(model_path):
        print(f"  [model] Qwen3_5ForConditionalGeneration from {model_path}")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
        )
        return model, True
    print(f"  [model] AutoModelForCausalLM from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    )
    return model, False


def mask_prompt_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = LABEL_IGNORE
    return labels


def format_example(tokenizer, ex: dict, max_length: int, system_prompt: str,
                   user_suffix: str = "") -> dict:
    user_content = ex["question"] + (user_suffix or "")
    messages_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    messages_full = messages_prompt + [
        {"role": "assistant", "content": ex["trace_text"]},
    ]
    text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    enc = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    enc["labels"] = mask_prompt_labels(enc["input_ids"], prompt_len)
    return enc


def load_jsonl_dataset(path: str, tokenizer, max_length: int, system_prompt: str,
                       user_suffix: str = "") -> Dataset:
    examples = [json.loads(line) for line in open(path)]
    ds = Dataset.from_list(examples)
    return ds.map(
        lambda ex: format_example(tokenizer, ex, max_length, system_prompt, user_suffix),
        remove_columns=ds.column_names,
        desc=f"Tokenizing {Path(path).name}",
    )


class VisionSFTDataset(TorchDataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _make_vision_collate(pad_token_id: int):
    def _vision_collate(features: list[dict]) -> dict:
        if len(features) == 1:
            f = features[0]
            batch = {}
            for key, val in f.items():
                if key in ("input_ids", "attention_mask", "labels"):
                    batch[key] = val.unsqueeze(0)
                elif key == "mm_token_type_ids":
                    batch[key] = val if val.dim() == 2 else val.unsqueeze(0)
                else:
                    batch[key] = val
            return batch

        max_len = max(f["input_ids"].shape[-1] for f in features)
        batch: dict = {}
        ids, attn, labels, mm_types = [], [], [], []
        for f in features:
            seq_len = f["input_ids"].shape[-1]
            pad = max_len - seq_len
            ids.append(torch.cat([
                f["input_ids"],
                torch.full((pad,), pad_token_id, dtype=f["input_ids"].dtype),
            ]))
            attn.append(torch.cat([
                f["attention_mask"],
                torch.zeros(pad, dtype=f["attention_mask"].dtype),
            ]))
            labels.append(torch.cat([
                f["labels"],
                torch.full((pad,), LABEL_IGNORE, dtype=f["labels"].dtype),
            ]))
            if "mm_token_type_ids" in f:
                mm = f["mm_token_type_ids"]
                if mm.dim() == 2:
                    mm = mm.squeeze(0)
                mm_types.append(torch.cat([
                    mm,
                    torch.zeros(pad, dtype=mm.dtype),
                ]))

        batch["input_ids"] = torch.stack(ids)
        batch["attention_mask"] = torch.stack(attn)
        batch["labels"] = torch.stack(labels)
        if mm_types:
            batch["mm_token_type_ids"] = torch.stack(mm_types)
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        return batch

    return _vision_collate


def load_vision_items(path: str, processor, max_length: int, system_prompt: str) -> list[dict]:
    from mllm_utils import resolve_image_path, tokenize_vision_sft_example

    examples = [json.loads(line) for line in open(path)]
    items: list[dict] = []
    skipped_long = 0
    skipped_no_image = 0
    with_image = 0

    for ex in examples:
        ip = ex.get("image_path")
        if resolve_image_path(ip) is None:
            skipped_no_image += 1
            continue
        with_image += 1
        feat = tokenize_vision_sft_example(
            processor, ex["question"], ex["trace_text"], ip,
            system_prompt, max_length,
        )
        if int(feat["input_ids"].shape[-1]) > max_length:
            skipped_long += 1
            continue
        items.append(feat)

    print(f"Train examples: {len(examples)} (with image: {with_image})")
    print(f"  [vision] kept {len(items)}, skipped long={skipped_long}, no_image={skipped_no_image}")
    if not items:
        sys.exit(f"No usable vision training examples in {path}")
    return items


def _latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


def train_one_vision(model_path, train_data_path, output_dir, sft_cfg, system_prompt,
                     cfg: dict, resume_from: str | None = None):
    torch.manual_seed(sft_cfg["seed"])
    tuning = "lora" if _use_lora(sft_cfg) else "full"
    print(f"\n=== [vision/{tuning}] Training {Path(train_data_path).name} → {output_dir} ===")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.max_pixels = MM_MAX_PIXELS
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    items = load_vision_items(train_data_path, processor, sft_cfg["max_length"], system_prompt)
    train_ds = VisionSFTDataset(items)

    model, qwen35 = _load_chart_model(model_path, cfg)
    model = _prepare_model(model, sft_cfg, qwen35=qwen35)
    bf16, fp16 = (True, False) if not _use_lora(sft_cfg) else (False, False)

    batch_size = sft_cfg.get("batch_size", 2)
    grad_accum = sft_cfg.get("grad_accum", 4)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["lr"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=[],
        seed=sft_cfg["seed"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=_make_vision_collate(pad_token_id),
    )
    if resume_from:
        print(f"  Resuming from {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    if qwen35:
        sync_vllm_multimodal_assets(model_path, output_dir)
    print(f"Saved → {output_dir}")


def train_one(model_path, train_data_path, output_dir, sft_cfg, system_prompt,
              cfg: dict, user_suffix: str = ""):
    torch.manual_seed(sft_cfg["seed"])
    tuning = "lora" if _use_lora(sft_cfg) else "full"
    print(f"\n=== [{tuning}] Training {Path(train_data_path).name} → {output_dir} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_jsonl_dataset(
        train_data_path, tokenizer, sft_cfg["max_length"], system_prompt, user_suffix,
    )
    print(f"Train examples: {len(train_ds)}")

    model, qwen35 = _load_chart_model(model_path, cfg)
    model = _prepare_model(model, sft_cfg, qwen35=qwen35)
    bf16, fp16 = (True, False) if not _use_lora(sft_cfg) else (False, False)
    print(f"  bf16={bf16} batch={sft_cfg['batch_size']} grad_accum={sft_cfg['grad_accum']}")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["grad_accum"],
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=sft_cfg["lr"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=[],
        seed=sft_cfg["seed"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt", label_pad_token_id=-100,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if qwen35:
        sync_vllm_multimodal_assets(model_path, output_dir)
    print(f"Saved → {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition",
                   choices=["paraphrase", "para_clear", "raw", "vanilla", "both", "all"],
                   default="paraphrase")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tuning", choices=["lora", "full"], default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(get_paths(cfg))
    sft_cfg = dict(cfg["sft"])
    if args.tuning:
        sft_cfg["tuning"] = args.tuning
    print(f"SFT tuning mode: {sft_cfg.get('tuning', 'lora')}")

    if cfg.get("task") == "chart":
        from chart_utils import CHART_SYSTEM_PROMPT
        system_prompt = CHART_SYSTEM_PROMPT
        user_suffix = ""
    elif cfg.get("task") == "mllm":
        from mllm_utils import MLLM_SYSTEM_PROMPT, use_vision_mode
        system_prompt = MLLM_SYSTEM_PROMPT
        user_suffix = ""
    else:
        system_prompt = SYSTEM_PROMPT_MATH
        user_suffix = ANSWER_FORMAT_HINT

    is_vision = cfg.get("task") == "mllm" and use_vision_mode(cfg)
    if cfg.get("task") == "mllm" and not is_vision:
        sys.exit("MLLM task requires eval.mode=vision for SFT.")

    data_map = {
        "paraphrase": paths.paraphrase_jsonl,
        "para_clear": paths.para_clear_jsonl,
        "raw":        paths.raw_jsonl,
        "vanilla":    paths.vanilla_jsonl,
    }
    if args.condition == "both":
        targets = ["raw", "paraphrase"]
    elif args.condition == "all":
        targets = ["raw", "paraphrase", "vanilla"]
    else:
        targets = [args.condition]

    for cond in targets:
        data_path = data_map[cond]
        if not data_path.exists():
            sys.exit(f"Training data missing: {data_path}")
        out_dir = str(paths.checkpoint_root / cond)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        resume_from = _latest_checkpoint(out_dir) if args.resume else None
        if is_vision:
            train_one_vision(
                str(paths.model_path), str(data_path), out_dir,
                sft_cfg, system_prompt, cfg, resume_from=resume_from,
            )
        else:
            train_one(
                str(paths.model_path), str(data_path), out_dir,
                sft_cfg, system_prompt, cfg, user_suffix,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
