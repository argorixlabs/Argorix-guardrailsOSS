from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class SaveStateCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_state.json").write_text(
            json.dumps({"status": "running", "global_step": 0}, indent=2),
            encoding="utf-8",
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        payload = {
            "status": "running",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "logs": logs or {},
        }
        (self.output_dir / "train_state.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def on_train_end(self, args, state, control, **kwargs):
        (self.output_dir / "train_state.json").write_text(
            json.dumps(
                {
                    "status": "finished",
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Gemma style guardrail with QLoRA over sft_text."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data_finetune/guardrail_es.parquet"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("models/guardrail-qwen25-1_5b-qlora"))
    parser.add_argument("--adapter-init", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def load_sft_splits(dataset_path: Path, max_train: int | None, max_eval: int | None, seed: int):
    dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
    keep_columns = {"sft_text", "split", "primary_label", "decision"}
    remove_columns = [column for column in dataset.column_names if column not in keep_columns]
    dataset = dataset.remove_columns(remove_columns)

    train_ds = dataset.filter(lambda row: row["split"] == "train")
    eval_ds = dataset.filter(lambda row: row["split"] == "validation")

    train_ds = train_ds.shuffle(seed=seed)
    eval_ds = eval_ds.shuffle(seed=seed)

    if max_train is not None:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_eval is not None:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return train_ds, eval_ds


def print_dataset_stats(train_ds: Dataset, eval_ds: Dataset) -> None:
    print(f"Train rows: {len(train_ds):,}")
    print(f"Eval rows: {len(eval_ds):,}")
    print("Train labels:")
    labels = {}
    for label in train_ds["primary_label"]:
        labels[label] = labels.get(label, 0) + 1
    print(json.dumps(dict(sorted(labels.items())), indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible. Ejecuta esto dentro de WSL con acceso a NVIDIA.")

    train_ds, eval_ds = load_sft_splits(
        args.dataset,
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
    )
    print_dataset_stats(train_ds, eval_ds)

    tokenizer_source = args.adapter_init if args.adapter_init and args.adapter_init.exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = None
    if args.adapter_init is not None:
        model = PeftModel.from_pretrained(model, args.adapter_init, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16 or not args.bf16,
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="sft_text",
        packing=False,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[SaveStateCallback(args.output_dir)],
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = {
        "base_model": args.model,
        "adapter_init": str(args.adapter_init) if args.adapter_init else None,
        "dataset": str(args.dataset),
        "max_length": args.max_length,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "type": "qlora_adapter",
    }
    (args.output_dir / "guardrail_training.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Adapter guardado en {args.output_dir}")


if __name__ == "__main__":
    main()
