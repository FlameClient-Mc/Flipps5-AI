"""Train Flipps V0.1 — LoRA fine-tuning on your own data (CPU-friendly).

Usage:
    python train.py --data data/train.jsonl
    python train.py --data data/train.jsonl --epochs 3 --rank 8 --merge
"""
import argparse
import json
import os

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from persona import SYSTEM_PROMPT

DEFAULT_MODEL = os.environ.get("FLAMEFLIPPS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_messages(row):
    if "messages" in row:
        return row["messages"]
    if "instruction" in row or "prompt" in row:
        instruction = row.get("instruction") or row.get("prompt")
        response = row.get("response") or row.get("completion") or row.get("output")
        if not response:
            raise ValueError(f"Row missing a response: {row}")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    raise ValueError(f"Unrecognized row format: {row}")


def tokenize(rows, tokenizer):
    result = []
    for row in rows:
        text = tokenizer.apply_chat_template(to_messages(row), tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=1024)
        # Labels are required explicitly: transformers >= 5 no longer derives them from input_ids.
        result.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Flipps V0.1 with LoRA on your own data")
    parser.add_argument("--data", default="data/train.jsonl", help="JSONL file of Q&A pairs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output", default="flipps-v0.1-lora", help="Where to save the trained adapter")
    parser.add_argument("--merge", action="store_true", help="Also save a merged full model")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(
            f"Dataset not found: {args.data} — add your data (see data/train.jsonl for the format)."
        )

    print(f"[Flipps V0.1] Loading dataset: {args.data}")
    rows = load_rows(args.data)
    print(f"[Flipps V0.1] {len(rows)} training examples")

    print(f"[Flipps V0.1] Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(tokenize(rows, tokenizer))

    model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="flipps-v0.1-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=tokenizer.pad_token_id
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    print(f"[Flipps V0.1] Saving adapter to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if args.merge:
        print("[Flipps V0.1] Merging LoRA into the base model...")
        merged = model.merge_and_unload()
        merged_dir = args.output + "-merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[Flipps V0.1] Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
