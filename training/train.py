"""
Training script for PIE reconstruction models.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --model google/byt5-small
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from tqdm import tqdm

# Wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(config: Dict) -> Dict[str, Dataset]:
    """Load train/val/test datasets."""
    data_config = config["data"]

    datasets = {}
    for split in ["train", "val", "test"]:
        file_key = f"{split}_file"
        if file_key in data_config:
            df = pd.read_csv(data_config[file_key])
            datasets[split] = Dataset.from_pandas(df)
            print(f"Loaded {split}: {len(df)} examples")

    return datasets


def preprocess_function(examples, tokenizer, config):
    """Tokenize inputs and targets."""
    data_config = config["data"]

    inputs = [str(x) for x in examples[data_config["input_column"]]]
    targets = [str(x) for x in examples[data_config["target_column"]]]

    model_inputs = tokenizer(
        inputs,
        max_length=data_config["max_input_length"],
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        targets,
        max_length=data_config["max_target_length"],
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute evaluation metrics."""
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad token id
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Exact match accuracy
    exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
    accuracy = exact_matches / len(decoded_preds)

    # Character-level accuracy
    char_correct = 0
    char_total = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred, label = pred.strip(), label.strip()
        min_len = min(len(pred), len(label))
        char_correct += sum(p == l for p, l in zip(pred[:min_len], label[:min_len]))
        char_total += max(len(pred), len(label))

    char_accuracy = char_correct / char_total if char_total > 0 else 0

    # Edit distance
    edit_distances = []
    for pred, label in zip(decoded_preds, decoded_labels):
        dist = levenshtein_distance(pred.strip(), label.strip())
        edit_distances.append(dist)

    avg_edit_distance = np.mean(edit_distances)
    normalized_edit_distance = np.mean([
        d / max(len(p), len(l), 1)
        for d, p, l in zip(edit_distances, decoded_preds, decoded_labels)
    ])

    return {
        "exact_match": accuracy,
        "char_accuracy": char_accuracy,
        "avg_edit_distance": avg_edit_distance,
        "normalized_edit_distance": normalized_edit_distance,
    }


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def generate_run_name(model_name: str) -> str:
    """Generate unique run name with model and timestamp."""
    # Extract short model name (e.g., "google/byt5-base" -> "byt5-base")
    short_name = model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{short_name}_{timestamp}"


def main():
    parser = argparse.ArgumentParser(description="Train PIE reconstruction model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (auto-generated if not provided)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.model:
        config["model"]["name"] = args.model

    model_name = config["model"]["name"]
    train_config = config["training"]

    # Generate unique output directory
    run_name = args.run_name or generate_run_name(model_name)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_output_dir = train_config.get("output_dir", "./outputs")
        output_dir = os.path.join(base_output_dir, run_name)

    train_config["output_dir"] = output_dir

    print(f"=" * 60)
    print(f"PIE Reconstruction Training")
    print(f"Model: {model_name}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir}")
    print(f"=" * 60)

    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb and config.get("wandb"):
        wandb.init(
            project=config["wandb"].get("project", "pie-reconstruction"),
            entity=config["wandb"].get("entity"),
            name=config["wandb"].get("run_name"),
            config=config,
        )

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with custom dropout if specified
    dropout_rate = train_config.get("dropout")
    if dropout_rate:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dropout_rate=dropout_rate,
        )
        print(f"Dropout rate: {dropout_rate}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print("\nLoading data...")
    datasets = load_data(config)

    # Tokenize datasets
    print("Tokenizing...")
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda x: preprocess_function(x, tokenizer, config),
            batched=True,
            remove_columns=dataset.column_names,
        )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Check if evaluation is enabled
    eval_strategy = train_config.get("eval_strategy", "no")
    do_eval = eval_strategy != "no" and "val" in tokenized_datasets

    if do_eval:
        print(f"Evaluation: enabled (strategy={eval_strategy})")
    else:
        print("Evaluation: disabled")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config.get("eval_batch_size", train_config["batch_size"]),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        learning_rate=train_config["learning_rate"],
        warmup_steps=train_config.get("warmup_steps", 500),
        weight_decay=train_config.get("weight_decay", 0.01),
        label_smoothing_factor=train_config.get("label_smoothing", 0.0),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", False),
        eval_strategy=eval_strategy,
        save_strategy=train_config.get("save_strategy", "epoch"),
        save_total_limit=train_config.get("save_total_limit"),  # None = save all
        load_best_model_at_end=train_config.get("load_best_model_at_end", False) if do_eval else False,
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss") if do_eval else None,
        greater_is_better=False,  # lower eval_loss is better
        logging_steps=train_config.get("logging_steps", 50),
        predict_with_generate=train_config.get("predict_with_generate", False),
        generation_max_length=config["generation"].get("max_length", 64) if do_eval else None,
        generation_num_beams=config["generation"].get("num_beams", 4) if do_eval else None,
        report_to="wandb" if (WANDB_AVAILABLE and not args.no_wandb) else "none",
    )

    # Early stopping callback (only if eval is enabled)
    callbacks = []
    early_stopping_patience = train_config.get("early_stopping_patience")
    if early_stopping_patience and do_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"Early stopping enabled: patience={early_stopping_patience}")

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("val") if do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=(lambda x: compute_metrics(x, tokenizer)) if (do_eval and train_config.get("predict_with_generate", False)) else None,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_config["output_dir"])

    print(f"\nTraining complete!")
    print(f"Model saved to: {train_config['output_dir']}")
    print(f"\nTo evaluate, run:")
    print(f"  python evaluate.py --model {train_config['output_dir']} --test ../dataset/splits/iecor/val.csv")

    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
