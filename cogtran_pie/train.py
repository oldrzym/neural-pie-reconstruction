"""
Train Cognate Transformer (MSA model) for PIE reconstruction on local CSV splits.

Example:
python cogtran_pie/train.py \
  --setup-repo \
  --train-file dataset/splits/iecor_kaikki_koebler_normalized/train.csv \
  --val-file dataset/splits/iecor_kaikki_koebler_normalized/val.csv \
  --test-file dataset/splits/iecor_kaikki_koebler_normalized/test.csv \
  --output-dir cogtran_pie/runs/pie_msat_run1 \
  --gpu 0
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pie_cogtran_utils import (
    build_vocab,
    compute_shape_stats,
    compute_string_metrics,
    decode_predictions_and_labels,
    ensure_msa_max_tokens_per_msa,
    get_character_tokenizer_compat_class,
    levenshtein_distance,
    load_split,
    tokenize_row,
    trainer_compute_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = SCRIPT_DIR / "repo"
COGTRAN_REPO_URL_DEFAULT = "https://github.com/mahesh-ak/CognateTransformer.git"


def _run(cmd: list[str], cwd: Path):
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        sys.exit(result.returncode)


def setup_repo(repo_dir: Path, repo_url: str):
    print(f"Cloning CognateTransformer into {repo_dir} ...")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], cwd=SCRIPT_DIR)


def ensure_repo_ready(repo_path: str, setup_if_missing: bool, repo_url: str) -> Path:
    repo = Path(repo_path).resolve()
    if not repo.exists():
        if not setup_if_missing:
            print(f"ERROR: CogTran repo directory not found: {repo}")
            print("  Run with --setup-repo to auto-clone it.")
            print(f"  Or clone manually: git clone {repo_url} {repo}")
            sys.exit(1)
        setup_repo(repo, repo_url)

    if not (repo / "src").exists():
        print(f"ERROR: Invalid CogTran repo path: {repo}")
        print("  Expected directory structure with 'src/' inside.")
        sys.exit(1)
    return repo


def add_cogtran_to_path(repo: Path):
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def validate_input_files(train_file: str, val_file: str, test_file: str):
    missing = [p for p in [train_file, val_file, test_file] if not Path(p).exists()]
    if missing:
        print("ERROR: Missing required split files:")
        for path in missing:
            print(f"  {Path(path).resolve()}")
        sys.exit(1)


def to_dataset(samples: list[dict], dataset_cls: Any):
    rows = [{"data": x["data"], "solns": x["solns"]} for x in samples]
    return dataset_cls.from_list(rows)


def tokenize_dataset(ds, tokenizer):
    return ds.map(
        lambda row: tokenize_row(row, tokenizer=tokenizer),
        remove_columns=ds.column_names,
    )


def save_prediction_table(records: list[dict], out_csv: Path):
    import pandas as pd

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)


def evaluate_split(
    trainer,
    split_name: str,
    tokenized_ds,
    samples: list[dict],
    tokenizer,
    output_dir: Path,
) -> dict[str, float]:
    pred_out = trainer.predict(tokenized_ds, metric_key_prefix=split_name)
    pred_texts, _ = decode_predictions_and_labels(pred_out.predictions, pred_out.label_ids, tokenizer)
    gold_texts = [x["target_text"] for x in samples]
    metrics = compute_string_metrics(pred_texts, gold_texts)

    records = []
    for sample, pred, gold in zip(samples, pred_texts, gold_texts):
        dist = levenshtein_distance(pred.strip(), gold.strip())
        records.append(
            {
                "input": sample["input_text"],
                "target": gold,
                "prediction": pred,
                "exact_match": pred.strip() == gold.strip(),
                "edit_distance": dist,
                "normalized_edit_distance": dist / max(len(pred.strip()), len(gold.strip()), 1),
            }
        )

    save_prediction_table(records, output_dir / f"{split_name}_predictions.csv")
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CogTran MSA model on PIE reconstruction CSV data")
    parser.add_argument(
        "--cogtran-repo",
        type=str,
        default=str(DEFAULT_REPO_DIR),
        help=f"Path to local CognateTransformer repo (default: {DEFAULT_REPO_DIR})",
    )
    parser.add_argument(
        "--setup-repo",
        action="store_true",
        help="Auto-clone CognateTransformer into --cogtran-repo if missing",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=COGTRAN_REPO_URL_DEFAULT,
        help=f"CogTran git URL for --setup-repo (default: {COGTRAN_REPO_URL_DEFAULT})",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="dataset/splits/iecor_kaikki_koebler_normalized/train.csv",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="dataset/splits/iecor_kaikki_koebler_normalized/val.csv",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="dataset/splits/iecor_kaikki_koebler_normalized/test.csv",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite-output", action="store_true", help="Allow writing into non-empty output dir")
    parser.add_argument("--proto-lang", type=str, default="PIE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-unique-langs", type=int, default=2)
    parser.add_argument("--no-lingpy", action="store_true", help="Disable lingpy MSA alignment")
    parser.add_argument(
        "--keep-all-protoforms",
        action="store_true",
        help="Do not cut target at first comma",
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=6)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-position-embeddings", type=int, default=256)
    parser.add_argument("--max-position-embeddings-per-msa", type=int, default=128)

    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use (-1 = default CUDA)")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print config without training")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.fp16 and args.bf16:
        print("ERROR: --fp16 and --bf16 are mutually exclusive.")
        sys.exit(1)
    if args.cpu and args.gpu >= 0:
        print("ERROR: --cpu and --gpu cannot be used together.")
        sys.exit(1)

    if not args.cpu and args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    repo_path = ensure_repo_ready(args.cogtran_repo, setup_if_missing=args.setup_repo, repo_url=args.repo_url)
    add_cogtran_to_path(repo_path)
    validate_input_files(args.train_file, args.val_file, args.test_file)

    print("=" * 72)
    print("CogTran PIE training")
    print("=" * 72)
    print(f"Repo:       {repo_path}")
    print(f"Train file: {Path(args.train_file).resolve()}")
    print(f"Val file:   {Path(args.val_file).resolve()}")
    print(f"Test file:  {Path(args.test_file).resolve()}")
    print(f"Output dir: {Path(args.output_dir).resolve()}")
    print(f"Device:     {'CPU' if args.cpu else f'GPU {args.gpu}' if args.gpu >= 0 else 'auto'}")

    if args.dry_run:
        print("\n[DRY RUN] Input checks passed. Exiting before training.")
        return

    try:
        import torch
        from datasets import Dataset
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    except ModuleNotFoundError as exc:
        print(f"ERROR: Missing Python package '{exc.name}'.")
        print("Install dependencies:")
        print("  pip install -r cogtran_pie/requirements.txt")
        print("  pip install -r <path-to-cogtran-repo>/requirements.txt")
        sys.exit(1)

    set_seed(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.data_collate import DataCollatorForMSALM
        from src.modelling_cogtran import MSATConfig, MSATForLM
    except ModuleNotFoundError as exc:
        print(f"ERROR: Missing package '{exc.name}' required by CognateTransformer code.")
        print("Install dependencies:")
        print("  pip install -r cogtran_pie/requirements.txt")
        print("Do NOT install cogtran_pie/repo/requirements.txt with modern pip.")
        sys.exit(1)
    CharacterTokenizer = get_character_tokenizer_compat_class()

    use_lingpy = not args.no_lingpy
    take_first_protoform = not args.keep_all_protoforms

    train_samples, train_stats = load_split(
        args.train_file,
        proto_lang=args.proto_lang,
        use_lingpy=use_lingpy,
        min_unique_langs=args.min_unique_langs,
        take_first_protoform=take_first_protoform,
    )
    val_samples, val_stats = load_split(
        args.val_file,
        proto_lang=args.proto_lang,
        use_lingpy=use_lingpy,
        min_unique_langs=args.min_unique_langs,
        take_first_protoform=take_first_protoform,
    )
    test_samples, test_stats = load_split(
        args.test_file,
        proto_lang=args.proto_lang,
        use_lingpy=use_lingpy,
        min_unique_langs=args.min_unique_langs,
        take_first_protoform=take_first_protoform,
    )

    if len(train_samples) == 0:
        raise RuntimeError("No train samples left after preprocessing/filtering.")
    if len(val_samples) == 0:
        raise RuntimeError("No val samples left after preprocessing/filtering.")

    shape_stats = compute_shape_stats(train_samples + val_samples + test_samples)
    vocab = build_vocab(train_samples)

    tokenizer = CharacterTokenizer(vocab, model_max_length=512, delim="|")

    train_ds = to_dataset(train_samples, Dataset)
    val_ds = to_dataset(val_samples, Dataset)
    test_ds = to_dataset(test_samples, Dataset)

    tokenized_train = tokenize_dataset(train_ds, tokenizer)
    tokenized_val = tokenize_dataset(val_ds, tokenizer)
    tokenized_test = tokenize_dataset(test_ds, tokenizer)

    # `max_position_embeddings` tracks sequence length; `max_position_embeddings_per_msa`
    # controls per-MSA row/token limits in fair-esm internals.
    max_pos = max(args.max_position_embeddings, shape_stats["max_alignment_length"] + 8)
    max_pos_per_msa = max(args.max_position_embeddings_per_msa, shape_stats["max_alignments"] + 8)

    config = MSATConfig(
        vocab_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=args.embed_dim,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.hidden_dim,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=max_pos,
        max_position_embeddings_per_msa=max_pos_per_msa,
        layer_norm_eps=1e-12,
    )
    model = MSATForLM(config=config)
    ensure_msa_max_tokens_per_msa(model, shape_stats["max_alignments"] + 8)
    data_collator = DataCollatorForMSALM(tokenizer=tokenizer, padding=True)

    use_cpu = args.cpu or not torch.cuda.is_available()
    if not args.cpu and args.gpu >= 0 and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    ta_kwargs = {
        "output_dir": str(out_dir),
        "overwrite_output_dir": args.overwrite_output,
        "save_strategy": "epoch",
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "save_total_limit": args.save_total_limit,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_mean_normalized_edit_distance",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "report_to": "none",
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
    }
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["do_eval"] = True

    if "no_cuda" in ta_params:
        ta_kwargs["no_cuda"] = use_cpu
    elif "use_cpu" in ta_params:
        ta_kwargs["use_cpu"] = use_cpu
    if "save_safetensors" in ta_params:
        ta_kwargs["save_safetensors"] = False

    training_args = TrainingArguments(**ta_kwargs)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=lambda x: trainer_compute_metrics(x, tokenizer),
        callbacks=callbacks if callbacks else None,
    )

    try:
        trainer.train()
    except ValueError as err:
        if "Output directory" in str(err) and "already exists and is not empty" in str(err):
            print("ERROR: output-dir already exists and is not empty.")
            print("  Use --overwrite-output to allow training in that directory.")
            sys.exit(1)
        raise

    try:
        trainer.save_model(str(out_dir), safe_serialization=False)
    except TypeError:
        trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    val_metrics = evaluate_split(
        trainer=trainer,
        split_name="val",
        tokenized_ds=tokenized_val,
        samples=val_samples,
        tokenizer=tokenizer,
        output_dir=out_dir,
    )

    test_metrics = {}
    if not args.skip_test_eval and len(test_samples) > 0:
        test_metrics = evaluate_split(
            trainer=trainer,
            split_name="test",
            tokenized_ds=tokenized_test,
            samples=test_samples,
            tokenizer=tokenizer,
            output_dir=out_dir,
        )

    train_config = vars(args).copy()
    train_config["cogtran_repo"] = str(repo_path)
    train_config["output_dir"] = str(out_dir)
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    run_info = {
        "repo_path": str(repo_path),
        "train_file": str(Path(args.train_file).resolve()),
        "val_file": str(Path(args.val_file).resolve()),
        "test_file": str(Path(args.test_file).resolve()),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "shape_stats": shape_stats,
        "vocab_size": tokenizer.vocab_size,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": "cpu" if use_cpu else "cuda",
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (out_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print("\nTraining complete.")
    print(f"Model dir: {out_dir}")
    print(f"Validation metrics: {json.dumps(val_metrics, ensure_ascii=False)}")
    if test_metrics:
        print(f"Test metrics: {json.dumps(test_metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
