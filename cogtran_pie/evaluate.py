"""
Evaluate a trained CogTran PIE model on a CSV split.

Example:
python cogtran_pie/evaluate.py \
  --setup-repo \
  --model-dir cogtran_pie/runs/pie_msat_run1 \
  --input-file dataset/splits/iecor_kaikki_koebler_normalized/test.csv \
  --output-dir cogtran_pie/runs/pie_msat_run1/eval_test \
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
    compute_shape_stats,
    compute_string_metrics,
    decode_predictions_and_labels,
    ensure_msa_max_tokens_per_msa,
    get_character_tokenizer_compat_class,
    levenshtein_distance,
    load_split,
    tokenize_row,
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


def to_dataset(samples: list[dict], dataset_cls: Any):
    rows = [{"data": x["data"], "solns": x["solns"]} for x in samples]
    return dataset_cls.from_list(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CogTran PIE model")
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
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument(
        "--input-file",
        "--test-file",
        dest="input_file",
        type=str,
        default="dataset/splits/iecor_kaikki_koebler_normalized/test.csv",
        help="CSV split path for evaluation",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path for metrics JSON (default: <output-dir>/metrics.json)",
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default=None,
        help="Optional path for predictions CSV (default: <output-dir>/predictions.csv)",
    )
    parser.add_argument("--proto-lang", type=str, default="PIE")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-unique-langs", type=int, default=2)
    parser.add_argument("--no-lingpy", action="store_true")
    parser.add_argument("--keep-all-protoforms", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use (-1 = default CUDA)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cpu and args.gpu >= 0:
        print("ERROR: --cpu and --gpu cannot be used together.")
        sys.exit(1)
    if not args.cpu and args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    repo_path = ensure_repo_ready(args.cogtran_repo, setup_if_missing=args.setup_repo, repo_url=args.repo_url)
    add_cogtran_to_path(repo_path)

    input_path = Path(args.input_file).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CogTran PIE evaluation")
    print("=" * 72)
    print(f"Repo:       {repo_path}")
    print(f"Model dir:  {model_dir}")
    print(f"Input file: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device:     {'CPU' if args.cpu else f'GPU {args.gpu}' if args.gpu >= 0 else 'auto'}")

    if args.dry_run:
        print("\n[DRY RUN] Input checks passed. Exiting before evaluation.")
        return

    try:
        import pandas as pd
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
    except ModuleNotFoundError as exc:
        print(f"ERROR: Missing Python package '{exc.name}'.")
        print("Install dependencies:")
        print("  pip install -r cogtran_pie/requirements.txt")
        print("  pip install -r <path-to-cogtran-repo>/requirements.txt")
        sys.exit(1)

    try:
        from src.data_collate import DataCollatorForMSALM
        from src.modelling_cogtran import MSATForLM
    except ModuleNotFoundError as exc:
        print(f"ERROR: Missing package '{exc.name}' required by CognateTransformer code.")
        print("Install dependencies:")
        print("  pip install -r cogtran_pie/requirements.txt")
        print("Do NOT install cogtran_pie/repo/requirements.txt with modern pip.")
        sys.exit(1)
    CharacterTokenizer = get_character_tokenizer_compat_class()

    test_samples, prep_stats = load_split(
        input_path,
        proto_lang=args.proto_lang,
        use_lingpy=not args.no_lingpy,
        min_unique_langs=args.min_unique_langs,
        take_first_protoform=not args.keep_all_protoforms,
    )
    if len(test_samples) == 0:
        raise RuntimeError("No samples left after preprocessing/filtering.")
    shape_stats = compute_shape_stats(test_samples)

    ds = to_dataset(test_samples, Dataset)

    tokenizer = CharacterTokenizer.from_pretrained(str(model_dir))
    model = MSATForLM.from_pretrained(str(model_dir))
    required_msa_tokens = shape_stats["max_alignments"] + 8
    ensure_msa_max_tokens_per_msa(model, required_msa_tokens)

    tokenized_test = ds.map(
        lambda row: tokenize_row(row, tokenizer=tokenizer),
        remove_columns=ds.column_names,
    )
    data_collator = DataCollatorForMSALM(tokenizer=tokenizer, padding=True)

    use_cpu = args.cpu or not torch.cuda.is_available()
    if not args.cpu and args.gpu >= 0 and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_kwargs = {
        "output_dir": str(out_dir),
        "per_device_eval_batch_size": args.batch_size,
        "do_train": False,
        "do_eval": False,
        "do_predict": True,
        "remove_unused_columns": False,
        "report_to": "none",
    }
    if "no_cuda" in ta_params:
        eval_kwargs["no_cuda"] = use_cpu
    elif "use_cpu" in ta_params:
        eval_kwargs["use_cpu"] = use_cpu

    eval_args = TrainingArguments(**eval_kwargs)

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
    )

    pred_out = trainer.predict(tokenized_test)
    pred_texts, _ = decode_predictions_and_labels(pred_out.predictions, pred_out.label_ids, tokenizer)
    gold_texts = [x["target_text"] for x in test_samples]

    metrics = compute_string_metrics(pred_texts, gold_texts)
    results = []
    for sample, pred, gold in zip(test_samples, pred_texts, gold_texts):
        dist = levenshtein_distance(pred.strip(), gold.strip())
        results.append(
            {
                "input": sample["input_text"],
                "target": gold,
                "prediction": pred,
                "exact_match": pred.strip() == gold.strip(),
                "edit_distance": dist,
                "normalized_edit_distance": dist / max(len(pred.strip()), len(gold.strip()), 1),
            }
        )

    predictions_path = Path(args.predictions_output).resolve() if args.predictions_output else out_dir / "predictions.csv"
    metrics_path = Path(args.metrics_output).resolve() if args.metrics_output else out_dir / "metrics.json"
    prep_stats_path = out_dir / "prep_stats.json"
    eval_config_path = out_dir / "eval_config.json"

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    prep_stats_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with prep_stats_path.open("w", encoding="utf-8") as f:
        payload = dict(prep_stats)
        payload["shape_stats"] = shape_stats
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with eval_config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_path": str(repo_path),
                "model_dir": str(model_dir),
                "input_file": str(input_path),
                "output_dir": str(out_dir),
                "batch_size": args.batch_size,
                "device": "cpu" if use_cpu else "cuda",
                "proto_lang": args.proto_lang,
                "min_unique_langs": args.min_unique_langs,
                "no_lingpy": args.no_lingpy,
                "keep_all_protoforms": args.keep_all_protoforms,
                "required_msa_tokens": required_msa_tokens,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Evaluation complete.")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
