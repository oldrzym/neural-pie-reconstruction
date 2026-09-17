#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Dict

import yaml


VARIANT_TO_SPLIT: Dict[str, str] = {
    "raw": "iecor_kaikki_koebler_normalized",
    "phonetic": "iecor_kaikki_koebler_normalized_phonetic_input",
    "hybrid_safe": "iecor_kaikki_koebler_normalized_phonetic_input_hybrid_safe",
    "hybrid_full": "iecor_kaikki_koebler_normalized_phonetic_input_hybrid_full",
    "hybrid_uroman": "iecor_kaikki_koebler_normalized_phonetic_input_hybrid_uroman",
}


def abs_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate on phonetic split variants using training/train.py")
    parser.add_argument("--variant", choices=sorted(VARIANT_TO_SPLIT.keys()), required=True)
    parser.add_argument("--model", default="google/t5-v1_1-base")
    parser.add_argument("--run-name", default=None, help="If omitted, auto-generated")
    parser.add_argument("--base-config", default="training/config.yaml")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true", default=True, help="Run evaluate.py after training")
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", default=None, help="Pass explicit device to evaluate.py (cuda/cpu)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    training_dir = repo_root / "training"
    base_cfg_path = abs_path(repo_root, args.base_config)

    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_name = VARIANT_TO_SPLIT[args.variant]
    split_dir = repo_root / "dataset" / "splits" / split_name
    for fn in ["train.csv", "val.csv", "test.csv"]:
        p = split_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing split file for variant '{args.variant}': {p}")

    cfg["data"]["train_file"] = str((split_dir / "train.csv").resolve())
    cfg["data"]["val_file"] = str((split_dir / "val.csv").resolve())
    cfg["data"]["test_file"] = str((split_dir / "test.csv").resolve())

    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        cfg["training"]["eval_batch_size"] = args.eval_batch_size
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.warmup_steps is not None:
        cfg["training"]["warmup_steps"] = args.warmup_steps
    if args.label_smoothing is not None:
        cfg["training"]["label_smoothing"] = args.label_smoothing
    if args.early_stopping_patience is not None:
        cfg["training"]["early_stopping_patience"] = args.early_stopping_patience

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_name = f"{args.variant}_{args.model.split('/')[-1]}_{ts}"
    run_name = args.run_name or auto_name

    gen_cfg_dir = Path(__file__).resolve().parent / "generated_configs"
    gen_cfg_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg_path = gen_cfg_dir / f"{run_name}.yaml"
    with gen_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_cmd = [
        "python3",
        "train.py",
        "--config",
        str(gen_cfg_path.resolve()),
        "--model",
        args.model,
        "--run-name",
        run_name,
    ]

    model_dir = training_dir / "outputs" / run_name
    eval_file = split_dir / f"{args.eval_split}.csv"
    eval_out = training_dir / "results" / f"{run_name}_{args.eval_split}"
    eval_cmd = [
        "python3",
        "evaluate.py",
        "--model",
        str(model_dir.resolve()),
        "--test",
        str(eval_file.resolve()),
        "--output",
        str(eval_out.resolve()),
        "--batch-size",
        str(args.eval_batch),
        "--num-beams",
        str(args.num_beams),
    ]
    if args.device:
        eval_cmd.extend(["--device", args.device])

    print("============================================================")
    print("Variant training launcher")
    print("============================================================")
    print(f"repo_root:   {repo_root}")
    print(f"variant:     {args.variant}")
    print(f"split_dir:   {split_dir}")
    print(f"model:       {args.model}")
    print(f"run_name:    {run_name}")
    print(f"config_out:  {gen_cfg_path}")
    print(f"model_dir:   {model_dir}")
    print(f"eval_file:   {eval_file}")
    print(f"eval_output: {eval_out}")

    if args.dry_run:
        print("Dry run requested, exiting before training.")
        return

    run(train_cmd, training_dir)
    if args.evaluate:
        run(eval_cmd, training_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
