"""
DPD Training Wrapper for PIE Reconstruction.

Handles:
1. Data conversion (CSV -> DPD pickle)
2. Training with proper checkpoint saving
3. Config persistence for evaluation

Usage:
    # Basic supervised training
    python train.py --split iecor_kaikki_koebler

    # Full DPD with Transformer
    python train.py --split iecor_kaikki_koebler --strat pimodel_bpall_cringe --arch Transformer

    # Custom epochs and batch size
    python train.py --split iecor_kaikki_koebler --epochs 300 --batch-size 32 --lr 0.0005
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_DIR = SCRIPT_DIR / "repo"
CONVERTER_SCRIPT = PROJECT_ROOT / "dataset" / "scripts" / "convert_to_dpd.py"
DPD_REPO_URL_DEFAULT = "https://github.com/cmu-llab/dpd.git"


def _run(cmd: list[str], cwd: Path):
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        sys.exit(result.returncode)


def setup_repo(repo_url: str):
    """Clone DPD repository into dpd/repo if missing."""
    print(f"Cloning DPD repo into {REPO_DIR} ...")
    _run(["git", "clone", "--depth", "1", repo_url, str(REPO_DIR)], cwd=SCRIPT_DIR)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "run"


def build_checkpoint_dir(args: argparse.Namespace, dataset_name: str) -> Path:
    base = REPO_DIR / "checkpoints" / f"{dataset_name}_{args.strat}_{args.arch}"
    if args.name:
        run_id = _slug(args.name)
    else:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.seed >= 0:
        run_id = f"{run_id}_seed{args.seed}"

    candidate = base / run_id
    suffix = 1
    while candidate.exists():
        candidate = base / f"{run_id}_{suffix:02d}"
        suffix += 1
    return candidate


def patch_exp_py(exp_file: Path):
    """
    Ensure exp.py has local-dataset support + stable checkpoint/config saving.
    Idempotent: safe to run multiple times.
    """
    txt = exp_file.read_text(encoding="utf-8")
    original = txt

    # 1) Remove restrictive dataset choices so custom datasets can be used.
    old_dataset_block = """parser.add_argument('--dataset', type=str, default="chinese_wikihan2022", help="Dataset to use", choices=[
    'chinese_wikihan2022',
    'Nromance_ipa',
])"""
    new_dataset_line = "parser.add_argument('--dataset', type=str, default=\"chinese_wikihan2022\", help=\"Dataset to use\")"
    if old_dataset_block in txt:
        txt = txt.replace(old_dataset_block, new_dataset_line)

    # 2) Add/upgrade local checkpoint dir definition (env override from wrapper).
    old_checkpoint_block = """
# Local checkpoint directory (saves regardless of WandB state)
CHECKPOINT_DIR = f"./checkpoints/{args.dataset}_{args.strat}_{args.architecture}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"== Checkpoints will be saved to: {CHECKPOINT_DIR} ==")
"""
    new_checkpoint_block = """
# Local checkpoint directory (saves regardless of WandB state)
CHECKPOINT_DIR = os.environ.get(
    "DPD_CHECKPOINT_DIR",
    f"./checkpoints/{args.dataset}_{args.strat}_{args.architecture}",
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"== Checkpoints will be saved to: {CHECKPOINT_DIR} ==")
"""
    if old_checkpoint_block in txt:
        txt = txt.replace(old_checkpoint_block, new_checkpoint_block, 1)
    elif "DPD_CHECKPOINT_DIR" not in txt:
        anchor = "# region === hyperparameters ==="
        if anchor not in txt:
            print("ERROR: Could not patch exp.py (missing hyperparameter anchor).")
            sys.exit(1)
        txt = txt.replace(anchor, anchor + new_checkpoint_block, 1)

    # 3) Route model checkpoints to local dir.
    old_ckpt_pattern = re.compile(
        r"'strategy_checkpoint_method': None if args\.sweeping else ModelCheckpoint\(\s*"
        r"monitor=f\"d2p/val/phoneme_edit_distance\",\s*"
        r"mode=\"min\",\s*"
        r"save_top_k=1,\s*"
        r"verbose=args\.dev,\s*"
        r"\),",
        re.MULTILINE,
    )
    new_ckpt_block = """'strategy_checkpoint_method': None if args.sweeping else ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="best-{epoch:03d}",
            monitor=f"d2p/val/phoneme_edit_distance",
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=args.dev,
        ),"""
    txt = old_ckpt_pattern.sub(new_ckpt_block, txt)

    # 4) Save config next to checkpoints for evaluate.py.
    config_save_marker = '_config_save_path = os.path.join(CHECKPOINT_DIR, "config.pkl")'
    if config_save_marker not in txt:
        needle = 'print("config:", config)'
        config_block = """

# Save config to checkpoint dir for later evaluation
import pickle as _pkl
_config_save_path = os.path.join(CHECKPOINT_DIR, "config.pkl")
with open(_config_save_path, 'wb') as _f:
    _pkl.dump(config, _f)
print(f"== Config saved to: {_config_save_path} ==")
"""
        if needle not in txt:
            print("ERROR: Could not patch exp.py (missing config print anchor).")
            sys.exit(1)
        txt = txt.replace(needle, needle + config_block, 1)

    # 5) Fix Lightning device semantics for explicit GPU index.
    # Old behavior produced Trainer(devices=0) for --gpu 0, which is invalid.
    devices_old = (
        "set_devices = 'auto' if args.cpu else (args.gpu if args.gpu != -1 else "
        "lib.getfreegpu.assign_free_gpus(threshold_vram_usage=args.vram_thresh, max_gpus=1, wait=True, sleep_time=10))"
    )
    devices_new = (
        "set_devices = 'auto' if args.cpu else ([args.gpu] if args.gpu != -1 else "
        "lib.getfreegpu.assign_free_gpus(threshold_vram_usage=args.vram_thresh, max_gpus=1, wait=True, sleep_time=10))"
    )
    if devices_old in txt:
        txt = txt.replace(devices_old, devices_new)

    # Fallback regex in case formatting differs slightly.
    devices_pattern = re.compile(
        r"set_devices\s*=\s*'auto'\s*if\s*args\.cpu\s*else\s*\(\s*args\.gpu\s*if\s*args\.gpu\s*!=\s*-1\s*else\s*lib\.getfreegpu\.assign_free_gpus\((.*?)\)\s*\)",
        re.DOTALL,
    )
    txt = devices_pattern.sub(
        r"set_devices = 'auto' if args.cpu else ([args.gpu] if args.gpu != -1 else lib.getfreegpu.assign_free_gpus(\1))",
        txt,
    )

    if txt != original:
        exp_file.write_text(txt, encoding="utf-8")
        print(f"Patched exp.py: {exp_file}")


def ensure_repo_ready(setup_if_missing: bool = False, repo_url: str = DPD_REPO_URL_DEFAULT):
    """Validate expected DPD repo layout and patch exp.py if needed."""
    if not REPO_DIR.exists():
        if not setup_if_missing:
            print(f"ERROR: DPD repo directory not found: {REPO_DIR}")
            print("  Run with --setup-repo to auto-clone it.")
            print(f"  Or clone manually: git clone {repo_url} {REPO_DIR}")
            sys.exit(1)
        setup_repo(repo_url)
    exp_file = REPO_DIR / "exp.py"
    if not exp_file.exists():
        print(f"ERROR: exp.py not found: {exp_file}")
        sys.exit(1)
    patch_exp_py(exp_file)
    if not CONVERTER_SCRIPT.exists():
        print(f"ERROR: Converter script not found: {CONVERTER_SCRIPT}")
        sys.exit(1)


def convert_data(split_name: str, force_reconvert: bool = False) -> str:
    """Convert CSV split to DPD pickle format. Returns DPD dataset name."""
    input_dir = PROJECT_ROOT / "dataset" / "splits" / split_name
    dpd_dataset_name = f"pie_{split_name}"
    output_dir = REPO_DIR / "data" / dpd_dataset_name

    if not input_dir.exists():
        print(f"ERROR: Split directory not found: {input_dir}")
        print(f"\nAvailable splits:")
        splits_dir = PROJECT_ROOT / "dataset" / "splits"
        if splits_dir.exists():
            for d in sorted(splits_dir.iterdir()):
                if d.is_dir():
                    csvs = list(d.glob("*.csv"))
                    print(f"  {d.name} ({len(csvs)} files)")
        sys.exit(1)

    # Check if already converted
    expected_files = ["train.pickle", "dev.pickle", "test.pickle"]
    if output_dir.exists() and all((output_dir / f).exists() for f in expected_files):
        if not force_reconvert:
            print(f"Data already converted: {output_dir}")
            print("Use --force-reconvert to regenerate pickles.")
            return dpd_dataset_name
        print(f"Force reconvert enabled: {output_dir}")

    print(f"\n{'='*60}")
    print(f"Converting data: {split_name}")
    print(f"  From: {input_dir}")
    print(f"  To:   {output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, str(CONVERTER_SCRIPT),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("ERROR: Data conversion failed!")
        sys.exit(1)

    return dpd_dataset_name


def build_exp_args(args, dpd_dataset_name: str) -> list:
    """Build command-line arguments for exp.py."""
    exp_args = [
        sys.executable, str(REPO_DIR / "exp.py"),
        "--dataset", dpd_dataset_name,
        "--architecture", args.arch,
        "--strat", args.strat,
        "--proportion_labelled", str(args.proportion_labelled),
        "--max_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--test_val_batch_size", str(args.test_batch_size),
        "--lr", str(args.lr),
        "--d2p_dropout_p", str(args.dropout),
        "--p2d_dropout_p", str(args.dropout),
        "--check_val_every_n_epoch", str(args.val_every),
        "--early_stopping_patience", str(args.patience),
        "--d2p_inference_decode_max_length", str(args.max_decode_len),
        "--p2d_inference_decode_max_length", str(args.max_decode_len),
        "--nowandb",
        "--skip_daughter_tone", "True",
        "--skip_protoform_tone", "True",
    ]

    if args.strat == "supervised_only":
        exp_args += ["--exclude_unlabelled"]

    if args.cpu:
        exp_args += ["--cpu"]
    elif args.gpu >= 0:
        exp_args += ["--gpu", str(args.gpu)]

    if args.name:
        exp_args += ["--name", args.name]

    if args.dev:
        exp_args += ["--dev"]

    if args.seed >= 0:
        exp_args += ["--forceseed", str(args.seed)]

    return exp_args


def main():
    parser = argparse.ArgumentParser(
        description="DPD Training for PIE Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Supervised only (baseline)
  python train.py --split iecor_kaikki_koebler

  # Full DPD strategy
  python train.py --split iecor_kaikki_koebler --strat pimodel_bpall_cringe --proportion-labelled 0.3

  # Transformer architecture
  python train.py --split iecor_kaikki_koebler --arch Transformer --batch-size 32

  # CPU training (debug)
  python train.py --split iecor_kaikki_koebler --cpu --epochs 5 --dev
        """
    )

    # Data
    parser.add_argument("--split", type=str, required=True,
                        help="Dataset split name (subdirectory of dataset/splits/)")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip data conversion (use existing pickles)")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="Force conversion even if pickle files already exist")
    parser.add_argument("--setup-repo", action="store_true",
                        help="Auto-clone dpd/repo if missing and auto-patch exp.py")
    parser.add_argument("--repo-url", type=str, default=DPD_REPO_URL_DEFAULT,
                        help=f"DPD git URL for --setup-repo (default: {DPD_REPO_URL_DEFAULT})")

    # Architecture
    parser.add_argument("--arch", type=str, default="GRU",
                        choices=["GRU", "Transformer"],
                        help="Model architecture (default: GRU)")
    parser.add_argument("--strat", type=str, default="supervised_only",
                        choices=["supervised_only", "pimodel", "bpall_cringe", "pimodel_bpall_cringe"],
                        help="Training strategy (default: supervised_only)")

    # Training
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=128,
                        help="Test/val batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Learning rate (default: 0.0003)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--proportion-labelled", type=float, default=1.0,
                        help="Proportion of labelled data (default: 1.0 for supervised, use 0.1-0.5 for semi-supervised)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience in val epochs (default: 30)")
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every N epochs (default: 5)")
    parser.add_argument("--max-decode-len", type=int, default=30,
                        help="Max decode length for PIE forms (default: 30)")

    # Device
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU index to use (-1 = auto)")

    # Misc
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 = random)")
    parser.add_argument("--name", type=str, default="",
                        help="Run name")
    parser.add_argument("--dev", action="store_true",
                        help="Development mode (verbose, check val every epoch)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without executing")

    args = parser.parse_args()

    ensure_repo_ready(setup_if_missing=args.setup_repo, repo_url=args.repo_url)

    if args.no_convert and args.force_reconvert:
        print("ERROR: --no-convert and --force-reconvert are mutually exclusive.")
        sys.exit(1)

    # Auto-adjust for semi-supervised strategies
    if args.strat != "supervised_only" and args.proportion_labelled == 1.0:
        print(f"NOTE: Strategy '{args.strat}' is semi-supervised.")
        print(f"  Setting --proportion-labelled to 0.3 (override with --proportion-labelled)")
        args.proportion_labelled = 0.3

    print(f"{'='*60}")
    print(f"DPD Training for PIE Reconstruction")
    print(f"{'='*60}")
    print(f"  Split:        {args.split}")
    print(f"  Architecture: {args.arch}")
    print(f"  Strategy:     {args.strat}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Labelled:     {args.proportion_labelled}")
    print(f"  Device:       {'CPU' if args.cpu else f'GPU {args.gpu}' if args.gpu >= 0 else 'auto'}")

    # Step 1: Convert data
    if not args.no_convert:
        dpd_dataset_name = convert_data(args.split, force_reconvert=args.force_reconvert)
    else:
        dpd_dataset_name = f"pie_{args.split}"
        data_dir = REPO_DIR / "data" / dpd_dataset_name
        expected_files = ["train.pickle", "dev.pickle", "test.pickle"]
        if not data_dir.exists() or not all((data_dir / f).exists() for f in expected_files):
            print(f"ERROR: Data dir not found: {data_dir}")
            print(f"  Required files: {expected_files}")
            print("  Run without --no-convert to convert the data first.")
            sys.exit(1)

    # Step 2: Build and run training command
    exp_args = build_exp_args(args, dpd_dataset_name)
    ckpt_dir = build_checkpoint_dir(args, dpd_dataset_name)
    run_env = os.environ.copy()
    run_env["DPD_CHECKPOINT_DIR"] = str(ckpt_dir)

    print(f"\n{'='*60}")
    print(f"Training command:")
    print(f"{'='*60}")
    print(" ".join(exp_args))
    print(f"\nRun checkpoint dir:\n  {ckpt_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Not executing.")
        return

    print(f"\nCheckpoints will be saved to: {ckpt_dir}")
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")

    result = subprocess.run(exp_args, cwd=str(REPO_DIR), env=run_env)

    if result.returncode != 0:
        print(f"\nTraining exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Step 3: Report results
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"\nCheckpoint directory: {ckpt_dir}")

    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if ckpt_files:
            print(f"Saved checkpoints:")
            for f in sorted(ckpt_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")
        else:
            print("WARNING: No checkpoint files found!")

        config_file = ckpt_dir / "config.pkl"
        if config_file.exists():
            print(f"Config: {config_file}")

    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --checkpoint-dir {ckpt_dir}")


if __name__ == "__main__":
    main()
