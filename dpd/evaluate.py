"""
Evaluate a trained DPD checkpoint on test data.

Usage:
    python evaluate.py --checkpoint-dir repo/checkpoints/pie_iecor_kaikki_koebler_supervised_only_GRU

    # Use specific checkpoint (default: best)
    python evaluate.py --checkpoint-dir <path> --ckpt last.ckpt

    # Save predictions to CSV
    python evaluate.py --checkpoint-dir <path> --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import os
from pathlib import Path

# Add repo to path so we can import DPD modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR / "repo"
sys.path.insert(0, str(REPO_DIR))

import logging
logging.getLogger('lingpy').setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings(action="ignore", message=".*num_workers.*")
warnings.filterwarnings(action="ignore", message=".*negatively affect performance.*")
warnings.filterwarnings(action="ignore", message=".*MPS available.*")

# Lazy-initialized in main() so `--help` works without full ML stack
torch = None
pl = None
DataloaderManager = None
biDirReconModelRNN = None
biDirReconModelTrans = None
models = None


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_checkpoint(checkpoint_dir: Path, ckpt_name: str = None) -> Path:
    """Find the checkpoint file to load."""
    if ckpt_name:
        ckpt_path = checkpoint_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        return ckpt_path

    # Try best checkpoint first (any file matching best-*)
    best_ckpts = list(checkpoint_dir.glob("best-*.ckpt"))
    if best_ckpts:
        # Pick the one with lowest edit distance in filename
        return sorted(best_ckpts)[0]

    # Try last.ckpt
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    # Any .ckpt file
    all_ckpts = list(checkpoint_dir.glob("*.ckpt"))
    if all_ckpts:
        return sorted(all_ckpts)[-1]

    print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
    print(f"  Available files: {list(checkpoint_dir.iterdir())}")
    sys.exit(1)


def load_config(checkpoint_dir: Path) -> dict:
    """Load config dict from pickle."""
    config_path = checkpoint_dir / "config.pkl"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        print("  Config should be saved alongside checkpoints during training.")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Nullify checkpoint/early stopping callbacks (not needed for eval)
    config["strategy_config"]["early_stopping_method"] = None
    config["strategy_config"]["strategy_checkpoint_method"] = None

    return config


def load_model(checkpoint_path: Path, config: dict, dm: DataloaderManager):
    """Load model from checkpoint."""
    c = AttributeDict(config)
    device = torch.device('cpu')

    if c.architecture == 'GRU':
        model = biDirReconModelRNN.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            map_location=device,
            ipa_vocab=dm.ipa_vocab,
            lang_vocab=dm.lang_vocab,
            has_p2d=c.strategy_config['has_p2d'],

            d2p_num_encoder_layers=c.d2p_num_encoder_layers,
            d2p_dropout_p=c.d2p_dropout_p,
            d2p_use_vae_latent=c.d2p_use_vae_latent,
            d2p_inference_decode_max_length=c.d2p_inference_decode_max_length,
            d2p_use_bidirectional_encoder=c.d2p_use_bidirectional_encoder,
            d2p_decode_mode=c.d2p_decode_mode,
            d2p_beam_search_alpha=c.d2p_beam_search_alpha,
            d2p_beam_size=c.d2p_beam_size,
            d2p_lang_embedding_when_decoder=c.d2p_lang_embedding_when_decoder,

            p2d_num_encoder_layers=c.p2d_num_encoder_layers,
            p2d_dropout_p=c.p2d_dropout_p,
            p2d_use_vae_latent=c.p2d_use_vae_latent,
            p2d_inference_decode_max_length=c.p2d_inference_decode_max_length,
            p2d_use_bidirectional_encoder=c.p2d_use_bidirectional_encoder,
            p2d_decode_mode=c.p2d_decode_mode,
            p2d_beam_search_alpha=c.p2d_beam_search_alpha,
            p2d_beam_size=c.p2d_beam_size,
            p2d_lang_embedding_when_decoder=c.p2d_lang_embedding_when_decoder,
            p2d_prompt_mlp_with_one_hot_lang=c.p2d_prompt_mlp_with_one_hot_lang,
            p2d_gated_mlp_by_target_lang=c.p2d_gated_mlp_by_target_lang,
            p2d_all_lang_summary_only=True,

            d2p_feedforward_dim=c.d2p_feedforward_dim,
            d2p_embedding_dim=c.d2p_embedding_dim,
            d2p_model_size=c.d2p_model_size,

            p2d_feedforward_dim=c.p2d_feedforward_dim,
            p2d_embedding_dim=c.p2d_embedding_dim,
            p2d_model_size=c.p2d_model_size,

            use_xavier_init=True,
            lr=c.lr,
            max_epochs=c.max_epochs,
            warmup_epochs=c.warmup_epochs,
            beta1=c.beta1,
            beta2=c.beta2,
            eps=c.eps,
            weight_decay=c.weight_decay,

            universal_embedding=c.universal_embedding,
            universal_embedding_dim=c.universal_embedding_dim,

            strategy=getattr(models.biDirReconStrategies,
                             c.strategy_config['strategy_class_name'])(
                **c.strategy_config['strategy_kwargs']),
        )
    elif c.architecture == 'Transformer':
        model = biDirReconModelTrans.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            map_location=device,
            ipa_vocab=dm.ipa_vocab,
            lang_vocab=dm.lang_vocab,
            has_p2d=c.strategy_config['has_p2d'],

            d2p_num_encoder_layers=c.d2p_num_encoder_layers,
            d2p_num_decoder_layers=c.d2p_num_decoder_layers,
            d2p_nhead=c.d2p_nhead,
            d2p_dropout_p=c.d2p_dropout_p,
            d2p_inference_decode_max_length=c.d2p_inference_decode_max_length,
            d2p_max_len=c.d2p_max_len,
            d2p_feedforward_dim=c.d2p_feedforward_dim,
            d2p_embedding_dim=c.d2p_embedding_dim,

            p2d_num_encoder_layers=c.p2d_num_encoder_layers,
            p2d_num_decoder_layers=c.p2d_num_decoder_layers,
            p2d_nhead=c.p2d_nhead,
            p2d_dropout_p=c.p2d_dropout_p,
            p2d_inference_decode_max_length=c.p2d_inference_decode_max_length,
            p2d_max_len=c.p2d_max_len,
            p2d_feedforward_dim=c.p2d_feedforward_dim,
            p2d_embedding_dim=c.p2d_embedding_dim,
            p2d_all_lang_summary_only=c.p2d_all_lang_summary_only,

            use_xavier_init=True,
            lr=c.lr,
            max_epochs=c.max_epochs,
            warmup_epochs=c.warmup_epochs,
            beta1=c.beta1,
            beta2=c.beta2,
            eps=c.eps,
            weight_decay=c.weight_decay,

            universal_embedding=c.universal_embedding,
            universal_embedding_dim=c.universal_embedding_dim,

            strategy=getattr(models.biDirReconStrategies,
                             c.strategy_config['strategy_class_name'])(
                **c.strategy_config['strategy_kwargs']),
        )
    else:
        print(f"ERROR: Unknown architecture: {c.architecture}")
        sys.exit(1)

    model.eval()
    return model


def evaluate_d2p(model, dm: DataloaderManager, split: str = 'test'):
    """Run D2P evaluation and return metrics."""
    evaluator = pl.Trainer(
        accelerator='cpu',
        max_epochs=1,
        enable_progress_bar=True,
    )

    if split == 'test':
        loader = dm.test_dataloader()
    elif split == 'val':
        loader = dm.val_dataloader()
    elif split == 'train':
        loader = dm.train_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")

    results = evaluator.validate(model.d2p, dataloaders=loader, verbose=False)
    return results[0] if results else {}


def generate_predictions(model, dm: DataloaderManager, config: dict, split: str = 'test'):
    """Generate predictions for all samples in a split."""
    import models.utils as utils
    from einops import repeat

    c = AttributeDict(config)

    if split == 'test':
        dataset = dm.test_set
    elif split == 'val':
        dataset = dm.val_set
    elif split == 'train':
        dataset = dm.train_set
    else:
        raise ValueError(f"Unknown split: {split}")

    predictions = []

    for i in range(dataset.length):
        minimal_singleton_batch = dataset.collate_fn([dataset[i]])

        if c.architecture == 'GRU':
            source_tokens, source_langs, source_seqs_lens, target_tokens, _target_lang_ipa_ids, target_lang_lang_ids = model.d2p.unpack_batch(minimal_singleton_batch)
            N = 1
            target_lang_lang_ids = repeat(
                (torch.LongTensor([model.d2p.lang_vocab.get_idx(model.d2p.protolang)]).to(model.d2p.device)),
                '1 -> N 1', N=N
            )
            prediction = model.d2p.greedy_decode(
                source_tokens=source_tokens,
                source_langs=source_langs,
                source_seqs_lens=source_seqs_lens,
                target_langs=target_lang_lang_ids,
            )
        elif c.architecture == 'Transformer':
            N, s_tkns, s_langs, s_indv_lens, t_tkns, t_tkns_in, t_tkns_out, t_ipa_lang, t_lang_lang, s_mask, t_mask, s_pad_mask, t_pad_mask = utils.unpack_batch_for_transformer(
                minimal_singleton_batch, model.d2p.device, model.d2p.task,
                model.d2p.ipa_vocab, model.d2p.lang_vocab, model.d2p.protolang
            )
            prediction = model.d2p.greedy_decode(
                s_tkns, s_indv_lens, s_langs, s_mask, s_pad_mask,
                decode_max_len=model.d2p.inference_decode_max_length
            )

        pred_tokens = model.d2p.ipa_vocab.to_tokens(prediction[0])

        # Get gold protoform
        gold_data = dataset.Pl[i]
        gold_proto = gold_data.get("PIE", gold_data.get(list(gold_data.keys())[0], []))

        # Get daughter data
        daughter_data = dataset.D[i]

        predictions.append({
            "daughters": str(daughter_data),
            "gold_proto": "".join(gold_proto),
            "pred_proto": "".join(pred_tokens),
            "gold_tokens": gold_proto,
            "pred_tokens": pred_tokens,
        })

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPD checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing checkpoint and config.pkl")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Specific checkpoint filename (default: auto-detect best)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to evaluate on (default: test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for predictions")
    parser.add_argument("--metrics-output", type=str, default=None,
                        help="Optional JSON path for metrics")
    parser.add_argument("--predictions", action="store_true",
                        help="Generate per-sample predictions")

    args = parser.parse_args()

    # Heavy imports are delayed until after arg parsing so `--help` is lightweight.
    global torch, pl, DataloaderManager, biDirReconModelRNN, biDirReconModelTrans, models
    import torch as _torch
    import pytorch_lightning as _pl
    from lib.dataloader_manager import DataloaderManager as _DataloaderManager
    from models.biDirReconIntegration import biDirReconModelRNN as _biDirReconModelRNN, biDirReconModelTrans as _biDirReconModelTrans
    import models.biDirReconStrategies as _strategies

    torch = _torch
    pl = _pl
    DataloaderManager = _DataloaderManager
    biDirReconModelRNN = _biDirReconModelRNN
    biDirReconModelTrans = _biDirReconModelTrans
    models = type("ModelsHolder", (), {"biDirReconStrategies": _strategies})

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Load config
    print(f"Loading config from {checkpoint_dir / 'config.pkl'}...")
    config = load_config(checkpoint_dir)
    c = AttributeDict(config)

    print(f"\nModel config:")
    print(f"  Architecture: {c.architecture}")
    print(f"  Strategy:     {c.strat}")
    print(f"  Dataset:      {c.dataset}")
    print(f"  Epochs:       {c.max_epochs}")
    print(f"  LR:           {c.lr}")

    # Find checkpoint
    ckpt_path = find_checkpoint(checkpoint_dir, args.ckpt)
    print(f"\nLoading checkpoint: {ckpt_path}")
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    # Load data
    data_dir = REPO_DIR / "data" / c.dataset
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("  Run train.py first to convert the data.")
        sys.exit(1)

    print(f"\nLoading data from {data_dir}...")
    dm = DataloaderManager(
        data_dir=str(data_dir),
        batch_size=c.batch_size,
        test_val_batch_size=c.test_val_batch_size,
        shuffle_train=True,
        lang_separators=c.d2p_use_lang_separaters,
        skip_daughter_tone=c.skip_daughter_tone,
        skip_protoform_tone=c.skip_protoform_tone,
        include_lang_tkns_in_ipa_vocab=True,
        transformer_d2p_d_cat_style=c.transformer_d2p_d_cat_style,
        daughter_subset=None,
        min_daughters=c.min_daughters,
        verbose=False,
        proportion_labelled=c.proportion_labelled,
        datasetseed=c.datasetseed,
    )

    # Load model
    print("Loading model...")
    model = load_model(ckpt_path, config, dm)

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating on {args.split} set...")
    print(f"{'='*60}\n")

    metrics = evaluate_d2p(model, dm, args.split)

    print(f"\n{'='*60}")
    print(f"Results ({args.split}):")
    print(f"{'='*60}")

    metric_names = {
        'd2p/val/accuracy': 'Exact Match Accuracy',
        'd2p/val/phoneme_edit_distance': 'Phoneme Edit Distance',
        'd2p/val/phoneme_error_rate': 'Phoneme Error Rate (PER)',
        'd2p/val/char_edit_distance': 'Character Edit Distance',
        'd2p/val/feature_error_rate': 'Feature Error Rate',
        'd2p/val/bcubed_f_score': 'BCubed F-Score',
        'd2p/val/loss': 'Loss',
    }

    for key, name in metric_names.items():
        if key in metrics:
            val = metrics[key]
            print(f"  {name:30s}: {val:.4f}")

    # Save metrics JSON if requested
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to: {metrics_path}")

    # Generate predictions if requested
    if args.predictions or args.output:
        print(f"\nGenerating predictions...")
        predictions = generate_predictions(model, dm, config, args.split)

        # Print sample predictions
        print(f"\nSample predictions (first 20):")
        print(f"  {'Gold':<30s} {'Predicted':<30s} {'Match?'}")
        print(f"  {'-'*65}")
        for p in predictions[:20]:
            match = "OK" if p['gold_proto'] == p['pred_proto'] else ""
            print(f"  {p['gold_proto']:<30s} {p['pred_proto']:<30s} {match}")

        # Save to CSV if requested
        if args.output:
            import pandas as pd
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([
                {"gold": p["gold_proto"], "predicted": p["pred_proto"]}
                for p in predictions
            ])
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nPredictions saved to: {output_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
