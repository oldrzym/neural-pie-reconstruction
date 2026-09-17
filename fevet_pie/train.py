#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prepared(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_fevet_split(entries: List[dict]) -> Dict[str, List[dict]]:
    return {
        "data": [e["inputs"] for e in entries],
        "solns": [e["target"] for e in entries],
    }


def parse_metrics(raw: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    per_lang: Dict[str, Dict[str, float]] = {}
    for k, v in raw.items():
        if k in {"Avg ED", "Avg NED", "B^3 F1"}:
            out[k] = float(v)
            continue
        if isinstance(v, str) and "\t" in v:
            ed, ned, b3 = v.split("\t")
            per_lang[k] = {"ED": float(ed), "NED": float(ned), "B3_F1": float(b3)}
    out["per_language"] = per_lang
    return out


def import_fevet_modules(fevet_repo: Path):
    code_dir = fevet_repo / "code"
    pkg = types.ModuleType("code")
    pkg.__path__ = [str(code_dir)]
    sys.modules["code"] = pkg
    sys.path.insert(0, str(fevet_repo))

    try:
        data_prep = importlib.import_module("code.data_prep")
        metrics = importlib.import_module("code.metrics")
        model_mod = importlib.import_module("code.model")
        tokenizer_mod = importlib.import_module("code.tokenizer")
        training_utils = importlib.import_module("code.training_utils")
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise RuntimeError(
            "Failed to import FeVeT dependencies. Missing package: "
            f"{missing}. Install FeVeT requirements first: "
            f"`pip install -r {fevet_repo / 'requirements.txt'}`"
        ) from e
    return data_prep, metrics, model_mod, tokenizer_mod, training_utils


def decode_batch_predictions(
    model,
    dataloader: DataLoader,
    entries: List[dict],
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    translate_to_string,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for langs, valids, target_langs, target, _ in dataloader:
            langs = langs.to(device)
            valids = valids.to(device)
            target_langs = target_langs.to(device)
            preds = model.generate(langs, valids, target_langs).cpu().numpy()
            target_np = target.cpu().numpy()
            target_lang_np = target_langs.cpu().numpy()

            pred_s = translate_to_string(preds, char2idx, idx2char)
            gold_s = translate_to_string(target_np, char2idx, idx2char)
            lang_s = translate_to_string(target_lang_np, char2idx, idx2char)

            bs = len(pred_s)
            for i in range(bs):
                src = entries[cursor + i]
                rows.append(
                    {
                        "id": src["id"],
                        "target_language": lang_s[i],
                        "target": gold_s[i],
                        "prediction": pred_s[i],
                    }
                )
            cursor += bs
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FeVeT model on prepared PIE data.")
    parser.add_argument(
        "--prepared-data",
        default="fevet_pie/data/pie_iecor_prepared.json",
    )
    parser.add_argument(
        "--fevet-repo",
        default="tmp/FeVeT_repo",
        help="Path to FeVeT repository root (must contain code/, data/).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--feat-vec-dim", type=int, default=39)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers-enc", type=int, default=1)
    parser.add_argument("--nlayers-dec", type=int, default=2)
    parser.add_argument("--print-loss", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and exit before training.")
    args = parser.parse_args()

    prepared_path = Path(args.prepared_data).resolve()
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_path}")

    fevet_repo = Path(args.fevet_repo).resolve()
    if not (fevet_repo / "code").exists():
        raise FileNotFoundError(f"FeVeT repo not found or invalid: {fevet_repo}")
    if not (fevet_repo / "data" / "clts").exists():
        raise FileNotFoundError(
            f"Missing CLTS data in FeVeT repo (expected {fevet_repo / 'data' / 'clts'})"
        )

    run_name = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (Path("fevet_pie") / "runs" / run_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    payload = load_prepared(prepared_path)
    train_entries = payload["splits"]["train"]
    val_entries = payload["splits"]["val"]
    test_entries = payload["splits"]["test"]

    train_split = to_fevet_split(train_entries)
    val_split = to_fevet_split(val_entries)
    test_split = to_fevet_split(test_entries)

    original_cwd = Path.cwd()
    os.chdir(fevet_repo)
    try:
        data_prep, metrics_mod, model_mod, tokenizer_mod, training_utils = import_fevet_modules(
            fevet_repo
        )
    finally:
        os.chdir(original_cwd)
    ProtoDataset = data_prep.ProtoDataset
    collate_fn = data_prep.collate_fn
    get_vocab = data_prep.get_vocab
    compute_metrics = metrics_mod.compute_metrics
    translate_to_string = metrics_mod.translate_to_string
    CognateS2S = model_mod.CognateS2S
    tokenizer = tokenizer_mod.tokenizer
    training_proto_finetune = training_utils.training_proto_finetune

    vocab_inputs, langs_inputs = get_vocab(train_split, key="data")
    vocab_targets, langs_targets = get_vocab(train_split, key="solns")
    vocab = sorted(set(vocab_inputs).union(vocab_targets))
    langs = sorted(set(langs_inputs).union(langs_targets))

    char2featvec, char2idx = tokenizer(vocab, langs)
    idx2char = {v: k for k, v in char2idx.items()}

    train_dataset = ProtoDataset(train_split, char2featvec, char2idx)
    val_dataset = ProtoDataset(val_split, char2featvec, char2idx)
    test_dataset = ProtoDataset(test_split, char2featvec, char2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = CognateS2S(
        char2idx=char2idx,
        feat_dim=args.feat_vec_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.nhead,
        num_encoder_layers=args.nlayers_enc,
        num_decoder_layers=args.nlayers_dec,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.dry_run:
        print("Dry run complete. Training was not started.")
        print(f"Prepared data: {prepared_path}")
        print(f"FeVeT repo: {fevet_repo}")
        print(f"Output dir: {output_dir}")
        print(
            f"rows(train/val/test)={len(train_entries)}/{len(val_entries)}/{len(test_entries)}, "
            f"vocab={len(char2idx)}, device={device}"
        )
        return

    model = training_proto_finetune(
        model=model,
        train_dataloader=train_loader,
        dev_dataloader=val_loader,
        optimizer=optimizer,
        n_epochs=args.epochs,
        save_name="unused",
        save=False,
        print_loss=args.print_loss,
    )

    raw_val_metrics = compute_metrics(model, val_loader, char2idx, idx2char)
    raw_test_metrics = compute_metrics(model, test_loader, char2idx, idx2char)
    val_metrics = parse_metrics(raw_val_metrics)
    test_metrics = parse_metrics(raw_test_metrics)

    val_preds = decode_batch_predictions(
        model=model,
        dataloader=val_loader,
        entries=val_entries,
        char2idx=char2idx,
        idx2char=idx2char,
        translate_to_string=translate_to_string,
        device=device,
    )
    test_preds = decode_batch_predictions(
        model=model,
        dataloader=test_loader,
        entries=test_entries,
        char2idx=char2idx,
        idx2char=idx2char,
        translate_to_string=translate_to_string,
        device=device,
    )

    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump((char2featvec, char2idx), f)
    (output_dir / "val_metrics.json").write_text(
        json.dumps(val_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    val_preds.to_csv(output_dir / "val_predictions.csv", index=False)
    test_preds.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "prepared_data": str(prepared_path),
                "fevet_repo": str(fevet_repo),
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
                "feat_vec_dim": args.feat_vec_dim,
                "hidden_dim": args.hidden_dim,
                "nhead": args.nhead,
                "nlayers_enc": args.nlayers_enc,
                "nlayers_dec": args.nlayers_dec,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training complete.")
    print(f"Output dir: {output_dir}")
    print("Validation metrics:")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
