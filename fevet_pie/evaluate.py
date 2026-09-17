#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader


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
    except ModuleNotFoundError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise RuntimeError(
            "Failed to import FeVeT dependencies. Missing package: "
            f"{missing}. Install FeVeT requirements first: "
            f"`pip install -r {fevet_repo / 'requirements.txt'}`"
        ) from e
    return data_prep, metrics, model_mod


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
    parser = argparse.ArgumentParser(description="Evaluate FeVeT model on prepared PIE split.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--prepared-data",
        default="fevet_pie/data/pie_iecor_prepared.json",
    )
    parser.add_argument(
        "--fevet-repo",
        default="tmp/FeVeT_repo",
        help="Path to FeVeT repository root (must contain code/, data/).",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and exit before evaluation.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    model_path = ckpt_dir / "model.pt"
    vocab_path = ckpt_dir / "vocab.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.pkl not found: {vocab_path}")

    prepared_path = Path(args.prepared_data).resolve()
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_path}")

    fevet_repo = Path(args.fevet_repo).resolve()
    if not (fevet_repo / "code").exists():
        raise FileNotFoundError(f"FeVeT repo not found or invalid: {fevet_repo}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (ckpt_dir / f"eval_{args.split}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_prepared(prepared_path)
    entries = payload["splits"][args.split]
    split_data = to_fevet_split(entries)

    with vocab_path.open("rb") as f:
        char2featvec, char2idx = pickle.load(f)
    idx2char = {v: k for k, v in char2idx.items()}
    train_cfg = {}
    train_cfg_path = ckpt_dir / "train_config.json"
    if train_cfg_path.exists():
        train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))

    original_cwd = Path.cwd()
    os.chdir(fevet_repo)
    try:
        data_prep, metrics_mod, model_mod = import_fevet_modules(fevet_repo)
    finally:
        os.chdir(original_cwd)
    ProtoDataset = data_prep.ProtoDataset
    collate_fn = data_prep.collate_fn
    compute_metrics = metrics_mod.compute_metrics
    translate_to_string = metrics_mod.translate_to_string
    CognateS2S = model_mod.CognateS2S

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = ProtoDataset(split_data, char2featvec, char2idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    model = CognateS2S(
        char2idx=char2idx,
        feat_dim=int(train_cfg.get("feat_vec_dim", 39)),
        hidden_dim=int(train_cfg.get("hidden_dim", 256)),
        num_heads=int(train_cfg.get("nhead", 4)),
        num_encoder_layers=int(train_cfg.get("nlayers_enc", 1)),
        num_decoder_layers=int(train_cfg.get("nlayers_dec", 2)),
        dropout=float(train_cfg.get("dropout", 0.1)),
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    if args.dry_run:
        print("Dry run complete. Evaluation was not started.")
        print(f"Checkpoint: {ckpt_dir}")
        print(f"Prepared data: {prepared_path}")
        print(f"Split rows: {len(entries)}")
        print(f"Device: {device}")
        print(f"Output dir: {output_dir}")
        return

    raw_metrics = compute_metrics(model, loader, char2idx, idx2char)
    metrics = parse_metrics(raw_metrics)
    preds = decode_batch_predictions(
        model=model,
        dataloader=loader,
        entries=entries,
        char2idx=char2idx,
        idx2char=idx2char,
        translate_to_string=translate_to_string,
        device=device,
    )

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preds.to_csv(output_dir / "predictions.csv", index=False)

    print("Evaluation complete.")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
