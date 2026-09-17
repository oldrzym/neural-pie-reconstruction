"""
Evaluation script for PIE reconstruction models.

Usage:
    python evaluate.py --model ./outputs --test ../dataset/splits/iecor/test.csv
"""

import argparse
import csv
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional


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


def infer_tokenizer_fallback(model_path: str) -> Optional[str]:
    """
    Infer a safe tokenizer source for checkpoints whose local tokenizer config
    is incompatible with the currently installed transformers version.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return None

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_type = cfg.get("model_type")
    vocab_size = int(cfg.get("vocab_size", -1))

    # ByT5-family checkpoints commonly have vocab=384.
    if model_type == "t5" and vocab_size == 384:
        return "google/byt5-base"
    return None


def evaluate_model(
    model_path: str,
    test_file: str,
    output_file: str = None,
    tokenizer_path: str = None,
    batch_size: int = 8,
    max_input_length: int = 512,
    max_output_length: int = 64,
    num_beams: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate a trained model on test set."""

    print(f"Loading model from {model_path}")
    tokenizer_source = tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception as e:
        fallback = infer_tokenizer_fallback(model_path)
        if fallback:
            print(
                f"Tokenizer load from '{tokenizer_source}' failed ({type(e).__name__}: {e}). "
                f"Falling back to '{fallback}'."
            )
            tokenizer = AutoTokenizer.from_pretrained(fallback)
        else:
            raise
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loading test data from {test_file}")
    df = pd.read_csv(test_file)

    # Run inference
    predictions = []
    print("Generating predictions...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        inputs = batch["input"].tolist()

        # Tokenize
        encoded = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_output_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    # Calculate metrics
    results = []
    exact_matches = 0
    total_edit_distance = 0
    total_chars = 0
    char_correct = 0

    for idx, (pred, row) in enumerate(zip(predictions, df.itertuples())):
        target = str(row.output)
        pred = pred.strip()
        target = target.strip()

        # Exact match
        is_exact = pred == target
        if is_exact:
            exact_matches += 1

        # Edit distance
        edit_dist = levenshtein_distance(pred, target)
        total_edit_distance += edit_dist

        # Character accuracy
        min_len = min(len(pred), len(target))
        char_correct += sum(p == t for p, t in zip(pred[:min_len], target[:min_len]))
        total_chars += max(len(pred), len(target))

        results.append({
            "input": row.input,
            "target": target,
            "prediction": pred,
            "exact_match": is_exact,
            "edit_distance": edit_dist,
            "normalized_edit_distance": edit_dist / max(len(pred), len(target), 1),
        })

    # Summary metrics
    metrics = {
        "exact_match_accuracy": exact_matches / len(df),
        "character_accuracy": char_correct / total_chars if total_chars > 0 else 0,
        "mean_edit_distance": total_edit_distance / len(df),
        "mean_normalized_edit_distance": sum(r["normalized_edit_distance"] for r in results) / len(results),
        "total_examples": len(df),
        "exact_matches": exact_matches,
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({exact_matches}/{len(df)})")
    print(f"Character Accuracy:   {metrics['character_accuracy']:.4f}")
    print(f"Mean Edit Distance:   {metrics['mean_edit_distance']:.2f}")
    print(f"Mean Norm. Edit Dist: {metrics['mean_normalized_edit_distance']:.4f}")
    print("=" * 50)

    # Show some examples
    print("\nExample predictions:")
    print("-" * 50)
    for i in range(min(10, len(results))):
        r = results[i]
        status = "✓" if r["exact_match"] else "✗"
        print(f"{status} Input:  {r['input'][:60]}...")
        print(f"  Target: {r['target']}")
        print(f"  Pred:   {r['prediction']}")
        print()

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            output_path.with_suffix(".csv"),
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
        )

        # Save metrics
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PIE reconstruction model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer source path/model id")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_model(
        model_path=args.model,
        test_file=args.test,
        output_file=args.output,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        device=device,
    )


if __name__ == "__main__":
    main()
