"""
Inference script for PIE reconstruction.

Usage:
    python inference.py --model ./outputs --input "[lat] canis [grc] kýōn [san] śvā́"

    # Interactive mode
    python inference.py --model ./outputs --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_path: str, device: str = None):
    """Load trained model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    return model, tokenizer, device


def predict(
    model,
    tokenizer,
    input_text: str,
    device: str,
    max_length: int = 64,
    num_beams: int = 4,
    num_return_sequences: int = 1,
) -> list:
    """Generate PIE reconstruction for input cognates."""

    # Tokenize
    encoded = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )

    # Decode
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return predictions


def interactive_mode(model, tokenizer, device):
    """Interactive mode for testing."""
    print("\n" + "=" * 50)
    print("PIE Reconstruction - Interactive Mode")
    print("=" * 50)
    print("Enter cognates in format: [lang] word [lang] word ...")
    print("Example: [lat] canis [grc] kýōn [san] śvā́")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    while True:
        try:
            input_text = input("Input: ").strip()
        except EOFError:
            break

        if input_text.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not input_text:
            continue

        predictions = predict(
            model, tokenizer, input_text, device,
            num_beams=4, num_return_sequences=3,
        )

        print(f"Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred}")
        print()


def main():
    parser = argparse.ArgumentParser(description="PIE reconstruction inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input", type=str, default=None, help="Input cognates string")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams")
    parser.add_argument("--top-k", type=int, default=1, help="Number of predictions to return")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model, tokenizer, device = load_model(args.model, args.device)
    print(f"Model loaded on {device}")

    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.input:
        predictions = predict(
            model, tokenizer, args.input, device,
            num_beams=args.num_beams,
            num_return_sequences=args.top_k,
        )
        print(f"\nInput: {args.input}")
        print(f"Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred}")
    else:
        parser.print_help()
        print("\nError: Provide --input or use --interactive mode")


if __name__ == "__main__":
    main()
