"""Show statistics for all validated chunks."""
import csv
import json
from pathlib import Path
import re

BASE_DIR = Path("dataset/splits/iecor_kaikki_koebler_normalized")
CHUNKS_DIR = BASE_DIR / "validated_chunks"

def extract_start_row(filename):
    """Extract start row number from chunk filename."""
    match = re.search(r'chunk_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

def parse_cognates(cognate_str):
    """Count cognates in a string."""
    pattern = r'\[([a-z]{3})\]'
    return len(re.findall(pattern, cognate_str))

def main():
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.csv"), key=lambda p: extract_start_row(p.name))

    if not chunk_files:
        print("No chunk files found in validated_chunks/")
        return

    print(f"\n{'='*100}")
    print(f"VALIDATED CHUNKS STATISTICS")
    print('='*100 + "\n")

    total_rows = 0
    total_input_cognates = 0
    total_output_cognates = 0
    total_cost = 0.0

    for chunk_file in chunk_files:
        chunk_name = chunk_file.stem
        results_file = CHUNKS_DIR / f"{chunk_name}_results.jsonl"

        # Read CSV
        with open(chunk_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Count cognates
        input_cogs = sum(parse_cognates(row.get('input', '')) for row in rows)
        output_cogs = sum(parse_cognates(row.get('output', '')) for row in rows)
        validation_rate = (output_cogs / input_cogs * 100) if input_cogs > 0 else 0

        # Get cost from results
        cost = 0.0
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    usage = result.get('response', {}).get('body', {}).get('usage', {})
                    total_tokens = usage.get('total_tokens', 0)
                    # GPT-5.2: $0.20/1M input, $0.40/1M output (rough average $0.30/1M)
                    cost += total_tokens * 0.30 / 1_000_000

        total_rows += len(rows)
        total_input_cognates += input_cogs
        total_output_cognates += output_cogs
        total_cost += cost

        print(f"{chunk_name}:")
        print(f"  Rows: {len(rows)}")
        print(f"  Input cognates: {input_cogs}")
        print(f"  Output cognates: {output_cogs}")
        print(f"  Validation rate: {validation_rate:.1f}%")
        print(f"  Cost: ${cost:.2f}")
        print()

    print('='*100)
    print(f"TOTAL:")
    print(f"  Rows: {total_rows}")
    print(f"  Input cognates: {total_input_cognates}")
    print(f"  Output cognates: {total_output_cognates}")
    print(f"  Overall validation rate: {(total_output_cognates/total_input_cognates*100):.1f}%")
    print(f"  Total cost: ${total_cost:.2f}")
    print('='*100 + "\n")

    # Show completion
    total_needed = 2880
    progress = total_rows / total_needed * 100
    print(f"Progress: {total_rows}/{total_needed} rows ({progress:.1f}%)")
    print(f"Remaining: {total_needed - total_rows} rows")
    print(f"Estimated remaining cost: ${(total_needed - total_rows) / max(1, total_rows) * total_cost:.2f}")

if __name__ == "__main__":
    main()
