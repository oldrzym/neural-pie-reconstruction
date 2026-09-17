"""Merge all validated chunks into a single file."""
import csv
from pathlib import Path
import re

BASE_DIR = Path("dataset/splits/iecor_kaikki_koebler_normalized")
CHUNKS_DIR = BASE_DIR / "validated_chunks"
OUTPUT_FILE = BASE_DIR / "train_wikt_validated_full.csv"

def extract_start_row(filename):
    """Extract start row number from chunk filename."""
    match = re.search(r'chunk_(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

def main():
    # Find all chunk CSV files
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.csv"), key=lambda p: extract_start_row(p.name))

    if not chunk_files:
        print("No chunk files found in validated_chunks/")
        return

    print(f"Found {len(chunk_files)} chunk files:")
    for f in chunk_files:
        print(f"  - {f.name}")

    # Read header from first file
    with open(chunk_files[0], 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

    # Merge all chunks
    total_rows = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                next(reader)  # Skip header
                chunk_rows = list(reader)
                for row in chunk_rows:
                    writer.writerow(row)
                total_rows += len(chunk_rows)
                print(f"  ✓ Merged {chunk_file.name}: {len(chunk_rows)} rows")

    print(f"\n✓ Merged {len(chunk_files)} chunks → {total_rows} total rows")
    print(f"✓ Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
