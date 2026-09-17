"""Poll and download results from an existing OpenAI Batch."""
import sys
import csv
import json
import time
import re
import os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# === CONFIGURATION ===
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "splits" / "paper_pipeline"
BASE_DIR = Path(os.environ.get("PIE_DATA_DIR", DEFAULT_DATA_DIR))
ORIGINAL_FILE = BASE_DIR / "train.csv"
ENRICHED_FILE = BASE_DIR / "train_synt.csv"  # Use train_synt.csv (larger file, 1016K)
CHUNKS_DIR = BASE_DIR / "validated_chunks"

# Reads OPENAI_API_KEY from the environment.
client = OpenAI()

# Get START_ROW and CHUNK_SIZE from batch ID or use defaults
START_ROW = 700  # IMPORTANT: Must match the batch being processed!
CHUNK_SIZE = 1000

def get_chunk_files(start, size):
    """Get file paths for a chunk."""
    end = start + size if size else "end"
    chunk_name = f"chunk_{start}_{end}"
    return {
        "output": CHUNKS_DIR / f"{chunk_name}.csv",
        "batch_requests": CHUNKS_DIR / f"{chunk_name}_requests.jsonl",
        "batch_results": CHUNKS_DIR / f"{chunk_name}_results.jsonl",
    }

def extract_languages(cognate_string: str) -> set[str]:
    return set(re.findall(r'\[([a-z]{3})\]', cognate_string.lower()))

def extract_added_cognates(original_string: str, enriched_string: str) -> list[tuple[str, str]]:
    original_langs = extract_languages(original_string)
    pattern = r'\[([a-z]{3})\]\s*([^\[\]]+?)(?=\s*\[|$)'
    enriched_pairs = re.findall(pattern, enriched_string, re.IGNORECASE)
    added = []
    for lang, form in enriched_pairs:
        if lang.lower() not in original_langs:
            added.append((lang.lower(), form.strip()))
    return added

def load_and_compare():
    """Load files and compare to find added cognates."""
    print("Loading files...")

    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        original_rows = list(reader)

    with open(ENRICHED_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        enriched_rows = list(reader)

    total_rows = min(len(original_rows), len(enriched_rows))
    start = START_ROW or 0
    end = min(start + CHUNK_SIZE, total_rows) if CHUNK_SIZE else total_rows

    # Find added cognates per row
    row_data = []
    for i in range(start, end):
        orig = original_rows[i]
        enr = enriched_rows[i]
        added = extract_added_cognates(orig[0] if orig else "", enr[0] if enr else "")
        row_data.append({
            "idx": i,
            "original_row": orig,
            "enriched_row": enr,
            "pie_form": enr[1] if len(enr) > 1 else "",
            "meaning": enr[2] if len(enr) > 2 else "",
            "added_cognates": added,
        })

    print(f"Loaded {end - start} rows (rows {start}-{end-1})")
    return header, row_data

def parse_validation_response(response_text: str, added_cognates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Parse GPT validation response."""
    valid_cognates = []
    added_dict = {lang: form for lang, form in added_cognates}

    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'\[([a-z]{3})\]\s*(.+?)\s*-\s*(VALID|INVALID)', line, re.IGNORECASE)
        if match:
            lang = match.group(1).lower()
            status = match.group(3).upper()
            if status == 'VALID' and lang in added_dict:
                valid_cognates.append((lang, added_dict[lang]))

    return valid_cognates

def download_and_assemble(batch, header: list, row_data: list):
    """Download batch results and assemble validated CSV."""
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("Error: No output file in batch!")
        return

    print(f"Downloading results from {output_file_id}...")
    content = client.files.content(output_file_id)
    result_text = content.text

    # Save raw results
    BATCH_RESULTS_JSONL = get_chunk_files(START_ROW, CHUNK_SIZE)["batch_results"]
    with open(BATCH_RESULTS_JSONL, 'w', encoding='utf-8') as f:
        f.write(result_text)
    print(f"Saved raw results to {BATCH_RESULTS_JSONL}")

    # Parse results into {row_idx: response_text}
    results_map = {}
    for line in result_text.strip().split('\n'):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result["custom_id"]  # "row-123"
        row_idx = int(custom_id.split("-")[1])

        resp_body = result.get("response", {}).get("body", {})
        # Extract output text from Responses API format
        output_text = ""
        for item in resp_body.get("output", []):
            if item.get("type") == "message":
                for content_block in item.get("content", []):
                    if content_block.get("type") == "output_text":
                        output_text += content_block.get("text", "")

        if result.get("response", {}).get("status_code") != 200:
            error = result.get("error", {})
            print(f"Warning: Row {row_idx} API error - {error}")

        results_map[row_idx] = output_text

    print(f"Parsed {len(results_map)} results")

    # Assemble validated CSV
    OUTPUT_FILE = get_chunk_files(START_ROW, CHUNK_SIZE)["output"]
    stats = {"total_checked": 0, "valid": 0, "invalid": 0, "skipped": 0}

    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in tqdm(row_data, desc="Assembling CSV"):
            idx = row["idx"]
            original_string = row["original_row"][0] if row["original_row"] else ""
            enriched_row = row["enriched_row"].copy()
            added = row["added_cognates"]

            if not added:
                stats["skipped"] += 1
                writer.writerow(enriched_row)
                continue

            stats["total_checked"] += len(added)

            response_text = results_map.get(idx, "")
            if response_text:
                valid_added = parse_validation_response(response_text, added)
                stats["valid"] += len(valid_added)
                stats["invalid"] += len(added) - len(valid_added)

                if valid_added:
                    valid_string = " ".join(f"[{lang}] {form}" for lang, form in valid_added)
                    enriched_row[0] = original_string.strip() + " " + valid_string
                else:
                    enriched_row[0] = original_string.strip()
            else:
                # No response — keep original only
                enriched_row[0] = original_string.strip()
                stats["invalid"] += len(added)

            writer.writerow(enriched_row)

    print("\n" + "=" * 60)
    print("VALIDATION STATS")
    print("=" * 60)
    print(f"Total checked:  {stats['total_checked']}")
    print(f"Valid:          {stats['valid']}")
    print(f"Invalid:        {stats['invalid']}")
    print(f"Skipped (0 new):{stats['skipped']}")
    if stats["total_checked"] > 0:
        print(f"Validation rate: {stats['valid'] / stats['total_checked'] * 100:.1f}%")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

def poll_and_download(batch_id: str, poll_interval: int = 5):
    """Poll batch status and download when complete."""
    print(f"Polling batch: {batch_id}")
    print(f"Checking every {poll_interval} seconds...\n")

    pbar = tqdm(desc="Batch progress", total=100, unit="%")
    last_pct = 0

    while True:
        batch = client.batches.retrieve(batch_id)
        completed = (batch.request_counts.completed or 0)
        failed = (batch.request_counts.failed or 0)
        total = (batch.request_counts.total or 1)
        pct = int((completed + failed) / total * 100)

        pbar.update(pct - last_pct)
        last_pct = pct

        print(f"\rStatus: {batch.status} | {completed}/{total} done, {failed} failed", end="")

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            pbar.update(100 - last_pct)
            pbar.close()
            print()
            break

        time.sleep(poll_interval)

    print("\n" + "=" * 60)
    if batch.status == "completed":
        print(f"[OK] Batch completed! {completed} succeeded, {failed} failed")
        print("=" * 60 + "\n")

        # Download and assemble results
        print("Loading row data for assembly...")
        header, row_data = load_and_compare()
        download_and_assemble(batch, header, row_data)

        print("\n[OK] Done! Results saved to validated_chunks/")

    else:
        print(f"[X] Batch ended with status: {batch.status}")
        if batch.errors:
            print("\nErrors:")
            for err in batch.errors.data[:5]:
                print(f"  {err}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_batch_status.py <batch_id>")
        print("\nExample:")
        print("  python check_batch_status.py batch_your_batch_id")
        print("\nThis script will:")
        print("  1. Poll the batch every 5 seconds")
        print("  2. Show progress")
        print("  3. Auto-download results when complete")
        print("  4. Save to validated_chunks/chunk_0_100.csv")
        sys.exit(1)

    batch_id = sys.argv[1]
    poll_and_download(batch_id, poll_interval=5)
