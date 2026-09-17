"""
Script to validate synthetic cognates by checking them against GPT.
Compares enriched file with original and removes false cognates.
"""

import csv
import re
import asyncio
import random
import os
from collections import defaultdict
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, RateLimitError

# === CONFIGURATION ===
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "splits" / "paper_pipeline"
DATA_DIR = Path(os.environ.get("PIE_DATA_DIR", DEFAULT_DATA_DIR))
ORIGINAL_FILE = DATA_DIR / "train.csv"
ENRICHED_FILE = DATA_DIR / "train_synt.csv"
OUTPUT_FILE = DATA_DIR / "train_synt_validated.csv"

# GPT model
GPT_MODEL = "gpt-5.2"
REASONING_EFFORT = "medium"  # need reasoning for accurate validation

# Async config
MAX_CONCURRENT = 3
BATCH_SIZE = 10
MAX_RETRIES = 10
RETRY_BASE_DELAY = 10
RETRY_MAX_DELAY = 300
REQUEST_DELAY = 5

# Test mode
TEST_MODE_LIMIT = 10  # Set to None for full dataset

# Reads OPENAI_API_KEY from the environment.
client = AsyncOpenAI()


def extract_languages(cognate_string: str) -> set[str]:
    """Extract set of language codes from cognate string."""
    pattern = r'\[([a-z]{3})\]'
    return set(re.findall(pattern, cognate_string.lower()))


def extract_added_cognates(original_string: str, enriched_string: str) -> list[tuple[str, str]]:
    """Find cognates that were added (in enriched but not in original)."""
    original_langs = extract_languages(original_string)

    # Extract all [lang] form pairs from enriched
    pattern = r'\[([a-z]{3})\]\s*([^\[\]]+?)(?=\s*\[|$)'
    enriched_pairs = re.findall(pattern, enriched_string, re.IGNORECASE)

    added = []
    for lang, form in enriched_pairs:
        if lang.lower() not in original_langs:
            added.append((lang.lower(), form.strip()))

    return added


def build_validation_prompt(pie_form: str, meaning: str, added_cognates: list[tuple[str, str]]) -> str:
    """Build prompt for GPT to validate cognates."""
    cognates_list = "\n".join(f"[{lang}] {form}" for lang, form in added_cognates)

    prompt = f"""You are an expert historical linguist specializing in Proto-Indo-European reconstruction.

TASK: Validate whether these words are TRUE COGNATES of the PIE root.

PIE ROOT: {pie_form}
MEANING: {meaning if meaning and meaning != 'nan' else 'unknown'}

CANDIDATES TO VALIDATE:
{cognates_list}

VALIDATION CRITERIA:
1. VALID = word is inherited from this PIE root through regular sound changes
2. INVALID = word has different etymology, is a loanword, or unrelated

EXAMPLES OF CORRECT VALIDATION:
- PIE *ped- "foot": [eng] foot VALID, [deu] Fuss VALID, [fra] pied VALID
- PIE *ped- "foot": [rus] noga INVALID (from *nogʷh- "nail/claw", not *ped-)
- PIE *(H)reidʰ- "ride": [eng] ride VALID, [deu] reiten VALID
- PIE *(H)reidʰ- "ride": [ita] correre INVALID (from *kers- "run"), [pol] rysowac INVALID (means "to draw")
- PIE *gʷet- "say": [eng] quoth VALID, [swe] kväda VALID
- PIE *gʷet- "say": [rus] gadat' VALID (if meaning "to guess/divine"), [pol] gadac VALID

GUIDELINES:
- If etymology is plausible and documented, mark VALID
- Only mark INVALID if clearly wrong etymology or unrelated meaning
- Germanic cognates often show Grimm's Law shifts (p→f, t→þ, k→h, etc.)
- Slavic cognates preserve many PIE features
- When uncertain but plausible, prefer VALID

OUTPUT FORMAT (exactly one per line):
[lang] form - VALID
[lang] form - INVALID

Validate:"""

    return prompt


def parse_validation_response(response: str, added_cognates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Parse GPT response and return only valid cognates."""
    valid_cognates = []
    added_dict = {lang: form for lang, form in added_cognates}

    for line in response.strip().split('\n'):
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


async def call_gpt_api(prompt: str, row_info: str, semaphore: asyncio.Semaphore) -> str:
    """Call GPT API with retry logic."""
    async with semaphore:
        await asyncio.sleep(random.uniform(0, REQUEST_DELAY))

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.responses.create(
                    model=GPT_MODEL,
                    instructions="You are an expert comparative linguist. Validate cognates accurately based on established etymology.",
                    input=prompt,
                    reasoning={"effort": REASONING_EFFORT, "summary": "auto"},
                    max_output_tokens=2000,
                    timeout=300
                )

                if response.status == 'failed':
                    raise RuntimeError(f"API call failed: {response.error}")

                return response.output_text.strip()

            except RateLimitError:
                base_wait = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                jitter = random.uniform(0, base_wait * 0.5)
                wait_time = base_wait + jitter
                print(f"\n  RATE LIMIT: {row_info}, retry {attempt+1}/{MAX_RETRIES} in {wait_time:.0f}s")
                await asyncio.sleep(wait_time)

            except Exception as e:
                print(f"\n  ERROR: {e} for {row_info}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)
                else:
                    return ""

        return ""


async def validate_row(
    row_idx: int,
    original_row: list,
    enriched_row: list,
    semaphore: asyncio.Semaphore
) -> tuple[int, list, dict]:
    """Validate a single row and return cleaned version."""
    stats = {
        'total_checked': 0,
        'valid': 0,
        'invalid': 0,
        'skipped': 0
    }

    # Get original string (preserve exactly as-is)
    original_string = original_row[0] if original_row else ""
    enriched_string = enriched_row[0] if enriched_row else ""
    pie_form = enriched_row[1] if len(enriched_row) > 1 else ""
    meaning = enriched_row[2] if len(enriched_row) > 2 else ""

    # Find added cognates (comparing language sets)
    added = extract_added_cognates(original_string, enriched_string)

    if not added:
        stats['skipped'] = 1
        # Keep enriched row as-is (no changes to validate)
        return row_idx, enriched_row, stats

    stats['total_checked'] = len(added)

    # Build and send validation prompt
    prompt = build_validation_prompt(pie_form, meaning, added)
    row_info = f"PIE={pie_form}"

    response = await call_gpt_api(prompt, row_info, semaphore)

    if response:
        valid_added = parse_validation_response(response, added)
        stats['valid'] = len(valid_added)
        stats['invalid'] = len(added) - len(valid_added)

        # FIXED: Keep original string intact, only append valid new cognates
        if valid_added:
            valid_string = " ".join(f"[{lang}] {form}" for lang, form in valid_added)
            enriched_row[0] = original_string.strip() + " " + valid_string
        else:
            enriched_row[0] = original_string.strip()
    else:
        # On error, keep original only (conservative)
        enriched_row[0] = original_string.strip()
        stats['invalid'] = len(added)

    return row_idx, enriched_row, stats


def count_existing_rows(filepath: str) -> int:
    """Count rows already processed."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in csv.reader(f)) - 1
    except FileNotFoundError:
        return 0


async def main():
    print("Loading files...")

    # Load original
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        original_rows = list(reader)

    # Load enriched
    with open(ENRICHED_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        enriched_rows = list(reader)

    print(f"Original rows: {len(original_rows)}")
    print(f"Enriched rows: {len(enriched_rows)}")

    # Match by index (assuming same order)
    rows_to_process = min(len(original_rows), len(enriched_rows))

    if TEST_MODE_LIMIT:
        rows_to_process = min(rows_to_process, TEST_MODE_LIMIT)
        print(f"TEST MODE: Processing only {TEST_MODE_LIMIT} rows")

    # Reset output file for test mode
    already_processed = 0
    if TEST_MODE_LIMIT:
        # Fresh start for test
        already_processed = 0
    else:
        already_processed = count_existing_rows(OUTPUT_FILE)
        if already_processed > 0:
            print(f"Resuming from row {already_processed}...")

    # Global stats
    total_stats = {
        'total_checked': 0,
        'valid': 0,
        'invalid': 0,
        'skipped': 0
    }

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    mode = 'a' if already_processed > 0 else 'w'
    with open(OUTPUT_FILE, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        if already_processed == 0:
            writer.writerow(header)

        # Process in batches
        for batch_start in range(already_processed, rows_to_process, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, rows_to_process)

            print(f"\nValidating batch {batch_start//BATCH_SIZE + 1} (rows {batch_start+1}-{batch_end})...")

            tasks = [
                validate_row(
                    row_idx=i,
                    original_row=original_rows[i].copy() if i < len(original_rows) else [],
                    enriched_row=enriched_rows[i].copy() if i < len(enriched_rows) else [],
                    semaphore=semaphore
                )
                for i in range(batch_start, batch_end)
            ]

            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Validating"):
                result = await coro
                results.append(result)

            results.sort(key=lambda x: x[0])

            for _, validated_row, stats in results:
                writer.writerow(validated_row)
                total_stats['total_checked'] += stats['total_checked']
                total_stats['valid'] += stats['valid']
                total_stats['invalid'] += stats['invalid']
                total_stats['skipped'] += stats['skipped']

            f.flush()
            print(f"Batch done. Valid: {total_stats['valid']}, Invalid: {total_stats['invalid']}")

    print("\n" + "="*60)
    print("VALIDATION STATISTICS")
    print("="*60)
    print(f"Total cognates checked: {total_stats['total_checked']}")
    print(f"Valid (kept): {total_stats['valid']}")
    print(f"Invalid (removed): {total_stats['invalid']}")
    print(f"Rows skipped (no additions): {total_stats['skipped']}")
    if total_stats['total_checked'] > 0:
        print(f"Validation rate: {total_stats['valid']/total_stats['total_checked']*100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
