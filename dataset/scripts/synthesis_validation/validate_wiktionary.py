"""
Validate cognates using Wiktionary etymology + OpenAI Batch API (50% cheaper).

Steps:
  1. Load files, find added cognates
  2. Fetch Wiktionary etymologies (async, parallel)
  3. Generate JSONL batch file
  4. Upload & submit batch to OpenAI
  5. Poll for completion
  6. Download results & assemble validated CSV
"""

import csv
import re
import json
import asyncio
import time
import ssl
import certifi
import aiohttp
import logging
import os
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from openai import OpenAI
from wiktionary_fetcher import WiktionaryFetcher

# === CONFIGURATION ===
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "splits" / "paper_pipeline"
BASE_DIR = Path(os.environ.get("PIE_DATA_DIR", DEFAULT_DATA_DIR))
ORIGINAL_FILE = BASE_DIR / "train.csv"
ENRICHED_FILE = BASE_DIR / "train_synt.csv"
WIKT_CACHE_FILE = BASE_DIR / "wiktionary_etymologies.json"  # Use main cache instead of wikt_cache.json

# Chunk processing (set START_ROW and CHUNK_SIZE, files will be saved to validated_chunks/)
START_ROW = 2700        # 0-indexed, skip first N rows
CHUNK_SIZE = 200      # process N rows from START_ROW (None = all remaining rows)

# Output directory for chunks
CHUNKS_DIR = BASE_DIR / "validated_chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

# Dynamic output files based on chunk
def get_chunk_files(start, size):
    end = start + size if size else "end"
    chunk_name = f"chunk_{start}_{end}"
    return {
        "output": CHUNKS_DIR / f"{chunk_name}.csv",
        "batch_requests": CHUNKS_DIR / f"{chunk_name}_requests.jsonl",
        "batch_results": CHUNKS_DIR / f"{chunk_name}_results.jsonl",
    }

OUTPUT_FILE = get_chunk_files(START_ROW, CHUNK_SIZE)["output"]
BATCH_JSONL = get_chunk_files(START_ROW, CHUNK_SIZE)["batch_requests"]
BATCH_RESULTS_JSONL = get_chunk_files(START_ROW, CHUNK_SIZE)["batch_results"]

# GPT config
GPT_MODEL = "gpt-5.2"
REASONING_EFFORT = "low"

# Wiktionary config
WIKT_CONCURRENT = 5
WIKT_DELAY = 0.3

# Backwards compatibility
TEST_MODE_LIMIT = CHUNK_SIZE

# Language code to Wiktionary language name
LANG_TO_WIKT = {
    "eng": "English", "deu": "German", "nld": "Dutch", "swe": "Swedish",
    "nor": "Norwegian", "dan": "Danish", "isl": "Icelandic", "fri": "Frisian",
    "rus": "Russian", "pol": "Polish", "ces": "Czech", "slk": "Slovak",
    "ukr": "Ukrainian", "bel": "Belarusian", "bul": "Bulgarian", "hbs": "Serbo-Croatian",
    "fra": "French", "ita": "Italian", "spa": "Spanish", "por": "Portuguese",
    "ron": "Romanian", "cat": "Catalan",
    "ell": "Greek", "lit": "Lithuanian", "lav": "Latvian",
    "gle": "Irish", "cym": "Welsh", "bre": "Breton", "gae": "Scottish Gaelic",
    "fas": "Persian", "hin": "Hindi", "sqi": "Albanian", "hye": "Armenian",
}

# Reads OPENAI_API_KEY from the environment.
client = OpenAI()

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
# Fix Windows encoding for logging
for handler in logging.root.handlers:
    if hasattr(handler, 'stream'):
        import io
        import sys
        handler.stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
log = logging.getLogger("wikt_validate")


# ============================
# STEP 1: Load & compare files
# ============================

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
    log.info("Loading files...")

    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        original_rows = list(reader)

    with open(ENRICHED_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        enriched_rows = list(reader)

    log.info(f"Original: {len(original_rows)} rows, Enriched: {len(enriched_rows)} rows")

    total_rows = min(len(original_rows), len(enriched_rows))
    start = START_ROW or 0
    end = min(start + TEST_MODE_LIMIT, total_rows) if TEST_MODE_LIMIT else total_rows
    log.info(f"Processing rows {start}-{end-1} ({end - start} rows)")

    # Find added cognates per row
    row_data = []
    total_added = 0
    for i in tqdm(range(start, end), desc="Comparing rows"):
        orig = original_rows[i]
        enr = enriched_rows[i]
        added = extract_added_cognates(orig[0] if orig else "", enr[0] if enr else "")
        total_added += len(added)
        row_data.append({
            "idx": i,
            "original_row": orig,
            "enriched_row": enr,
            "pie_form": enr[1] if len(enr) > 1 else "",
            "meaning": enr[2] if len(enr) > 2 else "",
            "added_cognates": added,
        })

    rows_with_added = sum(1 for r in row_data if r["added_cognates"])
    log.info(f"Found {total_added} added cognates in {rows_with_added}/{end - start} rows")

    return header, row_data


# ================================
# STEP 2: Fetch Wiktionary (async)
# ================================

async def fetch_wiktionary_etymology(word: str, lang_code: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> str:
    lang_name = LANG_TO_WIKT.get(lang_code, "")
    if not lang_name:
        return f"[No Wiktionary mapping for {lang_code}]"

    url = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "query",
        "titles": word.lower(),
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "rvslots": "main",
    }

    async with semaphore:
        await asyncio.sleep(WIKT_DELAY)
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return f"[Wiktionary HTTP {resp.status}]"

                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})

                for page_id, page in pages.items():
                    if page_id == "-1":
                        return "[Word not found in Wiktionary]"

                    content = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")

                    lang_pattern = rf"==\s*{re.escape(lang_name)}\s*==\n(.+?)(?=\n==[^=]|\Z)"
                    lang_match = re.search(lang_pattern, content, re.DOTALL | re.IGNORECASE)

                    if not lang_match:
                        return f"[No {lang_name} section found]"

                    lang_section = lang_match.group(1)

                    etym_pattern = r"===\s*Etymology[^=]*===\s*(.+?)(?=\n===|\Z)"
                    etym_match = re.search(etym_pattern, lang_section, re.DOTALL)

                    if etym_match:
                        etymology = etym_match.group(1).strip()
                        # Parse templates: {{inh|en|enm|riden}} → riden
                        def _parse_template(m):
                            inner = m.group(0)[2:-2]
                            parts = inner.split('|')
                            word = ''
                            trans = ''
                            for p in parts[1:]:
                                if p.startswith('t=') or p.startswith('gloss='):
                                    trans = p.split('=', 1)[1]
                                elif '=' not in p:
                                    word = p
                            if trans:
                                return f'{word} ("{trans}")'
                            return word
                        etymology = re.sub(r'\{\{[^}]+\}\}', _parse_template, etymology)
                        etymology = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', etymology)
                        etymology = re.sub(r'<[^>]+>', '', etymology)
                        etymology = re.sub(r'\n+', ' ', etymology)
                        etymology = re.sub(r'\s{2,}', ' ', etymology)
                        etymology = etymology[:500]
                        return etymology if etymology else "[Empty etymology]"

                    return "[No etymology section found]"

                return "[Page structure error]"

        except asyncio.TimeoutError:
            return "[Wiktionary timeout]"
        except Exception as e:
            return f"[Error: {str(e)[:80]}]"


async def fetch_all_etymologies(row_data: list) -> dict:
    """Fetch Wiktionary etymologies for all added cognates. Returns {(lang,form): etymology}."""

    # Collect unique (lang, form) pairs
    unique_pairs = set()
    for row in row_data:
        for lang, form in row["added_cognates"]:
            unique_pairs.add((lang, form))

    if not unique_pairs:
        log.info("No cognates to fetch etymologies for")
        return {}

    log.info(f"Fetching Wiktionary etymologies for {len(unique_pairs)} unique cognates...")

    # Use WiktionaryFetcher which automatically uses existing cache and saves results
    fetcher = WiktionaryFetcher()

    # Convert to list of (word, lang) tuples (WiktionaryFetcher expects this order)
    word_lang_pairs = [(form, lang) for lang, form in unique_pairs]

    # Fetch (will check cache first, only fetch missing ones)
    results = await fetcher.fetch_batch(word_lang_pairs)

    # Convert back to {(lang, form): etymology} format
    cache = {}
    for lang, form in unique_pairs:
        key = f"{lang}|{form}"
        cache[(lang, form)] = results.get(key, "[Not found]")

    found = sum(1 for v in cache.values() if not v.startswith("["))
    not_found = len(cache) - found
    log.info(f"Wiktionary: {found} etymologies found, {not_found} not found")

    return cache


# ==========================
# STEP 3: Generate JSONL
# ==========================

def build_validation_prompt(
    pie_form: str,
    meaning: str,
    cognates_with_etym: list[tuple[str, str, str]],
    original_cognates: str = "",
) -> str:
    cognate_info = ""
    for lang, form, etym in cognates_with_etym:
        # Optimization: skip etymology text if not found (saves tokens)
        if etym.startswith('['):  # Error: [No data], [Word not found], etc.
            cognate_info += f"[{lang}] {form}\n"
        else:  # Valid etymology found
            cognate_info += f"[{lang}] {form} - {etym}\n"

    return f"""You are validating cognates for a PIE root using Wiktionary etymology data.

PIE ROOT: {pie_form}
MEANING: {meaning if meaning and meaning != 'nan' else 'unknown'}
ORIGINAL COGNATES: {original_cognates if original_cognates else '[none]'}

CANDIDATES:
{cognate_info}

TASK: For each word, check if it's a cognate of the PIE root.

RULES:
- VALID: Etymology (if shown) confirms PIE origin or related proto-form
- VALID: Phonetically/semantically similar to PIE root (even without etymology)
- INVALID: Etymology shows different origin (loanword, different PIE root)
- INVALID: Clearly unrelated in meaning/sound
- NOTE: Words without etymology shown have no Wiktionary data. Use PIE root + phonetic/semantic similarity.

OUTPUT FORMAT (one per line):
[lang] form - VALID
[lang] form - INVALID

Validate:"""


def generate_batch_jsonl(row_data: list, etym_cache: dict) -> int:
    """Generate JSONL file for OpenAI Batch API. Returns number of requests."""
    request_count = 0

    with open(BATCH_JSONL, 'w', encoding='utf-8') as f:
        for row in tqdm(row_data, desc="Building JSONL"):
            if not row["added_cognates"]:
                continue

            cognates_with_etym = []
            for lang, form in row["added_cognates"]:
                etym = etym_cache.get((lang, form), "[No data]")
                cognates_with_etym.append((lang, form, etym))

            original_cognates = row["original_row"][0] if row.get("original_row") else ""
            prompt = build_validation_prompt(row["pie_form"], row["meaning"], cognates_with_etym, original_cognates)

            request = {
                "custom_id": f"row-{row['idx']}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": GPT_MODEL,
                    "instructions": "You are an expert etymologist. Validate cognates based on Wiktionary data.",
                    "input": prompt,
                    "reasoning": {"effort": REASONING_EFFORT, "summary": "auto"},
                    "max_output_tokens": 2000,
                },
            }

            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            request_count += 1

    log.info(f"Generated {BATCH_JSONL} with {request_count} requests")
    return request_count


# =================================
# STEP 4: Upload & submit batch
# =================================

def upload_and_submit_batch() -> str:
    """Upload JSONL and submit batch. Returns batch_id."""
    log.info("Uploading batch file to OpenAI...")
    with open(BATCH_JSONL, 'rb') as f:
        file_obj = client.files.create(file=f, purpose="batch")
    log.info(f"Uploaded file: {file_obj.id} ({file_obj.bytes} bytes)")

    log.info("Submitting batch...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    log.info(f"Batch submitted: {batch.id} (status: {batch.status})")
    return batch.id


# =================================
# STEP 5: Poll for completion
# =================================

def poll_batch(batch_id: str, poll_interval: int = 10) -> dict:
    """Poll batch status until completion."""
    log.info(f"Polling batch {batch_id}...")

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

        log.info(f"Status: {batch.status} | {completed}/{total} done, {failed} failed")

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            pbar.update(100 - last_pct)
            pbar.close()
            break

        time.sleep(poll_interval)

    if batch.status == "completed":
        log.info(f"Batch completed! {completed} succeeded, {failed} failed")
    else:
        log.error(f"Batch ended with status: {batch.status}")
        if batch.errors:
            for err in batch.errors.data[:5]:
                log.error(f"  Error: {err.message}")

    return batch


# =====================================
# STEP 6: Download results & assemble
# =====================================

def parse_validation_response(response_text: str, added_cognates: list[tuple[str, str]]) -> list[tuple[str, str]]:
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
        log.error("No output file in batch!")
        return

    log.info(f"Downloading results from {output_file_id}...")
    content = client.files.content(output_file_id)
    result_text = content.text

    # Save raw results
    with open(BATCH_RESULTS_JSONL, 'w', encoding='utf-8') as f:
        f.write(result_text)
    log.info(f"Saved raw results to {BATCH_RESULTS_JSONL}")

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
            log.warning(f"Row {row_idx}: API error - {error}")

        results_map[row_idx] = output_text

    log.info(f"Parsed {len(results_map)} results")

    # Assemble validated CSV
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

    log.info("=" * 60)
    log.info("VALIDATION STATS")
    log.info("=" * 60)
    log.info(f"Total checked:  {stats['total_checked']}")
    log.info(f"Valid:          {stats['valid']}")
    log.info(f"Invalid:        {stats['invalid']}")
    log.info(f"Skipped (0 new):{stats['skipped']}")
    if stats["total_checked"] > 0:
        log.info(f"Validation rate: {stats['valid'] / stats['total_checked'] * 100:.1f}%")
    log.info(f"Output: {OUTPUT_FILE}")


# =============
# MAIN
# =============

def main():
    log.info("=== WIKTIONARY + BATCH API VALIDATION ===")

    # Step 1: Load & compare
    header, row_data = load_and_compare()

    # Step 2: Fetch Wiktionary etymologies
    etym_cache = asyncio.run(fetch_all_etymologies(row_data))

    # Note: Cache is automatically saved by WiktionaryFetcher to wiktionary_etymologies.json
    log.info(f"Wiktionary cache updated in {WIKT_CACHE_FILE}")

    # Step 3: Generate JSONL
    request_count = generate_batch_jsonl(row_data, etym_cache)
    if request_count == 0:
        log.info("No requests to send, nothing to validate")
        return

    # Step 4: Upload & submit
    batch_id = upload_and_submit_batch()

    # Step 5: Poll
    batch = poll_batch(batch_id, poll_interval=5)
    if batch.status != "completed":
        log.error(f"Batch did not complete. Status: {batch.status}")
        log.error(f"Batch ID: {batch_id} — you can check later with: client.batches.retrieve('{batch_id}')")
        return

    # Step 6: Download & assemble
    download_and_assemble(batch, header, row_data)

    log.info("Done!")


if __name__ == "__main__":
    main()
