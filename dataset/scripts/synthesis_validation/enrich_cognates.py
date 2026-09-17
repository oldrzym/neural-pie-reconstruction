"""
Script to enrich PIE cognate dataset with synthetic data using GPT-5.2 API.
Adds cognates from underrepresented languages to rows with <20 language pairs.
ASYNC VERSION - processes multiple rows in parallel.
"""

import csv
import json
import re
import asyncio
import random
import os
from collections import defaultdict
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, RateLimitError

# === CONFIGURATION ===
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "splits" / "paper_pipeline"
DATA_DIR = Path(os.environ.get("PIE_DATA_DIR", DEFAULT_DATA_DIR))
INPUT_FILE = DATA_DIR / "train_synth.csv"
OUTPUT_FILE = DATA_DIR / "train_synt.csv"
LANGUAGES_FILE = DATA_DIR / "sorted_data.json"

# Number of cognates to request per row
COGNATES_TO_ADD = 10
# Minimum number of language pairs to skip enrichment
MIN_PAIRS_THRESHOLD = 20
# GPT model to use
GPT_MODEL = "gpt-5.2"
# Reasoning effort: none, low, medium, high, xhigh
REASONING_EFFORT = "none"
# Show reasoning summary in logs (verbose)
SHOW_REASONING = False
# For testing: limit to N rows (set to None for full dataset)
# >>>>>>> CHANGE THIS TO None TO PROCESS ALL ROWS <<<<<<<
TEST_MODE_LIMIT = None

# === ASYNC CONFIGURATION ===
# Max parallel requests (adjust based on your rate limits)
MAX_CONCURRENT = 5
# Batch size for writing results
BATCH_SIZE = 20
# Max retries for rate limit errors
MAX_RETRIES = 10
# Base delay for retry (seconds)
RETRY_BASE_DELAY = 10
# Max delay between retries (seconds)
RETRY_MAX_DELAY = 300
# Delay between starting new requests (seconds) - helps with TPM limits
REQUEST_DELAY = 10

# Reads OPENAI_API_KEY from the environment.
client = AsyncOpenAI()

# Modern languages only (excluding ancient/dead languages)
MODERN_LANGUAGES = {
    # Slavic
    "rus": "Russian", "pol": "Polish", "ces": "Czech", "slk": "Slovak",
    "ukr": "Ukrainian", "bel": "Belarusian", "bul": "Bulgarian", "mkd": "Macedonian",
    "hbs": "Serbo-Croatian", "sor": "Sorbian",
    # Romance
    "fra": "French", "spa": "Spanish", "por": "Portuguese", "ita": "Italian",
    "ron": "Romanian", "cat": "Catalan", "sar": "Sardinian", "fri": "Friulian",
    "nea": "Neapolitan", "wal": "Walloon",
    # Germanic
    "deu": "German", "eng": "English", "nld": "Dutch", "swe": "Swedish",
    "nor": "Norwegian", "dan": "Danish", "isl": "Icelandic", "far": "Faroese",
    # Celtic
    "gle": "Irish", "cym": "Welsh", "bre": "Breton", "gla": "Scottish Gaelic", "gae": "Manx",
    # Baltic
    "lit": "Lithuanian", "lav": "Latvian",
    # Indo-Iranian
    "fas": "Persian", "kur": "Kurdish", "oss": "Ossetian", "pas": "Pashto",
    "hin": "Hindi", "urd": "Urdu", "ben": "Bengali", "pun": "Punjabi",
    "mar": "Marathi", "nep": "Nepali", "sin": "Sinhalese", "ass": "Assamese",
    "bho": "Bhojpuri", "mai": "Maithili", "mag": "Magahi", "kas": "Kashmiri",
    # Other
    "ell": "Modern Greek", "sqi": "Albanian", "hye": "Armenian", "hyw": "Western Armenian",
}


def load_top_languages(filepath: str, top_n: int = 70) -> list[str]:
    """Load top N languages from sorted_data.json, filtered to modern languages only."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    modern_langs = [lang for lang in data.keys() if lang in MODERN_LANGUAGES]
    return modern_langs[:top_n]


def extract_languages_from_row(cognate_string: str) -> set[str]:
    """Extract all language codes from a cognate string."""
    pattern = r'\[([a-z]{3})\]'
    return set(re.findall(pattern, cognate_string.lower()))


def count_language_pairs(cognate_string: str) -> int:
    """Count number of [lang] pairs in the string."""
    pattern = r'\[[a-z]{3}\]'
    return len(re.findall(pattern, cognate_string.lower()))


def extract_cognates_with_forms(cognate_string: str) -> dict[str, str]:
    """Extract language -> word mapping from cognate string."""
    pattern = r'\[([a-z]{3})\]\s*([^\[\]]+?)(?=\s*\[|$)'
    matches = re.findall(pattern, cognate_string, re.IGNORECASE)
    return {lang.lower(): form.strip() for lang, form in matches}


def clean_quotes(text: str) -> str:
    """Remove surrounding quotes from text."""
    if text and len(text) >= 2:
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
    return text


def build_prompt(
    cognate_string: str,
    pie_form: str,
    meaning: str,
    available_languages: list[str],
    existing_languages: set[str],
    existing_cognates: dict[str, str]
) -> str:
    """Build a high-quality prompt for GPT to generate cognates."""
    target_languages = [
        lang for lang in available_languages
        if lang not in existing_languages and lang in MODERN_LANGUAGES
    ]

    existing_desc = ""
    if existing_cognates:
        examples = [f"[{lang}] {form}" for lang, form in existing_cognates.items()]
        existing_desc = f"Existing cognates ({len(examples)} total): {' '.join(examples)}"

    lang_ref = ", ".join([f"{code}={name}" for code, name in MODERN_LANGUAGES.items() if code in target_languages][:30])

    prompt = f"""You are an expert historical linguist specializing in Proto-Indo-European (PIE) language reconstruction and comparative linguistics.

TASK: Generate exactly {COGNATES_TO_ADD} cognate forms for the given PIE root from modern Indo-European languages.

PIE RECONSTRUCTION: {pie_form}
MEANING: {meaning if meaning and meaning != 'nan' else 'unknown'}

CURRENT ROW DATA:
{cognate_string}

{existing_desc}

TARGET LANGUAGES (choose {COGNATES_TO_ADD} from this list, only if you are certain the cognate exists):
{', '.join(target_languages[:40])}

LANGUAGE CODE REFERENCE:
{lang_ref}

REAL EXAMPLES FROM DATASET:
- PIE *peh₃- "to drink" → [ell] pino [sqi] pi [rus] pit' [pol] pic [ces] pit [hin] pina
- PIE *gʷʰen- "to strike, kill" → [rus] gnat' [pol] gnac [fas] zadan [sqi] gjuaj
- PIE *meh₁- "to measure" → [ell] metro [rus] mera [pol] miara [hin] matra [fra] mesure

STRICT REQUIREMENTS:
1. Output EXACTLY {COGNATES_TO_ADD} cognates, no more, no less
2. Each cognate MUST be from a DIFFERENT language in the target list
3. Use Latin transliteration ONLY (no Cyrillic, Greek, Arabic, Devanagari scripts)
4. Only provide cognates you are HIGHLY CONFIDENT are historically attested and etymologically related to the PIE root
5. Do NOT invent forms - only use forms documented in etymological literature
6. Include the inherited/derived word, not a loanword from another branch

OUTPUT FORMAT (one cognate per line, nothing else):
[lang] form
[lang] form
...

Generate {COGNATES_TO_ADD} cognates now:"""

    return prompt


def parse_gpt_response(response_text: str) -> list[tuple[str, str]]:
    """Parse GPT response into list of (language, form) tuples."""
    pattern = r'\[([a-z]{3})\]\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern, response_text.strip(), re.IGNORECASE | re.MULTILINE)

    result = []
    for lang, form in matches:
        form = form.strip()
        form = re.sub(r'[,;:]$', '', form).strip()
        if form and len(form) > 0:
            result.append((lang.lower(), form))

    return result


async def call_gpt_api(prompt: str, row_info: str, semaphore: asyncio.Semaphore) -> str:
    """Call GPT API with semaphore for concurrency control and retry logic."""
    async with semaphore:
        await asyncio.sleep(random.uniform(0, REQUEST_DELAY))  # Задержка перед запросом

        for attempt in range(MAX_RETRIES):
            try:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES} to call API for {row_info}")

                response = await client.responses.create(
                    model=GPT_MODEL,
                    instructions="You are an expert comparative linguist. Provide only the requested cognate forms in the exact format specified. Be precise and scholarly.",
                    input=prompt,
                    reasoning={"effort": REASONING_EFFORT, "summary": "auto"},
                    max_output_tokens=10000,
                    timeout=600
                )

                # Если reasoning включен, покажем логику
                if SHOW_REASONING and response.output:
                    for item in response.output:
                        if hasattr(item, 'summary') and item.summary:
                            print(f"[REASONING] {item.summary}")

                if response.status == 'incomplete':
                    print(f"WARNING: Response incomplete for {row_info}")

                if response.status == 'failed':
                    print(f"ERROR: API call failed for {row_info}, Error: {response.error}")
                    raise RuntimeError(f"API call failed: {response.error}")

                result = response.output_text.strip()
                if not result or not result.strip():
                    print(f"WARNING: Empty response for {row_info}")
                    return ""

                return result

            except RateLimitError as e:
                # Детализированное логирование ошибки RateLimitError
                print(f"RateLimitError encountered for {row_info}. Error: {e}. Retrying...")
                base_wait = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                jitter = random.uniform(0, base_wait * 0.5)  # добавление случайной вариации
                wait_time = base_wait + jitter
                print(f"RATE LIMIT: {row_info}, retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.0f}s")
                await asyncio.sleep(wait_time)

            except asyncio.TimeoutError:
                print(f"TIMEOUT: {row_info} due to timeout. Retrying...")
                await asyncio.sleep(5)  # Дополнительная задержка при таймауте
                return ""

            except Exception as e:
                # Логирование всех других ошибок
                print(f"ERROR: {e} for {row_info}, retrying...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
                else:
                    return ""

        return ""


async def process_row_async(
    row_idx: int,
    row: list,
    top_languages: list[str],
    semaphore: asyncio.Semaphore
) -> tuple[int, list, dict]:
    """Process a single row asynchronously and return (index, processed_row, stats_update)."""
    stats_update = {
        'languages': defaultdict(int),
        'total_added': 0,
        'rows_enriched': 0,
        'rows_processed': 0,
        'rows_skipped': 0
    }

    print(f"Processing row {row_idx}...")  # Логируем, какую строку обрабатываем
    if len(row) < 2:
        return row_idx, row, stats_update

    cognate_string = row[0]
    pie_form = row[1] if len(row) > 1 else ""
    meaning = row[2] if len(row) > 2 else ""

    cognate_string = clean_quotes(cognate_string)
    num_pairs = count_language_pairs(cognate_string)

    if num_pairs >= MIN_PAIRS_THRESHOLD:
        row[0] = cognate_string
        stats_update['rows_skipped'] = 1
        print(f"Skipping row {row_idx}, enough pairs already.")
        return row_idx, row, stats_update

    existing_languages = extract_languages_from_row(cognate_string)
    existing_cognates = extract_cognates_with_forms(cognate_string)

    prompt = build_prompt(
        cognate_string=cognate_string,
        pie_form=pie_form,
        meaning=meaning,
        available_languages=top_languages,
        existing_languages=existing_languages,
        existing_cognates=existing_cognates
    )

    row_info = f"PIE={pie_form}"
    response = await call_gpt_api(prompt, row_info, semaphore)

    if response and response.strip():
        print(f"Received response for row {row_idx}. Parsing cognates...")
        new_cognates = parse_gpt_response(response)

        added_cognates = []
        for lang, form in new_cognates:
            if lang not in existing_languages and lang in top_languages:
                added_cognates.append(f"[{lang}] {form}")
                existing_languages.add(lang)
                stats_update['languages'][lang] += 1
                stats_update['total_added'] += 1

        if added_cognates:
            enriched_string = cognate_string + " " + " ".join(added_cognates)
            row[0] = enriched_string.strip()
            stats_update['rows_enriched'] = 1
    else:
        row[0] = cognate_string

    stats_update['rows_processed'] = 1
    return row_idx, row, stats_update


def count_existing_rows(filepath: str) -> int:
    """Count rows already processed in output file (excluding header)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return sum(1 for _ in reader) - 1
    except FileNotFoundError:
        return 0


def merge_stats(global_stats: dict, local_stats: dict):
    """Merge local stats into global stats."""
    for lang, count in local_stats['languages'].items():
        global_stats['languages'][lang] += count
    global_stats['total_added'] += local_stats['total_added']
    global_stats['rows_enriched'] += local_stats['rows_enriched']
    global_stats['rows_processed'] += local_stats['rows_processed']
    global_stats['rows_skipped'] += local_stats['rows_skipped']


async def main():
    """Main async function to process the dataset."""
    print("Loading top 70 languages...")
    top_languages = load_top_languages(LANGUAGES_FILE, top_n=70)
    print(f"Top languages: {top_languages[:10]}...")

    stats = {
        'languages': defaultdict(int),
        'total_added': 0,
        'rows_enriched': 0,
        'rows_processed': 0,
        'rows_skipped': 0
    }

    print(f"\nReading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total_rows = len(rows)
    print(f"Total rows: {total_rows}")

    # Check how many rows already processed
    already_processed = count_existing_rows(OUTPUT_FILE)
    if already_processed > 0:
        print(f"Found {already_processed} rows already in output file, resuming from row {already_processed}...")
        rows = rows[already_processed:]

    if TEST_MODE_LIMIT and already_processed == 0:
        rows = rows[:TEST_MODE_LIMIT]
        print(f"TEST MODE: Processing only {TEST_MODE_LIMIT} row(s)")

    if not rows:
        print("All rows already processed!")
        return

    print(f"Rows to process: {len(rows)}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Batch size: {BATCH_SIZE}")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Open output file
    mode = 'a' if already_processed > 0 else 'w'
    with open(OUTPUT_FILE, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        if already_processed == 0:
            writer.writerow(header)

        # Process in batches
        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(rows))
            batch_rows = rows[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1} (rows {already_processed + batch_start + 1}-{already_processed + batch_end})...")

            # Create tasks for this batch
            tasks = [
                process_row_async(
                    row_idx=already_processed + batch_start + i,
                    row=row.copy(),
                    top_languages=top_languages,
                    semaphore=semaphore
                )
                for i, row in enumerate(batch_rows)
            ]

            # Run batch with progress bar
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Batch progress"):
                result = await coro
                results.append(result)

            # Sort results by original index to maintain order
            results.sort(key=lambda x: x[0])

            # Write results and merge stats
            for _, processed_row, local_stats in results:
                writer.writerow(processed_row)
                merge_stats(stats, local_stats)

            # Flush after each batch
            f.flush()
            print(f"Batch complete. Total enriched so far: {stats['rows_enriched']}")

    print("\n" + "="*60)
    print("STATISTICS (this session)")
    print("="*60)
    print(f"Total rows processed: {stats['rows_processed']}")
    print(f"Rows enriched: {stats['rows_enriched']}")
    print(f"Rows skipped (>=20 pairs): {stats['rows_skipped']}")
    print(f"Total cognates added: {stats['total_added']}")

    if stats['languages']:
        print("\nCognates added by language:")
        print("-"*40)
        sorted_langs = sorted(stats['languages'].items(), key=lambda x: -x[1])
        for lang, count in sorted_langs:
            print(f"  [{lang}]: {count}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
