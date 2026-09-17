"""Fetch Wiktionary etymologies for all unique cognates in the dataset."""
import asyncio
import json
from pathlib import Path
from wiktionary_fetcher import WiktionaryFetcher

BASE_DIR = Path(__file__).parent.parent.parent / "splits" / "iecor_kaikki_koebler_normalized"
COGNATES_FILE = BASE_DIR / "unique_cognates.json"
ETYMOLOGY_DICT = BASE_DIR / "wiktionary_etymologies.json"


async def main():
    print("="*80)
    print("WIKTIONARY ETYMOLOGY FETCHER")
    print("="*80 + "\n")

    # Load unique cognates
    if not COGNATES_FILE.exists():
        print(f"❌ Error: {COGNATES_FILE} not found!")
        print("Run extract_cognates.py first to extract unique cognates.")
        return

    with open(COGNATES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_cognates = data['unique_cognates']
    stats = data['statistics']

    print(f"📊 Dataset statistics:")
    print(f"  Total unique cognates: {stats['total_unique_cognates']}")
    print(f"  Total languages: {stats['total_languages']}")
    print()

    # Check existing cache
    fetcher = WiktionaryFetcher()
    cached_count = len(fetcher.cache)

    # Filter out already cached cognates
    to_fetch = [
        (word, lang) for lang, word in unique_cognates
        if f"{lang}|{word}" not in fetcher.cache
    ]

    print(f"📦 Cache status:")
    print(f"  Already cached: {cached_count}")
    print(f"  Need to fetch: {len(to_fetch)}")
    print(f"  Total: {len(unique_cognates)}")
    print()

    if not to_fetch:
        print("✓ All cognates already cached!")
        return

    # Estimate time
    avg_delay = 1.5  # Conservative estimate with retries
    estimated_time = len(to_fetch) * avg_delay
    hours = int(estimated_time // 3600)
    minutes = int((estimated_time % 3600) // 60)

    print(f"⏱️  Estimated time: {hours}h {minutes}m")
    print(f"   (assuming ~{avg_delay}s per request with retries)")
    print()

    # Confirm
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print("\n" + "="*80)
    print("FETCHING ETYMOLOGIES")
    print("="*80 + "\n")

    # Fetch all etymologies
    results = await fetcher.fetch_batch(to_fetch)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    # Count results
    valid = sum(1 for v in results.values() if not v.startswith('['))
    no_mapping = sum(1 for v in results.values() if '[No Wiktionary mapping' in v)
    not_found = sum(1 for v in results.values() if '[Word not found' in v)
    http_429 = sum(1 for v in results.values() if 'HTTP 429' in v)
    other_errors = len(results) - valid - no_mapping - not_found - http_429

    print(f"✅ Valid etymologies: {valid}")
    print(f"⚠️  No mapping: {no_mapping}")
    print(f"⚠️  Word not found: {not_found}")
    print(f"❌ HTTP 429 (rate limit): {http_429}")
    print(f"❌ Other errors: {other_errors}")
    print(f"📊 Total fetched: {len(results)}")
    print()

    total_in_cache = len(fetcher.cache)
    total_valid = sum(1 for v in fetcher.cache.values() if not v.startswith('['))

    print(f"📦 Total cache size: {total_in_cache}")
    print(f"✅ Total valid in cache: {total_valid}")
    print()
    print(f"💾 Saved to: {ETYMOLOGY_DICT}")

    if http_429 > 0:
        print()
        print("⚠️  WARNING: Some requests hit rate limit.")
        print("   You can re-run this script to retry failed requests.")
        print("   The script will automatically skip cached entries.")


if __name__ == "__main__":
    asyncio.run(main())
