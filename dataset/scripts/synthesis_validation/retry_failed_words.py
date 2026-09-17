"""Retry failed words with normalization."""
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from wiktionary_fetcher import WiktionaryFetcher

BASE_DIR = Path(__file__).parent.parent.parent / "splits" / "iecor_kaikki_koebler_normalized"
ERRORS_FILE = BASE_DIR / "wiktionary_errors.json"


def normalize_word(word, lang_code):
    """
    Normalize problematic word forms.

    Returns:
        str or None: Normalized word, or None if should skip (pure affix)
    """
    original = word

    # Remove parentheses: (com)periō → comperiō
    word = re.sub(r'\(([^)]+)\)', r'\1', word)

    # Remove asterisk: *‑kla → ‑kla
    word = word.lstrip('*')

    # Skip pure affixes (only dashes)
    word = word.strip('-')
    if not word:
        return None

    # Unicode NFC normalization
    word = unicodedata.normalize('NFC', word)

    # For Celtic languages: convert macron (ū) to acute accent (ú)
    if lang_code in ['air', 'sga', 'gla', 'gle', 'cym', 'bre']:
        macron_to_acute = {
            'ū': 'ú', 'ā': 'á', 'ē': 'é', 'ī': 'í', 'ō': 'ó',
            'Ū': 'Ú', 'Ā': 'Á', 'Ē': 'É', 'Ī': 'Í', 'Ō': 'Ó'
        }
        for macron, acute in macron_to_acute.items():
            word = word.replace(macron, acute)

    return word if word != original else None  # Return None if no change


async def main():
    print("="*80)
    print("RETRY FAILED WORDS WITH NORMALIZATION")
    print("="*80 + "\n")

    # Check if errors file exists
    if not ERRORS_FILE.exists():
        print(f"Error: {ERRORS_FILE} not found!")
        print("Run analyze_wiktionary_errors.py first.")
        return

    # Load error analysis
    with open(ERRORS_FILE, 'r', encoding='utf-8') as f:
        errors = json.load(f)

    # Prepare retry list with normalization
    retry_pairs = []
    original_to_normalized = {}  # Track what we normalized

    for pattern, keys in errors['fixable_patterns'].items():
        print(f"Processing {pattern}: {len(keys)} words...")

        for key in keys:
            lang, word = key.split('|', 1)
            normalized = normalize_word(word, lang)

            if normalized and normalized != word:
                retry_pairs.append((normalized, lang))
                original_to_normalized[key] = f"{lang}|{normalized}"

    print(f"\nTotal words to retry with normalization: {len(retry_pairs)}")

    if not retry_pairs:
        print("No fixable words found. All errors require different approach.")
        return

    # Fetch with normalization
    print("\nFetching etymologies for normalized words...\n")

    fetcher = WiktionaryFetcher()
    results = await fetcher.fetch_batch(retry_pairs)

    # Count improvements
    improved = sum(1 for v in results.values() if not v.startswith('['))
    still_failed = len(results) - improved

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print(f"Successfully found: {improved}/{len(retry_pairs)} ({improved/len(retry_pairs)*100:.1f}%)")
    print(f"Still failed: {still_failed}")

    # Show examples of successful normalizations
    if improved > 0:
        print("\nSuccessful normalizations (first 10):")
        count = 0
        for orig_key, norm_key in original_to_normalized.items():
            if norm_key in results and not results[norm_key].startswith('['):
                orig_lang, orig_word = orig_key.split('|', 1)
                norm_lang, norm_word = norm_key.split('|', 1)
                print(f"  {orig_word} → {norm_word} ({orig_lang})")
                count += 1
                if count >= 10:
                    break

    print(f"\nCache updated with {improved} new etymologies.")
    print("Run analyze_wiktionary_errors.py again to see updated statistics.")


if __name__ == "__main__":
    asyncio.run(main())
