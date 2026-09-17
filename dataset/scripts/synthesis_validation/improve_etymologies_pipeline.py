"""Unified pipeline: analyze errors, retry with normalization, show improvements."""
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from wiktionary_fetcher import WiktionaryFetcher

BASE_DIR = Path(__file__).parent.parent.parent / "splits" / "iecor_kaikki_koebler_normalized"
CACHE_FILE = BASE_DIR / "wiktionary_etymologies.json"
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


def analyze_errors(cache):
    """Analyze and categorize errors in the cache."""
    errors = {
        'no_mapping': [],           # [No Wiktionary mapping for XXX]
        'word_not_found': [],       # [Word not found in Wiktionary]
        'no_section': [],           # [No XXX section found]
        'no_etymology': [],         # [No etymology section found]
        'valid': [],                # Successfully parsed
    }

    for key, value in cache.items():
        if value.startswith('[No Wiktionary mapping'):
            errors['no_mapping'].append(key)
        elif '[Word not found' in value:
            errors['word_not_found'].append(key)
        elif 'section found' in value and '[No' in value:
            errors['no_section'].append(key)
        elif 'No etymology' in value:
            errors['no_etymology'].append(key)
        elif not value.startswith('['):
            errors['valid'].append(key)

    return errors


def identify_fixable_patterns(word_not_found_list):
    """Identify fixable patterns in word_not_found errors."""
    fixable_patterns = {
        'with_parentheses': [],   # (com)periō
        'with_dashes': [],        # -asts'
        'with_asterisk': [],      # *‑kla
        'with_macrons': [],       # grūad
    }

    for key in word_not_found_list:
        lang, word = key.split('|', 1)

        if '(' in word or ')' in word:
            fixable_patterns['with_parentheses'].append(key)
        elif word.startswith('-') or word.endswith('-'):
            fixable_patterns['with_dashes'].append(key)
        elif word.startswith('*'):
            fixable_patterns['with_asterisk'].append(key)
        elif any(c in word for c in ['ā', 'ē', 'ī', 'ō', 'ū', 'Ā', 'Ē', 'Ī', 'Ō', 'Ū']):
            fixable_patterns['with_macrons'].append(key)

    return fixable_patterns


async def main():
    print("="*80)
    print("WIKTIONARY ETYMOLOGY IMPROVEMENT PIPELINE")
    print("="*80 + "\n")

    # Step 1: Load cache and analyze
    print("Step 1: Analyzing errors in cache...")
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    errors_before = analyze_errors(cache)

    print(f"\nInitial statistics:")
    print(f"  Total entries: {len(cache)}")
    print(f"  Valid etymologies: {len(errors_before['valid'])} ({len(errors_before['valid'])/len(cache)*100:.1f}%)")
    print(f"  No mapping: {len(errors_before['no_mapping'])} ({len(errors_before['no_mapping'])/len(cache)*100:.1f}%)")
    print(f"  Word not found: {len(errors_before['word_not_found'])} ({len(errors_before['word_not_found'])/len(cache)*100:.1f}%)")
    print(f"  No section: {len(errors_before['no_section'])} ({len(errors_before['no_section'])/len(cache)*100:.1f}%)")
    print(f"  No etymology: {len(errors_before['no_etymology'])} ({len(errors_before['no_etymology'])/len(cache)*100:.1f}%)")

    # Step 2: Identify fixable patterns
    print("\nStep 2: Identifying fixable patterns...")
    fixable_patterns = identify_fixable_patterns(errors_before['word_not_found'])

    total_fixable = sum(len(v) for v in fixable_patterns.values())
    print(f"\nFixable patterns found:")
    for pattern, keys in fixable_patterns.items():
        print(f"  {pattern}: {len(keys)}")

    print(f"\nTotal potentially fixable: {total_fixable}")

    if total_fixable == 0:
        print("\nNo fixable words found. Pipeline complete.")
        return

    # Step 3: Prepare retry list with normalization
    print("\nStep 3: Preparing normalized retry list...")
    retry_pairs = []
    original_to_normalized = {}  # Track what we normalized

    for pattern, keys in fixable_patterns.items():
        for key in keys:
            lang, word = key.split('|', 1)
            normalized = normalize_word(word, lang)

            if normalized and normalized != word:
                retry_pairs.append((normalized, lang))
                original_to_normalized[key] = f"{lang}|{normalized}"

    print(f"Words to retry with normalization: {len(retry_pairs)}")

    if not retry_pairs:
        print("No words could be normalized. Pipeline complete.")
        return

    # Step 4: Fetch with normalization
    print("\nStep 4: Fetching etymologies for normalized words...")
    print("(This may take several minutes depending on number of words)\n")

    fetcher = WiktionaryFetcher()
    results = await fetcher.fetch_batch(retry_pairs)

    # Step 5: Count improvements
    print("\nStep 5: Analyzing improvements...")
    improved = sum(1 for v in results.values() if not v.startswith('['))
    still_failed = len(results) - improved

    # Reload cache to see updated state
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache_after = json.load(f)

    errors_after = analyze_errors(cache_after)

    # Step 6: Show results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print(f"Normalization retry results:")
    print(f"  Successfully found: {improved}/{len(retry_pairs)} ({improved/len(retry_pairs)*100:.1f}%)")
    print(f"  Still failed: {still_failed}")

    print(f"\nOverall improvement:")
    print(f"  Valid before: {len(errors_before['valid'])} ({len(errors_before['valid'])/len(cache)*100:.1f}%)")
    print(f"  Valid after: {len(errors_after['valid'])} ({len(errors_after['valid'])/len(cache)*100:.1f}%)")
    print(f"  Improvement: +{len(errors_after['valid']) - len(errors_before['valid'])} etymologies")

    # Show examples of successful normalizations
    if improved > 0:
        print(f"\nSuccessful normalizations (first 10):")
        count = 0
        for orig_key, norm_key in original_to_normalized.items():
            if norm_key in results and not results[norm_key].startswith('['):
                orig_lang, orig_word = orig_key.split('|', 1)
                norm_lang, norm_word = norm_key.split('|', 1)
                print(f"  {orig_word} -> {norm_word} ({orig_lang})")
                count += 1
                if count >= 10:
                    break

    # Save error analysis for reference
    output_data = {
        'statistics_before': {k: len(v) for k, v in errors_before.items()},
        'statistics_after': {k: len(v) for k, v in errors_after.items()},
        'fixable_patterns': {k: v for k, v in fixable_patterns.items()},
        'improvement': {
            'retried': len(retry_pairs),
            'succeeded': improved,
            'failed': still_failed,
        }
    }

    with open(ERRORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nError analysis saved to: {ERRORS_FILE}")
    print(f"Cache updated: {CACHE_FILE}")
    print("\nPipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
