"""Analyze Wiktionary parsing errors and identify fixable cases."""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent / "splits" / "iecor_kaikki_koebler_normalized"
CACHE_FILE = BASE_DIR / "wiktionary_etymologies.json"
OUTPUT_FILE = BASE_DIR / "wiktionary_errors.json"


def main():
    print("="*80)
    print("WIKTIONARY ERRORS ANALYSIS")
    print("="*80 + "\n")

    # Load completed cache
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    # Categorize errors
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

    # Print statistics
    print(f"Total entries: {len(cache)}")
    print(f"Valid etymologies: {len(errors['valid'])} ({len(errors['valid'])/len(cache)*100:.1f}%)")
    print(f"No mapping: {len(errors['no_mapping'])} ({len(errors['no_mapping'])/len(cache)*100:.1f}%)")
    print(f"Word not found: {len(errors['word_not_found'])} ({len(errors['word_not_found'])/len(cache)*100:.1f}%)")
    print(f"No section: {len(errors['no_section'])} ({len(errors['no_section'])/len(cache)*100:.1f}%)")
    print(f"No etymology: {len(errors['no_etymology'])} ({len(errors['no_etymology'])/len(cache)*100:.1f}%)")

    # Identify fixable cases in 'word_not_found'
    fixable_patterns = {
        'with_parentheses': [],   # (com)periō
        'with_dashes': [],        # -asts'
        'with_asterisk': [],      # *‑kla
        'with_macrons': [],       # grūad
    }

    for key in errors['word_not_found']:
        lang, word = key.split('|', 1)

        if '(' in word or ')' in word:
            fixable_patterns['with_parentheses'].append(key)
        elif word.startswith('-') or word.endswith('-'):
            fixable_patterns['with_dashes'].append(key)
        elif word.startswith('*'):
            fixable_patterns['with_asterisk'].append(key)
        elif any(c in word for c in ['ā', 'ē', 'ī', 'ō', 'ū', 'Ā', 'Ē', 'Ī', 'Ō', 'Ū']):
            fixable_patterns['with_macrons'].append(key)

    print("\nFixable cases:")
    total_fixable = 0
    for pattern, keys in fixable_patterns.items():
        print(f"  {pattern}: {len(keys)}")
        total_fixable += len(keys)
        if keys:
            print(f"    Examples: {keys[:5]}")

    print(f"\nTotal potentially fixable: {total_fixable}")

    # Save error lists for targeted retry
    output_data = {
        'statistics': {k: len(v) for k, v in errors.items()},
        'fixable_patterns': {k: v for k, v in fixable_patterns.items()},  # Save all
        'no_mapping_langs': sorted(list(set(k.split('|')[0] for k in errors['no_mapping']))),
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved error analysis to: {OUTPUT_FILE}")

    # Show missing languages
    if output_data['no_mapping_langs']:
        print(f"\nLanguages without mapping ({len(output_data['no_mapping_langs'])}):")
        for lang in output_data['no_mapping_langs']:
            count = sum(1 for k in errors['no_mapping'] if k.startswith(f'{lang}|'))
            print(f"  {lang}: {count} words")


if __name__ == "__main__":
    main()
