"""
Analyze PIE notation differences across datasets.
"""

import pandas as pd
from collections import Counter
import re
from pathlib import Path
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def extract_pie_forms(csv_path: str) -> list:
    """Extract all PIE forms from a dataset."""
    df = pd.read_csv(csv_path)
    return df['output'].dropna().tolist()

def analyze_special_chars(forms: list, dataset_name: str):
    """Analyze special characters and patterns in PIE forms."""
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print('='*60)

    # Find all unique characters
    all_chars = set()
    for form in forms:
        all_chars.update(str(form))

    # Categorize special chars
    special_chars = {
        'laryngeals': [],
        'palatalized': [],
        'aspirated': [],
        'long_vowels': [],
        'other_diacritics': []
    }

    for char in sorted(all_chars):
        code = ord(char)
        # Laryngeals (h with subscripts)
        if char in ['h', 'H', '₁', '₂', '₃', 'ₕ']:
            special_chars['laryngeals'].append(f"{char} (U+{code:04X})")
        # Palatalized k
        elif char in ['ḱ', 'k̑', 'ǵ', 'ǵʰ']:
            special_chars['palatalized'].append(f"{char} (U+{code:04X})")
        # Aspirated
        elif char in ['ʰ', 'ʷ', 'ʲ']:
            special_chars['aspirated'].append(f"{char} (U+{code:04X})")
        # Long vowels
        elif char in ['ā', 'ē', 'ī', 'ō', 'ū', 'ā́', 'ḗ', 'ṓ']:
            special_chars['long_vowels'].append(f"{char} (U+{code:04X})")
        # Other combining diacritics
        elif code > 127 and char not in ['*', '-', '(', ')']:
            special_chars['other_diacritics'].append(f"{char} (U+{code:04X})")

    # Print results
    print(f"Total PIE forms: {len(forms)}")
    print(f"Unique characters: {len(all_chars)}")

    for category, chars in special_chars.items():
        if chars:
            print(f"\n{category.upper()}:")
            for char in chars[:20]:  # Limit to 20
                try:
                    print(f"  {char}")
                except UnicodeEncodeError:
                    print(f"  {repr(char)}")
            if len(chars) > 20:
                print(f"  ... and {len(chars) - 20} more")

    # Pattern analysis
    print("\nPATTERNS:")

    # Laryngeal patterns
    h_patterns = [
        ('h₁', sum(1 for f in forms if 'h₁' in str(f))),
        ('h₂', sum(1 for f in forms if 'h₂' in str(f))),
        ('h₃', sum(1 for f in forms if 'h₃' in str(f))),
        ('H', sum(1 for f in forms if 'H' in str(f) and 'h' not in str(f).lower())),
    ]
    print("  Laryngeals:")
    for pattern, count in h_patterns:
        if count > 0:
            print(f"    {pattern}: {count} forms")

    # Palatalized patterns
    k_patterns = [
        ('ḱ (U+1E31)', sum(1 for f in forms if 'ḱ' in str(f))),
        ('k̑ (caron below)', sum(1 for f in forms if 'k̑' in str(f))),
        ('ǵ', sum(1 for f in forms if 'ǵ' in str(f))),
    ]
    print("  Palatalized velars:")
    for pattern, count in k_patterns:
        if count > 0:
            print(f"    {pattern}: {count} forms")

    # Sample forms
    print("\nSAMPLE FORMS (first 10):")
    for form in forms[:10]:
        try:
            print(f"  {form}")
        except UnicodeEncodeError:
            print(f"  {repr(form)}")

def compare_datasets():
    """Compare notation across all datasets."""
    print("="*60)
    print("PIE NOTATION COMPARISON")
    print("="*60)

    datasets = [
        ("dataset/processed/iecor.csv", "IE-CoR"),
        ("dataset/processed/kaikki.csv", "Kaikki"),
        ("dataset/processed_with_codes/etymology_db_pie_direct.csv", "Etymology-db (direct)"),
        ("dataset/processed_with_codes/etymology_db_pie_via_proto.csv", "Etymology-db (via proto)"),
    ]

    all_forms = {}
    for path, name in datasets:
        if Path(path).exists():
            forms = extract_pie_forms(path)
            all_forms[name] = forms
            analyze_special_chars(forms, name)

    # Find common PIE roots across datasets
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS")
    print("="*60)

    if len(all_forms) >= 2:
        # Normalize forms for comparison (remove -, *, etc.)
        def normalize_simple(form):
            return str(form).replace('*', '').replace('-', '').replace(' ', '').lower()

        normalized_sets = {
            name: set(normalize_simple(f) for f in forms)
            for name, forms in all_forms.items()
        }

        names = list(normalized_sets.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                overlap = normalized_sets[name1] & normalized_sets[name2]
                print(f"\n{name1} ∩ {name2}:")
                print(f"  Common forms: {len(overlap)}")
                print(f"  {name1} only: {len(normalized_sets[name1] - normalized_sets[name2])}")
                print(f"  {name2} only: {len(normalized_sets[name2] - normalized_sets[name1])}")

                # Show some common forms
                if overlap:
                    print(f"  Sample overlap: {list(overlap)[:5]}")

if __name__ == "__main__":
    compare_datasets()
