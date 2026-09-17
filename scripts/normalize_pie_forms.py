"""
Normalize PIE proto-forms across different datasets to unified notation.

Goals:
1. Unify palatalized velars: k̑ -> ḱ, g̑ -> ǵ
2. Normalize laryngeals: H -> h (keep subscripts h₁, h₂, h₃)
3. Apply Unicode NFC normalization
4. Remove extra spaces
5. Preserve all linguistic information

Test inside this script before applying to datasets.
"""

import pandas as pd
import unicodedata
import re
from collections import Counter
from pathlib import Path
import sys

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class PIENormalizer:
    """Normalize PIE proto-forms to unified notation."""

    def __init__(self):
        """Initialize normalization rules."""
        # Character replacements (specific cases)
        self.replacements = {
            # Palatalized velars: combining diacritics -> precomposed
            'k\u0311': 'ḱ',  # k + combining caron below -> ḱ
            'g\u0311': 'ǵ',  # g + combining caron below -> ǵ
            'k\u0301': 'ḱ',  # k + combining acute -> ḱ
            'g\u0301': 'ǵ',  # g + combining acute -> ǵ

            # Capital H to lowercase (only when not followed by subscript)
            # We'll handle this separately to preserve meaning

            # Remove zero-width characters
            '\u200e': '',  # LEFT-TO-RIGHT MARK
            '\u200f': '',  # RIGHT-TO-LEFT MARK
            '\ufeff': '',  # ZERO WIDTH NO-BREAK SPACE
        }

    def normalize_laryngeals(self, form: str) -> str:
        """
        Normalize laryngeal notation.

        Rules:
        - Keep h₁, h₂, h₃ as is (already normalized)
        - Convert standalone H to h (when not part of H₁, H₂, H₃)
        - Preserve H in compound forms like "temH-" -> "temh-"
        """
        # Replace H with h, but preserve subscripts
        # First, protect subscripted forms
        form = re.sub(r'H([₁₂₃])', r'h\1', form)  # H₁ -> h₁

        # Then convert standalone H to h
        # But only if not followed by a subscript digit
        form = re.sub(r'H(?![₁₂₃])', 'h', form)

        return form

    def normalize_form(self, form: str) -> str:
        """
        Apply full normalization pipeline to a PIE form.

        Steps:
        1. Strip whitespace
        2. Apply character replacements
        3. Unicode NFC normalization (compose diacritics)
        4. Normalize laryngeals
        5. Remove duplicate spaces
        6. Final cleanup
        """
        if pd.isna(form):
            return form

        form = str(form)

        # 1. Strip leading/trailing whitespace
        form = form.strip()

        # 2. Apply character-level replacements
        for old, new in self.replacements.items():
            form = form.replace(old, new)

        # 3. Unicode NFC normalization (compose combining diacritics)
        # This converts combining diacritics to precomposed characters where possible
        form = unicodedata.normalize('NFC', form)

        # 4. Normalize laryngeals
        form = self.normalize_laryngeals(form)

        # 5. Clean up multiple spaces
        form = re.sub(r'\s+', ' ', form)

        # 6. Final strip
        form = form.strip()

        return form

    def analyze_forms(self, forms: list, label: str = "Dataset") -> dict:
        """
        Analyze special characters in PIE forms.

        Returns:
        - unique_chars: set of unique characters
        - char_counts: Counter of character frequencies
        - special_chars: categorized special characters
        """
        all_chars = set()
        char_counter = Counter()

        for form in forms:
            if pd.notna(form):
                form_str = str(form)
                all_chars.update(form_str)
                char_counter.update(form_str)

        # Categorize special characters
        special_chars = {
            'laryngeals': set(),
            'palatalized': set(),
            'aspirated': set(),
            'long_vowels': set(),
            'combining_diacritics': set(),
            'subscripts': set(),
            'other': set()
        }

        for char in all_chars:
            code = ord(char)
            cat = unicodedata.category(char)

            # Laryngeals
            if char in ['h', 'H']:
                special_chars['laryngeals'].add(char)
            # Subscripts
            elif code in range(0x2080, 0x20A0):  # Subscript numbers
                special_chars['subscripts'].add(char)
            # Palatalized
            elif char in ['ḱ', 'ǵ']:
                special_chars['palatalized'].add(char)
            # Aspirated/labialized
            elif char in ['ʰ', 'ʷ', 'ʲ']:
                special_chars['aspirated'].add(char)
            # Long vowels (with macron)
            elif char in ['ā', 'ē', 'ī', 'ō', 'ū', 'ḗ', 'ṓ']:
                special_chars['long_vowels'].add(char)
            # Combining diacritics
            elif cat == 'Mn':  # Mark, Nonspacing
                special_chars['combining_diacritics'].add(char)
            # Other non-ASCII
            elif code > 127 and char not in ['*', '-', '(', ')', ',', ' ']:
                special_chars['other'].add(char)

        return {
            'label': label,
            'total_forms': len([f for f in forms if pd.notna(f)]),
            'unique_chars': all_chars,
            'char_counts': char_counter,
            'special_chars': special_chars
        }

    def print_analysis(self, analysis: dict):
        """Print analysis results."""
        print(f"\n{'='*60}")
        print(f"{analysis['label']}")
        print('='*60)
        print(f"Total forms: {analysis['total_forms']:,}")
        print(f"Unique characters: {len(analysis['unique_chars'])}")

        for category, chars in analysis['special_chars'].items():
            if chars:
                print(f"\n{category.upper()}:")
                for char in sorted(chars):
                    try:
                        name = unicodedata.name(char, 'UNKNOWN')
                        print(f"  {char} (U+{ord(char):04X}) - {name}")
                    except:
                        print(f"  {char} (U+{ord(char):04X})")


def test_normalization():
    """Test normalization on sample forms."""
    print("="*60)
    print("NORMALIZATION TESTING")
    print("="*60)

    normalizer = PIENormalizer()

    # Test cases from different datasets
    test_cases = [
        # IE-CoR examples (k with caron below)
        "*k̑elH-",
        "*k̑u̯ón-",
        "*h₂ōu̯ió-",
        "*h₂ster-",
        "*h₃mei̯gʰ-",

        # Kaikki examples (ḱ with acute)
        "*ḱwṓ",
        "*ǵʰóstis",
        "*h₁eḱwos",

        # Etymology-db examples
        "*ḱwn-i-",
        "*pṓds",
        "*priHós",
        "*preyH-",
        "*werdʰh₁om",

        # Edge cases
        "*temH-",  # Capital H at end
        "*H₂éwis",  # Capital H with subscript
        "* pḗd-s ",  # Extra spaces
    ]

    print("\nTest cases:")
    print(f"{'Original':<30} -> {'Normalized':<30} {'Changed?'}")
    print("-"*75)

    changes = []
    for original in test_cases:
        normalized = normalizer.normalize_form(original)
        changed = "YES" if original != normalized else ""
        print(f"{original:<30} -> {normalized:<30} {changed}")
        if original != normalized:
            changes.append((original, normalized))

    print(f"\nTotal test cases: {len(test_cases)}")
    print(f"Changed: {len(changes)}")

    # Analyze before/after
    print("\n" + "="*60)
    print("CHARACTER ANALYSIS - BEFORE")
    analysis_before = normalizer.analyze_forms(test_cases, "Before Normalization")
    normalizer.print_analysis(analysis_before)

    normalized_forms = [normalizer.normalize_form(f) for f in test_cases]
    print("\n" + "="*60)
    print("CHARACTER ANALYSIS - AFTER")
    analysis_after = normalizer.analyze_forms(normalized_forms, "After Normalization")
    normalizer.print_analysis(analysis_after)

    # Show what was eliminated
    print("\n" + "="*60)
    print("NORMALIZATION EFFECTS")
    print("="*60)

    eliminated = analysis_before['unique_chars'] - analysis_after['unique_chars']
    if eliminated:
        print("\nEliminated characters:")
        for char in sorted(eliminated):
            print(f"  {char} (U+{ord(char):04X}) - {unicodedata.name(char, 'UNKNOWN')}")
    else:
        print("\nNo characters eliminated")

    # Combining diacritics check
    before_combining = analysis_before['special_chars']['combining_diacritics']
    after_combining = analysis_after['special_chars']['combining_diacritics']

    print(f"\nCombining diacritics before: {len(before_combining)}")
    print(f"Combining diacritics after: {len(after_combining)}")
    if before_combining and not after_combining:
        print("YES All combining diacritics successfully composed!")


def test_on_real_datasets():
    """Test normalization on real datasets."""
    print("\n" + "="*60)
    print("TESTING ON REAL DATASETS")
    print("="*60)

    normalizer = PIENormalizer()

    datasets = [
        "dataset/processed/iecor.csv",
        "dataset/processed/kaikki.csv",
        "dataset/processed_with_codes/etymology_db_pie_direct.csv",
    ]

    for path in datasets:
        if not Path(path).exists():
            print(f"\nSkipping {path} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {Path(path).name}")
        print('='*60)

        df = pd.read_csv(path)
        original_forms = df['output'].dropna().tolist()

        # Normalize
        normalized_forms = [normalizer.normalize_form(f) for f in original_forms]

        # Count changes
        changes = sum(1 for o, n in zip(original_forms, normalized_forms) if o != n)

        print(f"Total forms: {len(original_forms):,}")
        print(f"Forms changed: {changes:,} ({100*changes/len(original_forms):.1f}%)")

        # Show samples of changes
        if changes > 0:
            print("\nSample changes (first 5):")
            count = 0
            for o, n in zip(original_forms, normalized_forms):
                if o != n and count < 5:
                    print(f"  {o} -> {n}")
                    count += 1

        # Character analysis
        analysis_before = normalizer.analyze_forms(original_forms, "Before")
        analysis_after = normalizer.analyze_forms(normalized_forms, "After")

        print(f"\nUnique characters before: {len(analysis_before['unique_chars'])}")
        print(f"Unique characters after: {len(analysis_after['unique_chars'])}")

        # Show combining diacritics
        before_comb = analysis_before['special_chars']['combining_diacritics']
        after_comb = analysis_after['special_chars']['combining_diacritics']

        if before_comb:
            print(f"\nCombining diacritics before: {len(before_comb)}")
            for char in sorted(before_comb):
                print(f"  {char} (U+{ord(char):04X})")

        if after_comb:
            print(f"\nCombining diacritics after: {len(after_comb)}")
            for char in sorted(after_comb):
                print(f"  {char} (U+{ord(char):04X})")
        else:
            if before_comb:
                print("\nYES All combining diacritics composed!")


if __name__ == "__main__":
    # Run tests
    test_normalization()
    test_on_real_datasets()
