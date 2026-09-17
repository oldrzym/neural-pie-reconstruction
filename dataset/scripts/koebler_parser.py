"""
Parser for Köbler's Indogermanisches Wörterbuch.
https://www.koeblergerhard.de/idgwbhin.html

Extracts PIE protoforms with meanings and cognates.

Usage:
    python koebler_parser.py --output ../raw/koebler.csv
    python koebler_parser.py --output ../raw/koebler.csv --pages a b c  # specific pages only
"""

import argparse
import html
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import pandas as pd


# All available letter pages in Köbler dictionary
KOEBLER_PAGES = [
    'a', 'b', 'bh', 'd', 'dh', 'e', 'g', 'gh', 'gw', 'gwh',
    'h', 'i', 'k', 'kw', 'l', 'm', 'n', 'o', 'p', 'r', 's',
    't', 'u', 'w', 'y'
]

BASE_URL = "http://www.koeblergerhard.de/idg/idg_{}.html"


def fetch_page(letter: str, delay: float = 1.0) -> Optional[str]:
    """Fetch a single page from Köbler dictionary."""
    url = BASE_URL.format(letter)
    print(f"  Fetching {url}...")

    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0 PIE-Research'})
        with urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8', errors='replace')
        time.sleep(delay)
        return content
    except (HTTPError, URLError) as e:
        print(f"  Error fetching {url}: {e}")
        return None


def extract_cognates_from_w_field(w_text: str) -> List[Dict]:
    """
    Extract cognates from W.: field.
    Format: W.: gr. word, POS, meaning; W.: lat. word, POS, meaning; ...
    """
    cognates = []

    # Skip markers (s. = siehe, vgl. = vergleiche, Lw. = Lehnwort, Hw. = Hinweis)
    skip_markers = {'s', 'vgl', 'Lw', 'Hw'}

    # Split by "W.:" to get individual cognate entries
    w_parts = re.split(r'W\.:\s*', w_text)

    for part in w_parts:
        if not part.strip():
            continue

        # Try to extract language code and word
        # Format: "gr. ἆ (a), Interj., ach!" or "lat. amnis, M., Gewässer"
        match = re.match(r'([a-z]+)\.?\s+([^,;]+)', part.strip())
        if match:
            lang = match.group(1)

            # Skip reference markers
            if lang in skip_markers:
                continue

            word = match.group(2).strip()

            # Clean up word - remove asterisks for reconstructed forms
            word = word.lstrip('*').strip()

            # Remove parenthetical transliterations like "(agōn)"
            word = re.sub(r'\s*\([^)]+\)\s*$', '', word)

            # Skip if word is too long (likely parsing error), empty, or reconstructed
            if word and len(word) < 50 and not word.startswith('*'):
                cognates.append({
                    'lang': lang,
                    'word': word
                })

    return cognates


def parse_entry(entry_text: str) -> Optional[Dict]:
    """
    Parse a single dictionary entry.

    Returns dict with:
    - protoform: the PIE root
    - pos: part of speech
    - meaning_de: German meaning
    - meaning_en: English meaning
    - pokorny_ref: Pokorny reference
    - lang_branches: attested language branches
    - cognates: list of cognates
    """
    # Extract protoform (starts with *)
    # Format: *root, idg., ... or *root (1), idg., ...
    # Stop at comma or "idg."
    proto_match = re.match(r'(\*[^\s,]+(?:\s*\(\d+\))?)', entry_text)
    if not proto_match:
        return None

    protoform = proto_match.group(1).strip()
    # Remove number suffixes like "(1)" or "(2)"
    protoform = re.sub(r'\s*\(\d+\)\s*$', '', protoform)

    # Extract part of speech after "idg.,"
    pos_match = re.search(r'idg\.,\s*(\w+)\.?:', entry_text)
    pos = pos_match.group(1) if pos_match else ''

    # Extract German meaning (nhd.)
    nhd_match = re.search(r'nhd\.\s*([^;]+)', entry_text)
    meaning_de = nhd_match.group(1).strip() if nhd_match else ''

    # Extract English meaning (ne.)
    ne_match = re.search(r'ne\.\s*([^;]+)', entry_text)
    meaning_en = ne_match.group(1).strip() if ne_match else ''

    # Extract Pokorny reference (RB.:)
    rb_match = re.search(r'RB\.:\s*Pokorny\s+(\d+)\s*\([^)]+\)', entry_text)
    pokorny_ref = rb_match.group(1) if rb_match else ''

    # Extract language branches from RB field
    branches_match = re.search(r'RB\.:[^;]+,\s*([^;]+)', entry_text)
    if branches_match:
        branches_text = branches_match.group(1)
        # Extract language codes (ind., gr., lat., germ., etc.)
        lang_branches = re.findall(r'([a-z]+)\.', branches_text)
    else:
        lang_branches = []

    # Extract cognates from W.: fields
    w_section = entry_text[entry_text.find('W.:'):] if 'W.:' in entry_text else ''
    cognates = extract_cognates_from_w_field(w_section)

    return {
        'protoform': protoform,
        'pos': pos,
        'meaning_de': meaning_de,
        'meaning_en': meaning_en,
        'pokorny_ref': pokorny_ref,
        'lang_branches': lang_branches,
        'cognates': cognates
    }


def split_entries(html_content: str) -> List[str]:
    """Split HTML content into individual entries."""
    # Decode HTML entities first
    text = html.unescape(html_content)

    # Remove HTML tags but keep content
    text = re.sub(r'<[^>]+>', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Split by entry pattern - entries start with asterisk followed by word
    # Pattern: lookbehind for space/newline, then * followed by letter/diacritic
    entries = re.split(r'(?<=[;\.\s])\s*(?=\*[a-zA-Zāăēĕīĭōŏūŭḱǵĝk̑g̑h₁h₂h₃])', text)

    # Clean up and filter
    result = []
    for e in entries:
        e = e.strip()
        if e.startswith('*') and len(e) > 5 and 'idg.' in e:
            result.append(e)

    return result


def parse_page(html_content: str) -> List[Dict]:
    """Parse all entries from a page."""
    entries_text = split_entries(html_content)

    parsed = []
    for entry_text in entries_text:
        try:
            entry = parse_entry(entry_text)
            if entry and entry['protoform']:
                parsed.append(entry)
        except Exception as e:
            # Skip problematic entries
            continue

    return parsed


def to_training_format(entries: List[Dict]) -> List[Dict]:
    """
    Convert parsed entries to training format.

    Input format (cognates): [lat] word1 [grc] word2 ...
    Output format: *protoform
    """
    training_data = []

    # Map Köbler lang codes to our standard codes
    lang_map = {
        'gr': 'grc',      # Greek
        'lat': 'lat',     # Latin
        'germ': 'gem',    # Germanic
        'ahd': 'goh',     # Old High German
        'as': 'osx',      # Old Saxon
        'ae': 'ang',      # Old English
        'an': 'non',      # Old Norse
        'got': 'got',     # Gothic
        'ind': 'san',     # Sanskrit/Indic
        'iran': 'ira',    # Iranian
        'arm': 'hye',     # Armenian
        'alb': 'sqi',     # Albanian
        'kelt': 'cel',    # Celtic
        'slaw': 'sla',    # Slavic
        'balt': 'bal',    # Baltic
        'toch': 'txb',    # Tocharian
        'heth': 'hit',    # Hittite
    }

    for entry in entries:
        if not entry['cognates']:
            continue

        # Build input string
        cognate_parts = []
        for cog in entry['cognates']:
            lang = lang_map.get(cog['lang'], cog['lang'])
            word = cog['word']
            cognate_parts.append(f"[{lang}] {word}")

        if not cognate_parts:
            continue

        input_str = ' '.join(cognate_parts)
        output_str = entry['protoform']
        meaning = entry['meaning_en'] or entry['meaning_de']

        training_data.append({
            'input': input_str,
            'output': output_str,
            'meaning': meaning,
            'num_cognates': len(entry['cognates']),
            'pos': entry['pos'],
            'pokorny_ref': entry['pokorny_ref'],
            'source': 'koebler'
        })

    return training_data


def main():
    parser = argparse.ArgumentParser(description="Parse Köbler Indogermanisches Wörterbuch")
    parser.add_argument("--output", type=str, default="../raw/koebler.csv", help="Output CSV file")
    parser.add_argument("--pages", nargs='+', default=None, help="Specific pages to parse (e.g., a b c)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--raw-output", type=str, default=None, help="Also save raw parsed data")

    args = parser.parse_args()

    pages = args.pages or KOEBLER_PAGES

    print(f"Parsing Koebler dictionary...")
    print(f"Pages to fetch: {pages}")
    print()

    all_entries = []

    for page in pages:
        print(f"Processing page '{page}'...")
        html = fetch_page(page, delay=args.delay)

        if html:
            entries = parse_page(html)
            print(f"  Found {len(entries)} entries")
            all_entries.extend(entries)
        else:
            print(f"  Skipped (fetch failed)")

    print()
    print(f"Total entries parsed: {len(all_entries)}")

    # Convert to training format
    training_data = to_training_format(all_entries)
    print(f"Entries with cognates: {len(training_data)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(training_data)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Optionally save raw data
    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_df = pd.DataFrame(all_entries)
        raw_df.to_csv(raw_path, index=False)
        print(f"Raw data saved to {raw_path}")

    # Show sample (ASCII-safe)
    print()
    print("Sample entries:")
    print("-" * 60)
    for i, row in df.head(5).iterrows():
        try:
            print(f"Input:  {row['input'][:60].encode('ascii', 'replace').decode()}...")
            print(f"Output: {row['output'].encode('ascii', 'replace').decode()}")
            print(f"Meaning: {str(row['meaning'])[:40].encode('ascii', 'replace').decode()}")
            print()
        except Exception:
            print(f"  (entry {i} - display error)")
            print()


if __name__ == "__main__":
    main()
