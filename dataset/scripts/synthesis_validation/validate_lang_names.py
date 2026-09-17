"""Validate language names against Wiktionary standard names."""

# Official Wiktionary language names (from https://en.wiktionary.org/wiki/Wiktionary:List_of_languages)
WIKTIONARY_OFFICIAL_NAMES = {
    # OLD/ANCIENT LANGUAGES - Check these carefully!
    "ang": "Old English",          # ✓ Correct
    "goh": "Old High German",      # ✓ Correct
    "osx": "Old Saxon",            # ✓ Correct
    "non": "Old Norse",            # ✓ Correct
    "sga": "Old Irish",            # ✓ Correct
    "chu": "Old Church Slavonic",  # ✓ Correct
    "grc": "Ancient Greek",        # ✓ Correct
    "lat": "Latin",                # ✓ Correct
    "san": "Sanskrit",             # ✓ Correct
    "xcl": "Old Armenian",         # ✓ Correct
    "hit": "Hittite",              # ✓ Correct
    "txb": "Tocharian B",          # ✓ Correct
    "xto": "Tocharian A",          # ✓ Correct
    "ave": "Avestan",              # ✓ Correct
    "pal": "Middle Persian",       # ✓ Correct
    "prg": "Old Prussian",         # ✓ Correct
    "osc": "Oscan",                # ✓ Correct
    "umb": "Umbrian",              # ✓ Correct
    "got": "Gothic",               # ✓ Correct
    "gmy": "Mycenaean Greek",      # Might not exist on Wiktionary
    "lyc": "Lycian",               # ✓ Correct
    "xlu": "Luwian",               # ✓ Correct
    "kho": "Khotanese",            # ✓ Correct
    "sog": "Sogdian",              # ✓ Correct

    # MODERN LANGUAGES
    "eng": "English",
    "deu": "German",
    "rus": "Russian",
    "pol": "Polish",
    "fra": "French",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",

    # DIALECTAL/REGIONAL - Check these!
    "aln": "Albanian",             # NOT "Gheg Albanian" - use standard Albanian
    "hbs": "Serbo-Croatian",       # ✓ Correct
    "nob": "Norwegian Bokmål",     # ✓ Correct (but "nor" = Norwegian also works)
    "hyw": "Western Armenian",     # ✓ Correct

    # POTENTIALLY PROBLEMATIC
    "old": "Old Occitan",          # Wiktionary might call it "Old Provençal" or "Old Occitan"
    "nea": "Neapolitan",           # ✓ Correct
    "sar": "Sardinian",            # ✓ Correct
    "wal": "Walloon",              # ✓ Correct
    "sor": "Sorbian",              # Might need "Upper Sorbian" or "Lower Sorbian"
    "fri": "Frisian",              # Might need "West Frisian"
    "gla": "Scottish Gaelic",      # ✓ Correct
    "gle": "Irish",                # ✓ Correct
    "cym": "Welsh",                # ✓ Correct
    "bre": "Breton",               # ✓ Correct

    # INDO-ARYAN
    "hin": "Hindi",
    "ben": "Bengali",
    "pun": "Punjabi",
    "urd": "Urdu",
    "nep": "Nepali",
    "sin": "Sinhala",              # or "Sinhalese"
    "mar": "Marathi",
    "bho": "Bhojpuri",
    "mai": "Maithili",
    "mag": "Magahi",

    # IRANIAN
    "fas": "Persian",
    "kur": "Kurdish",
    "oss": "Ossetian",
    "bal": "Baluchi",
    "pus": "Pashto",
    "kas": "Kashmiri",

    # Codes that might not exist or are ambiguous
    "air": None,    # Unknown - probably error in dataset
    "gem": "Proto-Germanic",  # Reconstruction language
    "ira": "Proto-Iranian",   # Reconstruction language
    "cel": "Proto-Celtic",    # Reconstruction language
}

# Compare with our mapping
from complete_lang_mapping import COMPLETE_LANG_TO_WIKT

print("="*80)
print("LANGUAGE NAME VALIDATION")
print("="*80 + "\n")

mismatches = []
for code, official_name in WIKTIONARY_OFFICIAL_NAMES.items():
    our_name = COMPLETE_LANG_TO_WIKT.get(code)

    if our_name != official_name:
        mismatches.append((code, our_name, official_name))

if mismatches:
    print(f"Found {len(mismatches)} potential mismatches:\n")
    for code, ours, official in mismatches:
        print(f"{code}:")
        print(f"  Our name:      {ours}")
        print(f"  Official name: {official}")
        print()
else:
    print("✓ All language names match!")

# Check for codes in our mapping that aren't validated
our_codes = set(COMPLETE_LANG_TO_WIKT.keys())
validated_codes = set(WIKTIONARY_OFFICIAL_NAMES.keys())
unvalidated = our_codes - validated_codes

if unvalidated:
    print(f"\n{len(unvalidated)} codes not yet validated:")
    for code in sorted(unvalidated):
        if COMPLETE_LANG_TO_WIKT[code] is not None:
            print(f"  {code}: {COMPLETE_LANG_TO_WIKT[code]}")
