"""Show example GPT request for a random row from train_synt.csv."""
import csv
import json
import re

# Read row 50 from train_synt.csv
with open('../../splits/iecor_kaikki_koebler_normalized/train_synt.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    row = rows[50]

# Read etymologies
with open('../../splits/iecor_kaikki_koebler_normalized/wiktionary_etymologies.json', encoding='utf-8') as f:
    etym_dict = json.load(f)

# Parse cognates
pattern = r'\[([a-z]{3})\]\s*([^\[\]]+?)(?=\s*\[|$)'
cognates = re.findall(pattern, row['input'])

# Filter valid cognates
valid_cognates = []
skipped_cognates = []

for lang, word in cognates:
    key = f'{lang}|{word.strip()}'
    etym = etym_dict.get(key)

    if etym and not etym.startswith('['):
        valid_cognates.append({
            'lang': lang,
            'word': word.strip(),
            'etymology': etym
        })
    else:
        skipped_cognates.append({
            'lang': lang,
            'word': word.strip(),
            'reason': etym or '[Not in cache]'
        })

# Write to file
with open('../../splits/iecor_kaikki_koebler_normalized/example_gpt_request.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("EXAMPLE GPT BATCH REQUEST (Row 50)\n")
    f.write("="*80 + "\n\n")

    f.write(f"PIE Root: {row['output']}\n")
    f.write(f"Meaning: {row['meaning']}\n")
    f.write(f"Source: {row['source']}\n\n")

    f.write(f"Original cognates: {len(cognates)}\n")
    f.write(f"Valid etymologies: {len(valid_cognates)}\n")
    f.write(f"Skipped (no etymology): {len(skipped_cognates)}\n\n")

    f.write("="*80 + "\n")
    f.write("GPT PROMPT (simplified version)\n")
    f.write("="*80 + "\n\n")

    f.write(f"You are an expert etymologist. Validate cognates for PIE root {row['output']} ({row['meaning']}).\n\n")

    f.write("For each cognate, check if its Wiktionary etymology supports connection to the PIE root.\n\n")

    f.write("COGNATES TO VALIDATE:\n\n")

    for i, cog in enumerate(valid_cognates, 1):
        f.write(f"{i}. [{cog['lang']}] {cog['word']}\n")
        f.write(f"   Wiktionary Etymology:\n")
        f.write(f"   {cog['etymology'][:200]}\n")
        if len(cog['etymology']) > 200:
            f.write("   ...\n")
        f.write("\n")

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("SKIPPED COGNATES (no Wiktionary etymology)\n")
    f.write("="*80 + "\n\n")

    for i, cog in enumerate(skipped_cognates[:10], 1):
        f.write(f"{i}. [{cog['lang']}] {cog['word']} -> {cog['reason'][:80]}\n")

    if len(skipped_cognates) > 10:
        f.write(f"\n... and {len(skipped_cognates) - 10} more skipped\n")

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("EXPECTED GPT RESPONSE FORMAT\n")
    f.write("="*80 + "\n\n")

    f.write("For each cognate, GPT would respond:\n")
    f.write("[eng] word - VALID (reasoning)\n")
    f.write("[deu] word - INVALID (reasoning)\n")
    f.write("...\n")

print("Example saved to: example_gpt_request.txt")
