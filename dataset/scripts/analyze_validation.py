"""Анализ результатов валидации: что было, что отфильтровано, что оставлено."""
import csv
import json
import re

SYNT_FILE = "dataset/splits/iecor_kaikki_koebler_normalized/train_synt.csv"
VALIDATED_FILE = "dataset/splits/iecor_kaikki_koebler_normalized/train_wikt_validated.csv"
BATCH_RESULTS = "dataset/splits/iecor_kaikki_koebler_normalized/batch_results.jsonl"

def parse_cognates(cognate_str):
    """Извлечь список [lang] word из строки когнатов."""
    pattern = r'\[([a-z]{3})\]\s*([^\[\]]+?)(?=\s*\[|$)'
    matches = re.findall(pattern, cognate_str)
    return [(lang.strip(), word.strip()) for lang, word in matches]

def main():
    # Загрузка данных
    with open(SYNT_FILE, 'r', encoding='utf-8') as f:
        synt_rows = list(csv.DictReader(f))

    with open(VALIDATED_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        validated_rows = list(reader)

    # Загрузка batch results
    batch_results = []
    with open(BATCH_RESULTS, 'r', encoding='utf-8') as f:
        for line in f:
            batch_results.append(json.loads(line))

    print(f"Loaded {len(synt_rows)} synthetic rows")
    print(f"Loaded {len(validated_rows)} validated rows")
    print(f"Loaded {len(batch_results)} batch results\n")
    print("="*100)

    # Анализ строк 10-19 (индексация с 0)
    # Validated file contains rows 10-19 at indices 0-9
    for local_idx in range(min(10, len(validated_rows))):
        actual_idx = 10 + local_idx  # Real row number in original dataset
        print(f"\n{'='*100}")
        print(f"ROW {actual_idx} (validated file index {local_idx})")
        print('='*100)

        # Validated row
        val_row = validated_rows[local_idx]
        val_input = val_row.get('input', '')
        val_output = val_row.get('output', '')
        pie_form = val_row.get('meaning', '') or val_row.get('PIE', '')

        # Synthetic row (исходные данные)
        synt_row = synt_rows[actual_idx]
        synt_cognates_str = synt_row['input']

        # Парсинг когнатов
        synt_cognates = parse_cognates(synt_cognates_str)
        val_input_cognates = parse_cognates(val_input)
        val_output_cognates = parse_cognates(val_output)

        print(f"PIE form: {pie_form}")
        print(f"\nSYNTHETIC ({len(synt_cognates)} cognates):")
        print(f"  {synt_cognates_str[:200]}...")

        print(f"\nVALIDATED INPUT ({len(val_input_cognates)} cognates):")
        print(f"  {val_input[:200]}...")

        print(f"\nVALIDATED OUTPUT ({len(val_output_cognates)} cognates):")
        print(f"  {val_output[:200]}...")

        # Что было отфильтровано между SYNT → VAL_INPUT
        synt_set = set(synt_cognates)
        val_input_set = set(val_input_cognates)
        val_output_set = set(val_output_cognates)

        removed_before_gpt = synt_set - val_input_set
        removed_by_gpt = val_input_set - val_output_set
        kept_by_gpt = val_input_set & val_output_set

        print(f"\nFILTERING SUMMARY:")
        print(f"  • Before GPT (Wiktionary filter): {len(removed_before_gpt)} removed")
        if removed_before_gpt:
            for lang, word in sorted(removed_before_gpt)[:5]:
                print(f"    - [{lang}] {word}")
            if len(removed_before_gpt) > 5:
                print(f"    ... and {len(removed_before_gpt) - 5} more")

        print(f"  • By GPT validation: {len(removed_by_gpt)} removed")
        if removed_by_gpt:
            for lang, word in sorted(removed_by_gpt)[:5]:
                print(f"    - [{lang}] {word}")
            if len(removed_by_gpt) > 5:
                print(f"    ... and {len(removed_by_gpt) - 5} more")

        print(f"  • Kept by GPT: {len(kept_by_gpt)} cognates")

        print(f"\nVALIDATION RATE: {len(val_output_cognates)}/{len(val_input_cognates)} = {100*len(val_output_cognates)/max(1,len(val_input_cognates)):.1f}%")

        # Найти reasoning из batch results
        if local_idx < len(batch_results):
            result = batch_results[local_idx]
            response = result.get('response', {})
            body = response.get('body', {})
            choices = body.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '')

                # Попытаться извлечь reasoning
                if '"valid_cognates"' in content:
                    # Есть JSON ответ
                    try:
                        data = json.loads(content)
                        reasoning = data.get('reasoning', '')
                        if reasoning:
                            print(f"\nGPT REASONING:")
                            print(f"  {reasoning[:300]}...")
                    except:
                        pass

if __name__ == "__main__":
    main()
