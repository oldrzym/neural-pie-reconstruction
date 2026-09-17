"""
Создаёт новый сплит для эксперимента zero-shot transfer на язык.

Логика:
- Train/Val: Все строки, но удалены подстроки целевого языка (например [lat] word)
- Test: Только подстроки целевого языка (из записей где он был)

Использование:
    python exclude_language.py \
        --input-dir ../splits/combined_all_simplified \
        --output-dir ../splits/combined_all_simplified_no_lat \
        --exclude-langs lat

    # Несколько языков:
    python exclude_language.py \
        --input-dir ../splits/combined_all_simplified \
        --output-dir ../splits/combined_all_simplified_no_greek \
        --exclude-langs grc ell gmy
"""

import argparse
import pandas as pd
import re
from pathlib import Path


def parse_input(input_str: str) -> list[tuple[str, str]]:
    """
    Парсит input строку в список пар (lang, word).

    Пример: "[lat] cortex [grc] phloios" -> [("lat", "cortex"), ("grc", "phloios")]
    """
    # Паттерн: [lang] word (word может содержать пробелы до следующего [)
    pattern = r'\[([^\]]+)\]\s*([^\[]+)'
    matches = re.findall(pattern, input_str)
    return [(lang.strip(), word.strip()) for lang, word in matches]


def build_input(pairs: list[tuple[str, str]]) -> str:
    """
    Собирает input строку из списка пар.
    """
    return ' '.join(f'[{lang}] {word}' for lang, word in pairs)


def process_row(row, exclude_langs: set[str]) -> dict:
    """
    Обрабатывает одну строку.

    Возвращает dict с:
    - input_without_target: input без целевых языков
    - target_only_input: input только с целевыми языками
    - had_target: была ли подстрока целевого языка
    """
    pairs = parse_input(row['input'])

    without_target = [(lang, word) for lang, word in pairs if lang not in exclude_langs]
    target_only = [(lang, word) for lang, word in pairs if lang in exclude_langs]

    return {
        'input_without_target': build_input(without_target) if without_target else '',
        'target_only_input': build_input(target_only) if target_only else '',
        'had_target': len(target_only) > 0,
        'output': row['output'],
        'meaning': row.get('meaning', ''),
        'num_cognates_original': row.get('num_cognates', ''),
        'num_cognates_without_target': len(without_target),
        'num_cognates_target_only': len(target_only),
        'cognate_set_id': row.get('cognate_set_id', ''),
        'source': row.get('source', ''),
        'pos': row.get('pos', ''),
    }


def main():
    parser = argparse.ArgumentParser(description='Create zero-shot transfer split')
    parser.add_argument('--input-dir', required=True, help='Input splits directory')
    parser.add_argument('--output-dir', required=True, help='Output splits directory')
    parser.add_argument('--exclude-langs', nargs='+', required=True,
                        help='Language codes to exclude (e.g., lat grc)')
    parser.add_argument('--min-cognates', type=int, default=1,
                        help='Minimum cognates in train/val after removal (default: 1)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exclude_langs = set(args.exclude_langs)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Excluding languages: {exclude_langs}")
    print()

    # Загрузить все сплиты
    train_df = pd.read_csv(input_dir / 'train.csv')
    val_df = pd.read_csv(input_dir / 'val.csv')
    test_df = pd.read_csv(input_dir / 'test.csv')

    # Объединить
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Total rows: {len(all_df)}")

    # Обработать каждую строку
    processed = [process_row(row, exclude_langs) for _, row in all_df.iterrows()]
    processed_df = pd.DataFrame(processed)

    # Статистика
    had_target = processed_df['had_target'].sum()
    print(f"Rows with target language(s): {had_target}")
    print(f"Rows without target language(s): {len(processed_df) - had_target}")
    print()

    # Train/Val: строки без целевого языка (или с удалённым)
    # Фильтруем строки где осталось >= min_cognates после удаления
    train_val_mask = processed_df['num_cognates_without_target'] >= args.min_cognates
    train_val_df = processed_df[train_val_mask].copy()

    # Переименовываем колонки для train/val
    train_val_df = train_val_df.rename(columns={
        'input_without_target': 'input',
        'num_cognates_without_target': 'num_cognates'
    })
    train_val_df = train_val_df[['input', 'output', 'meaning', 'num_cognates',
                                   'cognate_set_id', 'source', 'pos']]

    # Разделить train/val (80/20)
    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(train_val_df) * 0.8)
    new_train_df = train_val_df[:split_idx]
    new_val_df = train_val_df[split_idx:]

    print(f"Train rows: {len(new_train_df)}")
    print(f"Val rows: {len(new_val_df)}")

    # Test: только подстроки целевого языка
    test_mask = processed_df['had_target'] & (processed_df['num_cognates_target_only'] >= 1)
    test_df = processed_df[test_mask].copy()

    # Переименовываем колонки для test
    test_df = test_df.rename(columns={
        'target_only_input': 'input',
        'num_cognates_target_only': 'num_cognates'
    })
    test_df = test_df[['input', 'output', 'meaning', 'num_cognates',
                        'cognate_set_id', 'source', 'pos']]

    print(f"Test rows: {len(test_df)}")
    print()

    # Сохранить
    output_dir.mkdir(parents=True, exist_ok=True)

    new_train_df.to_csv(output_dir / 'train.csv', index=False)
    new_val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)

    print(f"Saved to {output_dir}")
    print()

    # Примеры
    print("=" * 60)
    print("EXAMPLES")
    print("=" * 60)

    print("\nTrain/Val sample (without target language):")
    for _, row in new_train_df.head(3).iterrows():
        print(f"  Input:  {row['input'][:80]}...")
        print(f"  Output: {row['output']}")
        print()

    print("\nTest sample (target language only):")
    for _, row in test_df.head(3).iterrows():
        print(f"  Input:  {row['input'][:80]}...")
        print(f"  Output: {row['output']}")
        print()


if __name__ == '__main__':
    main()
