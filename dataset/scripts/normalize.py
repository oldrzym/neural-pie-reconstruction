import pandas as pd
import re

# Маппинг для нормализации кодов
synonym_map = {
    'alb': 'sqi',
    'arm': 'hye',
    'baq': 'eus',
    'bur': 'mya',
    'chi': 'zho',
    'cze': 'ces',
    'dut': 'nld',
    'fre': 'fra',
    'geo': 'kat',
    'ger': 'deu',
    'gre': 'ell',
    'ice': 'isl',
    'mac': 'mkd',
    'mao': 'mri',
    'may': 'msa',
    'mol': 'ron',
    'per': 'fas',
    'rum': 'ron',
    'scc': 'srp',
    'scr': 'hrv',
    'slo': 'slk',
    'tib': 'bod',
    'wel': 'cym'
}

# Функция для замены кода в квадратных скобках
def normalize_language_codes(text):
    # Проверяем, что это строка, если нет - приводим к строке
    if not isinstance(text, str):
        text = str(text)

    # Находим все языковые коды в квадратных скобках
    matches = re.findall(r'\[([a-zA-Z]{3})\]', text)

    for match in matches:
        # Если код найден в маппинге, заменяем его на нормализованный
        normalized_code = synonym_map.get(match.lower(), match.lower())
        text = text.replace(f'[{match}]', f'[{normalized_code}]')

    return text

# Функция для обработки CSV
def process_csv(input_file, output_file):
    # Загружаем CSV в DataFrame
    df = pd.read_csv(input_file)

    # Применяем нормализацию ко всем строковым столбцам
    for column in df.columns:
        if df[column].dtype == 'object':  # Применяем только к строковым столбцам
            df[column] = df[column].apply(normalize_language_codes)

    # Сохраняем результат в новый CSV файл
    df.to_csv(output_file, index=False)

# Пример использования
input_file = 'dataset/splits/iecor_kaikki_koebler/val.csv'  # Путь к исходному CSV
output_file = 'dataset/splits/iecor_kaikki_koebler_normalized/val.csv'  # Путь к сохранённому файлу

process_csv(input_file, output_file)
