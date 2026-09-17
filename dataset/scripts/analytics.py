import pandas as pd
import re
import json
import os

# Функция для извлечения языков из данных
def extract_languages_from_csv(file_path):
    languages = {}
    df = pd.read_csv(file_path)
    for column in df.columns:
        if df[column].dtype == 'object':  # Работаем только с текстовыми столбцами
            for entry in df[column]:
                if isinstance(entry, str):
                    matches = re.findall(r'\[([a-zA-Z]{3})\]', entry)  # Находим языковые коды
                    for match in matches:
                        match = match.lower()
                        if match not in languages:
                            languages[match] = 1
                        else:
                            languages[match] += 1
    return languages

# Функция для обработки всех файлов в папке
def process_folder(folder_path):
    all_languages = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            file_languages = extract_languages_from_csv(file_path)
            # Суммируем данные из всех файлов
            for lang, count in file_languages.items():
                if lang not in all_languages:
                    all_languages[lang] = count
                else:
                    all_languages[lang] += count
    # Сохранение в JSON
    with open('language_counts.json', 'w', encoding='utf-8') as f:
        json.dump(all_languages, f, ensure_ascii=False, indent=4)
    print("JSON file has been created with language counts.")

# Пример использования
folder_path = '../splits/iecor_kaikki_koebler_normalized'
process_folder(folder_path)
