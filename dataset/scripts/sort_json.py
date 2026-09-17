import json

# Чтение JSON из файла
with open('language_counts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Сортировка словаря по значениям (убывающий порядок)
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# Запись отсортированного словаря обратно в файл
with open('sorted_data.json', 'w', encoding='utf-8') as file:
    json.dump(sorted_data, file, indent=4, ensure_ascii=False)

print("Данные отсортированы и сохранены в 'sorted_data.json'")
