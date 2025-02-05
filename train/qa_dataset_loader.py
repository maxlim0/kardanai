from datasets import Dataset
import glob
import os
import re

def constitution_load_dataset(ds_source_dir):
    # Проверяем, это директория или файл
    if os.path.isfile(ds_source_dir):
        files = [ds_source_dir]
    else:
        files = glob.glob(os.path.join(ds_source_dir, '*.qa'))

    texts = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            examples = text.strip().split('\n\n')
            texts.extend(examples)
        print(f"Количество строк в данных: {len(text)} {file_path}")
    
    data = []
    for example in texts:
        # Убираем лишние пробелы и переносы строк
        example = example.strip()

        # Отладка: выводим каждую строку
        #print(f"Обрабатываем строку: {repr(example)}")

        # Проверяем наличие ключевых слов
        if re.search(r'Вопрос:\s*', example, re.IGNORECASE) and re.search(r'\nОтвет:\s*', example, re.IGNORECASE):
            try:
                # Разделяем строку по ключевому слову "ответ:"
                parts = example.split('\nОтвет:')
                if len(parts) < 2:
                    print(f"Ошибка: не удалось разделить строку: {repr(example)}")
                    continue

                # Извлекаем вопрос и ответ
                question = parts[0].replace('"', '').strip()
                answer = "Ответ: " + parts[1].replace('"', '').strip()

                # Добавляем в итоговый список
                data.append({
                    "вопрос": question,
                    "ответ": answer
                })
                #print(f"Добавлено: Вопрос: {question}, Ответ: {answer}")

            except Exception as e:
                print(f"Ошибка при обработке строки: {example}. Ошибка: {e}")
                continue

    #print(data[4])    
    print(f"Количество примеров в датасете QA:{len(data)}") 
    return Dataset.from_list(data)

# dataset = constitution_load_dataset("data/export/")

# print(f"Количество примеров в датасете QA: {len(dataset)}")
# print(dataset.shape[0])  # Количество строк
# print(dataset.num_rows)  # Количество строк

# print(dataset[1000])
# или для красивого вывода
#print(f"Вопрос: {dataset[0]['question']}\nОтвет: {dataset[0]['answer']}")
