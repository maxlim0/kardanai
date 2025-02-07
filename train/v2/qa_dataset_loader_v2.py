import glob
import json
import os
import re
from datasets import Dataset 

def constitution_load_datasets(ds_source_dir):
    if os.path.isfile(ds_source_dir):
        files = [ds_source_dir]
    else:
        files = glob.glob(os.path.join(ds_source_dir, "*.qa"))
    
    texts = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            examples = text.strip().split('\n\n')
            texts.extend(examples)
        print(f"Количество строк в данных: {len(text)} {file_path}")

    data = []
    for example in texts:
        example = example.strip()

        if re.search(r'Вопрос:\s*', example, re.IGNORECASE) and re.search(r'\nОтвет:\s*', example, re.IGNORECASE):
            try:
                parts = example.split('\nОтвет:')
                if len(parts) < 2:
                    print(f"Ошибка: не удалось разделить строку: {repr(example)}")
                    continue

                question = parts[0].replace("Вопрос:", '').strip()
                answer = parts[1].replace("Ответ:", '').strip()

                data.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                        ]
                    }
                )

            except Exception as e:
                print(f"Ошибка при обработке строки: {example}. Ошибка: {e}")
                continue
    return Dataset.from_list(data)

# def main():
#     ds = (constitution_load_datasets("/Users/max/PycharmProjects/Topic/data/export"))

# if __name__ == "__main__":
#     main()

        