import json
import os
from datetime import datetime
from itertools import groupby

def safe_get_field(field, default="", join_char=" "):
    """
    Безопасно извлекает поле из сообщения.
    Если поле является строкой, возвращает её с удаленными пробелами.
    Если поле является списком, объединяет элементы в строку с указанным разделителем.
    В противном случае возвращает значение по умолчанию.
    """
    if isinstance(field, str):
        return field.strip()
    elif isinstance(field, list):
        # Объединяем элементы списка, предполагая, что они строки
        return join_char.join(str(item).strip() for item in field if isinstance(item, str)).strip()
    elif field is None:
        return default
    else:
        # Преобразуем другие типы данных в строку
        return str(field).strip()

def format_messages(input_file, output_root_path):
    total_messages = 0
    filtered_messages = 0
    grouped_days = 0
    grouped_users = 0

    with open(input_file, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile)  # Загружаем JSON-файл
        except json.JSONDecodeError as e:
            print(f"Ошибка при загрузке JSON-файла: {e}")
            return

        # Проверяем, есть ли ключ "messages" и он содержит список
        if "messages" in data and isinstance(data["messages"], list):
            # Фильтруем и сортируем сообщения по дате
            # Только сообщения типа "message" с text_entities типа "plain" и непустым текстом
            messages_to_process = []
            for message in data["messages"]:
                total_messages += 1

                if message.get("type") != "message":
                    continue  # Пропускаем, если тип не "message"

                text = message.get("text", "")
                # Обрабатываем поле "text" независимо от его типа
                text = safe_get_field(text, default="")

                if not text:
                    continue  # Пропускаем, если текст пустой

                text_entities = message.get("text_entities", [])
                # Проверяем, есть ли хотя бы один текстовый элемент типа "plain"
                has_plain = any(
                    isinstance(entity, dict) and entity.get("type") == "plain"
                    for entity in text_entities
                )
                if not has_plain:
                    continue  # Пропускаем, если нет "plain" текста

                # Парсим дату и сохраняем вместе с сообщением
                date_str = message.get("date", "")
                try:
                    date_obj = datetime.fromisoformat(date_str)
                except ValueError:
                    continue  # Пропускаем, если дата некорректная

                # Получаем имя пользователя с обработкой различных типов
                username = safe_get_field(message.get("from", "Unknown User"), default="Unknown User")

                if not username:
                    username = "Unknown User"

                # Добавляем в список фильтрованных сообщений
                messages_to_process.append({
                    "from": username,
                    "date": date_obj,
                    "text": text
                })
                filtered_messages += 1

            print(f"Всего сообщений в исходном файле: {total_messages}")
            print(f"Сообщений после фильтрации: {filtered_messages}")

            if filtered_messages == 0:
                print("Нет сообщений для обработки после фильтрации.")
                return

            # Сортируем сообщения по дате
            messages_to_process.sort(key=lambda x: x["date"])

            # Группируем сообщения по дате (суткам)
            for date, group in groupby(messages_to_process, key=lambda x: x["date"].date()):
                grouped_days += 1

                year = date.strftime('%Y')
                month = date.strftime('%m')
                day = date.strftime('%d')
                year_path = os.path.join(output_root_path, year)
                month_path = os.path.join(year_path, month)

                # Создаем папки, если они не существуют
                os.makedirs(month_path, exist_ok=True)

                # Формируем имя файла: DD_MM_YYYY.txt
                file_name = f"{day}_{month}_{year}.txt"
                file_path = os.path.join(month_path, file_name)

                # Открываем файл для записи сообщений данного дня
                with open(file_path, 'w', encoding='utf-8') as outfile:
                    # Записываем заголовок с датой
                    date_header = date.strftime('%Y-%m-%d')
                    outfile.write(f"Date: {date_header}\n\n")

                    # Преобразуем group в список для многократного использования
                    messages_list = list(group)

                    # Группируем подряд идущие сообщения от одного пользователя
                    user_group_count = 0
                    for user, user_group in groupby(messages_list, key=lambda x: x["from"]):
                        # Собираем все сообщения от этого пользователя в одну строку
                        combined_text = ' '.join(msg["text"] for msg in user_group)
                        outfile.write(f"User: {user} Msg: {combined_text}\n")
                        grouped_users += 1
                        user_group_count += 1

                    # Добавляем визуальный разделитель между днями (опционально)
                    # Если необходимо, можно раскомментировать следующую строку
                    # outfile.write("\n" + "#" * 50 + "\n\n")

                print(f"Дата: {date_header} - Обработано групп пользователей: {user_group_count}")
            
            print(f"\nФорматирование завершено. Структура папок и файлов сохранена в '{output_root_path}'.")
            print(f"Всего сгруппированных дней: {grouped_days}")
            print(f"Всего сгруппированных пользовательских сообщений: {grouped_users}")

        else:
            print("Входной JSON не содержит ключа 'messages' или он не является списком.")

if __name__ == "__main__":
    # Укажите имена входного файла и корневого пути для вывода
    input_json_file = 'data/raw/fz_original.json'  # Имя файла JSON с данными
    output_root_path = 'data/processed/'  # Корневой путь для сохранения структуры папок

    format_messages(input_json_file, output_root_path)
