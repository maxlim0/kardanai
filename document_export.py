async def export_variant(output_directory: str = "data/export"):
    """
    Экспортирует результаты обработки в файлы.
    
    Args:
        output_directory (str): Директория для сохранения файлов экспорта
    """
    from pathlib import Path
    from odm.structure import Constitution, Variant, LLMAnswer
    
    # Создаем директорию если её нет
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Получаем все документы где original_text полностью обработан
    documents = await Constitution.find(
        {"original_text": {"$not": {"$elemMatch": {"processed": False}}}}
    ).to_list()
    
    if not documents:
        print("Нет полностью обработанных документов для экспорта")
        return
    
    for document in documents:
        base_filename = document.file_name.rsplit('.', 1)[0]  # Убираем расширение
        
        # Для каждого варианта создаем отдельный файл
        for idx, variant in enumerate(document.variants):
            # Формируем имя файла
            if len(document.variants) > 1:
                output_filename = f"{base_filename}_variant_{idx}.txt"
            else:
                output_filename = f"{base_filename}.txt"
                
            output_file = output_path / output_filename
            
            try:
                # Записываем ответы в файл
                with open(output_file, 'w', encoding='utf-8') as f:
                    for answer in variant.llm_answers:
                        f.write(answer.answer)
                        f.write('\n\n')
                    
                print(f"Успешно экспортирован файл: {output_filename}")
                
            except Exception as e:
                print(f"Ошибка при экспорте файла {output_filename}: {e}")
                continue

