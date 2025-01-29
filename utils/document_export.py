import sys
sys.path.append("/Users/max/PycharmProjects/Topic")
from odm.structure import Constitution
import asyncio
from pathlib import Path
from beanie import init_beanie
#from main import init_db
from motor.motor_asyncio import AsyncIOMotorClient
import config

async def init_db():
    client = AsyncIOMotorClient(config.mongodb_connection)
    await init_beanie(
        database=client.LegalDocuments,
        document_models=[Constitution]
    )


async def export_variant(output_directory: str = "data/export"):
    """
    Экспортирует результаты обработки в файлы.
    
    Args:
        output_directory (str): Директория для сохранения файлов экспорта

    Удалить пустую строку между вопросом и ответом vscode
    (question:.+?\n)\n(answer:.+?)
    $1$2
    """
    
    # Создаем директорию если её нет
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Получаем все документы где original_text полностью обработан
    documents = await Constitution.find(
        {"original_text": {"$not": {"$elemMatch": {"processed": False}}}}).to_list()
    
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


# TODO не закончил выгрузку в файл, решил сделать напрямую в dataset из mongo
async def export_ru_original():
    constitution_ru = await Constitution.find_all().to_list()

    for state in constitution_ru:
        print(state.original_text)
        print()
    #print(constitution_uk[0].original_text[2].text)


    pass

async def main():
    await init_db()
    await export_variant()
    #await export_ukr_original()

if __name__ == "__main__":
    asyncio.run(main())