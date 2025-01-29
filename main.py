from doc_processor import DocumentProcessor
from odm.structure import Chunk
from odm.helper import create_constitution, init_db
import config
import asyncio
from pathlib import Path
import prompt


async def main():
    await init_db()
    processor = DocumentProcessor()

    data_source_dir = Path(config.DATA_SOURCE_DIR)
    if data_source_dir.exists() and data_source_dir.is_dir():
        txt_files = list(data_source_dir.glob("*.txt"))
        if txt_files:
            print("Найдены следующие файлы для обработки: ")

            # Загружаем и разбиваем документы
            result_arr = processor.load_and_split_documents(directory_path=config.DATA_SOURCE_DIR)
            for doc in result_arr:
                await create_constitution(file_name=doc["metadata_file_name"], size=doc["metadata_file_size"], 
                                        original_text=[Chunk(text=chunk) for chunk in doc["chunks"]])
        else:
            print("ТХТ файлы не найдены")
    else:
        print("Source директории не существует")

    await processor.process_chunks(
        chunk_process_limit=999,
        llm_provider="Anthropic",
        file_name="constitution-ru.txt",
        llm_model="claude-3-sonnet-20240229",
        llm_prompt=prompt.prompt_ru_man_low_worker
    )

    # TODO
    # Когда проставлены original_text.processed = True, новый вариант обработать невозможно
    # зафиксить
    #
    # После обработки и сохранения LLMAnswer не проставляет processed = True
    # связанно с моими изменениями сохранения каждого ответа сразу после получения от LLM
    
if __name__ == "__main__":
    asyncio.run(main())