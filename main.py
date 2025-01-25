from doc_processor import DocumentProcessor
from odm.structure import Chunk
from odm.helper import create_constitution, init_db
import config
import asyncio
from pathlib import Path
from document_export import export_variant
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

    # СРОЧНО добавить запись одного ответа сразу в мого, а то может не сохраниться
    await processor.process_chunks(
        chunk_process_limit=1,
        llm_provider="Anthropic",
        file_name="constitution-uk.txt",
        llm_model="claude-3-sonnet-20240229",
        llm_prompt=prompt.prompt_ukr_lawer
    )

    #await export_variant()

    # await add_variant(
    #     constitution_name=doc["metadata_file_name"],
    #     llm_provider="llm_provider2",
    #     llm_model="llm_model2",
    #     llm_prompt="llm_prompt2",
    #     llm_answers=[LLMAnswer(questionId="qwe", answer="qwe", llm_input_tokens=1, llm_output_tokens=2),
    #                  LLMAnswer(questionId="qwe2", answer="qwe2", llm_input_tokens=3, llm_output_tokens=4)]
        # )
    
if __name__ == "__main__":
    asyncio.run(main())