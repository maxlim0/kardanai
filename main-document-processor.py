from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import TextSplitter
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.schema import Node
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic
from typing import List, Optional
from pymongo import MongoClient
from pathlib import Path
import prompt
import asyncio
import config



class CustomSplitter(TextSplitter):
    chunk_size: int = 2000
    #chunk_size = config.CHUNK_SIZE
    
    def _find_stat_header(self, text: str) -> str:
        """
        Находит заголовок статьи в начале текста
        Пример: 'Статья 123', 'Статтья 1' и т.д.
        """
        # Ищем первые несколько слов текста
        words = text.split()[:4]  # Берем первые 4 слова для поиска
        for i in range(len(words)-1):
            current_word = words[i]
            next_word = words[i+1]
            # Проверяем паттерн: Стат* + число
            if current_word.startswith('Стат') and next_word.isdigit() and len(next_word) <= 3:
                return f"{current_word} {next_word}"
        return ""
    
    def split_text(self, text: str) -> List[str]:
        """
        Разделяет текст на чанки, используя '\nСтат' как обязательный разделитель,
        а также делит большие куски по размеру чанка с сохранением контекста статьи
        """
        # Сначала разделяем по '\nСтат'
        initial_chunks = text.split('\nСтат')
        
        # Если первый чанк пустой (текст начинался с '\nСтат'), убираем его
        if initial_chunks[0] == '':
            initial_chunks = initial_chunks[1:]
        
        # Добавляем префикс 'Стат' обратно ко всем чанкам кроме первого
        if len(initial_chunks) > 1:
            for i in range(1, len(initial_chunks)):
                initial_chunks[i] = 'Стат' + initial_chunks[i]
        
        final_chunks = []
        current_stat_header = ""
        
        # Проходим по каждому большому чанку и при необходимости делим его дальше
        for chunk in initial_chunks:
            # Проверяем, есть ли в начале чанка заголовок статьи
            new_stat_header = self._find_stat_header(chunk)
            if new_stat_header:
                current_stat_header = new_stat_header
            
            if len(chunk) <= self.chunk_size:
                if chunk.strip():  # Добавляем только непустые чанки
                    final_chunks.append(chunk)
            else:
                # Делим большой чанк на подчанки по размеру
                current_chunk = ''
                words = chunk.split()
                
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                        current_chunk += (word + ' ')
                    else:
                        if current_chunk.strip():
                            # Добавляем заголовок статьи к подчанку,
                            # если это не первый подчанк (у первого уже есть заголовок)
                            if current_chunk.strip() != chunk.strip() and current_stat_header:
                                final_chunks.append(f"{current_stat_header} (продолжение)\n{current_chunk.strip()}")
                            else:
                                final_chunks.append(current_chunk.strip())
                        current_chunk = word + ' '
                
                # Добавляем последний подчанк, если он не пустой
                if current_chunk.strip():
                    if current_chunk.strip() != chunk.strip() and current_stat_header:
                        final_chunks.append(f"{current_stat_header} (продолжение)\n{current_chunk.strip()}")
                    else:
                        final_chunks.append(current_chunk.strip())
        
        return final_chunks


class DocumentProcessor:
    def __init__(self, mongodb_uri: str, db_name: str, collection_name: str = None):
        """
        Инициализация процессора документов
        """        
        self.mongo_client = MongoClient(config.mongodb_connection)
        self.mongo_db = self.mongo_client.mydb
        self.anthropic = Anthropic()

        # Заменяем коллекцию в docstore
        Settings.llm = Anthropic(model="claude-3-opus-20240229",
                                 api_key=config.ANTHROPIC_API_KEY)

    async def load_and_split_documents(self, directory_path: str) -> List[Node]:
        """
        Загрузка документов и разбиение на ноды с сохранением метаданных
        """
        # Загружаем документы из директории
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        
        # Используем сплиттер для создания нод
        splitter = CustomSplitter(chunk_size=config.CHUNK_SIZE)
        for doc in documents:
            metadata_file_name = doc.metadata["file_name"]
            metadata_file_size = doc.metadata["file_size"]

            print("Chunking: " + metadata_file_name)

            collection_name = "ukr_laws_" + metadata_file_name
            collection = self.mongo_db[collection_name]
            
            chunks = splitter.split_text(doc.text)
            for chunk in chunks:
                document = {
                    "text": chunk,
                    "metadata": {
                        "file_name": metadata_file_name,
                        "file_size": metadata_file_size,
                        "processed": False,
                        "llm_provider": False,
                        "llm_answer": False,
                        "llm_model": False,
                        "llm_input_tokens": 0,
                        "llm_output_tokens": 0
                    }
                }
                collection.insert_one(document)
        print("Данные успешно сохранены в MongoDB Atlas.")


    def process_chunks_with_anthropic(
                self,
                source_file: Optional[str] = None,
                processed: Optional[bool] = None,
                chunk_limit: Optional[int] = None
            ):
            """
            Обработка нод через Anthropic API с сохранением ответов в метаданных
            """
            mongo_collection = self.mongo_db["ukr_laws_constitution-ru.txt"]

            mongo_query = {}
            if source_file and processed is not None:
                mongo_query = {
                    "$and": [
                        {"metadata.file_name": source_file},
                        {"metadata.processed": processed}        
                    ]
                }
            elif source_file:
                mongo_query["metadata.file_name"] = source_file
            elif processed is not None:
                mongo_query["metadata.processed"] = processed
    
            # Фильтруем по исходному файлу если указан
            mongo_documents = list(mongo_collection.find(mongo_query))
                            
            # Применяем ограничение если указано
            if chunk_limit:
                mongo_documents = mongo_documents[:chunk_limit]
            

            total_nodes = len(mongo_documents)
            print(f"Total nodes to process: {total_nodes}")
            
            # Обрабатываем каждую ноду
            for idx, doc in enumerate(mongo_documents, 1):
                print(f"ObjectID: {str(doc["_id"])}")  
                print(f"Processing node {idx}/{total_nodes} ({(idx/total_nodes*100):.1f}%)")
                
                #Отправляем запрос в Anthropic
                response = Settings.llm.chat([
                    ChatMessage(role="system", content=prompt.prompt_ukr_lawer),
                    ChatMessage(role="user", content=doc["text"])
                ])

                print(f"LLM token ussage: {response.raw["usage"].input_tokens + response.raw["usage"].output_tokens}")

                # Обновляем метаданные ноды
                mongo_collection.update_one(
                   {"_id": doc["_id"]},
                   {"$set": {
                       "metadata.processed": True,
                       "metadata.llm_provider": "Anthropic",
                       "metadata.llm_answer": response.message.blocks[0].text,
                       "metadata.llm_model": response.raw["model"],
                       "metadata.llm_input_tokens": response.raw["usage"].input_tokens,
                       "metadata.llm_output_tokens": response.raw["usage"].output_tokens,
                       "metedata.llm_prompt": prompt.prompt_ukr_lawer
                   }} 
                )

            print("Processing completed!")

async def main():
    processor = DocumentProcessor(
        mongodb_uri=config.mongodb_connection,
        db_name="mydb",
        collection_name="ukr_laws"
    )
    
    data_source_dir = Path(config.DATA_SOURCE_DIR)
    if data_source_dir.exists() and data_source_dir.is_dir():
        txt_files = list(data_source_dir.glob("*.txt"))
        if txt_files:
            print("Найдены следующие файлы для обработки: ")
            for file in txt_files:
                print(file.name)
            
            # Загружаем и разбиваем документы
            await processor.load_and_split_documents(config.DATA_SOURCE_DIR)
            #print(f"Загружено и разбито на чанки: {len(nodes)} документов")            
        else:
            print("ТХТ файлы не найдены")
    else:
        print("Директории не существует")

    processor.process_chunks_with_anthropic(processed=False, chunk_limit=1)


if __name__ == "__main__":
    asyncio.run(main())