from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TextSplitter
from llama_index.core.schema import Node
from llama_index.core.llms import ChatMessage
from typing import List, Optional
from pathlib import Path
import prompt
import asyncio
import config

from typing import Optional
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage
from prompt_similarity_checker import compare_prompts
from ratelimit import limits, sleep_and_retry
from odm.structure import LLMAnswer, Constitution, Variant


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
    def __init__(self):
        """
        Инициализация процессора документов
        """        
        # self.anthropic = Anthropic()

        # Settings.llm = Anthropic(model="claude-3-opus-20240229",
        #                          api_key=config.ANTHROPIC_API_KEY)

    def load_and_split_documents(self, directory_path: str) -> list:
        """
        Загрузка документов и разбиение на ноды с сохранением метаданных
        """
        # Загружаем документы из директории
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        
        # Используем сплиттер для создания нод
        splitter = CustomSplitter(chunk_size=config.CHUNK_SIZE)
        result_arr = []
        for doc in documents:
            result_dict = {}
            result_dict["metadata_file_name"] = doc.metadata["file_name"]
            result_dict["metadata_file_size"] = doc.metadata["file_size"]
            metadata_file_name = doc.metadata["file_name"]
            metadata_file_size = doc.metadata["file_size"]

            #print("Chunking: " + metadata_file_name)
            
            chunks = splitter.split_text(doc.text)
            result_dict["chunks"] = chunks
            result_arr.append(result_dict)

            #print(f"DocomentProcessor chunks: {len(chunks)}")
        return result_arr

    @staticmethod
    @sleep_and_retry
    @limits(calls=49, period=60)
    async def _rate_limited_llm_call(chunk):
        response = Settings.llm.chat([
            ChatMessage(role="system", content=prompt.prompt_ukr_lawer),
            ChatMessage(role="user", content=chunk.text)
        ])
        return response


    async def process_chunks(self, file_name: str, llm_provider: str, llm_model: str, 
                            llm_prompt: str, chunk_process_limit: Optional[int] = None):
        

        Settings.llm = Anthropic(model=llm_model,api_key=config.ANTHROPIC_API_KEY, max_tokens=3000)

        documents = await Constitution.find(
            {
                "file_name": file_name, 
                "original_text.processed": False}
        ).to_list()

        if not documents:
            print(f"Document {file_name} not found or already processed")
            return

        # перебор конституций
        for document in documents:
            unprocessed_chunks = [
                chunk for chunk in document.original_text #  list comprehension (списковое включение)
                if not chunk.processed
            ]

            if chunk_process_limit:
                unprocessed_chunks = unprocessed_chunks[:chunk_process_limit]
            
            if not unprocessed_chunks:
                break  # Выйдем из цикла, не будем перебирать документы Конституции

            # Проверим есть ли уже такой Вариант с одинаковыми парамтерами к LLM
            variant_index = None
            for idx, variant in enumerate(document.variants):
                if (
                    variant.llm_provider == llm_provider and
                    variant.llm_model == llm_model and
                    variant.llm_prompt == llm_prompt and
                    # Если разница больше 90% применяем новый промпт. Порядок слов не важен.
                    not compare_prompts(variant.llm_prompt, llm_prompt, threshold=90.0)  # False если промпты похожи
                ):
                    variant_index = idx
                    print("Variant already exists")
                    break

            if variant_index is None:
                new_variant = Variant(
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_prompt=llm_prompt,
                    llm_answers=[]
                )
                print("New Variant created")
                document.variants.append(new_variant)
                variant_index = len(document.variants) - 1


            print(f"Чанков будет обработанно {len(unprocessed_chunks)} chunks")
            for chunk in unprocessed_chunks:
                try:
                    #Отправляем запрос в Anthropic
                    response = await self._rate_limited_llm_call(chunk)

                    llm_answer = LLMAnswer(
                        questionId=str(chunk.id),
                        answer=response.message.blocks[0].text,
                        llm_input_tokens=response.raw["usage"].input_tokens,
                        llm_output_tokens=response.raw["usage"].output_tokens
                    )

                    document.variants[variant_index].llm_answers.append(llm_answer)
                    chunk.processed = True

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue

            await document.save()
            print(f"Document {document.file_name} has been saved")
