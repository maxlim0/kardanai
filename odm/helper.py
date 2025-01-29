from typing import List
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import config
from odm.structure import Constitution, Variant, LLMAnswer, Chunk

# --- Инициализация БД ---
async def init_db():
    client = AsyncIOMotorClient(config.mongodb_connection)
    await init_beanie(
        database=client.LegalDocuments,
        document_models=[Constitution]
    )

# --- Сервисные функции ---
async def create_constitution(file_name: str, size: int, original_text: List[Chunk]) -> Constitution:
    if await Constitution.find_one(Constitution.file_name == file_name):
        print(f"Document {file_name} already exists")
        return None
    
    return await Constitution(file_name=file_name, size=size, original_text=original_text).insert()
