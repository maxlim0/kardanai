import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odm.structure import Constitution
import asyncio
from pathlib import Path
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import config
from datasets import Dataset

async def init_db():
    client = AsyncIOMotorClient(config.mongodb_connection)
    await init_beanie(
        database=client.LegalDocuments,
        document_models=[Constitution]
    )


async def export_ru_original():
    await init_db()

    constitution_ru = await Constitution.find_one({"file_name": "constitution-ru.txt"})

    data = [{"text": state.text} for state in constitution_ru.original_text]
    # for state in constitution_ru.original_text:
    #     data.append(state.text)

    print(f"Всего примеров в original_ru constitution: {len(data)}")

    import statistics
    lengths = [len(d["text"]) for d in data]
    median_length = statistics.median(lengths)
    print(f"Медиана длины примера: {median_length}")

    return Dataset.from_list(data)

# async def main():
#     await export_ru_original()

# if __name__ == "__main__":
#     asyncio.run(main())