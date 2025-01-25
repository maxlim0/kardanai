from typing import List
from pydantic import Field, BaseModel
from beanie import Document, Indexed
from beanie import PydanticObjectId

class LLMAnswer(BaseModel):
    questionId: str
    answer: str
    llm_input_tokens: int
    llm_output_tokens: int

class Variant(BaseModel):
    llm_provider: str
    llm_model: str
    llm_prompt: str
    llm_answers: List[LLMAnswer] = []
    
    @property
    def variant_id(self) -> str:
        return f"{self.llm_model}-{hash(self.llm_prompt)}"

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(PydanticObjectId()))
    text: str
    processed: bool = False

# --- Основной документ ---
class Constitution(Document):
    file_name: Indexed(str, unique=True)
    size: int
    processed: bool = False
    original_text: List[Chunk] = []
    variants: List[Variant] = []

    class Settings:
        name = "Laws"

    def get_variant_number(self) -> int:
        return len(self.variants) + 1