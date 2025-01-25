from fuzzywuzzy import fuzz
from typing import Optional

def compare_prompts(new_llm_prompt: str, old_llm_prompt: str, 
                    threshold: Optional[float] = 90.0) -> bool:
    """
    Сравнивает два промпта и определяет, нужно ли применить новый промпт.
    
    Args:
        new_llm_prompt (str): Новый промпт для сравнения
        old_llm_prompt (str): Старый промпт для сравнения
        threshold (float): Пороговое значение схожести в процентах (по умолчанию 90%)
        
    Returns:
        bool: True если нужно применить новый промпт (различия больше порога),
              False если промпты достаточно похожи
    """
    similarity = fuzz.token_set_ratio(new_llm_prompt, old_llm_prompt)
    return similarity < threshold