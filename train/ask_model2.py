import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
import time
from threading import Thread

def create_pipeline(base_model_name, adapter_name=None):
    # Загружаем базовую модель
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Загружаем LoRA адаптер
    if adapter_name:
        print("Apply LoRa adapter")
        model = PeftModel.from_pretrained(
            model,
            adapter_name,
            torch_dtype=torch.float16
        )
    else:
        print("Using pure base model")    

    return model

def generate_response(model, tokenizer, messages, max_length=512):
    # Форматируем сообщения в prompt
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    # Создаем стример
    streamer = TextIteratorStreamer(tokenizer)
    
    # Параметры генерации
    generation_config = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": 0.7,  # Немного увеличена для большей вариативности
        "top_p": 0.92,
        "top_k": 50,        # Добавлен top_k
        "repetition_penalty": 1.2,  # Штраф за повторения
        "no_repeat_ngram_size": 4,  # Запрет повторения n-грамм
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
        # Добавляем стоп-токены для предотвращения продолжения диалога
        #"stop_words": ["User:", "<|endoftext|>", "\nUser", "\nSystem"],
    }
    
    # Запускаем генерацию в отдельном потоке
    thread = Thread(target=model.generate, kwargs={**inputs, **generation_config})
    thread.start()

    # Получаем и выводим токены
    for text in streamer:
        print(text, end='', flush=True)
        time.sleep(0.02)  # Небольшая задержка для имитации реального стриминга

# Пример использования
#base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
#base_model_name = "RefalMachine/RuadaptQwen2.5-1.5B-instruct"
adapter_name = "/Users/max/PycharmProjects/Topic/data/model/a5000-2h-p3-e12/llama_lora_output/checkpoint-1332"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Убираем pipeline и используем модель напрямую
#model = create_pipeline(base_model_name, adapter_name)
model = create_pipeline(base_model_name)

token_ids = tokenizer.encode("Ты ассистент юриста. Отвечай только на русском.")
tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

print(f"model name: {model._get_name}")
print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")

for i, token in enumerate(tokens, 1):
    print(f"{i}, '{token}'")

messages = [
    {"role": "system", "content": "Ты ассистент юриста. Отвечай только на русском."},
    {"role": "user", "content": "О чем говорит 10 статья конституции украины?"},
]

generate_response(model, tokenizer, messages)