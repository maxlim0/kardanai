import os
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from qa_dataset_loader_v2 import constitution_load_dataset
from datasets import Dataset, concatenate_datasets
from huggingface_hub import login
from utils.save_model_artifact import upload_to_gcs
from config import GCP_BUCKET_MODEL_ARTIFACT, HF_TOKEN, WANDB_API_KEY
import torch
from trainer_config import TrainingConfig
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["WANDB_PROJECT"] = "first"
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
 
async def start_train():
    #model_id = "meta-llama/Llama-3.2-1B"
    #model_id = "meta-llama/Llama-3.2-3B"
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True,)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 768
    #tokenizer.model_max_length = 128

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    except:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Загружаем модель в meta-режиме (Lazy Loading)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)  # Явно переместим на MPS

    # Явно инициализируем веса, если они в meta-режиме
    if next(model.parameters()).device == torch.device("meta"):
        model.to_empty(device)

    #model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        #Проверка структуры
        # print("\nСтруктура model_inputs после токенизации:")
        # print(f"Тип input_ids: {type(model_inputs['input_ids'])}")
        # if len(model_inputs['input_ids']) > 0:
        #     print(f"Длина первой последовательности: {len(model_inputs['input_ids'][0])}")

        model_inputs['labels'] = model_inputs['input_ids'].clone()    
        return model_inputs

    # Конфигурация LoRA
    lora_config = LoraConfig(
        r=32,                    # ранг матрицы
        lora_alpha=16,           # альфа параметр  # alpha = 2*r (стандартная практика)
        lora_dropout=0.05,       # dropout для регуляризации
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  
                        "gate_proj", "up_proj", "down_proj"] 
    )

    # Применение LoRA к модели
    model = get_peft_model(model, lora_config)

    # Загрузка и подготовка данных
    print("Debug: Constitution dataset loading...")
    primers_list = constitution_load_dataset("data/export")

    #print(dataset)
    print(f"Обьеденненный dataset: {len(primers_list)}")

    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 23 July 2024

    You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # Cutting Knowledge Date: December 2023
    # Today Date: 19 Oct 2024
    # You are a helpful AI assistant<|eot_id|>{%- for message in messages -%}
    # {%- if message['role'] == 'user' -%}<|start_header_id|>user<|end_header_id|>
    # {{ message['content'] }}<|eot_id|>{%- elif message['role'] == 'assistant' -%}<|start_header_id|>assistant<|end_header_id|>
    # {{ message['content'] }}<|eot_id|>{%- endif -%}
    # {%- endfor -%}"""

    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant<|eot_id|>{%- for message in messages -%}{%- if message['role'] == 'user' -%}<|start_header_id|>user<|end_header_id|>{{ message['content'] }}<|eot_id|>{%- elif message['role'] == 'assistant' -%}<|start_header_id|>assistant<|end_header_id|>{{ message['content'] }}<|eot_id|>{%- endif -%}{%- endfor -%}"""

    #from trl import apply_chat_template
    #result = tokenizer.apply_chat_template(dataset[0]["messages"], chat_template=chat_template, tokenize=False)

    def formatting_prompts(examples):
        samples = []
        for example in examples:  # Перебираем все примеры
            text = tokenizer.apply_chat_template(
                example["messages"],
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=False
            )
            samples.append({"text": text})
        return samples

    templated_primers_list = formatting_prompts(primers_list)
    dataset = Dataset.from_list(templated_primers_list)

    # Разделение на train/test
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.05, seed=42).values()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(valid_dataset)}")

    # # Токенизация данных
    tokenized_train = train_dataset.map(tokenize_function, batched=True, batch_size=128)
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True, batch_size=128)

    # TODO надо будет нормализовать ответы что бы не ела память при паддинге 
    # более 1000: 7 элементов
    # от 1000 до 500: 21 элементов
    # менее 500: 1064 элементов

    # # Вычисляем длину каждого вопроса
    # lengths = [len(question) for question in dataset['train']['ответ']]

    # # Создаем категории
    # categories = {'более 1000': 0, 'от 1000 до 500': 0, 'менее 500': 0}

    # # Подсчитываем длинну в символах ответов попадают в каждую категорию
    # for length in lengths:
    #     if length > 1000:
    #         categories['более 1000'] += 1
    #     elif 500 <= length <= 1000:
    #         categories['от 1000 до 500'] += 1
    #     else:
    #         categories['менее 500'] += 1

    # # Выводим результат
    # for category, count in categories.items():
    #     print(f"{category}: {count} элементов")


    # # Выведем точный формат одного элемента
    # sample = tokenized_valid[0]
    # print("\nТочная структура первого элемента:")
    # print("Тип:", type(sample))
    # print("Ключи:", sample.keys())
    # for key in sample:
    #     print(f"\n{key}:")
    #     print(f"Тип: {type(sample[key])}")
    #     if isinstance(sample[key], list):
    #         print(f"Тип первого элемента: {type(sample[key][0])}")

    if device.type == "cuda":
        training_args = TrainingConfig.from_yaml('training_configs.yaml', 'h100_config').to_training_arguments()
        print("Use CUDA trainer")
    elif device.type == "mps":
        training_args = TrainingConfig.from_yaml('training_configs.yaml', 'mac_config').to_training_arguments()
        print("Use MPS trainer")
    else:
        training_args = TrainingConfig.from_yaml('training_configs.yaml', 'mac_config').to_training_arguments()
        print("Use CPU trainer")

    # Инициализация тренера
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    import wandb
    wandb.init()
    
    trainer.train()
    model.save_pretrained("data/export/model")

    upload_to_gcs(
        GCP_BUCKET_MODEL_ARTIFACT,
        "data/model/",  # Директория для загрузки
        "model/"   # Имя папки в бакете
    )

async def main():
    login(token=HF_TOKEN)
    await start_train()

if __name__ == "__main__":
    asyncio.run(main())

# docker run --name train --gpus all --ipc=host  --entrypoint=/bin/bash --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 -v /in_container_app:/app/data/model -it maxsolyaris/kardanai
