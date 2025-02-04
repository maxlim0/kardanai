import os
import asyncio
from qa_dataset_loader import constitution_load_dataset
from original_dataset_loader import export_ru_original
from datasets import Dataset, concatenate_datasets
from huggingface_hub import login
from utils.save_model_artifact import upload_to_gcs
from config import GCP_BUCKET_MODEL_ARTIFACT, HF_TOKEN, NEPTUNE_API_TOKEN, NEPTUNE_PROJECT
import torch
import neptune
from trainer_config import TrainingConfig
from transformers.trainer_callback import EarlyStoppingCallback

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

os.environ["TOKENIZERS_PARALLELISM"] = "True"
 
async def start_train():
#    model_id = "meta-llama/Llama-3.2-1B"
    model_id = "meta-llama/Llama-3.2-3B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True,)

    # # Открываем файл для записи
    # with open("vocab.txt", "w", encoding="utf-8") as file:
    #     # Переменная, которую нужно записать
    #     data = "Пример данных для записи"
        
    #     # Печатаем переменную в файл
    #     print(tokenizer.get_vocab(), file=file)
    print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 768

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    except:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Загрузка модели с квантизацией
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    #model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    print(model.device)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        ).to(device)
        
        # #Проверка структуры
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
        task_type="CAUSAL_LM",   # тип задачи
        target_modules=["q_proj", "v_proj"]  # целевые слои для адаптации
    )

    # Применение LoRA к модели
    model = get_peft_model(model, lora_config)

    # Загрузка и подготовка данных
    dataset_qa_texts_not_formatted = constitution_load_dataset("data/export")
    dataset_qa_texts_formatted = [
        {"text": f"{tokenizer.bos_token}{q}{tokenizer.eos_token}{tokenizer.bos_token}{a}{tokenizer.eos_token}"} 
        for q, a in zip(
            dataset_qa_texts_not_formatted["вопрос"],
            dataset_qa_texts_not_formatted["ответ"]
        )
    ]

    # Преобразование в датасет
    ds_dataset_qa_texts_formatted = Dataset.from_list(dataset_qa_texts_formatted)
    #print(f"ds_dataset_qa_texts_formatted: {len(ds_dataset_qa_texts_formatted)}")

    # Загрузка второго датасета
    dataset_original_ru_texts = await export_ru_original()
    #print(f"dataset_original_ru_texts: {len(dataset_original_ru_texts)}")

    # Объединение датасетов
    dataset = concatenate_datasets([ds_dataset_qa_texts_formatted, dataset_original_ru_texts])
    print("-=-----------------------")
    print(f"Обьеденненный dataset: {len(dataset)}")

    # Разделение на train/test
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
    print(train_dataset)


    columns_to_remove = ['text']
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

    # И посмотрим на конфигурацию токенизатора
    print("\nКонфигурация токенизатора:")
    print(tokenizer.special_tokens_map)

    print("До настройки:")
    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("padding_side:", tokenizer.padding_side)

    # mac_config = TrainingConfig.from_yaml('training_configs.yaml', 'h100_config')
    # training_args = mac_config.to_training_arguments()

    if device == "cuda":
        training_args = TrainingConfig.from_yaml('training_configs.yaml', 'h100_config').to_training_arguments()
    else:
        training_args = TrainingConfig.from_yaml('training_configs.yaml', 'mac_config').to_training_arguments()
        

    # Инициализация тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        #compute_metrics=CustomMetrics().compute_metrics,
    )

    # Определение максимальной длины последовательности
    # 1380
    max_length = tokenizer.model_max_length
    print(f"tokenizer.model_max_length: {max_length}")
    print(f"model.config.max_position_embeddings {model.config.max_position_embeddings}")

    max_lengths = [len(x["input_ids"]) for x in tokenized_train]

    print(f"Max sequence length: {max(max_lengths)}")
    print(f"Model max length: {tokenizer.model_max_length}")

    os.environ["NEPTUNE_PROJECT"] = NEPTUNE_PROJECT
    os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_TOKEN
    neptune.init_run()

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
