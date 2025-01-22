import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return Dataset.from_dict({"text": [text.strip() for text in texts]})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

# Настройка авторизации для Hugging Face Hub
HF_TOKEN = "ваш_токен_здесь"  # Замените на ваш токен
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Инициализация модели и токенизатора
model_id = "meta-llama/Llama-2-1b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HF_TOKEN,
    use_auth_token=True
)
tokenizer.pad_token = tokenizer.eos_token

# Загрузка модели с квантизацией
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    use_auth_token=True,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Подготовка модели для обучения
model = prepare_model_for_kbit_training(model)

# Конфигурация LoRA
lora_config = LoraConfig(
    r=16,                     # ранг матрицы
    lora_alpha=32,           # альфа параметр
    lora_dropout=0.05,       # dropout для регуляризации
    bias="none",
    task_type="CAUSAL_LM",   # тип задачи
    target_modules=["q_proj", "v_proj"]  # целевые слои для адаптации
)

# Применение LoRA к модели
model = get_peft_model(model, lora_config)

# Загрузка и подготовка данных
train_dataset = load_dataset("qa.txt")
valid_dataset = load_dataset("valid.txt")

# Токенизация данных
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir="./llama_lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=100,
    save_total_limit=3,
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Запуск обучения
trainer.train()

# Сохранение адаптера
model.save_pretrained("./llama_lora_final")