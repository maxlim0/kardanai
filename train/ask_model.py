from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Путь к базовой модели и адаптерам
#base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#base_model_path = "meta-llama/Llama-3.2-1B"
#base_model_path = "meta-llama/Llama-3.2-3B"
base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
lora_model_path = "/Users/max/PycharmProjects/Topic/data/model/a5000-p4-e8/checkpoint-888"

# Загрузка базовой модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Применение LoRA-адаптера к модели
#model = PeftModel.from_pretrained(model, lora_model_path)

# print("lora conf ===========")
# print(model.peft_config)
# print(model.state_dict().keys())
# print("lora conf ===========")


# Функция генерации ответа
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, do_sample=True,top_p=0.5, temperature=0.5, 
                             early_stopping=True, eos_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=2, num_beams=2 )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

# do_sample, top_p, и temperature:
# Эти параметры работают вместе, чтобы контролировать случайность и разнообразие текста.
# Если do_sample=False, top_p и temperature игнорируются, и модель использует жадный поиск.
# Если do_sample=True, top_p и temperature влияют на выборку токенов.


messages = [
    {"role": "system", "content": "Ты ассистент юриста. Отвечающий на русском."},
    {"role": "user", "content": "Какого цвета флаг Украины"},
]

print(messages)

# Пример использования
question = "Что гарантирует 45 статья конституции украины?"
print(generate_answer(question))
