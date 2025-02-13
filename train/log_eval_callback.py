from transformers import TrainerCallback
import torch

class LogEvalCallback(TrainerCallback):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        print("\n🔹 Проверяем генерацию на тестовом примере...\n")
        for batch in eval_dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                output_ids = model.generate(inputs["input_ids"], max_new_tokens=50)
            print("🟢 Вопрос:", self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            print("🟡 Ожидаемый ответ:", self.tokenizer.decode(inputs["labels"][0], skip_special_tokens=True))
            print("🔴 Сгенерированный ответ:", self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
            break  # Логируем только один пример