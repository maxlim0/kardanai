from transformers import AutoModel

def print_layer_info(model):
    print("\nВыводим информацию о слоях:")
    for name, param in model.named_parameters():
        # Расширяем поиск для различных названий attention слоев
        if any(x in name.lower() for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']):
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(f"Parameters: {param.numel()}")
            print("---")

def print_model_structure(model, depth=0):
    print("\nВыводим структуру модели:")
    for name, module in model.named_modules():
        # Расширяем поиск
        if any(x in name.lower() for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj']):
            print(f"{'  ' * depth}{name}")

def analyze_model_architecture(model):
    print("\nАнализ конфигурации модели:")
    config = model.config
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Num hidden layers: {config.num_hidden_layers}")
    print(f"Intermediate size: {config.intermediate_size}")
    
    # Добавляем детальный анализ структуры
    print("\nДетальный анализ структуры:")
    for child_name, child in model.named_children():
        print(f"\nComponent: {child_name}")
        if hasattr(child, 'children'):
            for subchild_name, _ in child.named_children():
                print(f"  Subcomponent: {subchild_name}")

def main():
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_id = "meta-llama/Llama-3.2-1B"
    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Выводим базовую конфигурацию
    print("Model Configuration:")
    print(model.config)
    
    # Запускаем все анализы
    print_layer_info(model)
    print_model_structure(model)
    analyze_model_architecture(model)

if __name__ == "__main__":
    main()



# # Анализ активации слоев:
# 1. Запустите модель на целевых примерах
# 2. Посмотрите активации каждого слоя
# 3. Определите, какие слои наиболее активны для вашей задачи

# # Пример кода для анализа:
# def analyze_layer_activations(model, input_text):
#     outputs = model(input_text, output_attentions=True)
#     attentions = outputs.attentions
    
#     for layer_idx, layer_attention in enumerate(attentions):
#         attention_scores = layer_attention.mean().item()
#         print(f"Layer {layer_idx} average attention: {attention_scores}")