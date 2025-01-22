INPUT TEXT -> Токенизация -> Token IDs
     ↓
[Embedding Layer] (vocab_size × hidden_size)
     ↓
[Positional Embedding] (max_seq_length × hidden_size)
     ↓
Embedded Sequence (seq_length × hidden_size)
     ↓
TRANSFORMER BLOCKS (× num_hidden_layers):
┌─────────────────────────────────────────┐
│ Block 1:                                │
│   ┌─ Multi-Head Attention               │
│   │   ├─ Query (hidden_size × hidden_size)  │
│   │   ├─ Key   (hidden_size × hidden_size)  │
│   │   └─ Value (hidden_size × hidden_size)  │
│   │         ↓                           │
│   │   [Attention Outputs]               │
│   └──────────┐                          │
│              ↓                          │
│   [Layer Normalization 1]               │
│              ↓                          │
│   ┌─ Feed Forward Network               │
│   │   ├─ Linear 1 (hidden_size → 4×hidden_size)│
│   │   ├─ Activation (GELU)              │
│   │   └─ Linear 2 (4×hidden_size → hidden_size)│
│   └──────────┐                          │
│              ↓                          │
│   [Layer Normalization 2]               │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│ Block 2: [та же структура]              │
└─────────────────────────────────────────┘
     ↓
     ... (повторяется num_hidden_layers раз)
     ↓
[Final Layer Normalization]
     ↓
[Output Layer] (hidden_size × vocab_size)
     ↓
OUTPUT LOGITS

Размерности (пример для модели типа BERT-base):
- hidden_size = 768
- num_attention_heads = 12
- num_hidden_layers = 12
- vocab_size = ~30,000
- max_seq_length = 512

Внутри Multi-Head Attention:
- Каждая голова работает с размерностью: hidden_size/num_heads
- Для BERT: 768/12 = 64 на голову
- Q, K, V для каждой головы: (64 × 64)


=======
# Основные матрицы для LoRA адаптации:
target_modules = [
    "q_proj",  # Query - как модель задает вопросы к контексту
    "k_proj",  # Key - как модель индексирует информацию
    "v_proj",  # Value - как модель представляет информацию
    "o_proj",  # Output projection - как комбинируется внимание
    "gate_proj", # Gate в FFN - контроль потока информации
    "up_proj",   # Up projection в FFN - расширение представления
    "down_proj"  # Down projection в FFN - сжатие представления
]


# 1. Базовая адаптация (самая распространенная)
target_modules=["q_proj", "v_proj"]
# Применение:
# - Общая адаптация к новой области
# - Стилистическая настройка
# - Базовое обучение новым знаниям

# 2. Полная адаптация внимания
target_modules=["q_proj", "k_proj", "v_proj"]
# Применение:
# - Глубокая перестройка понимания контекста
# - Обучение сложным логическим паттернам
# - Работа с кодом или структурированными данными

# 3. Расширенная адаптация внимания
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# Применение:
# - Максимальная адаптация механизма внимания
# - Сложные задачи рассуждения
# - Специализированные задачи анализа

# 4. Адаптация FFN
target_modules=["gate_proj", "up_proj", "down_proj"]
# Применение:
# - Улучшение обработки информации
# - Адаптация к специфическим форматам данных
# - Изменение способа обобщения информации



# Можно выбирать конкретные слои для адаптации:
layers_to_transform=[0, 1, 2]  # только первые три слоя

# Общие паттерны по слоям:
# Нижние слои (0-3):
# - Базовое понимание языка
# - Синтаксические структуры
# - Простые паттерны

# Средние слои (4-8):
# - Семантическое понимание
# - Контекстуальные связи
# - Абстрактные концепции

# Верхние слои (9-12):
# - Сложные рассуждения
# - Высокоуровневое понимание
# - Генерация контента