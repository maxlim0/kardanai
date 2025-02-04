# config.py
from dataclasses import dataclass
from typing import List, Optional
import yaml
from transformers import TrainingArguments
from pathlib import Path

@dataclass
class TrainingConfig:
    name: str
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: int
    learning_rate: str
    weight_decay: float
    fp16: bool
    use_cpu: bool
    warmup_ratio: float
    save_total_limit: int
    logging_dir: str
    report_to: List[str]
    do_train: bool
    do_eval: bool
    disable_tqdm: bool
    logging_first_step: bool
    logging_steps: int
    overwrite_output_dir: bool
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool

    @classmethod
    def from_yaml(cls, path: str, config_name: str) -> 'TrainingConfig':
        config_path = Path(__file__).parent / path
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' not found in {path}")
            
        return cls(**configs[config_name])

    def to_training_arguments(self) -> TrainingArguments:
        # Преобразуем конфиг в TrainingArguments
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            evaluation_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            use_cpu=self.use_cpu,
            warmup_ratio=self.warmup_ratio,
            save_total_limit=self.save_total_limit,
            logging_dir=self.logging_dir,
            report_to=self.report_to,
            do_train=self.do_train,
            do_eval=self.do_eval,
            disable_tqdm=self.disable_tqdm,
            logging_first_step=self.logging_first_step,
            logging_steps=self.logging_steps,
            overwrite_output_dir=self.overwrite_output_dir,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
        )
    
# # usage_example.py
# from config import TrainingConfig

# # Загрузка конфигурации для Mac
# mac_config = TrainingConfig.from_yaml('training_configs.yaml', 'mac_config')
# training_args = mac_config.to_training_arguments()

# # Загрузка конфигурации для H100
# h100_config = TrainingConfig.from_yaml('training_configs.yaml', 'h100_config')
# h100_training_args = h100_config.to_training_arguments()