from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """
    Конфигурация для обучения ASR моделей.
    Обертка над аргументами HuggingFace TrainingArguments.
    """
    output_dir: str = "checkpoints"
    
    # WandB Integration
    project_name: str = "plantain2asr"
    run_name: Optional[str] = None
    
    # Основные параметры обучения
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.005
    
    # Оптимизация и железо
    fp16: bool = True  # Использовать mixed precision (рекомендуется для GPU)
    dataloader_num_workers: int = 0  # 0 для стабильности в Jupyter/macOS, >0 для скорости
    
    # Логирование и сохранение
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2  # Хранить только 2 последних чекпоинта
    
    # ASR специфика
    group_by_length: bool = True  # Группировать по длине аудио (ускоряет обучение)
    gradient_checkpointing: bool = True  # Экономит память ценой скорости
