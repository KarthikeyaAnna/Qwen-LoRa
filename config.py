from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device_map: str = "auto"

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list = None

@dataclass
class DPOHyperparams:
    beta: float = 0.1
    loss_type: str = "sigmoid"
    max_length: int = 4096
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    warmup_steps: int = 50
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "tensorboard"
    output_dir: str = "outputs/qwen2vl-dpo-lora"
    precompute_ref_log_probs: bool = True

@dataclass
class DataConfig:
    dataset_name: str = "juliozhao/hadpo-data"
    max_samples: int = 3000
    train_split_ratio: float = 0.9
    processed_data_dir: str = "data/processed"
    seed: int = 42
    # NOTE: Set this to the path containing MSCOCO / VG images on your server
    image_dir: str = "/path/to/your/images/directory"
