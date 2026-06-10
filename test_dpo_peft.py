from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32, device_map="cpu")
peft_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])

training_args = DPOConfig(
    output_dir="test_outputs", 
    per_device_train_batch_size=1, 
    precompute_ref_log_probs=True
)

try:
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
    )
    print("SUCCESS: DPOTrainer accepted precompute_ref_log_probs=True with peft_config!")
except Exception as e:
    print("FAILED with exception:", e)
