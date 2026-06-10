from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from trl import DPOTrainer, DPOConfig
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32, device_map="cpu")
peft_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])

# Wrap the model
model_peft = get_peft_model(model, peft_config)

training_args = DPOConfig(output_dir="test_outputs", per_device_train_batch_size=1)

try:
    trainer = DPOTrainer(
        model=model_peft,
        args=training_args,
        peft_config=peft_config,
    )
    print("SUCCESS: DPOTrainer accepted both!")
except Exception as e:
    print("FAILED with exception:", e)
