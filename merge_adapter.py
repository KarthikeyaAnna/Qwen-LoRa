import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from transformers.trainer_utils import get_last_checkpoint
import os

def merge_and_save(
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    adapter_path: str = "outputs/qwen2vl-dpo-lora/final",
    merged_output_dir: str = "outputs/qwen2vl-dpo-merged",
):
    if not os.path.exists(adapter_path):
        print(f"'{adapter_path}' not found. Checking for intermediate checkpoints...")
        parent_dir = os.path.dirname(adapter_path)
        if os.path.isdir(parent_dir):
            latest = get_last_checkpoint(parent_dir)
            if latest:
                print(f"Found checkpoint: {latest}. Using this instead.")
                adapter_path = latest
            else:
                raise FileNotFoundError(f"No adapters found in {parent_dir}.")
        else:
            raise FileNotFoundError(f"Directory {parent_dir} does not exist.")
            
    print(f"Loading base model: {base_model_id}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {merged_output_dir}")
    model.save_pretrained(merged_output_dir)
    
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(merged_output_dir)
    print("Merge complete!")

if __name__ == "__main__":
    merge_and_save()
