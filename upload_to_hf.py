import os
import argparse
from huggingface_hub import HfApi

def create_model_card(model_path: str, repo_id: str):
    readme_path = os.path.join(model_path, "README.md")
    if not os.path.exists(readme_path):
        print(f"Generating Hugging Face Model Card at '{readme_path}'...")
        card_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen2-VL-2B-Instruct
tags:
- vision
- vlm
- dpo
- lora
- preference-alignment
- hallucination-mitigation
- qwen2-vl
datasets:
- juliozhao/hadpo-data
pipeline_tag: image-text-to-text
library_name: transformers
---

# Qwen2-VL-2B-Instruct DPO (Hallucination Mitigated)

This model is a Direct Preference Optimization (DPO) aligned version of [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), fine-tuned using LoRA on the [HA-DPO](https://huggingface.co/datasets/juliozhao/hadpo-data) preference dataset to significantly reduce visual hallucinations.

## Quickstart & Usage

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "{repo_id}"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {{
        "role": "user",
        "content": [
            {{"type": "image", "image": "path/to/your/image.jpg"}},
            {{"type": "text", "text": "Describe this image in detail."}},
        ],
    }}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.2,
    )

generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Training Details
- **Base Model:** Qwen/Qwen2-VL-2B-Instruct
- **Method:** DPO (Direct Preference Optimization) + LoRA (BF16)
- **Dataset:** HA-DPO (`juliozhao/hadpo-data`)
"""
        with open(readme_path, "w") as f:
            f.write(card_content)

def upload(model_path: str, repo_id: str, private: bool = False, token: str = None):
    api = HfApi(token=token)
    
    # Auto-generate Model Card (README.md)
    create_model_card(model_path, repo_id)
    
    print(f"Ensuring repository '{repo_id}' exists on Hugging Face Hub...")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    
    print(f"Uploading files from '{model_path}' directly to Hugging Face Hub (zero extra disk space used)...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"\n🎉 Successfully uploaded to: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned Qwen2-VL model to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Target HF repo ID, e.g. username/Qwen2-VL-2B-DPO")
    parser.add_argument("--model-path", type=str, default="outputs/qwen2vl-dpo-merged", help="Path to local merged model folder.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token.")
    parser.add_argument("--private", action="store_true", help="Set repository as private on HF Hub.")
    
    args = parser.parse_args()
    upload(args.model_path, args.repo_id, args.private, args.token)

if __name__ == "__main__":
    main()
