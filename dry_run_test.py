from datasets import Dataset, Sequence, Image as DatasetsImage
from PIL import Image
import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# 1. Create a dummy image
os.makedirs("dummy_images", exist_ok=True)
dummy_img_path = "dummy_images/test_image.jpg"
Image.new("RGB", (224, 224), color="red").save(dummy_img_path)

# 2. Mock dataset
data = {
    "image_id": ["test_image"],
    "chosen": ["This is a chosen response."],
    "rejected": ["This is a rejected response."]
}
ds = Dataset.from_dict(data)

# 3. Apply the exact mapping function from prepare_dataset.py
def convert_hadpo_to_trl(example):
    image_path = os.path.join("dummy_images", f"{example['image_id']}.jpg")
    question = "Describe this image." 
    return {
        "prompt": [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}],
        "chosen": [{"role": "assistant", "content": [
            {"type": "text", "text": example["chosen"][0] if isinstance(example["chosen"], list) else example["chosen"]}
        ]}],
        "rejected": [{"role": "assistant", "content": [
            {"type": "text", "text": example["rejected"][0] if isinstance(example["rejected"], list) else example["rejected"]}
        ]}],
        "images": [image_path] 
    }

ds = ds.map(convert_hadpo_to_trl, remove_columns=ds.column_names)
ds = ds.cast_column("images", Sequence(DatasetsImage()))

print("Dataset prepared successfully. Schema:")
print(ds.features)

# 4. Load tiny model and processor (we'll just use a tiny LLM but fake it for VLM, or just use processor)
# Actually, loading Qwen2-VL-2B might take 4GB VRAM and a few mins.
print("Dry run complete without DPOTrainer instantiating (to save time).")
