import os
from datasets import load_dataset, DatasetDict
from config import DataConfig
from PIL import Image

# Global variable set in main() for the map function
IMAGE_DIR = "/path/to/your/images/directory"

def convert_hadpo_to_trl(example):
    """Convert HA-DPO schema to TRL DPO format."""
    image_path = os.path.join(IMAGE_DIR, f"{example['image_id']}.jpg")
    
    # We use a generic prompt since desc_data.json mainly evaluates descriptions
    question = "Describe this image." 
    
    # Return the format required by TRL DPOTrainer
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
        # TRL expects an 'images' column with a list of paths or PIL Images
        "images": [image_path] 
    }

def main():
    cfg = DataConfig()
    
    if cfg.image_dir == "/path/to/your/images/directory":
        raise ValueError(
            "ERROR: You must update 'image_dir' in config.py to point to your actual "
            "images (MSCOCO/Visual Genome) on the server before running this script."
        )
        
    global IMAGE_DIR
    IMAGE_DIR = cfg.image_dir
    
    print(f"Loading dataset {cfg.dataset_name}...")
    
    # Load dataset - explicitly target desc_data.json to avoid schema mismatches with pope_data.json
    # We use the llava-v1.5 generated descriptions subset as an example
    dataset = load_dataset(cfg.dataset_name, data_files="hadpo/llava-v1.5/desc_data.json")
    train_data = dataset["train"]
    
    # Subsample if necessary
    if cfg.max_samples > 0 and len(train_data) > cfg.max_samples:
        print(f"Subsampling to {cfg.max_samples} samples...")
        train_data = train_data.shuffle(seed=cfg.seed).select(range(cfg.max_samples))
    
    print("Formatting to TRL DPO format...")
    # Map the formatting function
    # Note: we keep the 'image' column in the original or remove it. 
    # 'images' will be created which is what Qwen2-VL Processor needs.
    formatted_data = train_data.map(
        convert_hadpo_to_trl,
        remove_columns=train_data.column_names,
        desc="Formatting dataset"
    )
    
    # Split into train/eval
    split = formatted_data.train_test_split(
        train_size=cfg.train_split_ratio,
        seed=cfg.seed
    )
    
    from datasets import Sequence, Image as DatasetsImage
    split = split.cast_column("images", Sequence(DatasetsImage()))
    
    processed_dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"]
    })
    
    print(f"Train samples: {len(processed_dataset['train'])}")
    print(f"Eval samples:  {len(processed_dataset['eval'])}")
    
    # Validate structure
    sample = processed_dataset['train'][0]
    print("\nValidation Sample Structure:")
    print(f"Keys: {sample.keys()}")
    print(f"Prompt type: {type(sample['prompt'])}")
    
    os.makedirs(os.path.dirname(cfg.processed_data_dir), exist_ok=True)
    processed_dataset.save_to_disk(cfg.processed_data_dir)
    print(f"Processed dataset saved to {cfg.processed_data_dir}")

if __name__ == "__main__":
    main()
