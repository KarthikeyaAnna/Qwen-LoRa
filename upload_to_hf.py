import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def upload(model_path: str, repo_id: str, private: bool = False):
    print(f"Loading model from '{model_path}'...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"Pushing model and processor to Hugging Face Hub: {repo_id}...")
    model.push_to_hub(repo_id, private=private)
    processor.push_to_hub(repo_id, private=private)
    print(f"\n🎉 Successfully uploaded to: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned Qwen2-VL model to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Target HF repo ID, e.g. username/Qwen2-VL-2B-DPO")
    parser.add_argument("--model-path", type=str, default="outputs/qwen2vl-dpo-merged", help="Path to local merged model folder.")
    parser.add_argument("--private", action="store_true", help="Set repository as private on HF Hub.")
    
    args = parser.parse_args()
    upload(args.model_path, args.repo_id, args.private)

if __name__ == "__main__":
    main()
