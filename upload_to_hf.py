import argparse
from huggingface_hub import HfApi

def upload(model_path: str, repo_id: str, private: bool = False, token: str = None):
    api = HfApi(token=token)
    
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
