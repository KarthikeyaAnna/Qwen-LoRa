import sys
import argparse
from pope_eval import run_inference

def main():
    parser = argparse.ArgumentParser(description="Run inference with the DPO-aligned Qwen2-VL model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask about the image.")
    parser.add_argument("--model", type=str, default="outputs/qwen2vl-dpo-merged", help="Path to the merged model.")
    
    args = parser.parse_args()
    
    print(f"\n=== Inference on {args.image} ===")
    print(f"Question: {args.question}")
    
    try:
        response = run_inference(args.model, args.image, args.question)
        print("\n=== Response ===")
        print(response)
    except Exception as e:
        print(f"\nError during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
