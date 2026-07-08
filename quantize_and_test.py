import torch
import time
import glob
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

def quantize_and_test():
    model_path = "outputs/qwen2vl-dpo-merged"
    output_4bit_dir = "outputs/qwen2vl-dpo-4bit"
    
    print(f"Loading model '{model_path}' with 4-bit NF4 Quantization (bitsandbytes)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    start_load = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    load_time = time.time() - start_load
    print(f"Model successfully loaded in {load_time:.2f} seconds!")
    
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"VRAM Allocated: {allocated_gb:.2f} GB | VRAM Reserved: {reserved_gb:.2f} GB")
    
    images = glob.glob("data/images/*.jpg")
    image_path = images[0] if images else None
    
    if not image_path:
        print("No images found in data/images/, skipping inference test.")
        return
        
    print(f"\nRunning 4-bit quantized inference test on: {image_path}")
    question = "Describe this image."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    start_infer = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.2,
        )
    infer_time = time.time() - start_infer
    
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\n=== 4-Bit Quantized Inference Result ===")
    print(f"Response: {response}")
    print(f"Inference Time: {infer_time:.2f} seconds")
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM Memory Usage during Inference: {peak_vram:.2f} GB")
        
    print(f"\nSaving 4-bit quantized model to '{output_4bit_dir}'...")
    model.save_pretrained(output_4bit_dir)
    processor.save_pretrained(output_4bit_dir)
    print(f"Successfully saved 4-bit quantized model!")

if __name__ == "__main__":
    quantize_and_test()
