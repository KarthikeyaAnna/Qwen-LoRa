import torch
import random
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_inference(model_path: str, image_path: str, question: str):
    print(f"Running inference on {image_path} with {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
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
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
        )
    
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def create_pope_questions(image_objects: list, all_objects: list, n_positive=5, n_negative=5):
    questions = []
    for obj in random.sample(image_objects, min(n_positive, len(image_objects))):
        questions.append({
            "question": f"Is there a {obj} in the image?",
            "answer": "yes"
        })
    negative_pool = [o for o in all_objects if o not in image_objects]
    for obj in random.sample(negative_pool, min(n_negative, len(negative_pool))):
        questions.append({
            "question": f"Is there a {obj} in the image?",
            "answer": "no"
        })
    random.shuffle(questions)
    return questions

def evaluate_pope(model_path: str, test_data: list):
    tp, fp, tn, fn = 0, 0, 0, 0
    for sample in test_data:
        response = run_inference(model_path, sample["image"], sample["question"])
        predicted = "yes" if "yes" in response.lower() else "no"
        actual = sample["answer"]
        
        if predicted == "yes" and actual == "yes": tp += 1
        elif predicted == "yes" and actual == "no": fp += 1
        elif predicted == "no" and actual == "no": tn += 1
        elif predicted == "no" and actual == "yes": fn += 1
        
    total = tp + fp + tn + fn
    if total == 0:
        return {}
        
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    hallucination_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": hallucination_rate,
        "yes_ratio": (tp + fp) / total,
    }

if __name__ == "__main__":
    print("Evaluate script ready. You can import and run `evaluate_pope`.")
