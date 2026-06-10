from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hello"}]}]

# Test 1: apply_chat_template
try:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print("apply_chat_template works:", text)
except Exception as e:
    print("apply_chat_template FAILED:", e)

# Test 2: process_vision_info
try:
    image_inputs, video_inputs = process_vision_info(messages)
    print("process_vision_info works:", image_inputs)
except Exception as e:
    print("process_vision_info FAILED:", e)
