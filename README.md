# Qwen2-VL-2B DPO LoRA Fine-tuning

This repository contains the full pipeline to mitigate visual hallucinations in Qwen2-VL-2B-Instruct using Direct Preference Optimization (DPO) and LoRA adapters. 

## Prerequisites for Server Deployment

1. **Hardware Requirements**: RTX 4090 / 5070 Ti (or any 16GB+ VRAM GPU), or higher.
2. **Dataset**: This pipeline uses the HA-DPO dataset. The dataset provides preference pairs but requires you to have the MSCOCO / Visual Genome images stored locally.

### Setup Instructions

1. Clone or upload this repository to your server.
2. Configure your image directory in `config.py`:
   ```python
   # Inside config.py
   image_dir: str = "/path/to/your/MSCOCO_or_VG_images/"
   ```
3. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch>=2.4.0
   pip install -r requirements.txt
   
   # Optional: If you want flash_attention_2 instead of sdpa
   # pip install flash-attn>=2.0.0 --no-build-isolation
   ```

### Execution Pipeline

**Step 1: Prepare the Dataset**
```bash
python data/prepare_dataset.py
```
*This validates your images, subsamples the data, structures it for TRL's conversational DPO format, and saves it to `data/processed/`.*

**Step 2: Run DPO Training**
```bash
python train_dpo.py
```
*This runs the parameter-efficient fine-tuning (PEFT) on the LLM backbone while keeping the vision encoder frozen. It uses gradient checkpointing and precomputed reference logprobs to fit within 10-12 GB VRAM. Adapters are saved to `outputs/qwen2vl-dpo-lora/final`.*

**Step 3: Merge Adapters**
```bash
python merge_adapter.py
```
*Merges the LoRA weights back into the base Qwen2-VL model for faster inference.*

**Step 4: Evaluate**
```bash
# You can use pope_eval.py to run POPE-style hallucination checks
# Or simply test inference
python inference.py
```

## Monitoring
Since `report_to="tensorboard"` is set, you can view the training metrics by running:
```bash
tensorboard --logdir runs/
```
Watch the `rewards/margins` — they should go up, while `loss` should go down.
