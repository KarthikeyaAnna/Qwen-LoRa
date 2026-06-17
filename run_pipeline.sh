#!/bin/bash
set -e

echo "=== Qwen2-VL DPO LoRA Pipeline ==="

# 1. Setup Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing requirements (this may take a few minutes for PyTorch)..."
pip install -r requirements.txt

# 2. Execute Pipeline
echo ""
echo ">>> STEP 1: Downloading & Preparing dataset..."
python data/prepare_dataset.py

echo ""
echo ">>> STEP 2: Starting DPO Training..."
python train_dpo.py

echo ""
echo ">>> STEP 3: Merging adapters..."
python merge_adapter.py

echo ""
echo "=== Pipeline finished successfully! ==="
echo "You can now run inference using:"
echo "python inference.py --image data/images/[ANY_ID].jpg --question 'Describe this image.'"
