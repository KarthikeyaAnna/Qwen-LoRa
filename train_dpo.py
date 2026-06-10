#!/usr/bin/env python3
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from transformers.trainer_utils import get_last_checkpoint
import os

from config import ModelConfig, LoRAConfig as LoRAConfigParams, DPOHyperparams, DataConfig

def load_model_and_processor(model_cfg: ModelConfig):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_cfg.model_id,
        torch_dtype=torch.bfloat16 if model_cfg.torch_dtype == "bfloat16" else torch.float32,
        attn_implementation=model_cfg.attn_implementation,
        device_map=model_cfg.device_map,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    return model, processor

def apply_lora(model, lora_cfg: LoRAConfigParams):
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
            
    # Let DPOTrainer handle get_peft_model to ensure flawless checkpoint resuming
    return model, peft_config

def create_training_args(dpo_cfg: DPOHyperparams):
    return DPOConfig(
        output_dir=dpo_cfg.output_dir,
        beta=dpo_cfg.beta,
        loss_type=dpo_cfg.loss_type,
        max_length=dpo_cfg.max_length,
        precompute_ref_log_probs=dpo_cfg.precompute_ref_log_probs,
        num_train_epochs=dpo_cfg.num_train_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        gradient_checkpointing=dpo_cfg.gradient_checkpointing,
        learning_rate=dpo_cfg.learning_rate,
        warmup_steps=dpo_cfg.warmup_steps,
        lr_scheduler_type=dpo_cfg.lr_scheduler_type,
        optim=dpo_cfg.optim,
        bf16=dpo_cfg.bf16,
        logging_steps=dpo_cfg.logging_steps,
        eval_strategy=dpo_cfg.eval_strategy,
        eval_steps=dpo_cfg.eval_steps,
        save_strategy=dpo_cfg.save_strategy,
        save_steps=dpo_cfg.save_steps,
        save_total_limit=dpo_cfg.save_total_limit,
        remove_unused_columns=dpo_cfg.remove_unused_columns,
        dataloader_num_workers=dpo_cfg.dataloader_num_workers,
        report_to=dpo_cfg.report_to,
    )

def main():
    model_cfg = ModelConfig()
    lora_cfg = LoRAConfigParams()
    dpo_cfg = DPOHyperparams()
    data_cfg = DataConfig()
    
    print("Loading datasets...")
    dataset = load_from_disk(data_cfg.processed_data_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    
    print("Loading model and processor...")
    model, processor = load_model_and_processor(model_cfg)
    
    print("Applying LoRA...")
    model, peft_config = apply_lora(model, lora_cfg)
    
    training_args = create_training_args(dpo_cfg)
    
    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    
    # Checkpoint resume logic
    last_checkpoint = None
    if os.path.isdir(dpo_cfg.output_dir):
        last_checkpoint = get_last_checkpoint(dpo_cfg.output_dir)
        if last_checkpoint is not None:
            print(f"Found existing checkpoint: {last_checkpoint}. Resuming training...")
            
    print("Starting Training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print("Saving adapter and processor...")
    trainer.save_model(dpo_cfg.output_dir + "/final")
    processor.save_pretrained(dpo_cfg.output_dir + "/final")
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    print("Training complete!")

if __name__ == "__main__":
    main()
