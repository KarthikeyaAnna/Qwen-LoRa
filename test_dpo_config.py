from trl import DPOConfig
cfg = DPOConfig(
    output_dir="outputs",
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
    precompute_ref_log_probs=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    report_to="tensorboard"
)
print("DPOConfig instantiated successfully!")
