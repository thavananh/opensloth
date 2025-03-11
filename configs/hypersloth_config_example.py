from HyperSloth.app_config import HyperSlothConfig


hyper_config = HyperSlothConfig(
    dataset_file="data/cod_1k.json",
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    test_ratio=0.05,
    max_seq_length=2048,
    loss_type="target_only",
    packing=False,
    gpus=[0],
)

# MUST NOT INITIALIZE DEVICE BEFORE threaded.run() IN HyperSloth/scripts/hypersloth.py
training_config = dict(
    output_dir="model_training_outputs/debug",
    per_device_train_batch_size=8,
    learning_rate=0.0002,  
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=2,
    eval_steps=100,
    logging_steps=1,
    report_to="tensorboard",
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    seed=3407,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",
    weight_decay=0.01,
)
