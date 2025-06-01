from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig.from_dataset_name("finetom"),
    training=TrainingConfig(
        gpus=[0, 1],  # Using GPU 3 as in your original script
        loss_type="response_only",
        shuffle_mode="on_dataset",
    ),
    fast_model_args=FastModelArgs(
        model_name="model_store/unsloth/Qwen3-8B-bnb-4bit",
        max_seq_length=2048,
    ),
    lora_args=LoraArgs(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_rslora=False,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-0.6b-2card.1/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=2,
    weight_decay=0.01,
    max_steps=100,
    optim="adamw_8bit",  # Using 8bit optimizer from original
    seed=3407,  # Adding seed for reproducibility
    report_to="wandb",  # Disable reporting
)
