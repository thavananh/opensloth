from HyperSloth.hypersloth_config import (
    HyperConfig, 
    TrainingArgsConfig,
    FastModelArgs,
    LoraArgs,
    TrainingConfig,
    DataConfig
)

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    grad_dir="/dev/shm/hypersloth",
    data=DataConfig(
        # dataset="data/cod_1k.json",
        dataset_name_or_path="../localization/data/sharegpt/train_234k.json",
        # split="train",
        test_ratio=0.05,
        dataset_num_proc=4,
        num_samples=234_00,
    ),
    training=TrainingConfig(
        # gpus=range(8),
        gpus=range(1),
        loss_type="all",
        
    ),
    fast_model_args=FastModelArgs(
        model_name="/mnt/data/huggingface-models/ModelSpace/GemmaX2-28-9B-v0.1-bnb-4bit",
        max_seq_length=16_000,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    ),
    lora_args=LoraArgs(
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="model_training_outputs/debug",
    per_device_train_batch_size=1,
    learning_rate=0.0002,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=2,
    eval_steps=100,
    logging_steps=1,
    report_to="tensorboard",
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    seed=42,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    packing=False,
)

# Keeping the old dict versions for backward compatibility
hyper_config = hyper_config_model.model_dump()
training_config = training_config_model.model_dump()
