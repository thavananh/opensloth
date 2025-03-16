from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    grad_dir="/dev/shm/hypersloth",
    data=DataConfig(
        dataset_name_or_path="../localization/data/sharegpt/train_50k.json",
        test_ratio=0.05,
        dataset_num_proc=16,
    ),
    training=TrainingConfig(
        gpus=range(1),
        loss_type="all",
        
    ),
    fast_model_args=FastModelArgs(
        model_name="/mnt/data/huggingface-models/ModelSpace/GemmaX2-28-2B-v0.1",
        max_seq_length=2048,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="outputs/2B/",
    per_device_train_batch_size=4,
    learning_rate=0.0002,
    gradient_accumulation_steps=16,
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
