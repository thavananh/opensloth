from opensloth.opensloth_config import *

# Main configuration using Pydantic models
opensloth_config = OpenSlothConfig(
    data=HFDatasetConfig(
        dataset_name="llamafactory/OpenThoughts-114k",
        split="train",
        num_samples=1000,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        chat_template="qwen3",
        nproc=16,
    ),
    training=TrainingConfig(
        gpus=[0, 1],
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8B-bnb-4bit",
        max_seq_length=4096,
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
training_config = TrainingArgsConfig(
    output_dir="outputs/qwen3-0.6b-openthoughts-10k-lora-2gpus",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    logging_steps=1,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=2,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="none",  # tensorboard or wawndb
)
