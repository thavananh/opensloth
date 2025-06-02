# To run this code hypersloth-train example_training_config.py
from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(

    data=DataConfigHF(
        dataset_name="llamafactory/OpenThoughts-114k",
        split="train",
        tokenizer_name="Qwen/Qwen3-8B",  # does not matter same family qwen3
    ),
    training=TrainingConfig(
        gpus=[0, 1],
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-0.6b-bnb-4bit",
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
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
)
