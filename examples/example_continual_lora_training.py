# Continual LoRA training from pretrained checkpoint
# To run: hypersloth-train example_continual_lora_training.py
from HyperSloth.hypersloth_config import *

# Configuration for continuing training from existing LoRA
hyper_config_model = HyperConfig(
    data=DataConfig.from_dataset_name("finetom"),  # Use prepared dataset from registry
    training=TrainingConfig(
        gpus=[0],  # Single GPU for continual training
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8b-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
        r=64,  # Medium rank for balanced performance
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        use_rslora=False,
    ),
    # Continue training from existing LoRA checkpoint
    pretrained_lora="outputs/previous-lora-training/checkpoint-500",
)

# Training arguments for continual training
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-8b-continual-lora/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,  # Lower LR for continual training
    logging_steps=10,
    num_train_epochs=1,  # Short training for adaptation
    lr_scheduler_type="linear",
    warmup_steps=20,  # Shorter warmup for continual training
    save_total_limit=3,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="tensorboard",
)
