# Full fine-tuning configuration with HuggingFace dataset
# To run: hypersloth-train example_hf_dataset_full_finetuning.py
from HyperSloth.hypersloth_config import *

# Configuration for full fine-tuning (no LoRA)
hyper_config_model = HyperConfig(
    data=DataConfigHF(
        dataset_name="llamafactory/OpenThoughts-114k",  # Reasoning dataset
        tokenizer_name="Qwen/Qwen3-8B",
        num_samples=5000,  # Moderate dataset size
        split="train",
        name="openthoughts-5k",
        columns=["messages"],  # OpenAI format messages
    ),
    training=TrainingConfig(
        gpus=[0, 1],  # Multi-GPU for full fine-tuning
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="model_store/unsloth/Qwen3-8B",
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=True,  # Enable full fine-tuning instead of LoRA
    ),
    # Note: LoRA args are ignored when full_finetuning=True
    lora_args=LoraArgs(),  # Default/empty LoRA config (will be ignored)
)

# Training arguments for full fine-tuning
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-8b-openthoughts-fullft/",
    per_device_train_batch_size=1,  # Very small batch for full FT memory usage
    gradient_accumulation_steps=32,  # Large accumulation for effective batch size
    learning_rate=5e-6,  # Much lower LR for full fine-tuning
    logging_steps=5,
    num_train_epochs=2,  # Fewer epochs for full FT
    lr_scheduler_type="cosine",  # Cosine annealing for full FT
    warmup_steps=100,
    save_total_limit=3,
    weight_decay=0.1,  # Higher weight decay for full FT
    optim="adamw_8bit",
    seed=3407,
    report_to="tensorboard",  # Use wandb for experiment tracking
)
