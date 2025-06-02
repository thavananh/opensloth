# Basic LoRA fine-tuning with HuggingFace dataset on single GPU
# To run: hypersloth-train example_hf_dataset_lora_single_gpu.py
from HyperSloth.hypersloth_config import *

# Configuration for single GPU LoRA training with HuggingFace dataset
hyper_config_model = HyperConfig(
    data=DataConfigHF(
        dataset_name="mlabonne/FineTome-100k",  # Popular instruction dataset
        tokenizer_name="Qwen/Qwen3-8B",  # Tokenizer from same model family
        num_samples=1000,  # Small subset for quick testing
        split="train",  # Dataset split to use
        name="finetome-1k",  # Local name for dataset caching
        columns=["conversations"],  # Column containing conversation data
    ),
    training=TrainingConfig(
        gpus=[0],  # Single GPU training
        loss_type="response_only",  # Only compute loss on assistant responses
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8b-bnb-4bit",  # 4-bit quantized model
        max_seq_length=1024,  # Shorter sequences for faster training
        load_in_4bit=True,  # Enable 4-bit quantization for memory efficiency
    ),
    lora_args=LoraArgs(
        r=16,  # LoRA rank - lower for efficiency
        lora_alpha=32,  # LoRA alpha (typically 2x rank)
        target_modules=[  # Standard transformer attention/MLP modules
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,  # Dropout for regularization
        bias="none",  # No bias adaptation
        use_rslora=False,  # Rank-stabilized LoRA (experimental)
    ),
)

# Training arguments optimized for single GPU
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-8b-finetome-lora/",
    per_device_train_batch_size=2,  # Small batch size for memory
    gradient_accumulation_steps=8,  # Simulate larger batch size
    learning_rate=2e-4,  # Standard LoRA learning rate
    logging_steps=10,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=50,  # Warmup for stability
    save_total_limit=2,
    weight_decay=0.01,
    optim="adamw_8bit",  # Memory-efficient optimizer
    seed=3407,
    report_to="tensorboard",  # Local logging (no wandb)
)
