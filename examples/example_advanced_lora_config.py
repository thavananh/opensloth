# Advanced LoRA configuration with custom target modules
# To run: hypersloth-train example_advanced_lora_config.py
from HyperSloth.hypersloth_config import *

# Advanced LoRA configuration with fine-grained control
hyper_config_model = HyperConfig(
    data=DataConfigHF(
        dataset_name="teknium/OpenHermes-2.5",  # High-quality instruction dataset
        tokenizer_name="Qwen/Qwen3-8B",
        num_samples=20000,
        split="train", 
        name="openhermes-20k",
        columns=["conversations"],
    ),
    training=TrainingConfig(
        gpus=[0, 1],  # Dual GPU training
        loss_type="all",  # Compute loss on all tokens (not just responses)
        chat_template=[  # Custom chat template
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
            "<|im_start|>user\n{instruction}<|im_end|>",
            "<|im_start|>assistant\n{response}<|im_end|>",
        ],
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8b-bnb-4bit",
        max_seq_length=3072,  # Longer context for complex instructions
        load_in_4bit=True,
        full_finetuning=False,  # Explicitly use LoRA
    ),
    lora_args=LoraArgs(
        r=128,  # High rank for complex adaptations
        lora_alpha=256,  # Alpha = 2x rank
        target_modules=[
            # Attention modules
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP modules  
            "gate_proj", "up_proj", "down_proj",
            # Additional modules for comprehensive adaptation
            "embed_tokens",  # Embedding layer
            "lm_head",  # Output projection
        ],
        lora_dropout=0.1,
        bias="lora_only",  # Adapt bias terms in LoRA layers only
        use_rslora=True,  # Rank-stabilized LoRA for high ranks
        random_state=3407,
        # Fine-grained layer control
        finetune_vision_layers=False,  # No vision layers
        finetune_language_layers=True,  # Language layers only
        finetune_attention_modules=True,  # All attention
        finetune_mlp_modules=True,  # All MLP layers
    ),
)

# Advanced training arguments with learning rate scheduling
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-8b-openhermes-advanced-lora/",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=12,  # Effective batch size = 3*12*2 = 72
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    lr_scheduler_type="cosine",  # Cosine annealing with restarts
    warmup_steps=200,  # Longer warmup for stability
    save_total_limit=5,
    weight_decay=0.05,  # Moderate regularization
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
    eval_strategy="no",  # Required for multi-GPU
)
