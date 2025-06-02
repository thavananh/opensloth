raise NotImplemented

# To run: hypersloth-train example_vision_language_lora.py
from HyperSloth.hypersloth_config import *

# Configuration for vision-language model training
hyper_config_model = HyperConfig(
    data=DataConfigHF(
        dataset_name="unsloth/Radiology_mini",  # Vision-QA dataset
        tokenizer_name="Qwen/Qwen3-VL-8B",  # Vision-language tokenizer
        num_samples=10000,
        split="train",
        name="vqav2-10k",
        columns=["conversations", "images"],  # Multi-modal data
    ),
    training=TrainingConfig(
        gpus=[0, 1],  # Multi-GPU for vision models
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-VL-8b-bnb-4bit",  # Vision-language model
        max_seq_length=1024,  # Shorter for vision tasks
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
        finetune_vision_layers=True,  # Enable vision layer fine-tuning
        finetune_language_layers=True,  # Enable language layer fine-tuning
        finetune_attention_modules=True,  # Attention modules
        finetune_mlp_modules=False,  # Skip MLP for efficiency
        r=32,  # Medium rank for vision tasks
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Text attention
            "vision_proj",  # Vision projection layers
        ],
        lora_dropout=0.1,
        bias="none",
        use_rslora=False,
    ),
)

# Training arguments for vision-language models
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-vl-8b-vqa-lora/",
    per_device_train_batch_size=1,  # Small batch for vision models
    gradient_accumulation_steps=32,  # Large accumulation
    learning_rate=1e-4,
    logging_steps=10,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    save_total_limit=3,
    weight_decay=0.05,
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
)
