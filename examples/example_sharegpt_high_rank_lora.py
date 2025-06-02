# ShareGPT format dataset with high-rank LoRA training
# To run: hypersloth-train example_sharegpt_high_rank_lora.py
from HyperSloth.hypersloth_config import *

# Configuration for ShareGPT format with high-rank LoRA
hyper_config_model = HyperConfig(
    data=DataConfigShareGPT(
        dataset_path="data/sharegpt_conversations.json",  # Local ShareGPT file
        tokenizer_name="Qwen/Qwen3-8B",
        num_samples=None,  # Use entire dataset
        seed=3407,
        instruction_part="<|im_start|>user\n",  # Custom instruction format
        response_part="<|im_start|>assistant\n",  # Custom response format
        print_samples=True,  # Debug: print sample conversations
        use_cache=True,  # Cache processed dataset
        name="custom_sharegpt",  # Local reference name
    ),
    training=TrainingConfig(
        gpus=[0, 1, 2, 3],  # 4-GPU training
        loss_type="response_only",
        chat_template=None,  # Use default chat template
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8b-bnb-4bit",
        max_seq_length=4096,  # Longer sequences for complex conversations
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
        r=256,  # High rank for more expressive adaptation
        lora_alpha=512,  # Alpha = 2x rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",  # MLP layers
        ],
        lora_dropout=0.05,  # Light dropout for high-rank LoRA
        bias="none",
        use_rslora=True,  # Rank-stabilized LoRA for high ranks
        random_state=3407,
    ),
)

# Training arguments for high-rank LoRA
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-8b-sharegpt-high-rank-lora/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # Effective batch size = 2*16*4 = 128
    learning_rate=1e-4,  # Slightly lower LR for high-rank LoRA
    logging_steps=5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",  # Cosine schedule for better convergence
    warmup_steps=100,
    save_total_limit=1, 
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
)
