# Memory-optimized training for large models
# To run: hypersloth-train example_memory_optimized_training.py
from HyperSloth.hypersloth_config import *


def build_messages(row): #each row have question and answer
    return {'messages': [
        {"role": "user", "content": row['question']},
        {"role": "assistant", "content": row['answer']}
    ]}

# Configuration optimized for maximum memory efficiency
hyper_config_model = HyperConfig(
    data=DataConfigHF(
        dataset_name="microsoft/orca-math-word-problems-200k",
        tokenizer_name="Qwen/Qwen3-8B",  # Large model tokenizer
        num_samples=50000,
        split="train",
        name="orca-math-50k",
        columns=["messages"],
        preprocess_fn=build_messages,  # Custom preprocessing to format messages
    ),
    training=TrainingConfig(
        gpus=[0, 1, 2, 3],  
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8B-bnb-4bit",  # Large 32B model
        max_seq_length=1024,  # Shorter sequences to save memory
        load_in_4bit=True,  # Essential for 32B model
        load_in_8bit=False,  # Use 4-bit instead
    ),
    lora_args=LoraArgs(
        r=8,  # Very low rank for memory efficiency
        lora_alpha=16,  # Alpha = 2x rank
        target_modules=[
            "q_proj", "v_proj",  # Only key modules for efficiency
            "gate_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_rslora=False,
    ),
)

# Memory-optimized training arguments
training_config_model = TrainingArgsConfig(
    output_dir="outputs/qwen3-32b-orca-math-memory-opt/",
    per_device_train_batch_size=1,  # Minimal batch size
    gradient_accumulation_steps=64,  # Large accumulation for effective training
    learning_rate=3e-4,  # Higher LR to compensate for low rank
    logging_steps=5,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=50,
    save_total_limit=2,  # Minimal checkpoints to save disk space
    weight_decay=0.01,
    optim="adamw_8bit",  # Memory-efficient optimizer
    seed=3407,
    report_to="wandb",
    eval_strategy="no",  # Required for multi-GPU training
)
