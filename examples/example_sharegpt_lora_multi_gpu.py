# ShareGPT format dataset with multi-GPU LoRA training
# To run: hypersloth-train example_sharegpt_lora_multi_gpu.py
from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    # data=DataConfigHF(
    #     dataset_name="mlabonne/FineTome-100k",
    #     tokenizer_name="Qwen/Qwen3-8B",  # does not matter same family qwen3
    #     num_samples=10000,
    #     split="train",
    #     name="finetom-10k",  # local name for later reference
    # ),
    data=DataConfigHF(
        dataset_name="llamafactory/OpenThoughts-114k",
        split="train",
        tokenizer_name="Qwen/Qwen3-8B",  # does not matter same family qwen3
        num_samples=1000,
        name="openthoughts-1k",  # local name for later reference
    ),
    # data=DataConfigShareGPT(
    #     dataset_path='/mnt/data/sharegpt/selfeval_retranslate_2025_05_30.json',
    #     tokenizer_name="Qwen/Qwen3-8B",  # does not matter same family qwen3
    #     num_samples=None,
    # ),
    training=TrainingConfig(
        gpus=[0, 1,2,3],
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-8b-bnb-4bit",
        max_seq_length=2048,
    ),
    lora_args=LoraArgs(
        r=512,
        lora_alpha=1024,
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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    logging_steps=3,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=2,
    weight_decay=0.01,
    # max_steps=100,
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
)
