from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset_name_or_path="data/alpaca-cleaned",
        split="train",
        group_by_length=False,
        instruction_part='<start_of_turn>user\n',
        response_part="<start_of_turn>model\n",
        num_samples=16_000,
    ),
    training=TrainingConfig(
        gpus=[7],  # Change this to the number of GPUs you have
        loss_type="all",  # Choices: ["all", "response_only"], the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-4b-it-bnb-4bit",
        max_seq_length=4_000,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    )
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="/data-4090/anhvth5/hypersloth_output/loras/gemma-3-4b-it/alpaca-cleaned-1gpus",
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=32,  # Meaning 32*4 = 128 samples per batch
    learning_rate=2e-4,
    per_device_eval_batch_size=4,
    eval_steps=100000,
    logging_steps=1,
    report_to="tensorboard",
    num_train_epochs=3,
    lr_scheduler_type="linear",
    warmup_steps=5,
    seed=42,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    packing=False,
    # dataset_kwargs={"skip_prepare_dataset": True},
)
