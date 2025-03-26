from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset_name_or_path="data/OpenO1-SFT",
        group_by_length=True,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_samples=10000,
    ),
    training=TrainingConfig(
        gpus=range(8),  # Change this to the number of GPUs you have
        loss_type="response_only",  # Choices: ["all", "response_only"], the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-27b-it-bnb-4bit",
        max_seq_length=8_000,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="/data-4090/anhvth5/hypersloth_output/loras/gemma-3-27b-it/openo1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8*16,
    learning_rate=1e-4,
    per_device_eval_batch_size=4,
    eval_steps=100000,
    logging_steps=1,
    report_to="tensorboard",
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=100,
    save_only_model=True,
    save_steps=200,
    seed=42,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    packing=False,
    include_num_input_tokens_seen=True,
)
