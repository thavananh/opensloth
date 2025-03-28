from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    grad_dir="/dev/shm/hypersloth",
    data=DataConfig(
        dataset_name_or_path="mlabonne/FineTome-100k",
        test_ratio=0.05,
        split="train",
        num_samples=1000,  # for debuging
        instruction_part="<start_of_turn>user\n",  # For gemma it is <bos><start_of_turn>user to train with loss
        response_part="<start_of_turn>model\n",  # For gemma it is <start_of_response> to train with loss
    ),
    training=TrainingConfig(
        gpus=range(2),  # Change this to the number of GPUs you have
        loss_type="response_only",  # all or response_only, the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-1b-it",
        max_seq_length=2048,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="outputs/2B/",
    per_device_train_batch_size=4,  #
    gradient_accumulation_steps=16,  # More GA help to reduce total communication time
    learning_rate=0.0002,
    per_device_eval_batch_size=2,
    eval_steps=100,
    logging_steps=1,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=2,
    weight_decay=0.01,
    packing=False,
)
