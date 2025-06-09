from opensloth.scripts.opensloth_trainer import run_mp_training, setup_envs
from opensloth.opensloth_config import (
    OpenSlothConfig,
    HFDatasetConfig,
    FastModelArgs,
    LoraArgs,
    TrainingArguments,
)
from loguru import logger

# from transformers.training_args import TrainingArguments


# # Main configuration using Pydantic models
def get_configs(n) -> tuple[OpenSlothConfig, TrainingArguments]:
    opensloth_config = OpenSlothConfig(
        data=HFDatasetConfig(
            tokenizer_name="Qwen/Qwen3-8B",
            chat_template="qwen3",
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            num_samples=1000,
            nproc=52,
            max_seq_length=4096,
            source_type="hf",
            dataset_name="mlabonne/FineTome-100k",
            split="train",
        ),
        devices=[0, 1, 2, 3][:n],
        fast_model_args=FastModelArgs(
            model_name="unsloth/Qwen3-0.6B-bnb-4bit",
            max_seq_length=4096,
            load_in_4bit=True,
        ),
        lora_args=LoraArgs(
            r=8,
            lora_alpha=16,
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
        disable_packing=True,
    )

    # # Training arguments using Pydantic model
    training_config = TrainingArguments(
        output_dir=f"outputs/exps/qwen3-0.6b-FineTome-{n}gpus-no-packing/",
        per_device_train_batch_size=8 // n,
        gradient_accumulation_steps=1,  # Adjust based on n_gpu
        learning_rate=1e-5,
        logging_steps=1,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        warmup_steps=5,
        save_total_limit=1,
        weight_decay=0.01,
        optim="adamw_8bit",
        seed=3407,
        report_to="tensorboard",  # tensorboard or wawndb
    )
    setup_envs(opensloth_config, training_config)
    return opensloth_config, training_config


if __name__ == "__main__":
    try:
        for n in [2, 4]:
            opensloth_config, training_config = get_configs(n)
            run_mp_training(opensloth_config.devices, opensloth_config, training_config)

    except Exception as e:
        logger.exception("An error occurred during training setup or execution.")
        raise e
