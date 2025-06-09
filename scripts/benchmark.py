from opensloth.opensloth_config import *
from opensloth.scripts.opensloth_trainer import run_mp_training, setup_envs

for n_gpu in [2]:
    # Main configuration using Pydantic models
    opensloth_config = OpenSlothConfig(
        data=HFDatasetConfig(
            dataset_name="llamafactory/OpenThoughts-114k",
            split="train",
            tokenizer_name="Qwen/Qwen3-8B",  # does not matter same family qwen3
            num_samples=5_000,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            chat_template="chatml",
        ),
        devices=[0, 1, 2, 3][:n_gpu],  # Adjust based on n_gpu
        fast_model_args=FastModelArgs(
            model_name="model_store/unsloth/Qwen3-8B-Base-bnb-4bit",
            max_seq_length=32_000,
            load_in_4bit=True,
        ),
        lora_args=LoraArgs(
            r=16,
            lora_alpha=32,
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
    training_config = TrainingArgsConfig(
        output_dir=f"outputs/experiment/Qwen3-8B-openthought5k-{n_gpu}gpu/",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8 // n_gpu,  # Adjust based on n_gpu
        learning_rate=1e-5,
        logging_steps=3,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        warmup_steps=5,
        save_total_limit=2,
        weight_decay=0.01,
        optim="adamw_8bit",
        seed=3407,
        report_to="wandb",  # tensorboard or wawndb
    )

    setup_envs(opensloth_config, training_config)
    if n_gpu == 1:
        # For single GPU, we can run the training directly
        from opensloth.scripts.opensloth_trainer import train_on_single_gpu

        train_on_single_gpu(
            gpu=0,
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
    else:
        run_mp_training(opensloth_config.devices, opensloth_config, training_config)
