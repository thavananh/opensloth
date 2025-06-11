from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs
import os
from opensloth.opensloth_config import (
    OpenSlothConfig,
    HFDatasetConfig,
    FastModelArgs,
    LoraArgs,
    TrainingArguments,
    PathDatasetConfig,
)
import datetime

todaytime = datetime.datetime.now().strftime("%y%m%d_%H%M")

MAX_SQL_LENGTH = 16000
opensloth_config = OpenSlothConfig(
    data=PathDatasetConfig(
        path="/mnt/data/sharegpt/translate_glossary_gen_1.6M_250610.json",
        chat_template="qwen3",
        tokenizer_name="unsloth/Qwen3-8B",
        max_seq_length=MAX_SQL_LENGTH,
    ),
    devices=[0, 1, 2, 3],
    fast_model_args=FastModelArgs(
        model_name="model_store/unsloth/Qwen3-32B-bnb-4bit",
        max_seq_length=MAX_SQL_LENGTH,
        load_in_4bit=True,
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
    sequence_packing=True,
)

# Training arguments with calculated gradient accumulation
training_config = TrainingArguments(
    output_dir=f"outputs/poly_{todaytime}_{opensloth_config.fast_model_args.model_name.split('/')[-1]}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=512,
    learning_rate=1e-5,
    logging_steps=1,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=1,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb",
)

if __name__ == "__main__":
    # Setup wandb with proper naming
    model_name = opensloth_config.fast_model_args.model_name.split("/")[-1]
    exp_name = f"poly_{model_name}_{todaytime}"
    os.environ["WANDB_PROJECT"] = "poly"
    os.environ["WANDB_NAME"] = exp_name
    num_devices = len(opensloth_config.devices)
    global_batch_size = (
        num_devices
        * training_config.per_device_train_batch_size
        * training_config.gradient_accumulation_steps
    )

    print(f"Running experiment with model: {model_name}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps}")

    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
