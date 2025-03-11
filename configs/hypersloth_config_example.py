from transformers.training_args import TrainingArguments
from HyperSloth.app_config import HyperSlothConfig


hyper_config = HyperSlothConfig(
    dataset_file="data/cod_1k.json",
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    test_ratio=0.05,
    max_seq_length=2048,
    loss_type="target_only",
    packing=False,
    gpus=[0, 1, 2, 3],
)


training_config = dict(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    logging_steps=1,
    eval_steps=10000,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=0.0002,  # 2e-4
    bf16=True,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="model_training_outputs/debug",
    save_total_limit=2,
    report_to="tensorboard",
)
