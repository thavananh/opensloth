import argparse
import os
from typing import List, Optional, Literal
import yaml
import fire
from loguru import logger
from transformers import TrainingArguments
from HyperSloth.app_config import HyperSlothConfig


def load_training_args(
    config_path: str, gpu_index: int, visible_devices: List[int], 
) -> TrainingArguments:
    """Load training arguments from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Override specific fields if needed based on GPU settings
    is_main = gpu_index == visible_devices[0]
    output_dir = (
        f"{config.get('output_dir', 'model_training_outputs/debug')}/{gpu_index}"
    )
    report_to = config.get("report_to", "tensorboard") if is_main else None

    return TrainingArguments(
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        logging_steps=config.get("logging_steps", 1),
        eval_strategy="steps" if is_main else "no",
        eval_steps=config.get("eval_steps", 10000),
        warmup_steps=config.get("warmup_steps", 5),
        do_eval=True,
        num_train_epochs=config.get("num_train_epochs", 1),
        learning_rate=config.get("learning_rate", 2e-4),
        fp16=not config.get("bf16", True),
        bf16=config.get("bf16", True),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=config.get("seed", 3407),
        output_dir=output_dir,
        save_total_limit=config.get("save_total_limit", 2),
        report_to=report_to,
    )


from fastcore.all import threaded


def _get_devices():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if gpus is None:
        logger.warning("CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0")
        return [0]

    visible_devices = [int(gpu) for gpu in gpus.split(",")]
    return visible_devices


@threaded(process=True)
def run(
    gpu_index,
    visible_devices,
    hyper_config: HyperSlothConfig,
    training_config_path: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    from HyperSloth.transformer_trainer_setup import setup_model_and_training
    from HyperSloth.dataset_utils import load_sharegpt_dataset

    is_main = gpu_index == visible_devices[0]

    # Load training arguments from YAML
    train_args = load_training_args(
        training_config_path, gpu_index, visible_devices, 
    )

    trainer = setup_model_and_training(
        hyper_config=hyper_config,
        hf_train_args=train_args,
        dataset_fn=lambda tokenizer, test_ratio: load_sharegpt_dataset(
            hyper_config.file, tokenizer, test_ratio
        ),
        gpu_index=gpu_index,
        visible_devices=visible_devices,
    )

    from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback

    if len(visible_devices) > 1:
        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=hyper_config.grad_dir,
            gpu_index=gpu_index,
            visible_devices=visible_devices,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu_index}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()


def main(
    config_file: str = "configs/hyper_config.yaml",
    training_args_file: str = "configs/training_args.yaml",
):
    """
    Main function to run training with multiple GPUs

    Args:
        config_file: Path to the hyperlevel configuration file
        training_args_file: Path to the training arguments YAML file
    """

    # Load hyperlevel config from YAML file
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    # Create HyperSlothConfig from the loaded dictionary
    hyper_config = HyperSlothConfig(**config_dict)
    visible_devices = _get_devices()
    
    for gpu_index in visible_devices:
        logger.debug(f"Running on GPU {gpu_index}")
        run(gpu_index, visible_devices, hyper_config, training_args_file)


def cli():
    """Command line interface for HyperSloth training framework."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
