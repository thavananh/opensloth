"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os
import random
from loguru import logger

from .hypersloth_config import HyperConfig, TrainingArgsConfig


def setup_model_and_training(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        gpu: GPU index
        hyper_config: Configuration arguments
        hf_train_args: Training arguments

    Returns:
        Trainer object configured for multi-GPU training
    """
    from unsloth import FastModel
    from HyperSloth.dataset_utils import get_chat_dataset
    from trl import SFTTrainer

    gpu_ith = hyper_config.training.gpus.index(gpu)

    # Initialize model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    if not hyper_config.fast_model_args.full_finetuning:
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())

    # Load dataset
    ds_train, ds_test = get_chat_dataset(
        tokenizer=tokenizer, **hyper_config.data.model_dump()
    )

    # Apply PEFT model

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test if gpu_ith == 0 else None,
        dataset_text_field="text",
        max_seq_length=hyper_config.fast_model_args.max_seq_length,
        dataset_num_proc=hyper_config.data.dataset_num_proc,
        args=hf_train_args,
    )

    # Adjust dataset for multi-GPU training
    max_len_ds = len(hyper_config.training.gpus) * (
        len(trainer.train_dataset) // len(hyper_config.training.gpus)
    )

    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    if hyper_config.data.group_by_length:  # This currently hurts performance
        from .patching import patch_sampler, select_dataset_by_length
        trainer = patch_sampler(trainer)
        trainer.train_dataset = select_dataset_by_length(
            trainer.train_dataset,
            gpu_ith,
            len(hyper_config.training.gpus),
            hf_train_args.gradient_accumulation_steps,
            hf_train_args.per_device_train_batch_size,
        )
    else:
        trainer.train_dataset = trainer.train_dataset.shard(
            num_shards=len(hyper_config.training.gpus), index=gpu_ith
        )

    # Handle specific training loss type
    if hyper_config.training.loss_type == "response_only":
        from unsloth.chat_templates import train_on_responses_only

        first_text = ds_train[0]["text"]
        instruction_part = hyper_config.data.instruction_part
        response_part = hyper_config.data.response_part
        assert instruction_part in first_text, f"{instruction_part} not in {first_text}"
        assert response_part in first_text, f"{response_part} not in {first_text}"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    if gpu_ith == 0:
        logger.info(f"Model setup complete for GPU {gpu_ith}")
        from ._debug_dataloader import _debug_dataloader
        _debug_dataloader(trainer)
    return trainer




