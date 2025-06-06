"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from speedy_utils import identify


from HyperSloth.dataset_utils import get_tokenized_dataset


from .hypersloth_config import (
    HyperConfig,
    TrainingArgsConfig,
)

from .logging_config import get_hypersloth_logger


def init_model_and_tokenizer(hyper_config: HyperConfig):
    """Initialize and optionally set up LoRA for the model."""
    from unsloth import FastModel

    logger = get_hypersloth_logger(log_level="INFO")

    logger.start_timing("model_loading")

    if hyper_config.pretrained_lora:
        logger.info(
            f"Loading model from {hyper_config.pretrained_lora} with LoRA weights"
        )
        hyper_config.fast_model_args.model_name = hyper_config.pretrained_lora
    from HyperSloth.nccl_grad_sync import setup_nccl_for_hypersloth

    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    logger.finish_timing("model_loading")

    logger.start_timing("nccl_setup")
    setup_nccl_for_hypersloth(
        gpu=int(os.environ["HYPERSLOTH_LOCAL_RANK"]), gpus=hyper_config.training.gpus
    )
    logger.finish_timing("nccl_setup")

    model_device = model.device
    logger.info(
        f"Model loaded on device {model_device}, tokenizer: {tokenizer.__class__.__name__}"
    )

    if (
        not hyper_config.fast_model_args.full_finetuning
        and not hyper_config.pretrained_lora
    ):
        logger.start_timing("lora_setup")
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())
        logger.finish_timing("lora_setup")

    # Allow custom chat templates
    if (
        hasattr(hyper_config.training, "chat_template")
        and hyper_config.training.chat_template is not None
    ):
        from transformers import AutoTokenizer  # type: ignore

        new_template = AutoTokenizer.from_pretrained(
            hyper_config.training.chat_template
        ).chat_template
        tokenizer.chat_template = new_template
        logger.warning(f"Using chat template of {new_template}")

    return model, tokenizer


def create_trainer(
    model,
    tokenizer,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """Load or prepare the dataset and create the SFTTrainer."""

    # Get enhanced logger for timing

    logger = get_hypersloth_logger(log_level="INFO")

    logger.start_timing("trainer_setup")

    trainer = _get_trainer(
        model,
        tokenizer,
        hyper_config,
        hf_train_args,
    )

    logger.finish_timing("trainer_setup")

    logger.start_timing("training_loop_patch")
    from HyperSloth.patching.inner_training_loop import patch_inner_training_loop

    patch_inner_training_loop(trainer)
    from .patching.patch_sampler import apply_patch_sampler

    trainer = apply_patch_sampler(trainer)
    logger.finish_timing("training_loop_patch")
    return trainer


def _get_trainer(
    model, tokenizer, hyper_config: HyperConfig, hf_train_args: TrainingArgsConfig
):
    """
    Returns an SFTTrainer instance with a tokenized dataset.
    """
    from trl import SFTTrainer
    from transformers import DataCollatorForSeq2Seq
    from .logging_config import get_hypersloth_logger

    logger = get_hypersloth_logger(log_level="INFO")

    # Get the tokenized dataset using the dataset_utils function
    train_dataset = get_tokenized_dataset(
        data_config=hyper_config.data,
        model=model,
        tokenizer=tokenizer,
        hf_train_args=hf_train_args,
    )

    logger.info("Creating final SFTTrainer with prepared dataset...")
    logger.start_timing("final_trainer_creation")
    hf_train_args.skip_prepare_dataset = True
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=hf_train_args,
        max_seq_length=hf_train_args.max_seq_len,
    )
    logger.finish_timing("final_trainer_creation")

    if hasattr(trainer, "data_collator") and not isinstance(
        trainer.data_collator, DataCollatorForSeq2Seq
    ):
        logger.info(
            f"Replacing {type(trainer.data_collator).__name__} with "
            f"DataCollatorForSeq2Seq for better sequence handling"
        )
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        logger.info(f"Data collator: {type(trainer.data_collator).__name__}")

    logger.info("Trainer setup completed successfully")
    return trainer


def configure_batch_size(hf_train_args, gpu_ith, num_gpus):
    if num_gpus != 1:
        hf_train_args.per_device_train_batch_size *= num_gpus  # This is the total batch size loaded by dataloader, the trainer later will chose the correct batch size for each GPU

    if not gpu_ith == 0:
        # disable reporting for all GPUs except the first one
        hf_train_args.report_to = "none"
        # disable evaluation for all GPUs except the first one
        hf_train_args.do_eval = False


__all__ = [
    "configure_batch_size",
    "init_model_and_tokenizer",
    "create_trainer",
]
