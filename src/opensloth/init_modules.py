"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from speedy_utils import identify


from opensloth.dataset_utils import get_tokenized_dataset


from .opensloth_config import (
    OpenSlothConfig,
    TrainingArguments,
)

from .logging_config import get_opensloth_logger


def init_model_and_tokenizer(opensloth_config: OpenSlothConfig):
    """Initialize and optionally set up LoRA for the model."""
    from unsloth import FastModel

    logger = get_opensloth_logger()

    logger.start_timing("model_loading")

    if opensloth_config.pretrained_lora:
        logger.info(
            f"Loading model from {opensloth_config.pretrained_lora} with LoRA weights"
        )
        opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora
    from opensloth.nccl_grad_sync import setup_nccl_for_opensloth

    model, tokenizer = FastModel.from_pretrained(
        **opensloth_config.fast_model_args.model_dump()
    )
    logger.finish_timing("model_loading")

    logger.start_timing("nccl_setup")
    setup_nccl_for_opensloth(
        rank=int(os.environ["OPENSLOTH_LOCAL_RANK"]),
        gpus=opensloth_config.devices,
    )
    logger.finish_timing("nccl_setup")

    model_device = model.device
    logger.info(
        f"Model loaded on device {model_device}, tokenizer: {tokenizer.__class__.__name__}"
    )

    # Monkey-patch pad for any ProcessorMixin tokenizer lacking it
    if not hasattr(tokenizer, "pad"):
        import types
        underlying = getattr(tokenizer, "tokenizer", None) or getattr(tokenizer, "hf_tokenizer", None)
        if underlying and hasattr(underlying, "pad"):
            def _pad(self, *args, **kwargs):
                return underlying.pad(*args, **kwargs)

            tokenizer.pad = types.MethodType(_pad, tokenizer)
            logger.info(f"Patched pad method for {tokenizer.__class__.__name__}")
        else:
            logger.warning(f"Could not patch pad method for {tokenizer.__class__.__name__}: underlying tokenizer has no pad")

    if (
        not opensloth_config.fast_model_args.full_finetuning
        and not opensloth_config.pretrained_lora
    ):
        logger.start_timing("lora_setup")
        model = FastModel.get_peft_model(
            model, **opensloth_config.lora_args.model_dump()
        )
        logger.finish_timing("lora_setup")

    # Allow custom chat templates
    if (
        hasattr(opensloth_config.data, "chat_template")
        and opensloth_config.data.chat_template is not None
    ):
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(
            tokenizer, chat_template=opensloth_config.data.chat_template
        )
        logger.info(f"Applied chat template: {opensloth_config.data.chat_template}")

    return model, tokenizer


def create_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Load or prepare the dataset and create the SFTTrainer."""

    # Get enhanced logger for timing

    logger = get_opensloth_logger()

    logger.start_timing("trainer_setup")

    trainer = _get_trainer(
        model,
        tokenizer,
        opensloth_config,
        hf_train_args,
    )

    logger.finish_timing("trainer_setup")

    logger.start_timing("training_loop_patch")
    from opensloth.patching.inner_training_loop import patch_inner_training_loop
    from opensloth.patching.patch_sampler import patch_sampler

    from opensloth.patching.patch_log import patch_log

    patch_log(type(trainer))
    patch_inner_training_loop(opensloth_config)

    from .patching.get_batch_samples import patch_get_batch_samples

    patch_get_batch_samples(opensloth_config)

    # ====
    trainer = patch_sampler(trainer)
    logger.finish_timing("training_loop_patch")

    # ===
    from .patching.patch_sampler import ShuffleData

    logger.info(f"Add callback ShuffleData to Trainer {trainer.__class__.__name__}")
    trainer.add_callback(ShuffleData())

    return trainer


def _get_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Returns an SFTTrainer instance with a tokenized dataset.
    """
    from trl import SFTTrainer
    from transformers import DataCollatorForSeq2Seq
    from .logging_config import get_opensloth_logger

    logger = get_opensloth_logger()

    # Get the tokenized dataset using the dataset_utils function
    tokenized_train_dataset = get_tokenized_dataset(
        config=opensloth_config.data,
    )

    logger.info("Creating final SFTTrainer with prepared dataset...")
    logger.start_timing("final_trainer_creation")
    hf_train_args.skip_prepare_dataset = True
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        args=hf_train_args,
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
        hf_train_args.report_to = "none"


__all__ = [
    "configure_batch_size",
    "init_model_and_tokenizer",
    "create_trainer",
]
