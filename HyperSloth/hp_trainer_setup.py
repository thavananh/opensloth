"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from HyperSloth.init_modules import (
    configure_batch_size,
    create_trainer,
    init_model_and_tokenizer,
)

from .hypersloth_config import HyperConfig, TrainingArgsConfig
from loguru import logger


def _change_compiler_location() -> None:
    import unsloth  # type: ignore
    from unsloth_zoo import compiler

    # ====== Patching the compiler location to avoid race conditions as it is shared between GPUs
    gpu_ith = int(os.environ["HYPERSLOTH_LOCAL_RANK"])

    compiler.UNSLOTH_COMPILE_LOCATION = ".cache/{}_{}".format(
        compiler.UNSLOTH_COMPILE_LOCATION, gpu_ith
    )
    logger.info(f"Using compiler location: {compiler.UNSLOTH_COMPILE_LOCATION}")


def setup_model_and_training(
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.
    """

    gpu_ith = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    num_gpus = int(os.environ["HYPERSLOTH_WORLD_SIZE"])

    # Get enhanced logger for timing
    from .logging_config import get_hypersloth_logger

    hp_logger = get_hypersloth_logger(log_level="INFO")

    # Start total setup timing
    hp_logger.start_timing("total_setup")

    _change_compiler_location()

    # Time batch size configuration
    configure_batch_size(hf_train_args, gpu_ith, num_gpus)


    # Time model initialization
    hp_logger.start_timing("model_init")
    model, tokenizer = init_model_and_tokenizer(hyper_config)
    hp_logger.finish_timing("model_init")


    # Time trainer creation
    hp_logger.start_timing("trainer_creation")
    trainer = create_trainer(model, tokenizer, hyper_config, hf_train_args)
    hp_logger.finish_timing("trainer_creation")


    # Finish total setup timing
    hp_logger.finish_timing("total_setup")

    return trainer, model, tokenizer
