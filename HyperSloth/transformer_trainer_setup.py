"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os
import time
from loguru import logger
import filelock

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

    gpu_ith = hyper_config.training.gpus.index(gpu)

    # Initialize model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    if not hyper_config.fast_model_args.full_finetuning:
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())

    # Load dataset
    trainer = _run(tokenizer, hyper_config, hf_train_args, gpu_ith, model)

    # Shard the dataset for multi-GPU training
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(hyper_config.training.gpus), index=gpu_ith
    )

    # Handle specific training loss type
    if hyper_config.training.loss_type == "response_only":
        from unsloth.chat_templates import train_on_responses_only

        first_text = trainer.train_dataset[0]["text"]
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


def _run(tokenizer, hyper_config, hf_train_args, gpu_ith, model):
    from trl import SFTTrainer
    from datasets import load_from_disk
    from speedy_utils import identify

    tokenizer_name = identify(str(tokenizer))
    dataset_cache_path = identify(hyper_config.data.dataset_name_or_path)

    from fastcore.all import Path
    is_file = Path(dataset_cache_path).is_file()
    dataset_name = identify(hyper_config.data.model_dump())
    if is_file:
        dataset_cache_path = (
            hyper_config.data.dataset_name_or_path
            + "_"
            + tokenizer_name
            + "_"
            + dataset_name
            + ".cache"
        )
        lock = dataset_cache_path + ".lock"
    else:
        dataset_cache_path = (
            "/dev/shm/"
            + hyper_config.data.dataset_name_or_path
            + "_"
            + tokenizer_name
            + ".cache"
        )
        lock = dataset_cache_path + ".lock"

    # Check if dataset cache already exists
    dataset_cache_exists = os.path.exists(dataset_cache_path)
    if dataset_cache_exists:
        # Dataset already cached, load from disk
        logger.info(
            f"GPU {gpu_ith}: Loading dataset from cached file {dataset_cache_path}"
        )
        dataset = load_from_disk(dataset_cache_path)
        hf_train_args.dataset_kwargs = {"skip_prepare_dataset": True}
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=None if gpu_ith != 0 else dataset,
            dataset_text_field="text",
            max_seq_length=hyper_config.fast_model_args.max_seq_length,
            dataset_num_proc=hyper_config.data.dataset_num_proc,
            args=hf_train_args,
        )
    elif gpu_ith == 0:
        with filelock.FileLock(lock):
            # GPU 0 needs to prepare the dataset
            logger.info(
                f"GPU {gpu_ith}: Preparing dataset and saving to {dataset_cache_path}"
            )
            from HyperSloth.dataset_utils import get_chat_dataset

            ds_train, ds_test = get_chat_dataset(
                tokenizer=tokenizer, **hyper_config.data.model_dump()
            )

            hf_train_args.dataset_kwargs = {"skip_prepare_dataset": False}
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=ds_train,
                eval_dataset=ds_test,
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
            if hyper_config.data.group_by_length:
                from .patching import patch_sampler, select_dataset_by_length

                trainer = patch_sampler(trainer)
                trainer.train_dataset = select_dataset_by_length(
                    trainer.train_dataset,
                    gpu_ith,
                    len(hyper_config.training.gpus),
                    hf_train_args.gradient_accumulation_steps,
                    hf_train_args.per_device_train_batch_size,
                )
            # Save dataset for other GPUs to use
            trainer.train_dataset.save_to_disk(dataset_cache_path)
        if os.path.exists(lock):
            os.remove(lock)
    else:
        # Non-GPU 0 waits for GPU 0 to prepare dataset
        # After the lock is acquired, the dataset should be available
        while not os.path.exists(dataset_cache_path):
            time.sleep(1)
            logger.info(f"GPU {gpu_ith}: Waiting for dataset to be prepared by GPU 0")
        # wait for the lock to be released
        t = time.time()
        while os.path.exists(lock):
            time.sleep(1)
            logger.info(f"GPU {gpu_ith}: Waiting for lock to be released by GPU 0")
            if time.time() - t > 5:
                raise TimeoutError(
                    f"The file is there but the lock is not released {lock}"
                )

        dataset = load_from_disk(dataset_cache_path)
        hf_train_args.dataset_kwargs = {"skip_prepare_dataset": True}
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            dataset_text_field="text",
            max_seq_length=hyper_config.fast_model_args.max_seq_length,
            dataset_num_proc=hyper_config.data.dataset_num_proc,
            args=hf_train_args,
        )
    return trainer
