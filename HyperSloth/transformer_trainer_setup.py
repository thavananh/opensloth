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
    """

    gpu_ith = hyper_config.training.gpus.index(gpu)

    # Initialize model and tokenizer
    model, tokenizer = _initialize_model_and_tokenizer(hyper_config)

    # Build trainer (loads/prepares dataset, sets up SFTTrainer)
    trainer = _create_trainer(tokenizer, hyper_config, hf_train_args, gpu_ith, model)

    # Shard dataset for multi-GPU
    ds = trainer.train_dataset
    global_bz = hf_train_args.per_device_train_batch_size * hf_train_args.gradient_accumulation_steps
    
    ds_shard0 = ds.shard(num_shards=2, index=0, contiguous=True, keep_in_memory=True)
    ds_shard1 = ds.shard(num_shards=2, index=1, contiguous=True, keep_in_memory=True)
    
    
    if gpu_ith == 0:
        batch0 = [len(ds_shard0[i]['input_ids']) for i in range(global_bz)]
        batch1 = [len(ds_shard1[i]['input_ids']) for i in range(global_bz)]   
        print(f"Shard 0: {batch0}")
        print(f"Shard 1: {batch1}")
    
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(hyper_config.training.gpus),
        index=gpu_ith,
        contiguous=True,
        keep_in_memory=True,# this will keep the dataset in memory
    )

    # Optionally train on response-only
    _maybe_train_on_responses_only(trainer, hyper_config)

    # Debug info for the main GPU
    if gpu_ith == 0:
        logger.info(f"Model setup complete for GPU {gpu_ith}")
        from ._debug_dataloader import _debug_dataloader

        _debug_dataloader(trainer)

    return trainer


def _initialize_model_and_tokenizer(hyper_config: HyperConfig):
    """Initialize and optionally set up LoRA for the model."""
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    if not hyper_config.fast_model_args.full_finetuning:
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())
    return model, tokenizer


def _create_trainer(tokenizer, hyper_config, hf_train_args, gpu_ith, model):
    """Load or prepare the dataset and create the SFTTrainer."""
    from trl import SFTTrainer
    from datasets import load_from_disk
    from speedy_utils import identify
    from fastcore.all import Path

    tokenizer_name = identify(str(tokenizer))
    num_gpus = len(hyper_config.training.gpus)
    dataset_cache_path = identify([hyper_config.data.dataset_name_or_path, num_gpus])
    dataset_name = identify(hyper_config.data.model_dump())
    dataset_cache_path = (
        "/dev/shm/dataset_"+ tokenizer_name+"_"+dataset_name
        + ".cache"
    )
    lock = dataset_cache_path + ".lock"
    dataset_cache_exists = os.path.exists(dataset_cache_path)

    # CASE 1: Dataset cache already exists, just load it
    if dataset_cache_exists:
        logger.info(f"GPU {gpu_ith}: Loading dataset from {dataset_cache_path}")
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

    # CASE 2: GPU 0 prepares the dataset and saves for others
    elif gpu_ith == 0:
        with filelock.FileLock(lock):
            logger.info(f"GPU {gpu_ith}: Preparing dataset -> {dataset_cache_path}")
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

            # Adjust dataset for multi-GPU
            max_len_ds = len(hyper_config.training.gpus) * (
                len(trainer.train_dataset) // len(hyper_config.training.gpus)
            )
            trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))


            trainer.train_dataset.save_to_disk(dataset_cache_path)



        if os.path.exists(lock):
            os.remove(lock)

    # CASE 3: Other GPUs wait for GPU 0
    else:
        while not os.path.exists(dataset_cache_path):
            time.sleep(1)
            logger.info(f"GPU {gpu_ith}: Waiting for dataset to be prepared by GPU 0")
        t = time.time()
        while os.path.exists(lock):
            time.sleep(1)
            logger.info(f"GPU {gpu_ith}: Waiting for lock to be released")
            if time.time() - t > 5:
                raise TimeoutError(f"Lock not released: {lock}")

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
    # Save for other GPUs
    return trainer


def _maybe_train_on_responses_only(trainer, hyper_config):
    """Use a specialized approach if 'response_only' loss is desired."""
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
    return trainer
