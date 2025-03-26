"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os
import random
import time
from loguru import logger
import filelock
from speedy_utils import dump_json_or_pickle


from .hypersloth_config import HyperConfig, TrainingArgsConfig


def setup_model_and_training(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.
    """

    gpu_ith = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    if not gpu_ith == 0:
        # disable reporting for all GPUs except the first one
        hf_train_args.report_to = "none"
        # disable evaluation for all GPUs except the first one
        hf_train_args.do_eval = False
    # Unsloth uses monkey patching thus it might have race conditions so we need to try until it works
    model, tokenizer = _initialize_model_and_tokenizer(hyper_config)
    trainer = _create_trainer(tokenizer, hyper_config, hf_train_args, gpu_ith, model)

    # Debug info for the main GPU
    if gpu_ith == 0:
        logger.info(f"Model setup complete for GPU {gpu_ith}")
        try:
            from ._debug_dataloader import _debug_dataloader

            _debug_dataloader(trainer)
            _debug_training_lengths(hf_train_args, gpu_ith, trainer)
        except:
            pass

    return trainer, model, tokenizer


def _debug_training_lengths(hf_train_args, gpu_ith, trainer):
    dataloader = trainer.get_train_dataloader()
    generator = iter(dataloader)
    with open(f"lengths_{gpu_ith}.txt", "w") as f:
        num_grad_accum = hf_train_args.gradient_accumulation_steps
        txt = ""
        for i in range(10):
            len_in_one_update = []
            for i in range(num_grad_accum):
                batch = next(generator)
                s1, s2 = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
                len_in_one_update.append(s2)
            total_len = sum([shape for shape in len_in_one_update])
            txt += (
                "|".join([str(x) for x in len_in_one_update])
                + "Total len:{}".format(total_len)
                + "\n"
            )
        f.write(txt)


def _initialize_model_and_tokenizer(hyper_config: HyperConfig):
    """Initialize and optionally set up LoRA for the model."""
    from unsloth import FastModel

    # ====== Patching the compiler location to avoid race conditions as it is shared between GPUs
    gpu_idx = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    from unsloth_zoo import compiler

    compiler.UNSLOTH_COMPILE_LOCATION = ".cache/{}_{}".format(
        compiler.UNSLOTH_COMPILE_LOCATION, gpu_idx
    )
    logger.info(f"Using compiler location: {compiler.UNSLOTH_COMPILE_LOCATION}")

    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    if not hyper_config.fast_model_args.full_finetuning:
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())
    return model, tokenizer


def _create_trainer(
    tokenizer,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
    gpu_ith: int,
    model: any,
):
    """Load or prepare the dataset and create the SFTTrainer."""

    from speedy_utils import identify

    tokenizer_name = identify(str(tokenizer))
    dataset_name = identify(hyper_config.data.model_dump())
    dataset_cache_name = "dataset_" + tokenizer_name + "_" + dataset_name + ".cache"
    dataset_cache_path = os.path.join(".cache/", dataset_cache_name)

    lock = dataset_cache_path + ".lock"
    dataset_cache_exists = os.path.exists(dataset_cache_path)

    # CASE 1: Dataset cache already exists, just load it
    trainer = get_trainer(
        tokenizer,
        hyper_config,
        hf_train_args,
        gpu_ith,
        model,
        dataset_cache_path,
        lock,
        dataset_cache_exists,
    )


    from HyperSloth.patching import patch_hf_trainer
    
    
    patch_hf_trainer()
    return trainer






def get_trainer(
    tokenizer,
    hyper_config,
    hf_train_args,
    gpu_ith,
    model,
    dataset_cache_path,
    lock,
    dataset_cache_exists,
):
    """
    Returns an SFTTrainer instance. If a cached dataset exists, load from disk.
    If not, GPU 0 will create and save it, and other GPUs will wait for GPU 0
    to finish.
    """
    import os
    import time
    import filelock
    import logging
    from trl import SFTTrainer
    from datasets import load_from_disk

    def _create_trainer(train_dataset, eval_dataset=None, skip_prepare=True):
        """Helper to build an SFTTrainer from given train/eval datasets."""
        # We use a dataset_kwargs override so that, once the dataset is prepared,
        # it won't attempt the same "prepare" logic again.
        hf_train_args.dataset_kwargs = {"skip_prepare_dataset": skip_prepare}
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=hyper_config.fast_model_args.max_seq_length,
            dataset_num_proc=hyper_config.data.dataset_num_proc,
            args=hf_train_args,
        )

    # ---------------------------
    # CASE 1: Cached dataset exists
    # ---------------------------
    if dataset_cache_exists:
        logger.info(f"GPU {gpu_ith}: Loading dataset from {dataset_cache_path}")
        dataset = load_from_disk(dataset_cache_path)
        trainer = _create_trainer(dataset, eval_dataset=None, skip_prepare=True)
        return trainer

    # ---------------------------
    # CASE 2: GPU 0 prepares dataset
    # ---------------------------
    elif gpu_ith == 0:
        with filelock.FileLock(lock):
            logger.info(f"GPU {gpu_ith}: Preparing dataset -> {dataset_cache_path}")

            from HyperSloth.dataset_utils import get_chat_dataset
            ds_train, ds_test = get_chat_dataset(
                tokenizer=tokenizer, **hyper_config.data.model_dump()
            )
            trainer = _create_trainer(ds_train, eval_dataset=ds_test, skip_prepare=False)

            # Optionally patch trainer or handle "response-only" logic
            _maybe_train_on_responses_only(trainer, hyper_config)

            from .dynamic_batching import encode_dynamic_batching_dataset
            trainer.train_dataset = encode_dynamic_batching_dataset(
                trainer.train_dataset,
                num_gpus=len(hyper_config.training.gpus),
                max_len_allow=hyper_config.fast_model_args.max_seq_length,
            )

            trainer.train_dataset.save_to_disk(dataset_cache_path)

        # Release the lock file
        if os.path.exists(lock):
            os.remove(lock)

        return trainer

    # ---------------------------
    # CASE 3: Other GPUs wait for GPU 0
    # ---------------------------
    else:
        logger.info(f"GPU {gpu_ith}: Waiting for dataset to be prepared by GPU 0")
        while not os.path.exists(dataset_cache_path):
            time.sleep(1)

        start_t = time.time()
        while os.path.exists(lock):
            time.sleep(1)
            logger.info(f"GPU {gpu_ith}: Waiting for lock to be released")
            if time.time() - start_t > 5:
                raise TimeoutError(f"Lock not released: {lock}")

        logger.info(f"GPU {gpu_ith}: Loading dataset from {dataset_cache_path}")
        dataset = load_from_disk(dataset_cache_path)
        trainer = _create_trainer(dataset, eval_dataset=None, skip_prepare=True)
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
