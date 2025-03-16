"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

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

    gpu_ith = hyper_config.training.gpus[gpu]

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

    # import numpy as np
    # from fastcore.all import chunked

    # def bz_same_size(dataset, device_effective_bz):
    #     ls = [len(x) for x in dataset["input_ids"]]  # length array

    #     # 1. Get sort indices by length
    #     ids = np.argsort(ls)

    #     # 2. Chunk those sorted indices
    #     chunks = list(chunked(ids, device_effective_bz))

    #     # 3. Shuffle the chunks
    #     # np.random.shuffle(chunks)

    #     # 4. Flatten the shuffled chunks
    #     flat_ids = [i for chunk in chunks for i in chunk]

    #     return dataset.select(flat_ids)

    # trainer.train_dataset = bz_same_size(
    #     trainer.train_dataset, hf_train_args.per_device_train_batch_size
    # )

    # Adjust dataset for multi-GPU training
    max_len_ds = len(hyper_config.training.gpus) * (
        len(trainer.train_dataset) // len(hyper_config.training.gpus)
    )
    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(hyper_config.training.gpus), index=gpu_ith
    )

    # Handle specific training loss type
    if hyper_config.training.loss_type == "target_only":
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

    _debug_dataloader(trainer)
    return trainer


def _debug_dataloader(trainer):
    """
    Debug function to log the first batch of the training dataloader.
    """
    dl = trainer.get_train_dataloader()
    g = iter(dl)
    batch = next(g)
    input_ids = batch["input_ids"]
    from copy import deepcopy

    tokenizer = deepcopy(trainer.tokenizer)
    text = tokenizer.decode(input_ids.cpu()[0])
    labels = batch["labels"]
    labels[labels < 0] = 0  # Fill < 0 with 0
    label_text = tokenizer.decode(labels.cpu()[0])
    logger.info(f"=====\nI: {text}\n-----\nL: {label_text}\n=====")
    assert (batch["labels"] > 0).sum() != 0, "NO LABELS???"
