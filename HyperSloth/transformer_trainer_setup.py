"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

from loguru import logger

from unsloth import FastLanguageModel
from .app_config import HyperSlothConfig
from transformers import TrainingArguments


def setup_model_and_training(
    hyper_config: HyperSlothConfig,
    hf_train_args: TrainingArguments,
    dataset_fn: callable, # return train_ds, test_ds
    gpu_index: int, # the raw gpu index (0, 1, 2, ...)
    visible_devices: list[int], # the list to be used in CUDA_VISIBLE_DEVICES
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        args: Configuration arguments
        train_args: Training arguments

    Returns:
        Trainer object configured for multi-GPU training
    """
    gpu_ith = visible_devices.index(gpu_index)
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hyper_config.model_name,
        max_seq_length=hyper_config.max_seq_length,
        dtype=None,
    )

    ds_train, ds_test = dataset_fn(tokenizer, test_ratio=0.1)

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    from trl import SFTTrainer

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test if gpu_ith == 0 else None,
        dataset_text_field="text",
        max_seq_length=hyper_config.max_seq_length,
        dataset_num_proc=2,
        packing=hyper_config.packing,
        args=hf_train_args,
    )
    max_len_ds = len(visible_devices) * (
        len(trainer.train_dataset) // len(visible_devices)
    )
    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(visible_devices), index=gpu_ith
    )

    if hyper_config.loss_type == "target_only":
        from unsloth.chat_templates import train_on_responses_only

        if "<|im_start|>" in tokenizer.chat_template:
            instruct_part = "<|im_start|>system\n"
            response_part = "<|im_start|>assistant\n"
        else:
            assert (
                "<｜Assistant｜>" in tokenizer.chat_template
            ), f'{tokenizer} does not have "<｜Assistant｜>" or "<|im_start|>"'
            instruct_part = "<｜begin▁of▁sentence｜><｜User｜>"
            response_part = "<｜Assistant｜>"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruct_part,
            response_part=response_part,
        )

    _debug_dataloader(trainer)
    #
    return trainer


def _debug_dataloader(trainer):
    dl = trainer.get_train_dataloader()
    g = iter(dl)
    batch = next(g)
    input_ids = batch["input_ids"]
    from copy import deepcopy

    tokenizer = deepcopy(trainer.tokenizer)
    text = tokenizer.decode(input_ids.cpu()[0])
    labels = batch["labels"]
    # fill < 0 with 0
    labels[labels < 0] = 0
    label_text = tokenizer.decode(labels.cpu()[0])
    logger.info(f"=====\nI: {text}\n-----\nL: {label_text}\n=====")
    assert (batch["labels"] > 0).sum() != 0, "NO LABELS???"
