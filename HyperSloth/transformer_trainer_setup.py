"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

from loguru import logger
from transformers import TrainingArguments

from .app_config import HyperSlothConfig

from unsloth.tokenizer_utils import load_correct_tokenizer,_load_correct_tokenizer


def setup_model_and_training(
    gpu: int,
    hyper_config: HyperSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        args: Configuration arguments
        train_args: Training arguments

    Returns:
        Trainer object configured for multi-GPU training
    """
    from unsloth import FastLanguageModel

    from HyperSloth.dataset_utils import load_sharegpt_dataset

    dataset_fn = lambda tokenizer, test_ratio: load_sharegpt_dataset(
        hyper_config.dataset_file, tokenizer, test_ratio
    )
    gpu_ith = hyper_config.gpus.index(gpu)
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hyper_config.model_name,
        max_seq_length=hyper_config.max_seq_length,
        dtype=None,
        fix_tokenizer=False,
    )

    ds_train, ds_test = dataset_fn(tokenizer, test_ratio=0.1)

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=hyper_config.lora_rank,
        target_modules=hyper_config.target_modules,
        lora_alpha=hyper_config.lora_alpha,
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
    max_len_ds = len(hyper_config.gpus) * (
        len(trainer.train_dataset) // len(hyper_config.gpus)
    )
    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(hyper_config.gpus), index=gpu_ith
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
            instruct_part = hyper_config.instruction_part or "<｜User｜>"
            response_part = hyper_config.response_part or "<｜Assistant｜>"
            
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
