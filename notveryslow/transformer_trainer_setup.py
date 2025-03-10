"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

from typing import Dict, List, Tuple, Any
import random
from transformers import (
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from loguru import logger

from notveryslow.think_chat_template_tokenier_fix import (
    fix_think_chat_template_tokenizer,
)
from speedy_utils.all import load_by_ext
from unsloth import FastLanguageModel


def setup_model_and_training(args, train_args):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        args: Configuration arguments
        train_args: Training arguments

    Returns:
        Trainer object configured for multi-GPU training
    """
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
    )
    gpu_ith = args.visible_devices.index(args.gpu_index)
    from .dataset_utils import get_alpaca

    ds_train, ds_test = get_alpaca(tokenizer, test_ratio=0.1)
    

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
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=args.packing,
        args=train_args,
    )
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(args.visible_devices), index=gpu_ith
    )

    if args.loss_type == "target_only":
        from unsloth.chat_templates import train_on_responses_only
        if '<|im_start|>' in tokenizer.chat_template:
            instruct_part = "<|im_start|>"
            response_part = "<|im_start|>assistant\n"
        else:
            assert "<｜Assistant｜>" in tokenizer.chat_template, f'{tokenizer} does not have "<｜Assistant｜>" or "<|im_start|>"'
            instruct_part = "<｜begin▁of▁sentence｜><｜User｜>"
            response_part = "<｜Assistant｜>"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruct_part,
            response_part=response_part,
        )
    
    _debug_dataloader(trainer)
    
    return trainer

def _debug_dataloader(trainer):
    dl = trainer.get_train_dataloader()
    g = iter(dl)
    batch = next(g)
    input_ids = batch["input_ids"]
    text = trainer.tokenizer.decode(input_ids.cpu()[0])
    labels = batch["labels"]
    # fill < 0 with 0
    labels[labels < 0] = 0
    label_text = trainer.tokenizer.decode(labels.cpu()[0])
    print('=====')
    print(f'I: {text}')
    print('-----')
    print(f'L: {label_text}')
    print('=====')
    assert (batch["labels"] > 0).sum() != 0, "NO LABELS???"
