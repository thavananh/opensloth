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
    ds_train, ds_test = get_alpaca(tokenizer, nsplits=len(args.visible_devices), split=gpu_ith, test_ratio=0.1)
    logger.debug(
        f"GPU {args.gpu_index}: Training on {len(args.visible_devices)} samples, testing on {len(ds_test)} samples"
    )

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
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=args.packing,
        args=train_args,
    )

    return trainer
