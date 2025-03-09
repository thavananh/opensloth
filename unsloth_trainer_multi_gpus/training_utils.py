"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import torch
from transformers import (
    TrainerCallback,
    TrainerState,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from datasets import Dataset
from loguru import logger

from unsloth_trainer_multi_gpus.sync_lora_cb import WeightSyncCallback
from unsloth_trainer_multi_gpus.think_chat_template_tokenier_fix import (
    fix_think_chat_template_tokenizer,
)


# from unsloth.chat_templates import train_on_responses_only
# from speedy_utils.all import load_by_ext

# Configuration constants
WEIGHT_SYNC_INTERVAL = 1  # Steps between weight synchronization
ACCUMULATION_STEPS = 16
PACKING = False
WEIGHT_FILE_WAIT_TIMEOUT = 1800  # 30 minutes in seconds
LOCK_FILE_WAIT_TIMEOUT = 60  # 60 seconds


def setup_model_and_training(
    gpu_id: int, all_gpu_ids: List[int], file="./data/cod_6k5.json"
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        gpu_id: Current GPU ID
        all_gpu_ids: List of all GPU IDs participating in training

    Returns:
        Tuple containing (model, tokenizer, dataset, trainer)
    """
    # Initialize model and tokenizer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from speedy_utils.all import load_by_ext

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        max_seq_length=16_000,
        dtype=None,
    )

    # Load and shard dataset for this GPU
    dataset_raw = load_by_ext(file)
    gpu_index = all_gpu_ids.index(gpu_id)
    dataset_raw = dataset_raw

    tokenizer = fix_think_chat_template_tokenizer(tokenizer)

    def format_chat_template(row: Dict[str, Any]) -> Dict[str, Any]:
        row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row

    dataset_raw = [format_chat_template(row) for row in dataset_raw]

    def split_item(
        items: List[Any], test_size: float = 0.05, train_fold: int = 5, seed: int = 42
    ) -> Tuple[List[Tuple[List[int], List[int]]], List[Any]]:
        """
        Split items into training and test sets, then further split the training set into folds.

        Args:
            items: List of items to split
            test_size: Proportion of the dataset to include in the test split
            train_fold: Number of folds for KFold cross-validation
            seed: Random seed for reproducibility

        Returns:
            A tuple containing the list of train/validation folds and the test set
        """
        # shufflt items
        import random

        r = random.Random(seed)
        r.shuffle(items)
        # Split into train and test sets
        test_size = int(len(items) * test_size)
        train = items[:-test_size]
        test = items[-test_size:]
        # Split into folds
        folds = [train[i::train_fold] for i in range(train_fold)]
        return folds, test

    trains, test = split_item(
        dataset_raw, test_size=0.05, train_fold=len(all_gpu_ids), seed=42
    )
    ds_train = Dataset.from_list(trains[gpu_index])
    ds_test = Dataset.from_list(test)
    logger.debug(
        f"GPU {gpu_id}: Training on {len(ds_train)} samples, testing on {len(ds_test)} samples"
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
    do_eval = gpu_index == 0
    logging_steps = 1 if gpu_index == 0 else int(1e12)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test if gpu_index == 0 else None,
        dataset_text_field="text",
        max_seq_length=16_000,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=PACKING,
        callbacks=[
            WeightSyncCallback(gpu_id, all_gpu_ids, sync_interval=WEIGHT_SYNC_INTERVAL)
        ],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=ACCUMULATION_STEPS,
            logging_steps=logging_steps,
            eval_strategy="steps" if do_eval else "no",
            eval_steps=0.2 if do_eval else 0,
            warmup_steps=5,
            do_eval=do_eval,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="model_training_outputs",
            report_to="none",
        ),
    )

    # Configure to train on responses only
    instruct_part = "<｜begin▁of▁sentence｜><｜User｜>"
    response_part = "<｜Assistant｜>"
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruct_part,
        response_part=response_part,
    )

    return trainer
