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


def load_dataset(file, tokenizer, test_ratio=0.05, num_gpus=1):
    # Load and shard dataset for this GPU
    dataset_raw = load_by_ext(file)

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
        dataset_raw, test_size=test_ratio, train_fold=num_gpus, seed=42
    )
    return trains, test


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
        max_seq_length=16_000,
        dtype=None,
    )
    trains, test = load_dataset(
        args.file, tokenizer, test_ratio=args.test_ratio, num_gpus=args.num_gpus
    )
    gpu_ith = args.visible_devices.index(args.gpu_index)
    ds_train = Dataset.from_list(trains[gpu_ith])
    ds_test = Dataset.from_list(test)
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
        max_seq_length=16_000,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=args.packing,
        args=train_args,
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
