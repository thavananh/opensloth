"""
Multi-GPU training script for Unsloth models.
Distributes training across specified GPUs with weight synchronization.
"""

import argparse
import os
from typing import List, Tuple

from transformers import (
    TrainingArguments,
)


def parse_arguments() -> Tuple[int, List[int]]:
    """
    Parse command line arguments for GPU configuration.

    Returns:
        Tuple containing current GPU ID and list of all GPU IDs
    """
    parser = argparse.ArgumentParser(
        description="Distributed training for Unsloth models"
    )
    parser.add_argument(
        "gpu_index", type=int, help="Index of current GPU in the GPU list"
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        default="0",
        help="Comma separated list of all GPUs to use",
    )
    parser.add_argument(
        "--weight_sync_every_update_steps",
        type=int,
        default=1,
        help="Steps between weight synchronization",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", '-ac',
        type=int,
        default=64,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Use packing for weight synchronization",
    )
    parser.add_argument(
        "--weight-file-wait-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for waiting for weight file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        help="Model name to use for training",
    )
    args = parser.parse_args()

    all_gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(",")]
    current_gpu_id = all_gpu_ids[int(args.gpu_index)]

    return current_gpu_id, all_gpu_ids, args


def main():
    """Main execution function for the training script."""
    current_gpu_id, all_gpu_ids, args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_gpu_id)
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth_trainer_multi_gpus.training_utils import setup_model_and_training

    # Setup and start training
    do_eval = current_gpu_id == all_gpu_ids[0]
    train_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        eval_strategy="epoch" if do_eval else "no",
        warmup_steps=5,
        do_eval=do_eval,
        num_train_epochs=5,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"model_training_outputs/{current_gpu_id}",
        save_total_limit=2,
        save_steps=args.weight_sync_every_update_steps,
        report_to="tensorboard",
    )
    trainer = setup_model_and_training(
        current_gpu_id,
        all_gpu_ids,
        file="./data/cod_6k5.json",
        packing=args.packing,
        args=train_args,
        model_name=args.model_name,
        weight_sync_every_update_steps=args.weight_sync_every_update_steps,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
