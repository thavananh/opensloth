import argparse
import os
import time
from typing import List, Tuple
import numpy as np
import torch
import filelock
from loguru import logger

# Hypothetical multi-threading utility from speedy
from speedy_utils.all import multi_thread
from functools import partial

# Disable "report" and "verbose" in multi_thread calls
multi_thread = partial(multi_thread, report=False, verbose=False)

# Transformers / Trainer imports
from transformers import TrainingArguments
from unsloth_trainer_multi_gpus.training_utils import setup_model_and_training
from unsloth_trainer_multi_gpus.mmap_gradient_sync import MmapGradientSync


def parse_args():
    """
    Example usage:
      python train_mmap.py 0 --gpus 0,1
      => This process is for GPU index=0, all_gpus=[0,1]
         => 'world_size' = 2 in that scenario
    """
    parser = argparse.ArgumentParser(
        description="Train model with memmap gradient sync"
    )
    parser.add_argument(
        "gpu_index",
        type=int,
        help="Index of current GPU in the GPU list (e.g. 0, 1, etc.)",
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[0],
        help="Comma-separated list of all GPUs to use (default: 0). Example: 0,1,2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # This ensures we only see the single GPU with index = args.gpu_index
    # (assuming you want each process to exclusively use one GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # For clarity: local GPU index among the job
    gpu_id = args.gpu_index
    all_gpus = args.gpus

    # Whether this is the "main" GPU or not (may decide logging/eval)
    is_main = gpu_id == all_gpus[0]

    train_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        # E.g., only the first GPU does evaluation in a naive multi-process setup
        eval_strategy="steps" if is_main else "no",
        eval_steps=100,
        warmup_steps=5,
        do_eval=True,
        num_train_epochs=5,
        learning_rate=1e-4,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="model_training_outputs/debug",
        save_total_limit=2,
        save_steps=1000,
        report_to="tensorboard",
    )

    # Example: your custom trainer-setup function
    trainer = setup_model_and_training(
        gpu_id=gpu_id,
        all_gpu_ids=all_gpus,
        file="./data/cod_6k5.json",
        packing=True,
        args=train_args,
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    )

    # Attach the MmapGradientSync to your trainer
    trainer.grad_sync = MmapGradientSync(
        model=trainer.model,
        grad_dir="./grads",
        gpu_id=gpu_id,
        gpus=all_gpus,
    )

    # Then run training
    trainer.train()


if __name__ == "__main__":
    main()
