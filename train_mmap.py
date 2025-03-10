import argparse
import os
from functools import partial
from loguru import logger
from notveryslow.mmap_gradient_sync import MmapGradSyncCallback
from speedy_utils.all import multi_thread

multi_thread = partial(multi_thread, report=False, verbose=False)



from transformers import (
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for multi-GPU setup.")
    parser.add_argument(
        "--gpu_index", type=int, default=0, help="Index of the GPU to use."
    )
    parser.add_argument(
        "--visible_devices",
        type=int,
        nargs="+",
        default=[0],
        help="List of visible GPU devices.",
    )
    parser.add_argument(
        "--file", type=str, default="./data/cod_6k5.json", help="Path to the data file."
    )
    parser.add_argument("--packing", action="store_true", help="Enable packing.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.05, help="Ratio of the test set."
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=16,
        help="Number of gradient accumulation steps.",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # from notveryslow.unsloth_trainer_setup import setup_model_and_training
    from notveryslow.transformer_trainer_setup import setup_model_and_training

    all_gpus = args.visible_devices
    args.is_main = args.gpu_index == args.visible_devices[0]

    train_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        eval_strategy="steps" if args.is_main else "no",
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
        output_dir=f"model_training_outputs/debug/{args.gpu_index}",
        save_total_limit=2,
        # save_steps=10,
        # max_steps=30,
        report_to="tensorboard",
    )

    # Example: your custom trainer-setup function
    trainer = setup_model_and_training(
        args=args,
        train_args=train_args,
    )

    grad_sync_cb = MmapGradSyncCallback(
        model=trainer.model,
        grad_dir="./grads",
        gpu_index=args.gpu_index,
        visible_devices=all_gpus,
    )
    if len(args.visible_devices) > 1:
        logger.info(f"Using gradient sync callback for GPU {args.gpu_index}")
        trainer.add_callback(grad_sync_cb)

    # Then run training
    trainer.train()


if __name__ == "__main__":
    main()
