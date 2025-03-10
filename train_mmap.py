import argparse
import os
from functools import partial

from loguru import logger

# Hypothetical multi-threading utility from speedy
from speedy_utils.all import multi_thread

# Disable "report" and "verbose" in multi_thread calls
multi_thread = partial(multi_thread, report=False, verbose=False)

# Transformers / Trainer imports
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class CustomCallback(TrainerCallback):
    def __init__(self, model, grad_sync):
        self.model = model
        self.grad_sync = grad_sync

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called before optimizer step.
        """
        logger.info("Before optimizer step")
        self.grad_sync.accumulate_local_grad(self.model)
        self.grad_sync.read_final_grad_into_model(self.model, average=True)

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called after optimizer step.
        """
        logger.info("After optimizer step")
        self.grad_sync.zero_mmaps()


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
    return parser.parse_args()


def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    from notveryslow.mmap_gradient_sync import MmapGradientSync
    from notveryslow.unsloth_trainer_setup import setup_model_and_training
    all_gpus = args.visible_devices
    args.num_gpus = len(all_gpus)  # Calculate the number of GPUs
    args.is_main = args.gpu_index == args.visible_devices[0]

    train_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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
        output_dir="model_training_outputs/debug",
        save_total_limit=2,
        save_steps=1000,
        report_to="tensorboard",
    )

    # Example: your custom trainer-setup function
    trainer = setup_model_and_training(
        args=args,
        train_args=train_args,
    )

    grad_sync = MmapGradientSync(
        model=trainer.model,
        grad_dir="./grads",
        gpu_index=args.gpu_index,
        visible_devices=all_gpus,
    )
    grad_sync_cb = CustomCallback(model=trainer.model, grad_sync=grad_sync)
    trainer.add_callback(grad_sync_cb)

    # Then run training
    trainer.train()


if __name__ == "__main__":
    main()
