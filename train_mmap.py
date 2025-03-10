import argparse
import os
from functools import partial
import filelock
from loguru import logger
from speedy_utils.all import multi_thread
import time

multi_thread = partial(multi_thread, report=False, verbose=False)

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import numpy as np



class CustomCallback(TrainerCallback):
    def __init__(self, model, grad_dir, gpu_index, visible_devices):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu_index = gpu_index
        self.visible_devices = visible_devices
        self.is_main = gpu_index == visible_devices[0]

        from notveryslow.mmap_gradient_sync import MmapGradientSync

        self.grad_sync = MmapGradientSync(
            model,
            grad_dir,
            gpu_index,
            visible_devices,
        )
        os.makedirs(self.grad_dir, exist_ok=True)
        self.loss_file = np.memmap(
            os.path.join(self.grad_dir, "loss.mmap"),
            dtype="float32",
            mode="w+",
            shape=(len(self.visible_devices),),
        )

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

        t = time.time()
        logger.info("After optimizer step")
        self.grad_sync.zero_mmaps()

    
    def on_log(self, args, state, control, **kwargs):
        if 'loss' not in state.log_history[-1]:
            self.loss_file[self.gpu_index] = np.float32(state.log_history[-1]["loss"])
            t = time.time()
            if self.is_main:
                # if main gpu, then read all the losses
                # wait for all the losses to be written
                while any(self.loss_file == 0):
                    time.sleep(0.01)
                losses = self.loss_file[:]
                state.log_history[-1]["mean_loss"] = np.mean(losses)
                logger.info(f"Mean loss: {state.log_history[-1]['mean_loss']}")
                # if all losses are not zero, then reset all the losses
                if np.all(losses != 0):
                    self.loss_file[:] = 0
            else:
                # if not main gpu, then wait for the main gpu to reset the losses
                while True:
                    losses = self.loss_file[:]
                    if np.all(losses == 0):
                        break
                    time.sleep(0.01)
            t = time.time() - t
    


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

    from notveryslow.unsloth_trainer_setup import setup_model_and_training

    all_gpus = args.visible_devices
    args.num_gpus = len(all_gpus)  # Calculate the number of GPUs
    args.is_main = args.gpu_index == args.visible_devices[0]

    train_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
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

    # grad_sync = MmapGradientSync(

    # )
    grad_sync_cb = CustomCallback(
        model=trainer.model,
        grad_dir="./grads",
        gpu_index=args.gpu_index,
        visible_devices=all_gpus,
    )
    trainer.add_callback(grad_sync_cb)

    # Then run training
    trainer.train()


if __name__ == "__main__":
    main()
