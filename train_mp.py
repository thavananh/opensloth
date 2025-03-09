"""
Multi-GPU training script for Unsloth models.
Distributes training across specified GPUs with weight synchronization.
"""

import os
import argparse
import atexit
from typing import List, Tuple
from multiprocessing import Manager, Process
from speedy_utils import setup_logger


def parse_arguments() -> Tuple[List[int], str]:
    """
    Parse command line arguments for GPU configuration.

    Returns:
        Tuple containing list of all GPU IDs and the file path
    """
    parser = argparse.ArgumentParser(
        description="Distributed training for Unsloth models"
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        default="0,2",
        help="Comma separated list of all GPUs to use",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="./data/cod_6k5.json",
        help="Path to the training data file",
    )
    args = parser.parse_args()
    all_gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(",")]
    return all_gpu_ids, args.file


def run(
    current_gpu_id: int, all_gpu_ids: List[int], file_path: str, shared_memory: dict
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(current_gpu_id)
    from unsloth_trainer_multi_gpus.training_utils import setup_model_and_training
    from loguru import logger
    # logger.remove()
    # logger should have [GPU ID] prefix
    # add terminal logger
    logger.add(lambda msg: f"[{current_gpu_id}] {msg}")
    # by gpu_id file
    logger.add(f"./logs/{current_gpu_id}.log")
    trainer = setup_model_and_training(
        current_gpu_id, all_gpu_ids, file=file_path, shared_memory=shared_memory
    )
    trainer.train()


def main():
    """Main execution function for the training script."""
    all_gpu_ids, file_path = parse_arguments()

    # Create a shared dictionary for state_dicts
    manager = Manager()
    shared_memory = manager.dict()

    # Register cleanup function to ensure shared memory is released

    # Start training processes
    processes = []
    for gpu_id in all_gpu_ids:
        p = Process(target=run, args=(gpu_id, all_gpu_ids, file_path, shared_memory))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
