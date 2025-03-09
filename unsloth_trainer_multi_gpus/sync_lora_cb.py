import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import torch
from transformers import TrainerCallback, TrainerState, DataCollatorForSeq2Seq, TrainingArguments
from datasets import Dataset
from loguru import logger

from unsloth_trainer_multi_gpus.think_chat_template_tokenier_fix import fix_think_chat_template_tokenizer


# from unsloth.chat_templates import train_on_responses_only
# from speedy_utils.all import load_by_ext

# Configuration constants
WEIGHT_SYNC_INTERVAL = 10  # Steps between weight synchronization
WEIGHT_FILE_WAIT_TIMEOUT = 1800  # 30 minutes in seconds
LOCK_FILE_WAIT_TIMEOUT = 60  # 60 seconds


def merge_and_save_weights(source_weight_files: List[str], target_weight_file: str) -> None:
    """
    Merge weights from multiple GPUs and save to a target file.
    
    Args:
        source_weight_files: List of weight files from different GPUs
        target_weight_file: Path where the merged weights will be saved
    
    Raises:
        FileNotFoundError: If source weight file creation times out
        FileExistsError: If lock file doesn't release in time
    """
    weights_by_param = defaultdict(list)

    # First check if all files exist before proceeding
    for weight_file in source_weight_files:
        wait_start = time.time()
        while not os.path.exists(weight_file):
            if time.time() - wait_start > WEIGHT_FILE_WAIT_TIMEOUT:
                logger.warning(f"Timeout waiting for {weight_file}")
                raise FileNotFoundError(f"Timeout waiting for {weight_file}")
            logger.debug(f"Waiting for {weight_file} to be created")
            time.sleep(1)
            
        # Wait for the lock to release
        lock_file = f"{weight_file}.lock"
        wait_start = time.time()
        while os.path.exists(lock_file):
            if time.time() - wait_start > LOCK_FILE_WAIT_TIMEOUT:
                logger.warning(f"Timeout waiting for lock to be released on {weight_file}")
                raise FileExistsError(f"Timeout waiting for lock to be released on {weight_file}")
            logger.debug(f"Waiting for lock to be released on {weight_file}")
            time.sleep(1)
        state_dict = None
        e = None
        for i in range(5):
            try:
                state_dict = torch.load(weight_file)
            except Exception as e:
                time.sleep(1)
        if not state_dict:
            logger.warning(f"Error loading {weight_file}: {e}")
            raise FileNotFoundError(f"Error loading {weight_file}: {e}") 
        
        for param_name, param_value in state_dict.items():
            weights_by_param[param_name].append(param_value)

    # Merge the weights by averaging across GPUs
    merged_weights = {
        param_name: torch.stack(param_values).mean(0)
        for param_name, param_values in weights_by_param.items()
    }

    # Create lock file then save weights
    lock_file = f"{target_weight_file}.lock"
    with open(lock_file, "w") as f:
        f.write("lock")

    # Save the weights
    torch.save(merged_weights, target_weight_file)

    # Remove lock file when done
    os.remove(lock_file)
    logger.debug(f"Saved merged weights to {target_weight_file}")


def wait_for_weights_and_load(model: Any, weight_file_path: str) -> None:
    """
    Wait for target weights file to be fully written and then load it into the model.
    
    Args:
        model: The model to load weights into
        weight_file_path: Path to the weights file to load
    """
    lock_file = f"{weight_file_path}.lock"

    # Wait for file to be created
    wait_start = time.time()
    while not os.path.exists(weight_file_path):
        if time.time() - wait_start > WEIGHT_FILE_WAIT_TIMEOUT:
            logger.warning(f"Timeout waiting for {weight_file_path}")
            return
        logger.debug(f"Waiting for {weight_file_path} to be created")
        time.sleep(1)

    # Wait for lock file to be removed (indicating write is complete)
    wait_start = time.time()
    while os.path.exists(lock_file):
        if time.time() - wait_start > LOCK_FILE_WAIT_TIMEOUT:
            logger.warning(f"Timeout waiting for lock to be released on {weight_file_path}")
            return
        logger.debug(f"Waiting for lock to be released on {weight_file_path}")
        time.sleep(1)
        
    logger.debug(f"Loading weights from {weight_file_path} to {model.device}")
    weights = torch.load(weight_file_path, map_location=f"cuda:{model.device.index}")
    result = model.load_state_dict(weights, strict=False)
    
    num_loaded = len(result[0])
    num_missing = len(result[1])
    logger.debug(f"Loaded {num_loaded} weights, {num_missing} missing")

class WeightSyncCallback(TrainerCallback):
    """
    Callback to synchronize weights across multiple GPUs during training.
    
    Periodically saves weights from each GPU, merges them on GPU 0,
    and loads the merged weights back to all GPUs.
    """
    
    def __init__(self, gpu_id: int, all_gpu_ids: List[int], sync_interval: int = WEIGHT_SYNC_INTERVAL):
        """
        Initialize the weight synchronization callback.
        
        Args:
            gpu_id: Current GPU ID
            all_gpu_ids: List of all GPU IDs participating in training
            sync_interval: Number of steps between synchronizations
        """
        self.gpu_id = gpu_id
        self.all_gpu_ids = all_gpu_ids
        self.sync_interval = sync_interval
        
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: Any, **kwargs):
        """
        Called after each optimizer step to potentially synchronize weights.
        
        Args:
            args: Training arguments
            state: Current trainer state
            control: Control object
        """
        output_dir = args.output_dir
        model = state.model
        current_step = state.global_step
        
        # Path for this GPU's weights
        this_gpu_weights_path = os.path.join(
            output_dir, f"checkpoint-{current_step}.gpu.{self.gpu_id}.pt"
        )
        
        # Path for merged weights
        merged_weights_path = os.path.join(output_dir, f"checkpoint-{current_step}.pt")

        # Synchronize only at specified intervals
        if current_step % self.sync_interval == 0:
            logger.debug(f"Step {current_step}: Saving weights for GPU {self.gpu_id}")
            
            # Extract trainable parameters
            state_dict = model.state_dict()
            trainable_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}

            # Create lock file before saving weights
            lock_file = f"{this_gpu_weights_path}.lock"
            with open(lock_file, "w") as f:
                f.write("lock")
            logger.debug(f"Created lock file {lock_file}")

            # Save the weights
            torch.save(trainable_state_dict, this_gpu_weights_path)
            logger.debug(f"Saved weights to {this_gpu_weights_path}")
            os.remove(lock_file)
            logger.debug(f"Removed lock file {lock_file}")
            
            # If this is GPU 0, merge weights from all GPUs
            if self.gpu_id == self.all_gpu_ids[0]:
                all_gpu_weights_paths = [
                    os.path.join(output_dir, f"checkpoint-{current_step}.gpu.{i}.pt")
                    for i in self.all_gpu_ids
                ]
                logger.debug(f"GPU 0: Merging weights from all GPUs: {all_gpu_weights_paths}")
                merge_and_save_weights(all_gpu_weights_paths, merged_weights_path)

            # Wait for the merged weights file and load it
            logger.debug(
                f"GPU {self.gpu_id}: Waiting for merged weights {merged_weights_path}"
            )
            wait_for_weights_and_load(state.model, merged_weights_path)
            logger.debug(f"GPU {self.gpu_id}: Loaded merged weights from {merged_weights_path}")
            
            # Clean up individual GPU weight files after all GPUs have loaded the merged weights
            # Only GPU 0 needs to clean up to avoid race conditions
            if self.gpu_id == self.all_gpu_ids[0]:
                # Wait a bit to ensure all GPUs have loaded the merged weights
                from fastcore.all import threaded
                @threaded
                def f_clean():
                    logger.debug(f"Starting cleanup for step {current_step} sleep 30s for all GPUs to load merged weights safely")
                    time.sleep(30)
                    for gpu_id in self.all_gpu_ids:
                        gpu_weight_path = os.path.join(
                            output_dir, f"checkpoint-{current_step}.gpu.{gpu_id}.pt"
                        )
                        if os.path.exists(gpu_weight_path):
                            try:
                                os.remove(gpu_weight_path)
                                logger.debug(f"Cleaned up {gpu_weight_path}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up {gpu_weight_path}: {e}")
                    # remove the merged weights file
                    if os.path.exists(merged_weights_path):
                        try:
                            os.remove(merged_weights_path)
                            logger.debug(f"Cleaned up {merged_weights_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up {merged_weights_path}: {e}")
                    logger.debug(f"Cleanup completed for step {current_step}")
                f_clean()