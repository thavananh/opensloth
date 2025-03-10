import os
import time
from collections import defaultdict
from typing import Any, List

import torch
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import TrainerCallback, TrainerState, TrainingArguments

# Configuration constants
WEIGHT_SYNC_INTERVAL = 10  # Steps between weight synchronization
WEIGHT_FILE_WAIT_TIMEOUT = 1800  # 30 minutes in seconds


def torch_load(path):
    lock_file = f"{path}.lock"
    while os.path.exists(lock_file):
        time.sleep(1)
    for _ in range(10):
        try:
            if path.endswith(".pt"):
                return torch.load(path)
            else:
                return load_file(path, device="cpu")
        except Exception as e:
            # logger.warning(f"Error loading file {path}: {e}")
            time.sleep(1)
    logger.error(f"Failed to load file {path}")
    raise RuntimeError(f"Failed to load file {path}")


def torch_save(obj, path):
    lock_file = f"{path}.lock"
    # Create a lock file
    with open(lock_file, "w") as f:
        f.write("lock")
    try:
        if path.endswith(".pt"):
            torch.save(obj, path)
        else:
            save_file(obj, path)
    finally:
        # Remove the lock file
        os.remove(lock_file)


def merge_optim(paths, output_path):
    """
    Merge optimizer states from multiple checkpoint files.

    Args:
        paths (List[str]): List of file paths for optimizer state files.
        output_path (str): Where to save the merged optimizer state.
    """
    merged_state = dict()
    merged_param_groups = None

    for i, optimizer_path in enumerate(paths):
        while not os.path.exists(optimizer_path):
            time.sleep(1)
            logger.debug(f"Waiting for file {optimizer_path} to be available")
        _optimizer_state = torch_load(optimizer_path)
        # Grab param_groups from the first file (assuming identical on each GPU).
        if i == 0:
            merged_param_groups = _optimizer_state["param_groups"]

        # Merge the 'state' dictionaries
        for param_id, state_dict in _optimizer_state["state"].items():
            if not param_id in merged_state:
                merged_state[param_id] = dict()

            for key, value in state_dict.items():
                # If not a tensor, simply store (assumed to be identical across GPUs)
                if not isinstance(value, torch.Tensor):
                    merged_state[param_id][key] = value
                else:
                    if key not in merged_state[param_id]:
                        merged_state[param_id][key] = []
                    merged_state[param_id][key].append(value)

    # Process the lists of tensors: if floating point, average them; else, choose first.
    for param_id, inner in merged_state.items():
        for key, value in inner.items():
            if isinstance(value, list):
                # If the tensor dtype is floating point, average; else, take the first.
                if value[0].dtype in (torch.float16, torch.float32, torch.float64):
                    merged_value = torch.stack(value).mean(dim=0)
                else:
                    merged_value = value[0]
                merged_state[param_id][key] = merged_value

    # Build the final optimizer state dict that matches PyTorch's expected format:
    final_optim_state = {
        "state": merged_state,
        "param_groups": merged_param_groups,
    }
    torch_save(final_optim_state, output_path)
    # torch.save(final_optim_state, output_path)
    logger.debug(f"Merged optimizer states -> {output_path}")


def merge_adapter(paths, output_path):
    state_dict = defaultdict(list)

    for adapter_model_path in paths:
        while not os.path.exists(adapter_model_path):
            time.sleep(1)
            logger.debug(f"Waiting for file {adapter_model_path} to be available")

        _adapter_model_state = torch_load(adapter_model_path)
        for k, v in _adapter_model_state.items():
            state_dict[k].append(v)
    merged_state = {
        key: torch.stack(values).mean(0) for key, values in state_dict.items()
    }
    # torch.save(merged_state, output_path)
    torch_save(merged_state, output_path)
    logger.debug(f"Merged adapter model states -> {output_path}")


def wait_for(file: str, timeout: int):
    lock_file = f"{file}.lock"
    start_time = time.time()
    # cond is the first exist and no lock
    while not os.path.exists(file) and not os.path.exists(lock_file):
        time.sleep(1)
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {file} to be available")
    logger.success(f"File {file} is available after {time.time() - start_time} seconds")
    return file


class WeightSyncCallback(TrainerCallback):
    """Synchronize weights across multiple GPUs during training."""

    def __init__(
        self,
        gpu_id: int,
        all_gpu_ids: List[int],
        sync_interval: int = WEIGHT_SYNC_INTERVAL,
        trainer=None,
    ):
        self.gpu_id = gpu_id
        self.all_gpu_ids = all_gpu_ids
        self.sync_interval = sync_interval
        self.trainer = trainer

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: Any, **kwargs
    ):
        # Only run sync on specified intervals
        if state.global_step % self.sync_interval != 0:
            return

        out_path_optimizer = f"model_training_outputs/0/checkpoint-{state.global_step}/optimizer.merge.pt"
        out_path_adapter_model = f"model_training_outputs/0/checkpoint-{state.global_step}/adapter_model.merge.pt"

        if self.gpu_id == 0:
            optimizer_paths = [
                f"model_training_outputs/{gpu}/checkpoint-{state.global_step}/optimizer.pt"
                for gpu in self.all_gpu_ids
            ]
            merge_optim(optimizer_paths, out_path_optimizer)
            # Merge adapter model states
            adapter_model_paths = [
                f"model_training_outputs/{gpu}/checkpoint-{state.global_step}/adapter_model.safetensors"
                for gpu in self.all_gpu_ids
            ]
            merge_adapter(adapter_model_paths, out_path_adapter_model)
            
        wait_for(file=out_path_optimizer, timeout=WEIGHT_FILE_WAIT_TIMEOUT)
        wait_for(file=out_path_adapter_model, timeout=WEIGHT_FILE_WAIT_TIMEOUT)

        # Load the merged weights
        model = state.model
        optimizer = state.optimizer

        model_st = torch_load(out_path_adapter_model)
        optimizer_st = torch_load(out_path_optimizer)

        model_st_updated = {}
        for k, v in model_st.items():
            model_st_updated[k.replace(".weight", ".default.weight")] = v
        ret = optimizer.load_state_dict(optimizer_st)

        
        # Suppose we already have merged checkpoint files in some path:
        trainer = self.trainer
        ret = trainer.model.load_state_dict(model_st_updated, strict=False)
        assert len(ret.unexpected_keys) == 0, f"Unexpected keys: {ret.unexpected_keys}"
        trainer.optimizer.load_state_dict(optimizer_st)
        # # ckpt_path = self.ckpt_path_getter(state.global_step)

        # # 1) Load model weights
        # # merged_model_dict = torch.load(f"{ckpt_path}/pytorch_model.bin", map_location="cpu")
        # merged_model_dict = torch.load(f"{ckpt_path}/adapter_model.merge.pt", map_location="cpu")
        # trainer.model.load_state_dict(merged_model_dict, strict=False)

        # # 2) Load optimizer state (if you want to resume optimizer)
        # if trainer.optimizer is not None:
        #     opt_dict = torch.load(f"{ckpt_path}/optimizer.pt", map_location="cpu")
        #     trainer.optimizer.load_state_dict(opt_dict)

        # # 3) If you also want to restore the LR scheduler:
        # if trainer.lr_scheduler is not None:
        #     sched_dict = torch.load(f"{ckpt_path}/scheduler.pt", map_location="cpu")
        #     trainer.lr_scheduler.load_state_dict(sched_dict)

        # # You might also want to adjust 'state.global_step' if you are fully resuming.
        # # But typically, we do not override TrainerState steps here unless we're doing a full resume.

        # # Return control if you need to adjust or stop training
        # return control