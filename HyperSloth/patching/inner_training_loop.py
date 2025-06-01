import os
from typing import Optional, Dict, Any
from fastcore.all import patch
from transformers.trainer import Trainer, TrainerState
from HyperSloth.logging_config import get_hypersloth_logger
from HyperSloth._patch_log import _patch_log
import torch


def smart_partition_batches(all_batches: list, num_devices: int) -> dict:
    """
    Partition batches across devices by sorting sequences by length.

    Args:
        all_batches: List of batch dictionaries with input_ids, labels, attention_mask
        num_devices: Number of devices to partition across

    Returns:
        Dictionary mapping device indices to lists of batches
    """
    perdevice_batch_size = all_batches[0]["input_ids"].shape[0] // num_devices

    # Flatten all sequences
    input_ids = [
        batch["input_ids"][i]
        for batch in all_batches
        for i in range(len(batch["input_ids"]))
    ]
    labels = [
        batch["labels"][i] for batch in all_batches for i in range(len(batch["labels"]))
    ]
    attention_mask = [
        batch["attention_mask"][i]
        for batch in all_batches
        for i in range(len(batch["attention_mask"]))
    ]

    # Sort by sequence length
    num_tokens = [mask.sum().item() for mask in attention_mask]
    sorted_ids = sorted(range(len(num_tokens)), key=lambda i: num_tokens[i])

    new_input_ids = [input_ids[i] for i in sorted_ids]
    new_labels = [labels[i] for i in sorted_ids]
    new_attention_mask = [attention_mask[i] for i in sorted_ids]
    lens = [num_tokens[i] for i in sorted_ids]

    def item_to_perdevice_batch(items, counts, padding_value) -> torch.Tensor:
        max_len = max(counts)
        output_tensor = torch.full(
            (len(items), max_len),
            padding_value,
            dtype=items[0].dtype,
            device=items[0].device,
        )
        for i, item in enumerate(items):
            output_tensor[i, : counts[i]] = item[: counts[i]]
        return output_tensor

    # Partition into device batches
    by_gpu_batches = {gpu_idx: [] for gpu_idx in range(num_devices)}
    gpu_idx = 0

    for i in range(0, len(new_input_ids), perdevice_batch_size):
        input_ids_tensor = item_to_perdevice_batch(
            new_input_ids[i : i + perdevice_batch_size],
            lens[i : i + perdevice_batch_size],
            0,
        )
        labels_tensor = item_to_perdevice_batch(
            new_labels[i : i + perdevice_batch_size],
            lens[i : i + perdevice_batch_size],
            -100,
        )
        attention_mask_tensor = item_to_perdevice_batch(
            new_attention_mask[i : i + perdevice_batch_size],
            lens[i : i + perdevice_batch_size],
            0,
        )

        by_gpu_batches[gpu_idx].append(
            {
                "input_ids": input_ids_tensor,
                "labels": labels_tensor,
                "attention_mask": attention_mask_tensor,
            }
        )
        gpu_idx = (gpu_idx + 1) % num_devices

    return by_gpu_batches


def patch_inner_training_loop(trainer):
    """
    Ultra-minimal patch that only adds essential HyperSloth customizations.
    This approach patches specific methods instead of duplicating the entire training loop.
    """
    # Get environment variables
    hp_local_rank = int(os.getenv("HYPERSLOTH_LOCAL_RANK", "0"))
    hp_wolrd_size = int(os.getenv("HYPERSLOTH_WORLD_SIZE", "1"))

    # Apply log patch
    trainer_class = type(trainer)
    _patch_log(trainer_class)

    # Get enhanced logger
    enhanced_logger = get_hypersloth_logger()

    # Patch 1: TrainerState creation with HyperSloth fields
    original_trainer_state_init = TrainerState.__init__

    def enhanced_trainer_state_init(self, **kwargs):
        original_trainer_state_init(self, **kwargs)
        # Add HyperSloth custom fields
        self.is_world_process_zero = hp_local_rank == 0

    TrainerState.__init__ = enhanced_trainer_state_init

    # Patch 2: GPU-specific batch slicing (if multi-GPU)
    if hp_wolrd_size > 1 and hasattr(trainer_class, "get_batch_samples"):
        original_get_batch_samples = trainer_class.get_batch_samples

        def select_gpu_slice(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Select this GPU's slice of the batch data."""
            if inputs is None:
                return inputs

            result = {}
            for key, value in inputs.items():
                if key in ["input_ids", "attention_mask", "labels"] and hasattr(
                    value, "__getitem__"
                ):
                    result[key] = value[hp_local_rank::hp_wolrd_size]
                else:
                    result[key] = value
            return result

        @patch
        def get_batch_samples(self: Trainer, epoch_iterator, num_batches, device=None):
            """Enhanced batch sampling with GPU slicing for HyperSloth."""
            batch_samples, num_items_in_batch = original_get_batch_samples(
                self, epoch_iterator, num_batches, device
            )

            splited_by_gpus = smart_partition_batches(batch_samples, hp_wolrd_size)
            processed_samples = splited_by_gpus[hp_local_rank]
            return processed_samples, num_items_in_batch

    enhanced_logger.info(
        f"HyperSloth ultra-minimal patches applied successfully "
        f"(rank {hp_local_rank}/{hp_wolrd_size})"
    )
