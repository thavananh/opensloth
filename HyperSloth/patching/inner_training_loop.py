import os
import time
from typing import Dict, Any
from fastcore.all import patch
from transformers.trainer import Trainer, TrainerState
from HyperSloth.logging_config import get_hypersloth_logger
from HyperSloth._patch_log import _patch_log
import torch

try:
    from tabulate import tabulate
except ImportError:
    print("Warning: tabulate not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "tabulate"])
    from tabulate import tabulate


class TokenStatsTracker:
    """Track token statistics for HyperSloth optimization."""

    def __init__(self, log_interval_seconds: int = 10):
        self.log_interval_seconds = log_interval_seconds
        self.last_log_time = time.time()
        self.batch_counter = 0
        self.total_tokens_before = 0
        self.total_tokens_after = 0
        self.total_actual_tokens = 0
        self.logger = get_hypersloth_logger()

    def update_and_maybe_log(
        self, batch_samples: list, splited_by_gpus: dict, hp_local_rank: int
    ) -> None:
        """Update statistics and log if interval has passed."""
        stats = self._compute_batch_stats(batch_samples, splited_by_gpus)

        self.batch_counter += 1
        self.total_tokens_before += stats["tokens_before"]
        self.total_tokens_after += stats["tokens_after"]
        self.total_actual_tokens += stats["actual_tokens"]

        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval_seconds:
            self._log_stats_table(hp_local_rank)
            self.last_log_time = current_time

    def _compute_batch_stats(self, batch_samples: list, splited_by_gpus: dict) -> dict:
        """Compute statistics for a single batch."""
        # Calculate tokens before partitioning
        tokens_before = sum(batch["input_ids"].numel() for batch in batch_samples)
        actual_tokens = sum(
            batch["attention_mask"].sum().item() for batch in batch_samples
        )

        # Calculate tokens after partitioning
        tokens_after = 0
        for gpu_batches in splited_by_gpus.values():
            tokens_after += sum(batch["input_ids"].numel() for batch in gpu_batches)

        return {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "actual_tokens": actual_tokens,
            "tokens_saved": tokens_before - tokens_after,
        }

    def _log_stats_table(self, hp_local_rank: int) -> None:
        """Log statistics in a nice tabular format."""
        if self.total_tokens_before == 0:
            return

        # Calculate token utilization metrics
        before_optimization_utilization = (
            self.total_actual_tokens / self.total_tokens_before
        ) * 100
        actual_efficiency = (self.total_actual_tokens / self.total_tokens_after) * 100

        # Create table data
        table_data = [
            [
                "Before Optimization",
                f"{self.total_tokens_before:,}",
                f"{before_optimization_utilization:.2f}%",
            ],
            [
                "After Optimization",
                f"{self.total_tokens_after:,}",
                f"{actual_efficiency:.2f}%",
            ],
        ]

        table_str = tabulate(
            table_data,
            headers=["Stage", "Total Tokens", "Token Utilization %"],
            tablefmt="grid",
            colalign=["left", "right", "right"],
        )

        self.logger.info(
            f"\n🚀 HyperSloth Token Efficiency Report (Rank {hp_local_rank})\n"
            f"{table_str}\n"
        )


def improve_partition_batches(all_batches: list, num_devices: int) -> dict:
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
    # This is for debugging purposes, it does not affect the training logics
    _patch_log(trainer_class)

    # Get enhanced logger
    hp_logger = get_hypersloth_logger()

    # Initialize statistics tracker
    stats_tracker = TokenStatsTracker(
        log_interval_seconds=int(os.getenv("HYPERSLOTH_LOG_INTERVAL", "180"))
    )

    # Patch 1: TrainerState creation with HyperSloth fields
    original_trainer_state_init = TrainerState.__init__

    def setup_hyper_sloth_trainer_state(self, **kwargs):
        original_trainer_state_init(self, **kwargs)
        # Add HyperSloth custom fields
        self.is_world_process_zero = hp_local_rank == 0

    TrainerState.__init__ = setup_hyper_sloth_trainer_state

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

            splited_by_gpus = improve_partition_batches(batch_samples, hp_wolrd_size)
            processed_samples = splited_by_gpus[hp_local_rank]

            # Update statistics and log periodically
            stats_tracker.update_and_maybe_log(
                batch_samples, splited_by_gpus, hp_local_rank
            )

            return processed_samples, num_items_in_batch

    hp_logger.info(
        f"HyperSloth ultra-minimal patches applied successfully "
        f"(rank {hp_local_rank}/{hp_wolrd_size})"
    )
