from typing import Dict, List

from fastcore.all import patch
from transformers.trainer import *

from opensloth.patching.patch_log import patch_log

from ..logging_config import get_opensloth_logger

DISABLE_PACKING = False


def pack(
    input_ids_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    attention_mask_list: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Pack multiple sequences into a single batch with proper attention masking."""

    if not input_ids_list:
        raise ValueError("Cannot pack empty sequence list.")

    # All tensors in the list should be on the same device.
    # Get device from the first tensor.
    device = input_ids_list[0].device

    # Concatenate all sequences
    # These are 1D tensors after concatenation (total_len,)
    packed_input_ids_1d = torch.cat(input_ids_list, dim=0)
    packed_labels_1d = torch.cat(labels_list, dim=0)
    # This 1D mask indicates real vs. padded tokens across the whole pack
    packed_original_1d_mask = torch.cat(attention_mask_list, dim=0)

    total_len = packed_input_ids_1d.shape[0]

    # Add batch dimension (B=1)
    # Shape: (1, total_len)
    packed_input_ids = packed_input_ids_1d.unsqueeze(0)
    packed_labels = packed_labels_1d.unsqueeze(0)

    # --- Create 2D attention mask for packed sequences ---
    # Shape: (total_len, total_len)
    # This mask will ensure causality within each sequence and no attention
    # between sequences.
    correct_attention_mask_2d = torch.zeros(
        total_len, total_len, device=device, dtype=torch.bool
    )

    sequence_lengths = [len(seq) for seq in input_ids_list]
    current_pos = 0
    for seq_len in sequence_lengths:
        segment_end = current_pos + seq_len
        # Create a causal mask for the current sequence segment
        # torch.tril creates a lower triangular matrix.
        # True means attention is allowed.
        segment_causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        )
        correct_attention_mask_2d[current_pos:segment_end, current_pos:segment_end] = (
            segment_causal_mask
        )
        current_pos = segment_end

    # Account for original padding within sequences.
    # A token k cannot be attended to if it was a padding token in its
    # original sequence.
    # packed_original_1d_mask has shape (total_len,). Convert to bool.
    # unsqueeze(0) makes it (1, total_len) for broadcasting with (total_len, total_len).
    # mask[q, k] = mask[q, k] & original_mask[k]
    correct_attention_mask_2d &= packed_original_1d_mask.bool().unsqueeze(0)

    # Add batch dimension for the final attention mask
    # Shape: (1, total_len, total_len)
    final_packed_attention_mask = correct_attention_mask_2d.unsqueeze(0)

    return {
        "input_ids": packed_input_ids,
        "labels": packed_labels,
        "attention_mask": final_packed_attention_mask,
        # "position_ids": packed_position_ids, # NO NEED becase ROPE already handles it
    }


from ..opensloth_config import OpenSlothConfig


def patch_get_batch_samples(opensloth_config: OpenSlothConfig):
    """
    Ultra-minimal patch that only adds essential opensloth customizations.
    This approach patches specific methods instead of duplicating the entire training loop.
    """
    # Get environment variables
    hp_local_rank = int(os.getenv("HYPERSLOTH_LOCAL_RANK", "0"))
    hp_world_size = int(os.getenv("HYPERSLOTH_WORLD_SIZE", "1"))
    # Get enhanced logger
    logger = get_opensloth_logger("DEBUG")

    # TrainerState.__init__ = setup_hyper_sloth_trainer_state
    original_get_batch_samples = Trainer.get_batch_samples

    @patch
    def get_batch_samples(self: Trainer, epoch_iterator, num_batches, device=None):
        """Enhanced batch sampling with GPU slicing and token tracking for opensloth."""
        batch_samples, num_items_in_batch = original_get_batch_samples(
            self, epoch_iterator, num_batches, device
        )
        if opensloth_config.disable_packing:
            # all_items = {gpu: {} for gpu in range(hp_world_size)}
            ga_batches = []
            for accumulated_batch in batch_samples:
                b = {}
                b["input_ids"] = accumulated_batch["input_ids"][
                    hp_local_rank::hp_world_size
                ]
                b["labels"] = accumulated_batch["labels"][hp_local_rank::hp_world_size]
                b["attention_mask"] = accumulated_batch["attention_mask"][
                    hp_local_rank::hp_world_size
                ]
                ga_batches.append(b)
            return ga_batches, num_items_in_batch

        # Apply GPU-specific optimizations if multi-GPU
        if hp_world_size > 1:
            max_seq_len = opensloth_config.fast_model_args.max_seq_length

            all_items = []
            for accumulated_batch in batch_samples:
                input_ids, labels, attention_mask = (
                    accumulated_batch["input_ids"],
                    accumulated_batch["labels"],
                    accumulated_batch["attention_mask"],
                )
                for i in range(len(input_ids)):
                    single_input_ids = input_ids[i]
                    single_labels = labels[i]
                    single_attention_mask = attention_mask[i]
                    num_non_padding_tokens = single_attention_mask.sum().item()
                    single_input_ids = single_input_ids[:num_non_padding_tokens]
                    single_labels = single_labels[:num_non_padding_tokens]
                    single_attention_mask = single_attention_mask[
                        :num_non_padding_tokens
                    ]
                    all_items.append(
                        {
                            "input_ids": single_input_ids,
                            "labels": single_labels,
                            "attention_mask": single_attention_mask,
                            "num_non_padding_tokens": num_non_padding_tokens,
                        }
                    )
            # Sort items by length
            all_items.sort(key=lambda item: item["num_non_padding_tokens"])
            items_this_device = all_items[hp_local_rank::hp_world_size]
            cumulative_len = 0
            packed_items = []
            pack_items_pending = []
            while items_this_device:
                item = items_this_device.pop(0)
                ft_len = cumulative_len + item["num_non_padding_tokens"]
                if ft_len > max_seq_len:  # check if we can pack it
                    # Pack current batch
                    packed = pack(
                        [item["input_ids"] for item in pack_items_pending],
                        [item["labels"] for item in pack_items_pending],
                        [item["attention_mask"] for item in pack_items_pending],
                    )
                    # Add packed batch to the list
                    logger.debug(
                        f"Packed {len(pack_items_pending)} items into batch of length {packed['input_ids'].shape[1]}"
                    )
                    packed_items.append(packed)

                    # Reset for next batch
                    pack_items_pending = []
                    cumulative_len = 0

                # Add item to current batch
                pack_items_pending.append(item)
                cumulative_len += item["num_non_padding_tokens"]

            # Pack any remaining items
            if pack_items_pending:
                packed = pack(
                    [item["input_ids"] for item in pack_items_pending],
                    [item["labels"] for item in pack_items_pending],
                    [item["attention_mask"] for item in pack_items_pending],
                )
                logger.debug(
                    f"Packed {len(pack_items_pending)} items into batch of length {packed['input_ids'].shape[1]}"
                )
                packed_items.append(packed)

            # Use packed items as batch_samples
            batch_samples = packed_items

        return batch_samples, num_items_in_batch
