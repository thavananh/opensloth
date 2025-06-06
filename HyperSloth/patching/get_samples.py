from fastcore.all import patch
from transformers.trainer import *
from HyperSloth._patch_log import _patch_log
from ..logging_config import get_hypersloth_logger

DISABLE_PATCHING = False


def pack(input_ids_list, labels_list, attention_mask_list):
    """
    Pack multiple sequences into batched tensors with proper attention masking.

    Based on the packed sequences strategy from the HuggingFace article:
    "Efficient LLM Pretraining: Packed Sequences and Masked Attention"

    Args:
        input_ids_list: List of input_ids tensors
        labels_list: List of labels tensors
        attention_mask_list: List of attention_mask tensors

    Returns:
        Dict with keys: input_ids, labels, attention_mask, position_ids
    """
    import torch

    if not input_ids_list:
        raise ValueError("Cannot pack empty sequence list")

    # Get device from first tensor
    device = input_ids_list[0].device

    # Concatenate all sequences
    packed_input_ids = torch.cat(input_ids_list, dim=0)
    packed_labels = torch.cat(labels_list, dim=0)
    packed_attention_mask = torch.cat(attention_mask_list, dim=0)

    # Add batch dimension
    packed_input_ids = packed_input_ids.unsqueeze(0)
    packed_labels = packed_labels.unsqueeze(0)
    packed_attention_mask = packed_attention_mask.unsqueeze(0)

    B, T = packed_input_ids.shape

    # Find EOS token positions to create sequence boundaries
    # We'll use the transition points where attention mask changes
    # from 1 to 0 between sequences as EOS positions
    sequence_lengths = [len(seq) for seq in input_ids_list]

    # Create EOS indices based on sequence boundaries
    eos_indices = []
    cumulative_pos = 0
    for seq_len in sequence_lengths:
        cumulative_pos += seq_len
        eos_indices.append(cumulative_pos - 1)  # Last position of each sequence

    eos_indices = torch.tensor(eos_indices, device=device)

    # Create position IDs that reset for each sequence
    pos_ids = torch.zeros(T, device=device, dtype=torch.long)
    start_pos = 0
    for seq_len in sequence_lengths:
        pos_ids[start_pos : start_pos + seq_len] = torch.arange(seq_len, device=device)
        start_pos += seq_len

    pos_ids = pos_ids.unsqueeze(0)  # Add batch dimension

    return {
        "input_ids": packed_input_ids,
        "labels": packed_labels,
        "attention_mask": packed_attention_mask,
        "position_ids": pos_ids,
    }


def patch_get_batch_samples(trainer):
    """
    Ultra-minimal patch that only adds essential HyperSloth customizations.
    This approach patches specific methods instead of duplicating the entire training loop.
    """
    # Get environment variables
    hp_local_rank = int(os.getenv("HYPERSLOTH_LOCAL_RANK", "0"))
    hp_wolrd_size = int(os.getenv("HYPERSLOTH_WORLD_SIZE", "1"))

    trainer_class = type(trainer)
    _patch_log(trainer_class)

    # Get enhanced logger
    logger = get_hypersloth_logger("DEBUG")

    # TrainerState.__init__ = setup_hyper_sloth_trainer_state
    original_get_batch_samples = Trainer.get_batch_samples

    @patch
    def get_batch_samples(self: Trainer, epoch_iterator, num_batches, device=None):
        """Enhanced batch sampling with GPU slicing and token tracking for HyperSloth."""
        batch_samples, num_items_in_batch = original_get_batch_samples(
            self, epoch_iterator, num_batches, device
        )

        # Apply GPU-specific optimizations if multi-GPU
        if hp_wolrd_size > 1:
            max_seql_len = trainer.args.max_seq_length

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
            item_this_gpus = all_items[hp_local_rank::hp_wolrd_size]

            cumulative_len = 0
            packed_items = []
            pack_items_pending = []
            while item_this_gpus:
                item = item_this_gpus.pop(0)
                ft_len = cumulative_len + item["num_non_padding_tokens"]
                if ft_len > max_seql_len:  # check if we can pack it
                    # Pack current batch
                    packed = pack(
                        [item["input_ids"] for item in pack_items_pending],
                        [item["labels"] for item in pack_items_pending],
                        [item["attention_mask"] for item in pack_items_pending],
                    )
                    # Add packed batch to the list
                    logger.info(
                        f"Packing multiple items of lens {', '.join(str(item['num_non_padding_tokens']) for item in pack_items_pending)} "
                        f"into a single batch of length {packed['input_ids'].shape[1]}."
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
                logger.info(
                    f"Packing remaining items of lens {', '.join(str(item['num_non_padding_tokens']) for item in pack_items_pending)} "
                    f"into a single batch of length {packed['input_ids'].shape[1]}."
                )
                packed_items.append(packed)

            # Use packed items as batch_samples
            batch_samples = packed_items

        return batch_samples, num_items_in_batch
