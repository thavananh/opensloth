import itertools
import random
from typing import Literal

import torch
import torch.nn.functional as F

SPECIAL_TOKEN_SPLIT_IN_GPU = -1000
SPECIAL_TOKEN_SPLIT_OUT_GPU = -1001


def _create_batches(indexed_lengths, max_seq_len=5000, num_gpus=8):
    """
    Split sequence lengths into batches per GPU where each batch has total numel <= max_seq_len.

    Args:
        indexed_lengths (List[int]): list of (index, length) pairs.
        max_seq_len (int): maximum total number of elements per batch.
        num_gpus (int): number of GPUs.

    Returns:
        List[List[List[int]]]: A list of length `num_gpus`. Each element is a list of batches,
        and each batch is a list of original indices.
    """
    # Sort descending by length
    indexed_lengths = sorted(indexed_lengths, key=lambda x: x[1], reverse=True)

    batches = []
    current_batch = []
    current_sum = 0

    # Create batches so that sum of lengths in a batch <= max_seq_len
    for idx, length in indexed_lengths:
        if current_sum + length <= max_seq_len:
            current_batch.append(idx)
            current_sum += length
        else:
            batches.append(current_batch)
            current_batch = [idx]
            current_sum = length

    if current_batch:
        batches.append(current_batch)

    # Distribute batches round-robin across GPUs
    gpu_batches = [[] for _ in range(num_gpus)]
    for i, batch in enumerate(batches):
        gpu_batches[i % num_gpus].append(batch)

    return gpu_batches


def encode_dynamic_batching_dataset(dataset, num_gpus, max_len_allow, filter_by_max_len=True):
    """
    Create a dynamic-batching version of the dataset.

    1) Filter out sequences longer than max_len_allow (if specified).
    2) Use `_create_batches()` to group these items so that each batch
       has a total number of tokens <= max_len_allow.
    3) Round-robin distribute these batches across the GPUs.
    4) For each "training step" (update_ith), merge the batches from each GPU,
       inserting special boundary tokens to mark:
          - boundary between samples in the same GPU batch
          - boundary between different GPUs
    5) Finally, append a count of all valid (>=0) label tokens at the end.

    Returns:
        A HuggingFace Dataset containing merged items. Each item is a dict like:
            {
                "input_ids": [...],
                "attention_mask": [...],
                "labels": [...],
            }
        with special boundary tokens inserted.
    """
    lengths = [len(ids) for ids in dataset["input_ids"]]
    indexed_lengths = list(enumerate(lengths))

    # Optionally filter by maximum length
    if filter_by_max_len:
        indexed_lengths = [
            (idx, length) for idx, length in indexed_lengths if length <= max_len_allow
        ]

    gpu_batches = _create_batches(
        indexed_lengths, max_seq_len=max_len_allow, num_gpus=num_gpus
    )

    def merge_items(items):
        # Merge multiple samples (dicts) into one by inserting SPECIAL_TOKEN_SPLIT_IN_GPU
        # between them. Each item has ["input_ids"], ["attention_mask"], ["labels"].
        merged = {
            "input_ids": items[0]["input_ids"][:],
            "attention_mask": items[0]["attention_mask"][:],
            "labels": items[0]["labels"][:],
        }
        for _item in items[1:]:
            merged["input_ids"] += [SPECIAL_TOKEN_SPLIT_IN_GPU] + _item["input_ids"]
            merged["attention_mask"] += [SPECIAL_TOKEN_SPLIT_IN_GPU] + _item["attention_mask"]
            merged["labels"] += [SPECIAL_TOKEN_SPLIT_IN_GPU] + _item["labels"]
        return merged

    merged_items = []
    # Instead of using gpu_batches[0], we take the max # of batches across all GPUs
    num_updates = max(len(gb) for gb in gpu_batches) if gpu_batches else 0
    from tqdm import tqdm
    from datasets import Dataset
    for update_ith in tqdm(range(num_updates), desc="Encoding dynamic batching"):
        # Collect one merged batch *per GPU* (if available)
        gpu_merged = []
        for gpu_ith in range(num_gpus):
            if update_ith < len(gpu_batches[gpu_ith]):
                batch_indices = gpu_batches[gpu_ith][update_ith]
                items = [dataset[idx] for idx in batch_indices]
                merge_item = merge_items(items)
                gpu_merged.append(merge_item)

        if not gpu_merged:
            continue

        # Merge the GPU batches into a single item, separated by SPECIAL_TOKEN_SPLIT_OUT_GPU
        global_item = gpu_merged[0]
        for gpu_item in gpu_merged[1:]:
            global_item["input_ids"] += [SPECIAL_TOKEN_SPLIT_OUT_GPU] + gpu_item["input_ids"]
            global_item["attention_mask"] += [SPECIAL_TOKEN_SPLIT_OUT_GPU] + gpu_item["attention_mask"]
            global_item["labels"] += [SPECIAL_TOKEN_SPLIT_OUT_GPU] + gpu_item["labels"]


        merged_items.append(global_item)

    return Dataset.from_list(merged_items)


def torch_batch_pad(tensors, dim=0, pad_val=0):
    """
    Given a list of tensors of shape [1, length], pad them to the same length and cat along 'dim'.
    If the list is empty, returns an empty float tensor.
    """
    if not tensors:  # empty
        return torch.empty(0, dtype=torch.long)

    max_len = max(t.size(1) for t in tensors)

    def pad_to(tensor, target_len):
        pad_amount = target_len - tensor.size(1)
        if pad_amount > 0:
            return F.pad(tensor, (0, pad_amount), value=pad_val)
        return tensor

    padded_tensors = [pad_to(t, max_len) for t in tensors]
    return torch.cat(padded_tensors, dim=dim)


def split_by_token(seq, boundary_token):
    """
    Split a 1D tensor `seq` by `boundary_token`.
    Return a list of sub-tensors (excluding the boundary token itself).

    Example:
      seq = [101, 102, -1000, 103, 104]
      boundary_token = -1000
      => return [ [101, 102], [103, 104] ]
    """
    chunks = []
    start = 0
    for i, val in enumerate(seq):
        if val.item() == boundary_token:
            # sub-chunk is from [start..i)
            if i > start:
                chunks.append(seq[start:i])
            start = i + 1
    # final chunk from [start..end]
    if start < seq.shape[0]:
        chunks.append(seq[start:])
    return chunks


def decode_dynamic_batching(item, gpu_boundary=-1001, sample_boundary=-1000):
    """
    Perform two-level splitting of a single packed item (shape [1, L]):
      1) Split by `gpu_boundary` (e.g. -1001) to get each GPU partition
      2) Within each GPU partition, split by `sample_boundary` (e.g. -1000) to get individual samples
      3) Pad those samples to shape [num_samples, max_seq_len] (per GPU)

    Returns a list of dictionaries, each corresponding to one GPU partition:
        {
          "input_ids":      Tensor([num_samples, max_seq_len]),
          "attention_mask": Tensor([num_samples, max_seq_len]),
          "labels":         Tensor([num_samples, max_seq_len]),
          "num_items_in_batch": (int),
        }
    """
    # We stored the "valid token count" in the last position of the labels
    # item["labels"] is shape [1, L], so we split off the last column to get the count
    assert len(item["input_ids"]) == 1, "Expected a single item with shape [1, L]"
    # labels, num_items_in_batch = item['labels'][:, :-1], item['labels'][:, -1]
    # num_items_in_batch = num_items_in_batch.item()

    # Now item["input_ids"].shape is [1, L], labels.shape is [1, (L-1)]
    # Typically these match if you appended exactly 1 to the end of labels
    # assert labels.shape[1] == item["input_ids"].shape[1], (
    #     f"Mismatch in shapes: input_ids {item['input_ids'].shape}, "
    #     f"labels {labels.shape}"
    # )
    # item['labels'] = labels

    inp_ids = item["input_ids"][0]        # shape [L]
    attn    = item["attention_mask"][0]   # shape [L]
    labels  = item["labels"][0]           # shape [L]
    
    
    num_items_in_batch = (labels>=0).sum().item()

    # 1) Split by GPU boundary
    gpu_partitions_inp   = split_by_token(inp_ids,  gpu_boundary)
    gpu_partitions_attn  = split_by_token(attn,     gpu_boundary)
    gpu_partitions_label = split_by_token(labels,   gpu_boundary)

    # Check they all have the same # of partitions
    assert len(gpu_partitions_inp) == len(gpu_partitions_attn) == len(gpu_partitions_label), (
        "Mismatch in # of GPU partitions after splitting"
    )

    gpu_results = []

    # 2) For each GPU partition, split again by sample boundary
    for gpu_inp, gpu_attn, gpu_lbl in zip(
        gpu_partitions_inp, gpu_partitions_attn, gpu_partitions_label
    ):
        sample_inp   = split_by_token(gpu_inp,  sample_boundary)
        sample_attn  = split_by_token(gpu_attn, sample_boundary)
        sample_label = split_by_token(gpu_lbl,  sample_boundary)

        assert len(sample_inp) == len(sample_attn) == len(sample_label), (
            "Mismatch in # of samples after splitting by sample boundary"
        )

        # Convert each sub-chunk to shape [1, sub_len] so we can pad
        inp_list = [si.unsqueeze(0) for si in sample_inp]
        attn_list = [sa.unsqueeze(0) for sa in sample_attn]
        label_list = [sl.unsqueeze(0) for sl in sample_label]

        # 3) Pad to [num_samples, max_seq_len]
        pad_inp   = torch_batch_pad(inp_list,   dim=0, pad_val=0)
        pad_attn  = torch_batch_pad(attn_list,  dim=0, pad_val=0)
        pad_label = torch_batch_pad(label_list, dim=0, pad_val=-100)

        gpu_results.append({
            "input_ids":         pad_inp,
            "attention_mask":    pad_attn,
            "labels":            pad_label,
            "num_items_in_batch": num_items_in_batch,
        })

    return gpu_results
