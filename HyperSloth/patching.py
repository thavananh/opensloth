from collections import defaultdict
import random

from loguru import logger


def patch_sampler(trainer):
    from torch.utils.data import SequentialSampler
    from transformers import Trainer
    from fastcore.all import patch

    @patch
    def _get_train_sampler(self: Trainer) -> SequentialSampler:
        """Get a sequential sampler for the training dataset."""
        return SequentialSampler(self.train_dataset)

    return trainer


def select_dataset_by_length(
    dataset, gpu_index: int, num_gpus: int, grad_accum_steps: int, batch_size: int
):
    from fastcore.all import chunked
    from typing import List, Dict
    import numpy as np

    def split_batch_evenly(
        lengths: List[int], global_ids: List[int], num_gpus: int
    ) -> Dict[int, Dict[str, List[int]]]:
        if len(lengths) % num_gpus != 0:
            raise ValueError("The list length must be divisible by num_gpus")

        indices_sorted = np.argsort(-np.array(lengths))
        splits = [[] for _ in range(num_gpus)]
        length_sums = [0] * num_gpus
        max_items_per_gpu = len(lengths) // num_gpus

        for idx in indices_sorted:
            gpu_candidates = sorted(
                range(num_gpus),
                key=lambda gpu: (
                    len(splits[gpu]) >= max_items_per_gpu,
                    length_sums[gpu],
                ),
            )
            chosen_gpu = gpu_candidates[0]
            splits[chosen_gpu].append(idx)
            length_sums[chosen_gpu] += lengths[idx]

        splits = [sorted(split) for split in splits]

        gpu_batches = {
            gpu: {
                "global_ids": [global_ids[i] for i in splits[gpu]],
                "lengths": [lengths[i] for i in splits[gpu]],
            }
            for gpu in range(num_gpus)
        }
        for gpu in range(num_gpus):
            lens = gpu_batches[gpu]["lengths"]
            ids = gpu_batches[gpu]["global_ids"]
            new_lens, new_ids = zip(*sorted(zip(lens, ids), key=lambda x: x[0]))
            total_len = sum(new_lens)
            gpu_batches[gpu] = {
                "global_ids": list(new_ids),
                "lengths": list(new_lens),
                "total_len": total_len,
            }

        return gpu_batches

    dataset_indices = list(range(len(dataset)))
    id_to_length = {idx: len(dataset[idx]["input_ids"]) for idx in dataset_indices}
    random.Random(42).shuffle(dataset_indices)

    global_batch_size = grad_accum_steps * batch_size * num_gpus
    global_batches = list(chunked(dataset_indices, global_batch_size))

    selected_ids = defaultdict(list)
    for batch_indices in global_batches[:-1]:  # Exclude potentially smaller last batch
        batch_lengths = [id_to_length[i] for i in batch_indices]
        splits = split_batch_evenly(batch_lengths, batch_indices, num_gpus)
        this_gpu_split = splits[gpu_index]
        if len(selected_ids) == 0:
            print(f"{splits=}")
        for gpu, split in splits.items():
            selected_ids[gpu].extend(split["global_ids"])
    # ensure all gpus have the same number of samples
    n1 = len(selected_ids[0])
    for gpu in range(1, num_gpus):
        n2 = len(selected_ids[gpu])
        if n1 != n2:
            raise ValueError(f"GPU {gpu} has {n2} samples, while GPU 0 has {n1}")
    logger.info(f"GPU {gpu_index}: Selected {n1} samples for training")
    selected_ids_flat = []
    for gpu, ids in selected_ids.items():
        selected_ids_flat.extend(ids)
    return dataset.select(selected_ids_flat)
