# import os
# import random
# from typing import Iterator, Literal, Union

# from datasets import Dataset, IterableDataset
# from fastcore.all import patch
# from torch.utils.data import SequentialSampler
# from transformers import Trainer, TrainerCallback

# from HyperSloth.logging_config import get_hypersloth_logger


# logger = get_hypersloth_logger(log_level="INFO")
# from fastcore.all import num_cpus


# def _compute_reordered_and_shuffled_ids(
#     dataset: Dataset,
#     epoch,
#     seed=42,
#     mode=None,  # not use
# ) -> list[int]:
#     from fastcore.all import chunked

#     local_rank = int(os.environ.get("HYPERSLOTH_LOCAL_RANK", "0"))

#     # Calculate optimal number of processes
#     dataset_size = len(dataset)
#     nproc = dataset_size // 5_000
#     cpu_count = num_cpus()
#     if cpu_count is not None and nproc > cpu_count - 2:
#         nproc = cpu_count - 2
#     nproc = max(nproc, 1)  # ensure at least one process

#     if local_rank == 0:
#         logger.info(
#             f"🔢 Computing sequence lengths for {dataset_size:,} samples using {nproc} processes"
#         )

#     lens = dataset.map(
#         lambda x: {"len": len(x["input_ids"])},
#         remove_columns=dataset.column_names,
#         desc="Computing sequence lengths",
#         num_proc=nproc,  # type: ignore
#     )
#     lens = [x["len"] for x in lens]  # type: ignore

#     sorted_ids = sorted(range(len(lens)), key=lambda k: lens[k])

#     global_bz = int(os.environ["HYPERSLOTH_FORWARD_BZ"])
#     chunked_ids = list(chunked(sorted_ids, global_bz))

#     R = random.Random(seed + epoch)
#     R.shuffle(chunked_ids)

#     if local_rank == 0:
#         min_len, max_len = min(lens), max(lens)
#         avg_len = sum(lens) / len(lens)
#         logger.info(
#             f"📏 Sequence lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}"
#         )

#     return [idx for chunk in chunked_ids for idx in chunk]


# def get_callback_shuffle_data(trainer) -> TrainerCallback:
#     "return a callback to shuffle data on_epoch_begin"

#     class ShuffleData(TrainerCallback):
#         def __init__(self, trainer):
#             self.trainer: Trainer = trainer

#         def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
#             local_rank = int(os.environ["HYPERSLOTH_LOCAL_RANK"])

#             # Only log from rank 0 to reduce verbosity
#             if local_rank == 0:
#                 logger.info(f"🔄 Starting epoch {state.epoch + 1}")

#                 try:
#                     from ._debug_dataloader import _debug_dataloader

#                     tok = kwargs["processing_class"]
#                     _debug_dataloader(
#                         self.trainer.get_train_dataloader(), tokenizer=tok
#                     )
#                     logger.info(
#                         "📋 Dataloader examples logged to .log/dataloader_examples.html"
#                     )
#                 except Exception as e:
#                     logger.debug(f"Dataloader debugging failed (non-critical): {e}")
#                     pass

#     return ShuffleData(trainer)


# from torch.utils.data.sampler import SequentialSampler


# class CustomSampler(SequentialSampler):
#     def __init__(self, data_source, shuffle_mode: str = "on_dataset") -> None:
#         self.data_source = data_source
#         self.ids = _compute_reordered_and_shuffled_ids(
#             data_source,
#             epoch=0,
#             seed=42,
#             mode=shuffle_mode,
#         )

#     def __iter__(self) -> Iterator[int]:
#         return iter(self.ids)


# from speedy_utils import Clock
# from warnings import warn


# def patch_sampler(trainer: Trainer):
#     warn(
#         "This function is deprecated and will be removed in future versions. "
#         "Use `trainer.add_callback(get_callback_shuffle_data(trainer))` instead."
#     )
#     return trainer
#     # clock = Clock(start_now=True)
#     # local_rank = int(os.environ.get("HYPERSLOTH_LOCAL_RANK", "0"))

#     # @patch
#     # def _get_train_sampler(self: Trainer, train_dataset=None) -> CustomSampler:
#     #     """Get a custom sampler for the training dataset."""
#     #     if train_dataset is None:
#     #         train_dataset = self.train_dataset

#     #     # Log dataset info with better formatting
#     #     if hasattr(train_dataset, "__len__"):
#     #         try:
#     #             dataset_size = len(train_dataset)  # type: ignore
#     #             if local_rank == 0:  # Only log from rank 0
#     #                 logger.info(
#     #                     f"📊 Dataset: {dataset_size:,} samples | Creating custom sampler"
#     #                 )
#     #         except Exception:
#     #             if local_rank == 0:
#     #                 logger.info("📊 Dataset: Creating custom sampler")
#     #     else:
#     #         if local_rank == 0:
#     #             logger.info("📊 Dataset: Iterable (no fixed length)")

#     #     assert isinstance(
#     #         train_dataset, (Dataset, IterableDataset)
#     #     ), "train_dataset must be a Dataset or IterableDataset"

#     #     # Get shuffle mode from trainer's hypersloth config if available
#     #     shuffle_mode = getattr(
#     #         getattr(self, "hypersloth_config", None), "shuffle_mode", "on_dataset"
#     #     )
#     #     return CustomSampler(
#     #         train_dataset,
#     #         shuffle_mode=os.environ.get("HYPERSLOTH_SHUFFLE_MODE", shuffle_mode),
#     #     )

#     # trainer.add_callback(get_callback_shuffle_data(trainer))

#     # if local_rank == 0:
#     #     clock.log_elapsed_time("Sampler patching completed")

#     # return trainer
