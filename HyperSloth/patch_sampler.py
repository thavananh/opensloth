import os
import random
from datasets import Dataset
from fastcore.all import patch
from loguru import logger
from torch.utils.data import SequentialSampler
from transformers import Trainer


def reorder_and_shuffle_data(
    dataset: Dataset,
    per_device_train_batch_size: int,
    epoch=0,
    seed=42,
):

    lens = [len(x["input_ids"]) for x in dataset]
    sorted_ids = sorted(range(len(lens)), key=lambda k: lens[k])
    dataset = dataset.select(sorted_ids)

    from fastcore.all import chunked

    chunked_lens = list(
        chunked(
            range(len(lens)),
            per_device_train_batch_size,
        )
    )
    random.Random(seed + epoch).shuffle(chunked_lens)  # the 8 continous value are similar
    ids = [idx for chunk in chunked_lens for idx in chunk]
    dataset = dataset.select(ids)
    lens = [len(x["input_ids"]) for x in dataset]

    gpu_ith = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    with open(f"lengths_{gpu_ith}.txt", "w") as f:
        # jsut write all
        f.write("|".join([str(x) for x in lens]) + "\n")
    return dataset


# callback to shuffle data on_epoch_begin
def get_callback_shuffle_data():
    from transformers import TrainerCallback

    class ShuffleData(TrainerCallback):
        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            logger.info("[on_epoch_begin] Shuffling data")
            
            local_rank = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
            # Debug info for the main GPU
            train_dataloader.dataset = reorder_and_shuffle_data(
                train_dataloader.dataset,
                args.per_device_train_batch_size,
                epoch=state.epoch,
                seed=args.seed,
            )
            
            if local_rank == 0:
                try:
                    from ._debug_dataloader import _debug_dataloader
                    tok = kwargs['processing_class']
                    _debug_dataloader(train_dataloader, tokenizer=tok)
                except:
                    logger.exception("Failed to debug dataloader this is not a problem")
                    pass

    return ShuffleData()


def patch_sampler(trainer:Trainer):

    @patch
    def _get_train_sampler(self: Trainer) -> SequentialSampler:
        """Get a sequential sampler for the training dataset."""
        return SequentialSampler(self.train_dataset)

    trainer.add_callback(get_callback_shuffle_data())

    return trainer
