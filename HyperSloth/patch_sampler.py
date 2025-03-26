def patch_sampler(trainer):
    from fastcore.all import patch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer

    @patch
    def _get_train_sampler(self: Trainer) -> SequentialSampler:
        """Get a sequential sampler for the training dataset."""
        return SequentialSampler(self.train_dataset)

    return trainer