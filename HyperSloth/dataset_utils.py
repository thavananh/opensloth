from typing import Any

from datasets import Dataset

import warnings
import os
from datasets import load_dataset
from typing import Any

from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from datasets import Dataset


warnings.filterwarnings("ignore", category=UserWarning)


def get_chat_dataset(
    dataset_name_or_path: str,
    split: str = None,
    num_samples: int = None,
    test_ratio: float = 0,
    tokenizer: Any = None,
    **kwargs,
) -> Any:
    """
    Load and preprocess the dataset.

    Args:
        dataset_name (str): The name or json path
        split (str): The dataset split to load.
        num_samples (int): The number of samples to select from the dataset.

    Returns:
        Any: The preprocessed dataset.
    """

    if os.path.exists(dataset_name_or_path):
        dataset = Dataset.from_json(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, split=split)
        try:
            first_item = dataset[0]
        except:
            raise ValueError(f"You might forget to set split? {split=}")

    dataset = standardize_data_formats(dataset)

    def apply_chat_template(examples):
        messages_key = "messages" if "messages" in examples else "conversations"
        assert messages_key in examples, f"Dataset does not have {messages_key} key"
        texts = tokenizer.apply_chat_template(examples[messages_key], tokenize=False)
        return {"text": texts}

    if num_samples:
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(num_samples))
    if tokenizer:
        dataset = dataset.map(apply_chat_template, batched=True)

    if test_ratio > 0:
        ds = dataset.train_test_split(test_size=test_ratio, shuffle=True, seed=42)
        ds_train, ds_test = ds["train"], ds["test"]
    else:
        ds_train, ds_test = dataset, None
    return ds_train, ds_test
