from pathlib import Path
from typing import Optional

from datasets import Dataset

from HyperSloth import HYPERSLOTH_DATA_DIR
from HyperSloth.hypersloth_config import DataConfig, DataConfigHF, DataConfigShareGPT


def get_chat_dataset(input: str |DataConfigShareGPT | DataConfigHF | DataConfig) -> Dataset:
    """Load a chat dataset from disk."""

    if isinstance(input, DataConfig):
        path_to_text_dataset = Path(input.path_to_text_dataset)
    else:
        raise TypeError(
            "Input must be an instance of DataConfig, DataConfigHF, or DataConfigShareGPT"
        )

    if isinstance(path_to_text_dataset, str):
        path_to_text_dataset = Path(path_to_text_dataset)
    if not path_to_text_dataset.exists():
        path_to_text_dataset = Path(HYPERSLOTH_DATA_DIR) / path_to_text_dataset
        if not path_to_text_dataset.exists():
            raise FileNotFoundError(
                f"Dataset path {path_to_text_dataset} does not exist"
            )
    data = Dataset.load_from_disk(path_to_text_dataset)

    if "text" not in data.column_names:
        raise ValueError("Dataset must contain a 'text' column")

    return data
