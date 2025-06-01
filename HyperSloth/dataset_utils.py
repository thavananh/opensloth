from datasets import Dataset
from HyperSloth import HYPERSLOTH_DATA_DIR
from pathlib import Path


def get_chat_dataset(path: str | Path) -> Dataset:
    """Load a chat dataset from disk."""
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        # try to load from the HyperSloth data directory
        path = Path(HYPERSLOTH_DATA_DIR) / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset path {path} does not exist")
    data = Dataset.load_from_disk(path)

    if "text" not in data.column_names:
        raise ValueError("Dataset must contain a 'text' column")

    return data
