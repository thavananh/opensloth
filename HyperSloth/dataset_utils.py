from datasets import Dataset
from HyperSloth import HYPERSLOTH_DATA_DIR

# from pathlib import Path


def get_chat_dataset(path) -> Dataset:
    # hypersloth_path = pathlib.Path(__file__).parent.parent

    data_dir = HYPERSLOTH_DATA_DIR
    assert isinstance(path, str), "Path must be a string"
    if not path.exists():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    data = Dataset.load_from_disk(path)
    assert "text" in data.column_names, "Dataset must contain a 'text' column"
    return data
