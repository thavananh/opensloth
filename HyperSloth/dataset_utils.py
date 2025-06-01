import pathlib
from datasets import Dataset


def get_chat_dataset(path):
    assert isinstance(path, str), "Path must be a string"
    path = pathlib.Path(path).expanduser().resolve()
    return Dataset.load_from_disk(path)
