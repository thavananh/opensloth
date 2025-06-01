import pathlib

from datasets import Dataset


def get_chat_dataset(path):
    hypersloth_path = pathlib.Path(__file__).parent.parent
    data_dir = hypersloth_path / "data"
    assert isinstance(path, str), "Path must be a string"
    path = pathlib.Path(path)
    if not path.exists():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    return Dataset.load_from_disk(path)
