from datasets import Dataset
from HyperSloth import HYPERSLOTH_DATA_DIR
from pathlib import Path
from typing import Optional


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


def get_sharegpt_dataset(
    dataset_path: str,
    tokenizer_name: str,
    num_samples: Optional[int] = None,
    seed: int = 3407,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
    print_samples: bool = False,
) -> Dataset:
    from HyperSloth.scripts.build_dataset import build_sharegpt_dataset

    share_gpt_path = build_sharegpt_dataset(
        dataset_path=dataset_path,
        tokenizer_name=tokenizer_name,
        num_samples=num_samples,
        seed=seed,
        instruction_part=instruction_part,
        response_part=response_part,
        print_samples=print_samples,
    )
    return get_chat_dataset(share_gpt_path)
