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
    message_key: str = None,
    chat_template=None,
    **kwargs,
) -> tuple[Dataset, Dataset | None]:
    """
    Load and preprocess a chat dataset.

    Args:
        dataset_name_or_path (str): Path to a JSON file or name of a dataset in the HuggingFace Hub
        split (str): The dataset split to load
        num_samples (int): The number of samples to select from the dataset
        test_ratio (float): Ratio to split into test set (0 means no test set)
        tokenizer (Any): Tokenizer with apply_chat_template method
        message_key (str): Optional custom key for accessing message data

    Returns:
        tuple: (train_dataset, test_dataset) - test_dataset may be None if test_ratio is 0
    """
    # Load dataset based on input type
    if os.path.exists(dataset_name_or_path):
        dataset = Dataset.from_json(dataset_name_or_path)
    else:
        try:
            dataset = load_dataset(dataset_name_or_path, split=split)
            # Check if dataset is empty
            dataset[0]
        except IndexError:
            raise ValueError(
                f"Dataset is empty. Check dataset name and split: {dataset_name_or_path}, split={split}"
            )
        except Exception as e:
            if split is None:
                raise ValueError(
                    f"Failed to load dataset '{dataset_name_or_path}'. You might need to specify a split."
                ) from e
            else:
                raise ValueError(
                    f"Failed to load dataset '{dataset_name_or_path}' with split '{split}': {str(e)}"
                ) from e

    # Show dataset structure for debugging
    example = dataset[0]
    dataset_keys = list(example.keys())
    if num_samples:
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(num_samples))

    # Try to standardize data format
    # dataset = standardize_data_formats(dataset)

    # Apply chat template if tokenizer is provided
    if tokenizer:

        def apply_chat_template(examples):
            # Use custom message key if provided
            if message_key and message_key in examples:
                messages_key = message_key
            # Try to detect the correct key for messages
            elif "messages" in examples:
                messages_key = "messages"
            elif "conversations" in examples:
                messages_key = "conversations"
            else:
                available_keys = list(examples.keys())
                raise ValueError(
                    "Dataset does not contain recognized message keys ('messages' or 'conversations'). "
                    f"Available keys: {available_keys}. "
                    "Use the message_key parameter to specify the correct key. "
                    "Expected chat format example: (mlabonne/FineTome-100k) https://huggingface.co/datasets/mlabonne/FineTome-100k"
                )
            if chat_template:
                from transformers import AutoTokenizer

                tokenizer.chat_template = AutoTokenizer.from_pretrained(
                    chat_template
                ).chat_template

            texts = tokenizer.apply_chat_template(
                examples[messages_key], tokenize=False
            )
            return {"text": texts}

        try:
            dataset = dataset.map(apply_chat_template, batched=True)
        except Exception as e:
            # Provide helpful error with dataset structure information
            raise ValueError(
                f"Failed to apply chat template: {str(e)}\n"
                f"Dataset structure: Keys in first example = {dataset_keys}\n"
                f"Make sure the dataset has conversation data in a recognized format."
            ) from e

    # Sample dataset if requested

    # Create train/test split if needed
    if test_ratio > 0:
        ds = dataset.train_test_split(test_size=test_ratio, shuffle=True, seed=42)
        ds_train, ds_test = ds["train"], ds["test"]
    else:
        ds_train, ds_test = dataset, None

    return ds_train, ds_test
