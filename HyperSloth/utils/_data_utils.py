import argparse
import json
import time
import fcntl
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger
from speedy_utils.all import jdumps, dump_json_or_pickle, load_by_ext

from HyperSloth import HYPERSLOTH_DATA_DIR

from trl.trainer.sft_trainer import SFTTrainer


@contextmanager
def file_lock(lock_path: str, timeout: int = 3000):
    """
    Cross-platform file locking context manager.

    Args:
        lock_path: Path to the lock file
        timeout: Maximum time to wait for lock acquisition in seconds
    """
    lock_file = None
    try:
        # Create lock file
        lock_file = open(lock_path, "w")

        # Try to acquire lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"Acquired lock: {lock_path}")
                break
            except (OSError, IOError):
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock after {timeout}s: {lock_path}"
                    )
                logger.info(f"Waiting for lock: {lock_path}")
                time.sleep(1)

        # Write process info to lock file
        lock_file.write(f"pid:{os.getpid()}\ntime:{time.time()}\n")
        lock_file.flush()

        yield

    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Remove lock file
                if os.path.exists(lock_path):
                    os.unlink(lock_path)
                logger.info(f"Released lock: {lock_path}")
            except Exception as e:
                logger.warning(f"Error releasing lock {lock_path}: {e}")


mapping_chattemplate = {
    "chatgpt": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "gemma": {
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
    "qwen": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
}


def check_existing_dataset(
    source_dataset: str,
    tokenizer_name: str,
    num_samples: int,
    instruction_part: str,
    response_part: str,
    seed: int = 3407,
    name: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """Return dataset name if already exists in registry or None if not found."""
    registry_path = Path(HYPERSLOTH_DATA_DIR) / "data_config.json"

    if not registry_path.exists():
        return None

    registry = load_by_ext(registry_path)
    # Generate ID for current configuration

    # if kwargs:
    # data.append(kwargs)
    config_id = identify_dataset(
        source_dataset,
        tokenizer_name,
        num_samples,
        instruction_part,
        response_part,
        seed,
        **kwargs,  # Include any additional parameters for uniqueness
    )
    # Check each existing dataset for matching ID
    assert isinstance(
        registry, list
    ), f"Registry must be a list of dataset configurations, got {type(registry)}"
    for config in registry:
        assert isinstance(
            config, dict
        ), f"Each dataset config must be a dict, got {type(config)}"
        if config.get("id") == config_id:
            # Verify both datasets exist
            step1_path = Path(HYPERSLOTH_DATA_DIR) / config.get("path", "")
            tokenized_path = Path(HYPERSLOTH_DATA_DIR) / config.get(
                "path_tokenized", ""
            )

            if not step1_path.exists():
                logger.warning(f"Text dataset not found: {step1_path}")
                return None
            if not tokenized_path.exists():
                logger.warning(f"Tokenized dataset not found: {tokenized_path}")
                return None

            if name is not None and config.get("name") != name:
                # update the
                logger.info(
                    f"Updating existing dataset name from '{config['name']}' to '{name}'"
                )
                config["name"] = name
                dump_json_or_pickle(
                    registry,
                    registry_path,
                )
            return config["name"]
    return None


def identify_dataset(
    source_dataset,
    tokenizer_name,
    num_samples,
    instruction_part,
    response_part,
    seed,
    **kwargs,
):
    from speedy_utils.all import identify

    config_id = identify(
        [
            source_dataset,
            tokenizer_name,
            num_samples,
            instruction_part,
            response_part,
            seed,
            kwargs,  # Include any additional parameters for uniqueness
        ]
    )

    return config_id



def compute_tokenized_dataset(
    dataset_path: str, tokenizer_name: str, output_dir: str = HYPERSLOTH_DATA_DIR, train_on_responses_only: bool = True, instruction_part: Optional[str] = None, response_part: Optional[str] = None
) -> str:
    """
    Compute token indices for the dataset using the specified tokenizer.

    Args:
        dataset_path: Path to the text dataset
        tokenizer_name: Name/path of the tokenizer to use
        output_dir: Directory to save the tokenized dataset

    Returns:
        Path to the saved tokenized dataset
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = Dataset.load_from_disk(dataset_path)

    def tokenize(
        example, processing_class, dataset_text_field, add_special_tokens=True
    ):
        processed = processing_class(
            text=example[dataset_text_field], add_special_tokens=add_special_tokens
        )
        return processed

    processed_dataset = dataset.map(
        tokenize,
        fn_kwargs={
            "processing_class": tokenizer,
            "dataset_text_field": "text",
            "add_special_tokens": True,
        },
        num_proc=32,
    )

    # Create tokenized dataset path
    dataset_name = Path(dataset_path).name
    tokenized_path = Path(output_dir) / f"{dataset_name}_tokenized"

    # Save the processed dataset with token indices
    # if train_on_responses_only:
    #     from HyperSloth.utils.train_on_responses_only import convert_dataset_train_on_responses_only
    #     assert instruction_part is not None, "instruction_part must be provided when train_on_responses_only is True"
    #     assert response_part is not None, "response_part must be provided when train_on_responses_only is True"
    #     processed_dataset = convert_dataset_train_on_responses_only(
    #         dataset=processed_dataset,
    #         tokenizer=tokenizer,
    #         instruction_part=instruction_part,
    #         response_part=response_part,
    #     )

    processed_dataset.save_to_disk(str(tokenized_path))

    logger.info(f"Tokenized dataset saved to: {tokenized_path}")
    return str(tokenized_path)


