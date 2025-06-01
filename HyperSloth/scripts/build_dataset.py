# -*- coding: utf-8 -*-
"""Build and save processed datasets for training."""

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


def build_hf_dataset(
    dataset_name: str,
    tokenizer_name: str,
    num_samples: int = 1000,
    output_dir: str = HYPERSLOTH_DATA_DIR,
    seed=3407,
    split: str = "train",
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
    name: Optional[str] = None,
    print_samples: bool = True,
    use_cache: bool = True,
    columns: Optional[str] = [
        "conversations",
        "messages",
    ],  # Default columns for conversational datasets
) -> str:
    """
    Build and save HuggingFace dataset in conversational format.

    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to select
        output_dir: Directory to save processed dataset
        tokenizer_name: Name/path of the tokenizer to use
        instruction_part: Instruction part template or None for auto-detection
        response_part: Response part template or None for auto-detection
        custom_name: Custom name for the dataset
        use_cache: Whether to check for existing datasets with same config

    Returns:
        Dataset name for use with DataConfig.from_dataset_name()
    """
    from transformers import AutoTokenizer

    # Load tokenizer
    # Auto-detect instruction and response parts if set to None
    if instruction_part is None or response_part is None:
        tokenizer_lower = tokenizer_name.lower()
        detected_type = None

        if "qwen" in tokenizer_lower:
            detected_type = "qwen"
        elif "gemma" in tokenizer_lower:
            detected_type = "gemma"
        else:
            detected_type = "chatgpt"  # Default to chatgpt if not qwen/gemma

        logger.info(
            f"Tokenizer {tokenizer_name} - "
            f"Defaulting to {detected_type} template with instruction_part and response_part: "
            f"{mapping_chattemplate[detected_type]}"
        )

        if instruction_part is None:
            instruction_part = mapping_chattemplate[detected_type]["instruction_part"]
        if response_part is None:
            response_part = mapping_chattemplate[detected_type]["response_part"]

    # Generate dataset ID and use as name if no name provided
    if name is None:
        name = identify_dataset(
            source_dataset=dataset_name,
            tokenizer_name=tokenizer_name,
            num_samples=num_samples,
            instruction_part=instruction_part,
            response_part=response_part,
            seed=seed,
        )

    # Check for existing dataset with same configuration
    if use_cache:
        existing_dataset = check_existing_dataset(
            source_dataset=dataset_name,
            tokenizer_name=tokenizer_name,
            num_samples=num_samples,
            instruction_part=instruction_part,
            response_part=response_part,
            seed=seed,
            name=name,
        )
        if existing_dataset:
            logger.success('Use existing dataset: "{}"'.format(existing_dataset))
            return existing_dataset

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create lock file path based on dataset name
    lock_path = output_path / f"{name}.lock"

    # Use file lock to prevent race conditions
    with file_lock(str(lock_path)):
        # Double-check cache after acquiring lock
        if use_cache:
            existing_dataset = check_existing_dataset(
                source_dataset=dataset_name,
                tokenizer_name=tokenizer_name,
                num_samples=num_samples,
                instruction_part=instruction_part,
                response_part=response_part,
                seed=seed,
                name=name,
            )
            if existing_dataset:
                logger.success(
                    'Use existing dataset (found after lock): "{}"'.format(
                        existing_dataset
                    )
                )
                return existing_dataset

        # Load dataset
        print(f"Loading {num_samples} samples from {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split).select(range(num_samples))

        # Convert dataset to conversational format
        from unsloth.chat_templates import standardize_sharegpt

        dataset = standardize_sharegpt(dataset)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # detect column
        for column in columns:
            if column in dataset.column_names:
                break
        else:
            raise ValueError(
                f"None of the specified columns {columns} found in dataset. "
                f"Available columns: {dataset.column_names}"
            )
        conversations = tokenizer.apply_chat_template(
            dataset[column],
            tokenize=False,
        )

        # Create final dataset
        data = pd.Series(conversations)
        data.name = "text"
        processed_dataset = Dataset.from_pandas(pd.DataFrame(data))
        processed_dataset = processed_dataset.shuffle(seed=seed)

        # Print samples if requested
        if print_samples:
            print("\n" + "=" * 80)
            print("SAMPLE TEXTS FROM PROCESSED DATASET:")
            print("=" * 80)
            sample_count = min(3, len(processed_dataset))
            for i in range(sample_count):
                print(f"\n--- Sample {i+1} ---")
                text = processed_dataset[i]["text"]
                # Truncate very long texts for readability
                if len(text) > 555:
                    print(f"{text[:555]}...")
                    print(f"[Text truncated - full length: {len(text)} chars]")
                else:
                    print(text)
            print("=" * 80 + "\n")

        # Save dataset
        dataset_filename = name
        dataset_path = output_path / dataset_filename
        processed_dataset.save_to_disk(str(dataset_path))

        # Create tokenized dataset with token indices
        tokenized_path = compute_tokenized_dataset(
            str(dataset_path), tokenizer_name, str(output_path)
        )

        # Update dataset registry - use standardized location
        registry_path = Path(HYPERSLOTH_DATA_DIR) / "data_config.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry = []
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)

        # Store relative path from data directory for both datasets
        try:
            # Convert both to absolute paths for comparison
            abs_dataset_path = dataset_path.resolve()
            abs_tokenized_path = Path(tokenized_path).resolve()
            abs_data_path = Path(HYPERSLOTH_DATA_DIR).resolve()
            relative_dataset_path = abs_dataset_path.relative_to(abs_data_path)
            relative_tokenized_path = abs_tokenized_path.relative_to(abs_data_path)
        except ValueError:
            # If dataset is not under data directory, store absolute path
            relative_dataset_path = dataset_path.resolve()
            relative_tokenized_path = Path(tokenized_path).resolve()

        dataset_config = {
            "name": dataset_filename,
            "path": str(relative_dataset_path),
            "path_tokenized": str(relative_tokenized_path),
            "source_dataset": dataset_name,
            "num_samples": num_samples,
            "text_field": "text",
            "tokenizer_name": tokenizer_name,
            "instruction_part": instruction_part,
            "response_part": response_part,
            "id": identify_dataset(
                source_dataset=dataset_name,
                tokenizer_name=tokenizer_name,
                num_samples=num_samples,
                instruction_part=instruction_part,
                response_part=response_part,
                seed=seed,
            ),
        }
        logger.info(f"\n{jdumps(dataset_config)}")

        # Update or add new config
        existing_idx = next(
            (
                i
                for i, cfg in enumerate(registry)
                if cfg.get("id") == dataset_config["id"]
            ),
            None,
        )
        if existing_idx is not None:
            # Update existing config and potentially the name
            old_name = registry[existing_idx]["name"]
            registry[existing_idx] = dataset_config
            if old_name != dataset_filename:
                logger.info(
                    f"Updated dataset name from '{old_name}' to '{dataset_filename}' for same configuration"
                )
        else:
            registry.append(dataset_config)

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    return dataset_filename


def build_sharegpt_dataset(
    dataset_path: str,
    tokenizer_name: str,
    name: Optional[str] = None,
    num_samples: Optional[int] = None,
    output_dir: str = HYPERSLOTH_DATA_DIR,
    seed: int = 3407,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
    print_samples: bool = False,
    use_cache: bool = True,
) -> str:
    """
    Build and save ShareGPT dataset from local file in conversational format.

    Args:
        dataset_path: Path to local ShareGPT format file (JSON/JSONL)
        tokenizer_name: Name/path of the tokenizer to use
        num_samples: Number of samples to select (None for all)
        output_dir: Directory to save processed dataset
        seed: Random seed for shuffling
        instruction_part: Instruction part template or None for auto-detection
        response_part: Response part template or None for auto-detection
        custom_name: Custom name for the dataset
        use_cache: Whether to check for existing datasets with same config

    Returns:
        Dataset name for use with DataConfig.from_dataset_name()
    """
    assert name is None or isinstance(name, str), "custom_name must be a string or None"
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Auto-detect instruction and response parts if set to None
    if instruction_part is None or response_part is None:
        tokenizer_lower = tokenizer_name.lower()
        detected_type = None

        if "qwen" in tokenizer_lower:
            detected_type = "qwen"
        elif "gemma" in tokenizer_lower:
            detected_type = "gemma"
        else:
            raise ValueError(
                f"Cannot auto-detect template type for tokenizer: {tokenizer_name}. "
                f"Supported types: {list(mapping_chattemplate.keys())}"
            )

        if instruction_part is None:
            instruction_part = mapping_chattemplate[detected_type]["instruction_part"]
        if response_part is None:
            response_part = mapping_chattemplate[detected_type]["response_part"]

    # Get absolute path for consistent comparison
    abs_dataset_path = str(Path(dataset_path).resolve())

    # Determine actual num_samples for cache check
    if num_samples is None:
        # Need to load and count samples for cache key
        dataset_file = Path(dataset_path)
        if dataset_file.suffix == ".jsonl":
            with open(dataset_file, "r", encoding="utf-8") as f:
                actual_num_samples = sum(1 for _ in f)
        else:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                actual_num_samples = len(data)
    else:
        actual_num_samples = num_samples

    # Generate dataset ID and use as name if no name provided
    if name is None:
        name = identify_dataset(
            source_dataset=abs_dataset_path,
            tokenizer_name=tokenizer_name,
            num_samples=actual_num_samples,
            instruction_part=instruction_part,
            response_part=response_part,
            seed=seed,
        )

    # Check for existing dataset with same configuration
    if use_cache:
        existing_dataset = check_existing_dataset(
            source_dataset=abs_dataset_path,
            tokenizer_name=tokenizer_name,
            num_samples=actual_num_samples,
            instruction_part=instruction_part,
            response_part=response_part,
            seed=seed,
            name=name,
        )
        if existing_dataset:
            logger.success('Use existing dataset: "{}"'.format(existing_dataset))
            return existing_dataset

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create lock file path based on dataset name
    lock_path = output_path / f"{name}.lock"

    # Use file lock to prevent race conditions
    with file_lock(str(lock_path)):
        # Double-check cache after acquiring lock
        if use_cache:
            existing_dataset = check_existing_dataset(
                source_dataset=abs_dataset_path,
                tokenizer_name=tokenizer_name,
                num_samples=actual_num_samples,
                instruction_part=instruction_part,
                response_part=response_part,
                seed=seed,
                name=name,
            )
            if existing_dataset:
                logger.success(
                    'Use existing dataset (found after lock): "{}"'.format(
                        existing_dataset
                    )
                )
                return existing_dataset

        # Load local dataset
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        print(f"Loading dataset from {dataset_path}...")

        # Load JSON/JSONL file
        if dataset_file.suffix == ".jsonl":
            data = []
            with open(dataset_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Normalize conversation format
        normalized_conversations = []
        for item in data:
            # Handle both 'conversation'/'conversations' and 'messages' keys
            conversation = None
            if "conversation" in item:
                conversation = item["conversation"]
            elif "conversations" in item:
                conversation = item["conversations"]
            elif "messages" in item:
                conversation = item["messages"]
            else:
                logger.warning(
                    f"No conversation/messages key found in item: {item.keys()}"
                )
                continue

            normalized_conversations.append(conversation)

        # Select samples if specified
        if num_samples is not None and num_samples < len(normalized_conversations):
            print(
                f"Selecting {num_samples} samples from {len(normalized_conversations)}..."
            )
            normalized_conversations = normalized_conversations[:num_samples]
        else:
            num_samples = len(normalized_conversations)

        # Create dataset and apply chat template
        dataset_dict = {"conversations": normalized_conversations}
        dataset = Dataset.from_dict(dataset_dict)

        # conversations = tokenizer.apply_chat_template(
        #     dataset["conversations"],
        #     tokenize=False,
        # )

        # Create final dataset
        # data_series = pd.Series(conversations)
        # data_series.name = "text"
        # processed_dataset = Dataset.from_pandas(pd.DataFrame(data_series))
        processed_dataset = dataset.map(
            lambda x: {
                "text": tokenizer.apply_chat_template(
                    x["conversations"], tokenize=False
                )
            },
            batched=True,
            num_proc=32,
        )
        processed_dataset = processed_dataset.shuffle(seed=seed)

        # Print samples if requested
        if print_samples:
            print("\n" + "=" * 80)
            print("SAMPLE TEXTS FROM PROCESSED DATASET:")
            print("=" * 80)
            sample_count = min(3, len(processed_dataset))
            for i in range(sample_count):
                print(f"\n--- Sample {i+1} ---")
                text = processed_dataset[i]["text"]
                # Truncate very long texts for readability
                if len(text) > 500:
                    print(f"{text[:500]}...")
                    print(f"[Text truncated - full length: {len(text)} chars]")
                else:
                    print(text)
            print("=" * 80 + "\n")

        # Save dataset
        dataset_filename = name
        dataset_save_path = output_path / dataset_filename
        processed_dataset.save_to_disk(str(dataset_save_path))

        # Create tokenized dataset with token indices
        tokenized_path = compute_tokenized_dataset(
            str(dataset_save_path), tokenizer_name, str(output_path)
        )

        # Update dataset registry
        registry_path = Path(HYPERSLOTH_DATA_DIR) / "data_config.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry = []
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)

        # Store relative path from data directory for both datasets
        try:
            abs_dataset_path = dataset_save_path.resolve()
            abs_tokenized_path = Path(tokenized_path).resolve()
            abs_data_path = Path(HYPERSLOTH_DATA_DIR).resolve()
            relative_dataset_path = abs_dataset_path.relative_to(abs_data_path)
            relative_tokenized_path = abs_tokenized_path.relative_to(abs_data_path)
        except ValueError:
            relative_dataset_path = dataset_save_path.resolve()
            relative_tokenized_path = Path(tokenized_path).resolve()

        dataset_config = {
            "name": dataset_filename,
            "path": str(relative_dataset_path),
            "path_tokenized": str(relative_tokenized_path),
            "source_dataset": str(dataset_file.resolve()),
            "num_samples": num_samples,
            "text_field": "text",
            "tokenizer_name": tokenizer_name,
            "instruction_part": instruction_part,
            "response_part": response_part,
            "id": identify_dataset(
                source_dataset=str(dataset_file.resolve()),
                tokenizer_name=tokenizer_name,
                num_samples=actual_num_samples,
                instruction_part=instruction_part,
                response_part=response_part,
                seed=seed,
            ),
        }
        logger.info(f"{dataset_config=}")

        # Update or add new config
        existing_idx = next(
            (
                i
                for i, cfg in enumerate(registry)
                if cfg.get("id") == dataset_config["id"]
            ),
            None,
        )
        if existing_idx is not None:
            # Update existing config and potentially the name
            old_name = registry[existing_idx]["name"]
            registry[existing_idx] = dataset_config
            if old_name != dataset_filename:
                logger.info(
                    f"Updated dataset name from '{old_name}' to '{dataset_filename}' for same configuration"
                )
        else:
            registry.append(dataset_config)

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    return dataset_filename


def compute_tokenized_dataset(
    dataset_path: str, tokenizer_name: str, output_dir: str = HYPERSLOTH_DATA_DIR
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
    processed_dataset.save_to_disk(str(tokenized_path))

    logger.info(f"Tokenized dataset saved to: {tokenized_path}")
    return str(tokenized_path)


def main():
    """Build datasets when run as main script."""
    parser = argparse.ArgumentParser(
        description="Build and save processed datasets for training"
    )
    parser.add_argument(
        "--hf_dataset", help="HuggingFace dataset name (e.g., mlabonne/FineTome-100k)"
    )
    parser.add_argument(
        "--local_path",
        help="Path to local ShareGPT format file (alternative to dataset_name)",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--seed", type=int, default=3407, help="Random seed for shuffling"
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument(
        "--output_path",
        default=HYPERSLOTH_DATA_DIR,
        help="Output path for the dataset",
    )
    parser.add_argument(
        "--tokenizer_name",
        required=True,
        help="Tokenizer name/path to use for processing",
    )
    parser.add_argument(
        "--name",
        help="Custom name for the dataset",
    )
    parser.add_argument(
        "--print_samples",
        action="store_true",
        help="Print sample texts from the processed dataset",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching and force rebuild dataset",
    )
    parser.add_argument(
        "--columns",
        default="conversations,messages",
        help="Comma-separated list of columns to use for conversational datasets",
    )

    args = parser.parse_args()

    # Build dataset with parsed arguments
    if args.local_path:
        dataset_name = build_sharegpt_dataset(
            dataset_path=args.local_path,
            tokenizer_name=args.tokenizer_name,
            num_samples=args.num_samples,
            output_dir=args.output_path,
            seed=args.seed,
            name=args.name,
            print_samples=args.print_samples,
            use_cache=not args.no_cache,
        )
    else:
        dataset_name = build_hf_dataset(
            dataset_name=args.hf_dataset,
            tokenizer_name=args.tokenizer_name,
            num_samples=args.num_samples,
            output_dir=args.output_path,
            seed=args.seed,
            split=args.split,
            name=args.name,
            print_samples=args.print_samples,
            use_cache=not args.no_cache,
            columns=args.columns.split(",") if args.columns else None,
        )

    # Success message with usage instructions
    success_msg = f"""Dataset "{dataset_name}" is ready! 🎉

📁 Registry: {Path(HYPERSLOTH_DATA_DIR) / "data_config.json"}
📊 Two datasets created:
   • Text: Conversational text format
   • Tokenized: Tokenized format with indices
🚀 Usage in training scripts:

```python
from HyperSloth.hypersloth_config import HyperConfig, DataConfig

hyper_config_model = HyperConfig(
    data=DataConfig.from_dataset_name("{dataset_name}"),
)
...
```
"""
    logger.success(success_msg)
    logger.info(f'Config path: {Path(HYPERSLOTH_DATA_DIR) / "data_config.json"}')


if __name__ == "__main__":
    main()
