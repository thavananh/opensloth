import json
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger
from speedy_utils.all import jdumps

from HyperSloth import HYPERSLOTH_DATA_DIR

from ._data_utils import (
    identify_dataset,
    check_existing_dataset,
    compute_tokenized_dataset,
    mapping_chattemplate,
    file_lock)

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
            str(dataset_path), tokenizer_name, str(output_path), instruction_part=instruction_part, response_part=response_part,
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

