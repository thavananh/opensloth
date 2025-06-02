import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger

from HyperSloth import HYPERSLOTH_DATA_DIR

from ._data_utils import (
    identify_dataset,
    check_existing_dataset,
    compute_tokenized_dataset,
    mapping_chattemplate,
    file_lock)


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
        tokenized_dataset_path = compute_tokenized_dataset(
            str(dataset_save_path), tokenizer_name, str(output_path), instruction_part=instruction_part, response_part=response_part,
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
            abs_tokenized_path = Path(tokenized_dataset_path).resolve()
            abs_data_path = Path(HYPERSLOTH_DATA_DIR).resolve()
            relative_dataset_path = abs_dataset_path.relative_to(abs_data_path)
            relative_tokenized_path = abs_tokenized_path.relative_to(abs_data_path)
        except ValueError:
            relative_dataset_path = dataset_save_path.resolve()
            relative_tokenized_path = Path(tokenized_dataset_path).resolve()

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

