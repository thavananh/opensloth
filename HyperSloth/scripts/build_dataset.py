# -*- coding: utf-8 -*-
"""Build and save processed datasets for training."""

import argparse
import json
from pathlib import Path
from typing import Optional
from speedy_utils.all import jdumps
import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger
from HyperSloth import HYPERSLOTH_DATA_DIR
from unsloth.chat_templates import standardize_sharegpt

hypersloth_path = Path(__file__).parents[3]
print(f"Using hypersloth path: {hypersloth_path}")

mapping_chattemplate = {
    "chatgpt": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "gemma": {
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
}


def build_hf_dataset(
    dataset_name: str,
    tokenizer_name: str,
    num_samples: int = 1000,
    output_dir: str = HYPERSLOTH_DATA_DIR,
    seed=3407,
    split: str = "train",
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
    custom_name: Optional[str] = None,
    print_samples: bool = True,
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

    Returns:
        Dataset name for use with DataConfig.from_dataset_name()
    """
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Auto-detect instruction and response parts if set to None
    if instruction_part is None or response_part is None:
        tokenizer_lower = tokenizer_name.lower()
        detected_type = None

        if "gemma" in tokenizer_lower:
            detected_type = "gemma"
        else:
            detected_type = "chatgpt"  # Default to chatgpt if not gemma
        logger.info(
            f"Tokenizer {tokenizer_name} - "
            f"Defaulting to chatgpt template with instruction_part and response_part: "
            f'{mapping_chattemplate["chatgpt"]}'
        )

        if instruction_part is None:
            instruction_part = mapping_chattemplate[detected_type]["instruction_part"]
        if response_part is None:
            response_part = mapping_chattemplate[detected_type]["response_part"]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {num_samples} samples from {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split).select(range(num_samples))

    # Convert dataset to conversational format
    dataset = standardize_sharegpt(dataset)
    conversations = tokenizer.apply_chat_template(
        dataset["conversations"],
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
            if len(text) > 2000:
                print(f"{text[:2000]}...")
                print(f"[Text truncated - full length: {len(text)} chars]")
            else:
                print(text)
        print("=" * 80 + "\n")

    # Save dataset
    dataset_filename = custom_name or f"hf_dataset_{num_samples}_samples"
    dataset_path = output_path / dataset_filename
    processed_dataset.save_to_disk(str(dataset_path))

    # Update dataset registry - use standardized location
    registry_path = Path(HYPERSLOTH_DATA_DIR) / "data_config.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)

    # Store relative path from data directory
    try:
        # Convert both to absolute paths for comparison
        abs_dataset_path = dataset_path.resolve()
        abs_data_path = Path(HYPERSLOTH_DATA_DIR).resolve()
        relative_dataset_path = abs_dataset_path.relative_to(abs_data_path)
    except ValueError:
        # If dataset is not under data directory, store absolute path
        relative_dataset_path = dataset_path.resolve()

    dataset_config = {
        "name": dataset_filename,
        "path": str(relative_dataset_path),
        "source_dataset": dataset_name,
        "num_samples": num_samples,
        "text_field": "text",
        "tokenizer_name": tokenizer_name,
        "instruction_part": instruction_part,
        "response_part": response_part,
    }
    logger.info(f"\n{jdumps(dataset_config)}")

    # Update or add new config
    existing_idx = next(
        (i for i, cfg in enumerate(registry) if cfg["name"] == dataset_filename), None
    )
    if existing_idx is not None:
        registry[existing_idx] = dataset_config
    else:
        registry.append(dataset_config)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Dataset saved to: {dataset_path}")
    print(f"Registry updated: {registry_path}")

    return dataset_filename


def build_sharegpt_dataset(
    dataset_path: str,
    tokenizer_name: str,
    num_samples: Optional[int] = None,
    output_dir: str = HYPERSLOTH_DATA_DIR,
    seed: int = 3407,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
    custom_name: Optional[str] = None,
    print_samples: bool = False,
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

    Returns:
        Dataset name for use with DataConfig.from_dataset_name()
    """
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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
            logger.warning(f"No conversation/messages key found in item: {item.keys()}")
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

    conversations = tokenizer.apply_chat_template(
        dataset["conversations"],
        tokenize=False,
    )

    # Create final dataset
    data_series = pd.Series(conversations)
    data_series.name = "text"
    processed_dataset = Dataset.from_pandas(pd.DataFrame(data_series))
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
    dataset_filename = custom_name or f"sharegpt_{num_samples}_samples"
    dataset_save_path = output_path / dataset_filename
    processed_dataset.save_to_disk(str(dataset_save_path))

    # Update dataset registry
    registry_path = Path(HYPERSLOTH_DATA_DIR) / "data_config.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)

    # Store relative path from data directory
    try:
        abs_dataset_path = dataset_save_path.resolve()
        abs_data_path = Path(HYPERSLOTH_DATA_DIR).resolve()
        relative_dataset_path = abs_dataset_path.relative_to(abs_data_path)
    except ValueError:
        relative_dataset_path = dataset_save_path.resolve()

    dataset_config = {
        "name": dataset_filename,
        "path": str(relative_dataset_path),
        "source_dataset": str(dataset_file.resolve()),
        "num_samples": num_samples,
        "text_field": "text",
        "tokenizer_name": tokenizer_name,
        "instruction_part": instruction_part,
        "response_part": response_part,
    }
    logger.info(f"{dataset_config=}")

    # Update or add new config
    existing_idx = next(
        (i for i, cfg in enumerate(registry) if cfg["name"] == dataset_filename), None
    )
    if existing_idx is not None:
        registry[existing_idx] = dataset_config
    else:
        registry.append(dataset_config)

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Dataset saved to: {dataset_save_path}")
    print(f"Registry updated: {registry_path}")

    return dataset_filename


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
        "--tokenizer",
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

    args = parser.parse_args()

    # Build dataset with parsed arguments
    if args.local_path:
        dataset_name = build_sharegpt_dataset(
            dataset_path=args.local_path,
            tokenizer_name=args.tokenizer_name,
            num_samples=args.num_samples,
            output_dir=args.output_path,
            seed=args.seed,
            custom_name=args.name,
            print_samples=args.print_samples,
        )
    else:
        dataset_name = build_hf_dataset(
            dataset_name=args.hf_dataset,
            tokenizer_name=args.tokenizer_name,
            num_samples=args.num_samples,
            output_dir=args.output_path,
            seed=args.seed,
            split=args.split,
            custom_name=args.name,
            print_samples=args.print_samples,
        )

    # Success message with usage instructions
    success_msg = f"""Dataset "{dataset_name}" has been successfully built and saved!

📁 Registry: {Path(HYPERSLOTH_DATA_DIR) / "data_config.json"}
🚀 Usage in training scripts:

from HyperSloth.hypersloth_config import HyperConfig, DataConfig

hyper_config_model = HyperConfig(
    data=DataConfig.from_dataset_name("{dataset_name}"),
)
"""
    logger.success(success_msg)


if __name__ == "__main__":
    main()
