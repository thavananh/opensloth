# -*- coding: utf-8 -*-
"""Build and save processed datasets for training."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from loguru import logger
from unsloth.chat_templates import standardize_sharegpt


def build_finetome_dataset(
    dataset_name: str = "mlabonne/FineTome-100k",
    num_samples: int = 1000,
    output_dir: str = "./data/built_dataset",
    tokenizer=None,
    tokenizer_name: Optional[str] = None,
    seed=3407,
    split: str = "train",
    instruction_part="<|im_start|>user\n",  # Qwen chat template
    response_part="<|im_start|>assistant\n",  # Qwen chat template
    custom_name: Optional[str] = None,
) -> str:
    """
    Build and save FineTome dataset in conversational format.

    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to select
        output_dir: Directory to save processed dataset
        tokenizer: Tokenizer for chat template application
        tokenizer_name: Name/path of the tokenizer used
        custom_name: Custom name for the dataset

    Returns:
        Path to saved dataset
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for dataset processing")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {num_samples} samples from {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train").select(range(num_samples))

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
    processed_dataset = processed_dataset.shuffle(seed=3407)

    # Save dataset
    dataset_filename = custom_name or f"finetome_{num_samples}_samples"
    dataset_path = output_path / dataset_filename
    processed_dataset.save_to_disk(str(dataset_path))

    # Update dataset registry - use standardized location
    registry_path = Path("data/data_config.json")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)

    # Store relative path from data directory
    try:
        # Convert both to absolute paths for comparison
        abs_dataset_path = dataset_path.resolve()
        abs_data_path = Path("data").resolve()
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

    print(f"Dataset saved to: {dataset_path}")
    print(f"Registry updated: {registry_path}")

    return str(dataset_path)


def main():
    """Build datasets when run as main script."""
    parser = argparse.ArgumentParser(
        description="Build and save processed datasets for training"
    )
    parser.add_argument(
        "dataset_name", help="HuggingFace dataset name (e.g., mlabonne/FineTome-100k)"
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
        default="./data/built_dataset",
        help="Output path for the dataset",
    )
    parser.add_argument(
        "--tokenizer_name",
        "--tokenizer",
        default="model_store/unsloth/Qwen3-0.6B-bnb-4bit",
        help="Tokenizer name/path to use for processing",
    )
    parser.add_argument(
        "--name",
        help="Custom name for the dataset",
    )

    args = parser.parse_args()

    from transformers import AutoTokenizer

    # Load tokenizer for dataset processing
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Build dataset with parsed arguments
    dataset_path = build_finetome_dataset(
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        output_dir=args.output_path,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer_name,
        seed=args.seed,
        split=args.split,
        custom_name=args.name,
    )

    print(f"Dataset built successfully: {dataset_path}")


if __name__ == "__main__":
    main()
