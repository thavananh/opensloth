# -*- coding: utf-8 -*-
"""Build and save processed datasets for training."""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from datasets import Dataset, load_dataset
from unsloth.chat_templates import standardize_sharegpt


def build_finetome_dataset(
    dataset_name: str = "mlabonne/FineTome-100k",
    num_samples: int = 1000,
    output_dir: str = "./data/built_dataset",
    tokenizer=None,
    seed=3407,
    split: str = "train",
) -> str:
    """
    Build and save FineTome dataset in conversational format.

    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to select
        output_dir: Directory to save processed dataset
        tokenizer: Tokenizer for chat template application

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
    dataset_filename = f"finetome_{num_samples}_samples"
    dataset_path = output_path / dataset_filename
    processed_dataset.save_to_disk(str(dataset_path))

    # Update dataset registry
    registry_path = output_path / "data_config.json"
    registry = []
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)

    dataset_config = {
        "name": dataset_filename,
        "path": str(dataset_path),
        "source_dataset": dataset_name,
        "num_samples": num_samples,
        "text_field": "text",
    }

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

    args = parser.parse_args()

    from unsloth import FastLanguageModel

    # Load tokenizer for dataset processing
    print("Loading tokenizer...")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name="model_store/unsloth/Qwen3-0.6B-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Build dataset with parsed arguments
    dataset_path = build_finetome_dataset(
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        output_dir=args.output_path,
        tokenizer=tokenizer,
        seed=args.seed,
        split=args.split,
    )

    print(f"Dataset built successfully: {dataset_path}")


if __name__ == "__main__":
    main()
