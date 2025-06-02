# -*- coding: utf-8 -*-
"""Build and save processed datasets for training."""

import argparse
from pathlib import Path

from loguru import logger

from HyperSloth import HYPERSLOTH_DATA_DIR


from HyperSloth.utils.build_hf_dataset import build_hf_dataset
from HyperSloth.utils.build_sharegpt_dataset import build_sharegpt_dataset


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
