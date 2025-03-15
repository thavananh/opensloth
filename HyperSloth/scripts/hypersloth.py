import argparse
from fastcore.all import threaded
from loguru import logger
from transformers.training_args import TrainingArguments
from typing import Union, Dict, Any


def run(
    gpu: int,
    hyper_config,
    train_args: Union[Dict[str, Any], TrainingArguments],
):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from HyperSloth.transformer_trainer_setup import setup_model_and_training
    from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback

    # Convert dictionary to TrainingArguments if needed
    if isinstance(train_args, dict):
        hf_train_args = TrainingArguments(**train_args)
    else:
        hf_train_args = train_args

    trainer = setup_model_and_training(
        gpu=gpu,
        hyper_config=hyper_config,
        hf_train_args=hf_train_args,
    )

    if len(hyper_config.training.gpus) > 1:
        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=hyper_config.grad_dir,
            gpu=gpu,
            gpus=hyper_config.training.gpus,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()


run_in_process = threaded(process=True)(run)

import importlib.util


def load_config_from_path(config_path: str):
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def train(config_file: str, **kwargs):
    import os

    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    config_module = load_config_from_path(config_file)
    import tabulate
    
    # Get configurations from the module
    from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig

    # Use Pydantic models directly or create from dictionaries if needed
    if hasattr(config_module, 'hyper_config_model'):
        hyper_config = config_module.hyper_config_model
    elif hasattr(config_module, 'hyper_config'):
        hyper_config = HyperConfig(**config_module.hyper_config)
    else:
        hyper_config = HyperConfig()
    
    if hasattr(config_module, 'training_config_model'):
        training_config = config_module.training_config_model
    elif hasattr(config_module, 'training_config'):
        training_config = TrainingArgsConfig(**config_module.training_config)
    else:
        training_config = TrainingArgsConfig()
    
    # Display configuration
    combined_config = {**hyper_config.model_dump(), **training_config.model_dump()}
    config_table = tabulate.tabulate(combined_config.items(), headers=["Key", "Value"])
    logger.info("\n" + config_table)

    # Clean up previous runs
    logger.info("Cleaning up previous runs")
    os.system(f"rm -rf {hyper_config.grad_dir}/*")

    # Run training
    if len(hyper_config.training.gpus) > 1:
        for gpu_index in hyper_config.training.gpus:
            logger.debug(f"Running on GPU {gpu_index}")
            run_in_process(
                gpu_index,
                hyper_config=hyper_config,
                train_args=training_config.to_dict(),
            )
    else:
        run(
            gpu=hyper_config.training.gpus[0],
            hyper_config=hyper_config,
            train_args=training_config.to_dict(),
        )


def init_config():
    """Initialize a new configuration file with example content."""
    import os
    
    # Create a minimal example configuration file
    with open("hypersloth_config.py", "w") as f:
        f.write("""# filepath: hypersloth_config.py
from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig, DataConfig

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset="data/cod_1k.json",
    ),
    fast_model_args={
        "model_name": "unsloth/gemma-3-4b-it",
    }
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="model_training_outputs/my_model",
    num_train_epochs=1,
)
""")
    logger.info("Created example configuration in hypersloth_config.py")


def main():
    parser = argparse.ArgumentParser(description="HyperSloth CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("config_file", type=str, help="Path to the config file")

    init_parser = subparsers.add_parser("init", help="Initialize the configuration")

    args = parser.parse_args()

    if args.command == "train":
        train(args.config_file)
    elif args.command == "init":
        init_config()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
