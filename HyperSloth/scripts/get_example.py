
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
