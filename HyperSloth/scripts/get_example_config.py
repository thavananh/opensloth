from fastcore.all import *
from loguru import logger


def main():
    out_path = "./example_training_config.py"
    path = Path(__file__).parent.parent / "example_training_config.py"
    import os

    os.system(f"cp {path} {out_path}")
    print(f"Example training config: {out_path}")
    logger.info(f"Edit/Run it with: hypersloth-train {out_path}")
