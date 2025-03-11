from dataclasses import dataclass
from typing import Literal


@dataclass
class HyperSlothConfig:
    """Configuration for HyperSloth training."""

    file: str = "./data/cod_6k5.json"
    packing: bool = False
    model_name: str = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
    test_ratio: float = 0.05
    max_seq_length: int = 2048
    loss_type: Literal["all", "target_only"] = "target_only"
    grad_dir: str = "/dev/shm/hypersloth"
