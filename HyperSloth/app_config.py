from dataclasses import dataclass
from typing import Literal
from dataclasses import field


@dataclass
class HyperSlothConfig:
    """Configuration for HyperSloth training."""

    dataset_file: str = "./data/cod_6k5.json"
    packing: bool = False
    model_name: str = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
    test_ratio: float = 0.05
    max_seq_length: int = 2048
    loss_type: Literal["all", "target_only"] = "target_only"
    grad_dir: str  = "/dev/shm/hypersloth"
    gpus: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
