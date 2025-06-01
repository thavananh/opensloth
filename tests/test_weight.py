import torch
from typing import Dict, Tuple
from safetensors.torch import load_file
from loguru import logger


def ensure_close_weight(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
) -> None:
    """Ensure that the weights in two dictionaries are close within a tolerance."""
    for key in dict1.keys():
        assert torch.allclose(dict1[key], dict2[key], atol=1e-6)
        distance = torch.dist(dict1[key], dict2[key]) # type: ignore


def compare_weights(
    output_dir: str, gpu_ids: Tuple[int, ...], steps: Tuple[int, ...]
) -> None:
    """Compare weights from different checkpoints in the specified output directory."""
    for step in steps:
        path1 = f"{output_dir}/{gpu_ids[0]}/checkpoint-{step}/adapter_model.safetensors"
        path2 = f"{output_dir}/{gpu_ids[1]}/checkpoint-{step}/adapter_model.safetensors"

        d1 = load_file(path1)
        d2 = load_file(path2)
        ensure_close_weight(d1, d2)


# Example usage
output_dir = "model_training_outputs/debug/"
gpu_ids = (0, 1)
steps = (20, 30)
compare_weights(output_dir, gpu_ids, steps)

logger.info("Success!")
