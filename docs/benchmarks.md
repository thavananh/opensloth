# opensloth Performance Benchmarks

## Training Time Comparison (4x RTX 4090)

| Framework    | Training Time | VRAM Peak | Notes |
| ------------ | ------------- | --------- | ----- |
| **opensloth** | **19 minutes** | **6 GB** | 4 GPUs, adaptive batching |
| LlamaFactory | 30 minutes | 21 GB | 4 GPUs, standard approach |
| Unsloth (1X) | ~70 minutes | 6 GB | Single GPU baseline |

## Loss Curves Analysis

The loss scale between Hyper-Sloth and LlamaFactory looks comparable, indicating similar training quality with significantly improved training speed.

| ![Hyper-Sloth Tensorboard](../images/hyper-sloth-tb.png){ width=150 } | ![LlamaFactory Tensorboard](../images/llama-factory-tb.png){ width=150 } |
| ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Hyper-Sloth Tensorboard[^1]                                        | LlamaFactory Tensorboard[^2]                                          |

[^1]: Hyper-Sloth Tensorboard.
[^2]: LlamaFactory Tensorboard.

## Additional Performance Notes

- Training quality remains consistent across frameworks
- Memory efficiency improvements allow for larger batch sizes
- Multi-GPU scaling shows near-linear performance gains
- Sequence packing provides additional 15-30% efficiency gains on conversation datasets
