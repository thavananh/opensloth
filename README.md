<p align="center">
    <img src="images/opensloth.png" alt="opensloth Logo" width="200" />
</p>

# OpenSloth

A multi-GPU training framework that combines [Unsloth](https://github.com/unslothai/unsloth) with multi-GPU support and [sequence packing](https://huggingface.co/blog/sirluk/llm-sequence-packing) optimizations.

**Core Components:**
- **Unsloth**: 2x faster training with 75% VRAM savings
- **Multi-GPU**: NCCL-based distributed training across multiple GPUs  
- **Sequence Packing**: Smart batching that reduces padding waste by up to 40%

**The Result:** Unsloth's efficiency × GPU count × sequence packing optimizations = speedups that often exceed theoretical maximums.

## 💾 Installation

```bash
conda create --name opensloth_env python=3.11
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth xformers opensloth
# or from source pip install git+https://github.com/anhvth/opensloth.git
```

## ⚡ Quickstart

```bash
# Basic multi-GPU training
python scripts/train.py
```

| Example | Description | Link/Command |
|---------|-------------|--------------|
| **Kaggle Notebook (T4x2)** | Live training example on Kaggle's dual T4 GPU environment | [🔗 Qwen3 OpenSloth 2GPUs](https://www.kaggle.com/code/anhvth226/qwen3-opensloth-2gpus?scriptVersionId=244825301) |
| **Local Training Script** | Check out the training script for configuration examples | `python scripts/train.py` |
| **Local Jupyter Notebook** | Interactive training notebook for local development | [`notebooks/train.ipynb`](notebooks/train.ipynb) |

## ⚡ Performance Benchmarks

**[📊 View Full WandB Comparison](https://wandb.ai/anhvth/CompareUnsloth)**

### opensloth vs Unsloth Direct Comparison

Controlled comparison with identical configurations:
- **Model**: Qwen3-8B-bnb-4bit  
- **Training Steps**: 100 steps
- **Global Batch Size**: 32

**Results:**
- **opensloth (2 GPUs)**: 8m 28s ⚡
- **Unsloth (1 GPU)**: 19m 34s
- **Performance Gain**: ~2.3x faster 

**Why 2.3x Speedup on 2 GPUs?**

OpenSloth achieves **2.3x speedup** through three optimizations:
- ✅ **Sequence packing**: Smart batching reduces padding waste ([learn more](https://huggingface.co/blog/sirluk/llm-sequence-packing))
- ✅ **Multi-GPU scaling**: NCCL-based distributed training
- ✅ **Load balancing**: Even workload distribution across GPUs

**Scaling Expectations:**
- **2 GPUs**: ~2.3x faster than single GPU
- **4 GPUs**: ~4.6x faster than single GPU
- **8 GPUs**: ~9.2x faster than single GPU

## 🏗 How It Works

OpenSloth combines three proven techniques:

### 1. Unsloth Foundation
- 2x faster training through optimized kernels
- 75% VRAM savings through memory efficiency
- Quality preservation with performance gains

### 2. Multi-GPU Distribution
- **PyTorch DDP**: Standard distributed data parallel training
- **NCCL backend**: Efficient gradient synchronization
- **Process spawning**: One process per GPU for optimal scaling

### 3. Sequence Packing
Based on the [Hugging Face sequence packing approach](https://huggingface.co/blog/sirluk/llm-sequence-packing):
- **Length sorting**: Groups sequences by similar lengths
- **Smart batching**: Minimizes padding tokens within batches
- **Round-robin distribution**: Balances workload across GPUs
- **Up to 40% token savings**: Reduces computational waste

## 🔧 Quick Tips

**For faster iteration:**
- Start with smaller models: `unsloth/Qwen3-0.6b-bnb-4bit`
- Test single GPU first: modify `gpus=[0]` in script
- Use fewer samples for quick testing

**Recommended Configuration:**
```python
# Optimize for sequence packing and multi-GPU efficiency
TrainingConfig(
    per_device_train_batch_size=4,      # Larger batches per GPU
    gradient_accumulation_steps=8,       # Fewer gradient sync operations
    # Effective batch size = 4 * 8 * num_gpus
)
```

## 🔧 Troubleshooting

   **Single GPU Testing:**
   ```python
   # In your training script, change:
   gpus = [0]  # Use only first GPU for debugging
   ```
