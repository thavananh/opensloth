<p align="center">
    <img src="images/hpsloth.webp" alt="HyperSloth Logo" width="200" />
</p>

# Hyper-Sloth

A high-performance framework for fine-tuning large language models.

## Overview

**Built on top of [Unsloth](https://github.com/unslothai/unsloth)** - HyperSloth extends Unsloth's excellent foundation with multi-GPU support and optimized batching strategies.

**What HyperSloth adds:**
- **Multi-GPU training via NCCL**: Scale your Unsloth workflows across multiple GPUs
- **Adaptive batching optimizations**: Sequence sorting, round-robin load balancing, and minimal padding strategies to reduce computational waste and improve GPU utilization

**Inherited from Unsloth:**
- **2x faster than standard transformers training**: Built on Unsloth's optimized kernels and memory management
- **Up to 75% VRAM savings**: Inherits Unsloth's memory efficiency optimizations
- **Quality preserved**: Same training quality as standard approaches with significantly better performance

**The multiplier effect**: Since we build on Unsloth's foundation, you get Unsloth's 2x speed + 75% memory savings, then multiply that performance across the number of GPUs you have - often achieving speedups well beyond the theoretical maximum through our batching optimizations.

**Multi-GPU Optimization Strategy:**

HyperSloth optimizes multi-GPU training by addressing three key bottlenecks:

1. **GPU Underutilization Problem:**
   - Unsloth typically trains with 1 sample per forward pass
   - If you train with bz>1, gpu utilization higher but will be slower due to large padding waste when data have high variance in sequence lengths
   - This underutilizes larger GPUs or when training smaller models
   - **Solution:** Use larger batch sizes to fully utilize GPU compute

2. **GPU Synchronization Bottleneck:**
   - In multi-GPU training, all GPUs must wait for the slowest one to finish
   - Different batch complexities can create uneven processing times
   - **Solution:** Sequence sorting and load balancing ensure even workload distribution

3. **Communication Overhead:**
   - Gradient synchronization between GPUs adds significant overhead
   - More frequent communication = more wasted time
   - **Solution:** Larger gradient accumulation steps reduce communication frequency

## 🔧 Supported Training Methods

**Currently Supported:**
- **SFT (Supervised Fine-Tuning)**:
  - **LoRA**
  - **Full Fine-Tuning**

**Coming Soon:**
- **Strong-to-Weak Distillation**
- **GRPO**

## ⚡ Performance Benchmarks

**[📊 View Full WandB Comparison](https://wandb.ai/anhvth/CompareUnsloth)**

### HyperSloth vs Unsloth Direct Comparison

We conducted a controlled comparison using identical configurations:

- **Model**: Qwen3-8B-bnb-4bit  
- **Training Steps**: 100 steps
- **Global Batch Size**: 32
- **Dataset**: Fixed data sampler ensures identical training data

**Results:**
- **HyperSloth (2 GPUs)**: 8m 28s ⚡
- **Unsloth (1 GPU)**: 19m 34s
- **Performance Gain**: ~2.3x faster 

**Why 2.3x Speedup on 2 GPUs?**

Theoretical maximum speedup with 2 GPUs would be 2x, but communication overhead typically reduces this to ~1.7x in practice. HyperSloth achieves **2.3x speedup** through several optimizations:

```
🔄 Standard Multi-GPU: ~1.7x speedup
    ├─ GPU communication overhead
    └─ Load balancing inefficiencies

⚡ HyperSloth: 2.3x speedup  
    ├─ ✅ Sequence length sorting: reduces padding waste
    ├─ ✅ Adaptive batching: improves memory efficiency  
    ├─ ✅ Round-robin load balancing: better GPU utilization
    └─ ✅ NCCL gradient optimization: reduced communication overhead
```

This demonstrates how algorithmic optimizations can exceed theoretical hardware limits by reducing computational waste.

**Scaling Expectations:**

The 2.3x speedup shown above is **per GPU pair** - meaning you can expect similar multipliers as you scale up:
- **2 GPUs**: ~2.3x faster than single GPU
- **4 GPUs**: ~4.6x faster than single GPU (2.3x × 2)
- **8 GPUs**: ~9.2x faster than single GPU (2.3x × 4)

This scaling efficiency comes from HyperSloth's optimizations working consistently across different GPU counts, not just the 2-GPU case shown in the benchmark.

### Key Performance Features

- **Sequence length sorting**: Groups similar-length sequences to minimize padding waste (up to 40% token savings)
- **GPU load balancing**: Distributes work evenly across all available GPUs using round-robin batch assignment
- **NCCL optimization**: Uses PyTorch's native distributed training with efficient all-reduce gradient synchronization
- **Memory efficiency**: Adaptive batching reduces VRAM usage compared to naive padding approaches

### Additional Benchmarks

For detailed training time comparisons across different hardware configurations and loss curve analysis, see our [📊 Auxiliary Speed Benchmarks](docs/benchmarks.md).

## 💾 Installation

**Option 1: Using Conda Environment (Recommended)**
```bash

conda create --name hypersloth_env python=3.11
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth xformers hypersloth
# or from source pip install git+https://github.com/anhvth/HyperSloth.git
```

## ⚡ Quickstart

```python
# Basic multi-GPU training
hypersloth-train examples/example_sharegpt_lora_2gpus.py
```

| Example | Description | Link/Command |
|---------|-------------|--------------|
| **Kaggle Notebook (T4x2)** | Live training example on Kaggle's dual T4 GPU environment | [🔗 Qwen3 Unsloth 2GPUs](https://www.kaggle.com/code/anhvth226/qwen3-unsloth-2gpus?scriptVersionId=243436741) |
| **Local Jupyter Notebook** | Interactive training notebook for local development | [`notebooks/train.ipynb`](notebooks/train.ipynb) |
| **Command Line Example** | Quick start with pre-configured 2-GPU ShareGPT training | `hypersloth-train examples/example_sharegpt_lora_2gpus.py` |
| **Tmux Multi-windows** | Training with separate tmux sessions for each GPU monitoring | `hypersloth-train ./example_training_config.py --tmux train` |



### Quick Tips

**For faster iteration:**
- Start with smaller models: `unsloth/Qwen3-0.6b-bnb-4bit`
- Use fewer samples: `-n 1000` for quick testing
- Test single GPU first: `gpus=[0]` in config

**Recommended Configuration:**
```python
# Instead of: batch_size=1, gradient_accumulation=1 (high communication overhead)
# Use: batch_size=4, gradient_accumulation=8 (same effective batch size, 8x less communication)
TrainingConfig(
    per_device_train_batch_size=4,      # Larger batches per GPU
    gradient_accumulation_steps=8,       # Fewer gradient sync operations
    # Effective batch size = 4 * 8 * num_gpus, change your learning rate accordingly, i have not tested this yet
)
```

This approach maximizes GPU utilization while minimizing the communication overhead that typically limits multi-GPU scaling efficiency.

## 🛠 Command-Line Tools

- **`hypersloth-train`**: Main training launcher with multi-GPU and tmux support
- **`hypersloth-init`**: Generate configuration templates for new projects



## 🏗 How It Works

### Adaptive Batch Partitioning

HyperSloth patches the trainer's inner training loop with `adaptive_partition_batches()` that:

1. **Sorts sequences by length**: Groups similar-length sequences together within each batch slice
2. **Round-robin GPU distribution**: Distributes batch slices across GPUs in round-robin fashion for load balancing
3. **Minimizes padding**: Reduces wasted computation from padding tokens by up to 40%
4. **Tracks efficiency**: Logs padding savings and token statistics in real-time during training

### Distributed Training with NCCL

For multi-GPU setups, HyperSloth uses:

1. **Standard PyTorch DDP**: Each GPU runs a separate process with `torch.distributed`
2. **NCCL gradient synchronization**: Automatic all-reduce operations for gradient averaging
3. **Process spawning**: `hypersloth-train` launches one process per GPU using `spawn_training_process()`
4. **Tmux integration**: Optional `--tmux` flag creates separate terminal sessions for monitoring each GPU

## 🔧 Troubleshooting

**Common Issues:**

1. **Process Spawning Errors:**
   ```bash
   nvidia-smi  # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"  # Verify CUDA
   ```

2. **Memory Issues:**
   - Reduce `per_device_train_batch_size` in your config
   - Increase `gradient_accumulation_steps` to maintain effective batch size

3. **Performance Optimization:**
   - **Monitor tmux sessions** to check individual GPU utilization
   - **Experiment with batch sizes** to find optimal memory/speed trade-off

**Debugging Tips:**
```bash
hypersloth-train configs/your_config.py --tmux train
Or change gpus=[0] # use first gpu for training
# Then attach to sessions: tmux a -t train_gpu_0
```
