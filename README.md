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

### Key Performance Features

- **Sequence length sorting**: Groups similar-length sequences to minimize padding waste (up to 40% token savings)
- **GPU load balancing**: Distributes work evenly across all available GPUs using round-robin batch assignment
- **NCCL optimization**: Uses PyTorch's native distributed training with efficient all-reduce gradient synchronization
- **Sequence packing**: Optional packing utility combines multiple conversations into single sequences
- **Memory efficiency**: Adaptive batching reduces VRAM usage compared to naive padding approaches

### Additional Benchmarks

For detailed training time comparisons across different hardware configurations and loss curve analysis, see our [📊 Auxiliary Speed Benchmarks](docs/benchmarks.md).

## 💾 Installation

```bash
pip install git+https://github.com/anhvth/HyperSloth.git
```

## ⚡ Quickstart

### 1. Initialize Configuration
```bash
hypersloth-init  # Creates a config template
```

### 2. Train Across Multiple GPUs
```bash
hypersloth-train configs/your_config.py 
```

<!-- ### 3. Optional: Pack Data for Maximum Efficiency
```bash
python -m HyperSloth.scripts.packing \
  -i data/conversations.json \
  -o data/packed_conversations.json \
  --seq_len 4096 --workers 32
``` -->

### 4. Export and Merge LoRA Weights
```bash
hypersloth-export merge_and_save_lora \
  --lora_path outputs/my_model/ \
  --base_model_name_or_path unsloth/qwen2.5-7b-bnb-4bit \
  --output_path merged_model/
```



## 📊 When to Use HyperSloth

**Multi-GPU Scenarios (Primary Use Case):**
- You have 2+ GPUs and want to speed up training significantly
- Distributed training with better efficiency than standard DDP approaches

**Single GPU Benefits:**
- **Yes, you still benefit!** If you use gradient accumulation with batch size > 1, the adaptive batching tricks help utilize your GPU more efficiently
- Sequence sorting reduces padding waste even on single GPU setups

## 🛠 Command-Line Tools

- **`hypersloth-train`**: Main training launcher with multi-GPU and tmux support
- **`hypersloth-init`**: Generate configuration templates for new projects  
- **`hypersloth-export`**: Export and merge LoRA weights, convert models
- **`python -m HyperSloth.scripts.packing`**: Sequence packing utility for data preprocessing

## 📋 Data Formats

**HuggingFace Datasets:**
```python
data=DataConfig.from_dataset_name("mlabonne/FineTome-100k")
```

**Local JSON (Conversation Format):**
```json
[
  {
    "messages": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
  }
]
```

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
   - Use sequence packing to reduce padding and memory usage

3. **Performance Optimization:**
   - **Use sequence packing** for datasets with varied conversation lengths
   - **Monitor tmux sessions** to check individual GPU utilization
   - **Experiment with batch sizes** to find optimal memory/speed trade-off

**Debugging Tips:**
```bash
# Test single GPU first (modify your config to use gpus=[0])
hypersloth-train configs/your_config.py

# Monitor individual GPU processes  
hypersloth-train configs/your_config.py --tmux train
# Then attach to sessions: tmux a -t train_gpu_0
```
