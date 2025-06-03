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

Get up and running with HyperSloth in 3 simple steps:

### Step 1: Build Your Dataset

First, prepare your training data using any Hugging Face dataset:

```bash
hypersloth-build-dataset --hf_dataset mlabonne/FineTome-100k -n 1000 --split train --name finetom-1k --tokenizer Qwen/Qwen3-8B --print_samples
```

**What this does:**
- Downloads 1000 samples from `mlabonne/FineTome-100k`
- Tokenizes using `Qwen/Qwen3-8B` tokenizer 
- Saves as `finetom-1k` dataset
- Shows sample conversations with `--print_samples`

**Expected output:**
```
Loading 1000 samples from mlabonne/FineTome-100k...

================================================================================
SAMPLE TEXTS FROM PROCESSED DATASET:
================================================================================

--- Sample 1 ---
<|im_start|>user
[Sample conversation]
<|im_end|>
<|im_start|>assistant
[Sample response]
<|im_end|>

Dataset saved to: data/built_dataset/finetom-1k
Registry updated: data/data_config.json
Dataset "finetom-1k" has been successfully built and saved!
```

### Step 2: Initialize Training Configuration

Generate a configuration template:

```bash
hypersloth-init
```

This creates `example_training_config.py` with default settings. Edit it to use your dataset:

```python
# Update the data section to use your built dataset
hyper_config_model = HyperConfig(
    data=DataConfig.from_dataset_name("finetom-1k"),  # Your dataset name
    training=TrainingConfig(
        gpus=[0, 1],  # Adjust to your available GPUs
        loss_type="response_only",  # Calculate loss only on assistant responses
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-0.6b-bnb-4bit",  # Smaller model for quick testing
        max_seq_length=2048,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)
```

### Step 3: Start Multi-GPU Training

Launch training across your GPUs:

```bash
hypersloth-train ./example_training_config.py
```

**Expected output:**
```
21:32:54 | INFO | 🔧 GPU 0 (Rank 0/1) | Model: unsloth/Qwen3-0.6b-bnb-4bit
21:32:54 | INFO | 🔧 GPU 1 (Rank 1/1) | Model: unsloth/Qwen3-0.6b-bnb-4bit
21:32:54 | INFO | 🚀 Starting total training timer
[Training progress with adaptive batching and NCCL synchronization]
```

**Optional: Monitor with tmux**
```bash
hypersloth-train ./example_training_config.py --tmux train
# Then attach to sessions: tmux a -t train_gpu_0
```

### Quick Tips

**For faster iteration:**
- Start with smaller models: `unsloth/Qwen3-0.6b-bnb-4bit`
- Use fewer samples: `-n 1000` for quick testing
- Test single GPU first: `gpus=[0]` in config

**For production:**
- Scale up dataset size: `-n 50000` or more
- Use larger models: `unsloth/Qwen3-8B-bnb-4bit`
- Add more GPUs: `gpus=[0, 1, 2, 3]`

**Memory management:**
- Reduce `per_device_train_batch_size` if you hit OOM
- Increase `gradient_accumulation_steps` to maintain effective batch size

That's it! You now have HyperSloth running multi-GPU training with optimized batching. Check the logs for padding savings and performance metrics.

## 🛠 Command-Line Tools

- **`hypersloth-train`**: Main training launcher with multi-GPU and tmux support
- **`hypersloth-init`**: Generate configuration templates for new projects

## 📓 Demo Notebook

For interactive training and experimentation, check out our demo training notebooks:

- **[`notebooks/train.ipynb`](notebooks/train.ipynb)**: Complete training example equivalent to `hypersloth-train examples/example_sharegpt_lora_2gpus.py`  
- **[Kaggle: Qwen3 Unsloth 2GPUs](https://www.kaggle.com/code/anhvth226/qwen3-unsloth-2gpus)**: Live training example with HyperSloth on Kaggle's GPU environment

## 📊 How to Prepare Data

To prepare your dataset for training, use the build_dataset.py script:

```bash
python scripts/build_dataset.py mlabonne/FineTome-100k -n 50000 --seed 3407 --split train --name finetom --tokenizer Qwen/Qwen3-8B
```

After running the script, use the built dataset in your configuration:

```python
hyper_config_model = HyperConfig(
    data=DataConfig.from_dataset_name("finetom") # Use the dataset name you created
    training=TrainingConfig(
        gpus=[0, 1],  # Change this to the number of GPUs you have
        loss_type="response_only",  # all or response_only, the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-1b-it",
        max_seq_length=2048,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)
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

3. **Performance Optimization:**
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
