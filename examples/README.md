# HyperSloth Training Examples

This directory contains comprehensive examples showcasing different training configurations and use cases for HyperSloth.

## Example Configurations

### 1. Basic Single GPU LoRA Training
**File:** `example_hf_dataset_lora_single_gpu.py`
- **Dataset:** HuggingFace dataset (FineTome-100k)
- **Setup:** Single GPU, basic LoRA configuration
- **Use case:** Getting started, development, small experiments
- **Key features:**
  - Low-rank LoRA (r=16) for efficiency
  - 4-bit quantization
  - Small dataset subset (1000 samples)
  - Tensorboard logging

### 2. Full Fine-tuning
**File:** `example_hf_dataset_full_finetuning.py`
- **Dataset:** HuggingFace dataset (OpenThoughts-114k)
- **Setup:** Multi-GPU, full parameter fine-tuning
- **Use case:** When you need to modify all model parameters
- **Key features:**
  - `full_finetuning=True` in FastModelArgs
  - Very low learning rate (5e-6)
  - Cosine annealing scheduler
  - Higher weight decay for regularization

### 3. High-Rank LoRA with ShareGPT Format
**File:** `example_sharegpt_high_rank_lora.py`
- **Dataset:** ShareGPT format (local JSON file)
- **Setup:** 4-GPU training, high-rank LoRA
- **Use case:** Complex adaptations requiring more parameters
- **Key features:**
  - High LoRA rank (r=256)
  - Rank-stabilized LoRA (`use_rslora=True`)
  - Custom instruction/response formatting
  - Longer sequence length (4096)

### 4. Continual LoRA Training
**File:** `example_continual_lora_training.py`
- **Dataset:** Pre-registered dataset from registry
- **Setup:** Continue training from existing LoRA checkpoint
- **Use case:** Incremental learning, domain adaptation
- **Key features:**
  - `pretrained_lora` parameter to load existing weights
  - Lower learning rate for stable adaptation
  - Uses dataset registry (`DataConfig.from_dataset_name()`)

### 5. Vision-Language Model Training
**File:** `example_vision_language_lora.py`
- **Dataset:** Vision-QA dataset (VQAv2)
- **Setup:** Multi-modal LoRA training
- **Use case:** Fine-tuning vision-language models
- **Key features:**
  - `finetune_vision_layers=True`
  - Vision-specific target modules
  - Multi-modal data handling
  - Qwen3-VL model architecture

### 6. Memory-Optimized Training
**File:** `example_memory_optimized_training.py`
- **Dataset:** Math reasoning dataset (Orca-Math)
- **Setup:** 8-GPU training for large 32B model
- **Use case:** Training large models with limited memory
- **Key features:**
  - Very low LoRA rank (r=8)
  - Minimal batch size with large gradient accumulation
  - 32B parameter model
  - Memory-efficient target module selection

### 7. Advanced LoRA Configuration
**File:** `example_advanced_lora_config.py`
- **Dataset:** High-quality instruction dataset (OpenHermes-2.5)
- **Setup:** Advanced LoRA with custom configurations
- **Use case:** Fine-grained control over adaptation
- **Key features:**
  - Custom target modules including embeddings and LM head
  - `bias="lora_only"` for bias adaptation
  - Custom chat templates
  - `loss_type="all"` for full sequence loss

### 8. Legacy Examples (Renamed)
- **`example_finetome_lora_2gpu.py`** - Original Qwen3_2gpus.py example
- **`example_sharegpt_lora_multi_gpu.py`** - Original example_training_config.py

## Data Configuration Types

### HuggingFace Datasets (`DataConfigHF`)
```python
data=DataConfigHF(
    dataset_name="mlabonne/FineTome-100k",  # HF dataset name
    tokenizer_name="Qwen/Qwen3-8B",  # Tokenizer to use
    num_samples=1000,  # Number of samples to use
    split="train",  # Dataset split
    name="local_name",  # Local reference name
    columns=["conversations"],  # Data columns to use
)
```

### ShareGPT Format (`DataConfigShareGPT`)
```python
data=DataConfigShareGPT(
    dataset_path="path/to/sharegpt.json",  # Local file path
    tokenizer_name="Qwen/Qwen3-8B",
    num_samples=None,  # Use all samples
    instruction_part="<|im_start|>user\n",  # Custom formatting
    response_part="<|im_start|>assistant\n",
    print_samples=True,  # Debug option
    use_cache=True,  # Cache processed data
)
```

### Registry Datasets (`DataConfig.from_dataset_name()`)
```python
data=DataConfig.from_dataset_name("finetom")  # Pre-registered dataset
```

## Key Configuration Parameters

### LoRA Parameters
- **`r`**: LoRA rank (8-512) - higher rank = more parameters
- **`lora_alpha`**: Scaling factor (typically 2x rank)
- **`target_modules`**: Which layers to adapt
- **`use_rslora`**: Rank-stabilized LoRA for high ranks
- **`lora_dropout`**: Dropout for regularization

### Training Parameters
- **`gpus`**: List of GPU indices to use
- **`loss_type`**: "response_only" or "all"
- **`per_device_train_batch_size`**: Batch size per GPU
- **`gradient_accumulation_steps`**: Steps to accumulate gradients
- **`learning_rate`**: Learning rate (2e-4 for LoRA, 5e-6 for full FT)

### Model Parameters
- **`max_seq_length`**: Maximum sequence length
- **`load_in_4bit`**: Enable 4-bit quantization
- **`full_finetuning`**: Enable full fine-tuning instead of LoRA

## Running Examples

To run any example:
```bash
hypersloth-train example_filename.py
```

Make sure to adjust paths and dataset names according to your setup.
