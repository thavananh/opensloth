<p align="center">
    <img src="images/hpsloth.webp" alt="HyperSloth Logo" width="200" />
</p>

# Hyper-Sloth

A high-performance framework for fine-tuning large language models.

## Overview

HyperSloth is an extension of Unsloth for distributed training of Large Language Models across multiple GPUs. It focuses on efficient gradient synchronization using memory-mapped files for seamless communication between GPU processes.

## Features

- **Memory-mapped gradient synchronization**: Coordinate training across multiple GPUs without requiring NCCL/DDP
- **Efficient weight synchronization**: Ensure model consistency across all GPUs during training
- **Template fixes**: Custom tokenizer chat template fixes for proper handling of "think" tags
- **Customizable loss types**: Support for full sequence or response-only training

## Installation

```bash
# Clone the repository
pip install git+https://github.com/anhvth/HyperSloth.git
```

## Quick Start

```kaggle
https://www.kaggle.com/code/anhvth226/kaggle-mistral-7b-hypersloth-notebook?scriptVersionId=228204516
```

### Train a model across multiple GPUs

```bash
# create a config file for training
[>training| ~/projects/hyper-sloth ] hypersloth-init
# Example training config: ./hs_training_config.py
hypersloth-train ./hs_training_config.py

# [>training| ~/projects/hyper-sloth ] hypersloth-train ./hs_training_config.py
# 2025-03-16 06:53:56.861 | INFO     | HyperSloth.scripts.trainner:train:94 -
# Key                          Value
# ---------------------------  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# grad_dir                     /dev/shm/hypersloth
# data                         {'dataset_name_or_path': 'mlabonne/FineTome-100k', 'test_ratio': 0.05, 'dataset_num_proc': 4, 'instruction_part': '<start_of_turn>user\n', 'response_part': '<start_of_turn>model\n', 'num_samples': 1000, 'split': 'train'}
# training                     {'gpus': [0, 1], 'loss_type': 'response_only'}
# fast_model_args              {'model_name': 'unsloth/gemma-3-1b-it', 'max_seq_length': 2048, 'load_in_4bit': True, 'load_in_8bit': False, 'full_finetuning': False, 'token': None}
# lora_args                    {'finetune_vision_layers': False, 'finetune_language_layers': True, 'finetune_attention_modules': True, 'finetune_mlp_modules': True, 'r': 16, 'lora_alpha': 16, 'lora_dropout': 0.0, 'bias': 'none', 'random_state': 3407}
# output_dir                   outputs/2B/
# per_device_train_batch_size  4
# learning_rate                0.0002
# gradient_accumulation_steps  16
# per_device_eval_batch_size   2
# eval_steps                   100
# logging_steps                1
# report_to                    tensorboard
# num_train_epochs             1
# lr_scheduler_type            linear
# warmup_steps                 5
# seed                         42
# save_total_limit             2
# bf16                         True
# fp16                         False
# optim                        adamw_8bit
# weight_decay                 0.01
# 2025-03-16 06:53:56.861 | INFO     | HyperSloth.scripts.trainner:train:97 - Cleaning up previous runs
# 2025-03-16 06:53:56.868 | DEBUG    | HyperSloth.scripts.trainner:train:103 - Running on GPU 0
# 2025-03-16 06:53:57.870 | DEBUG    | HyperSloth.scripts.trainner:train:103 - Running on GPU 1
```

see

## Performance Benchmarks

Hyper-Sloth demonstrates significant performance improvements over other popular fine-tuning frameworks.

### Training Time Comparison (4x RTX 4090)

| Framework    | Training Time | VRAM Peak Consumption |
| ------------ | ------------- | --------------------- |
| Hyper-Sloth  | 19 minutes    | 6 GB                  |
| LlamaFactory | 30 minutes    | 21 GB                 |
| Unsloth (1X) | ~70 minutes   | 6 GB                  |

### Loss Curves

The loss scale between Hyper-Sloth and LlamaFactory looks comparable, indicating similar training quality with significantly improved training speed.

| ![Hyper-Sloth Tensorboard](images/hyper-sloth-tb.png){ width=150 } | ![LlamaFactory Tensorboard](images/llama-factory-tb.png){ width=150 } |
| ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Hyper-Sloth Tensorboard[^1]                                        | LlamaFactory Tensorboard[^2]                                          |

[^1]: Hyper-Sloth Tensorboard.
[^2]: LlamaFactory Tensorboard.
