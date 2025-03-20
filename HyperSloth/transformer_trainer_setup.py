"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os
import random
from loguru import logger

from .hypersloth_config import HyperConfig, TrainingArgsConfig


def setup_model_and_training(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        gpu: GPU index
        hyper_config: Configuration arguments
        hf_train_args: Training arguments

    Returns:
        Trainer object configured for multi-GPU training
    """
    from unsloth import FastModel
    from HyperSloth.dataset_utils import get_chat_dataset
    from trl import SFTTrainer

    gpu_ith = hyper_config.training.gpus.index(gpu)

    # Initialize model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    if not hyper_config.fast_model_args.full_finetuning:
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())

    # Load dataset
    ds_train, ds_test = get_chat_dataset(
        tokenizer=tokenizer, **hyper_config.data.model_dump()
    )

    # Apply PEFT model

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test if gpu_ith == 0 else None,
        dataset_text_field="text",
        max_seq_length=hyper_config.fast_model_args.max_seq_length,
        dataset_num_proc=hyper_config.data.dataset_num_proc,
        args=hf_train_args,
    )

    # Adjust dataset for multi-GPU training
    max_len_ds = len(hyper_config.training.gpus) * (
        len(trainer.train_dataset) // len(hyper_config.training.gpus)
    )

    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    if hyper_config.data.group_by_length:  # This currently hurts performance
        from .patching import patch_sampler, select_dataset_by_length
        trainer = patch_sampler(trainer)
        trainer.train_dataset = select_dataset_by_length(
            trainer.train_dataset,
            gpu_ith,
            len(hyper_config.training.gpus),
            hf_train_args.gradient_accumulation_steps,
            hf_train_args.per_device_train_batch_size,
        )
    else:
        trainer.train_dataset = trainer.train_dataset.shard(
            num_shards=len(hyper_config.training.gpus), index=gpu_ith
        )

    # Handle specific training loss type
    if hyper_config.training.loss_type == "response_only":
        from unsloth.chat_templates import train_on_responses_only

        first_text = ds_train[0]["text"]
        instruction_part = hyper_config.data.instruction_part
        response_part = hyper_config.data.response_part
        assert instruction_part in first_text, f"{instruction_part} not in {first_text}"
        assert response_part in first_text, f"{response_part} not in {first_text}"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    if gpu_ith == 0:
        logger.info(f"Model setup complete for GPU {gpu_ith}")
        _debug_dataloader(trainer)
    return trainer


# def patch_sampler(trainer):
#     from torch.utils.data import SequentialSampler
#     from transformers import Trainer
#     from fastcore.all import patch

#     @patch
#     def _get_train_sampler(self: Trainer) -> SequentialSampler:
#         """Get a sequential sampler for the training dataset."""
#         return SequentialSampler(self.train_dataset)

#     return trainer


# def select_dataset_by_length(
#     dataset, gpu_index: int, num_gpus: int, grad_accum_steps: int, batch_size: int
# ):
#     from fastcore.all import chunked
#     from typing import List, Dict
#     import numpy as np

#     def split_batch_evenly(
#         lengths: List[int], global_ids: List[int], num_gpus: int
#     ) -> Dict[int, Dict[str, List[int]]]:
#         if len(lengths) % num_gpus != 0:
#             raise ValueError("The list length must be divisible by num_gpus")

#         indices_sorted = np.argsort(-np.array(lengths))
#         splits = [[] for _ in range(num_gpus)]
#         length_sums = [0] * num_gpus
#         max_items_per_gpu = len(lengths) // num_gpus

#         for idx in indices_sorted:
#             gpu_candidates = sorted(
#                 range(num_gpus),
#                 key=lambda gpu: (
#                     len(splits[gpu]) >= max_items_per_gpu,
#                     length_sums[gpu],
#                 ),
#             )
#             chosen_gpu = gpu_candidates[0]
#             splits[chosen_gpu].append(idx)
#             length_sums[chosen_gpu] += lengths[idx]

#         splits = [sorted(split) for split in splits]

#         gpu_batches = {
#             gpu: {
#                 "global_ids": [global_ids[i] for i in splits[gpu]],
#                 "lengths": [lengths[i] for i in splits[gpu]],
#             }
#             for gpu in range(num_gpus)
#         }
#         for gpu in range(num_gpus):
#             lens = gpu_batches[gpu]["lengths"]
#             ids = gpu_batches[gpu]["global_ids"]
#             new_lens, new_ids = zip(*sorted(zip(lens, ids), key=lambda x: x[0]))
#             total_len = sum(new_lens)
#             gpu_batches[gpu] = {
#                 "global_ids": list(new_ids),
#                 "lengths": list(new_lens),
#                 "total_len": total_len,
#             }

#         return gpu_batches

#     dataset_indices = list(range(len(dataset)))
#     id_to_length = {idx: len(dataset[idx]["input_ids"]) for idx in dataset_indices}
#     random.Random(42).shuffle(dataset_indices)

#     global_batch_size = grad_accum_steps * batch_size * num_gpus
#     global_batches = list(chunked(dataset_indices, global_batch_size))

#     selected_ids = []
#     for batch_indices in global_batches[:-1]:  # Exclude potentially smaller last batch
#         batch_lengths = [id_to_length[i] for i in batch_indices]
#         splits = split_batch_evenly(batch_lengths, batch_indices, num_gpus)
#         this_gpu_split = splits[gpu_index]
#         if gpu_index == 0:
#             print(f"{splits=}")
#         selected_ids.extend(this_gpu_split["global_ids"])

#     return dataset.select(selected_ids)


def _debug_dataloader(trainer, n_example=10):
    """
    Debug function to log samples from the training dataloader in an HTML format.
    Outputs to both terminal (with colors) and an HTML file with CSS styling.
    """
    from copy import deepcopy

    tokenizer = deepcopy(trainer.tokenizer)
    dl = trainer.get_train_dataloader()
    g = iter(dl)
    html_path = ".log/dataloader_examples.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # Create HTML file with CSS styling
    with open(html_path, "w") as html_file:
        html_file.write(
            """<!DOCTYPE html>
    <html>
    <head>
        <title>Dataloader Examples</title>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #333; }
            .trainable { background-color: #FFEBCD; color: #333; }
            .context { background-color: #E0FFE0; color: #333; }
            th { background-color: #f2f2f2; }
            th, td { border-color: #ddd; }
        }
        
        @media (prefers-color-scheme: dark) {
            body { background-color: #222; color: #f0f0f0; }
            .trainable { background-color: #664a20; color: #f0f0f0; }
            .context { background-color: #2a5a2a; color: #f0f0f0; }
            th { background-color: #444; color: #f0f0f0; }
            th, td { border-color: #555; }
        }
        
        .trainable, .context { padding: 2px; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid; padding: 8px; text-align: left; }
        h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Dataloader Examples</h1>
        <p>This file contains examples of training data with context and trainable parts.</p>
    """
        )

        for i in range(n_example):
            batch = next(g)
            input_ids = batch["input_ids"][0]
            label_ids = batch["labels"][0]
            parts_mask = label_ids >= 0  # True is trainable, False is context

            # Find split points where trainable/non-trainable sections change
            split_points = (
                [0]
                + [
                    i
                    for i, val in enumerate(parts_mask)
                    if i > 0 and val != parts_mask[i - 1]
                ]
                + [len(parts_mask)]
            )

            colored_parts = []
            html_file.write(f"\n    <h2>Example {i+1}</h2>\n")
            html_file.write(
                "    <table>\n        <tr><th>Text</th><th>Label</th></tr>\n"
            )

            for a, b in zip(split_points[:-1], split_points[1:]):
                text = tokenizer.decode(input_ids[a:b])
                is_trainable = parts_mask[a]

                # Colored text for terminal
                colored_text = (
                    f"\033[93m{text}\033[0m"
                    if is_trainable
                    else f"\033[92m{text}\033[0m"
                )
                colored_parts.append(colored_text)

                # HTML with CSS classes
                css_class = "trainable" if is_trainable else "context"
                label = "🟠 TRAIN" if is_trainable else "🟢 CONTEXT"

                # Escape HTML special characters
                text_escaped = (
                    text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )

                # Add row to HTML table
                html_file.write(
                    f'        <tr>\n            <td><span class="{css_class}">{text_escaped}</span></td>\n'
                    f"            <td>{label}</td>\n        </tr>\n"
                )

            html_file.write("    </table>\n")

            # Colored text for terminal
            colored_output = "".join(colored_parts)
            terminal_msg = f"\n=== EXAMPLE #{i+1} ===\n" + colored_output + "\n"
            if i == 0:
                print(terminal_msg)

        html_file.write("</body>\n</html>")

    print(f"More training debug examples written to {html_path}")
