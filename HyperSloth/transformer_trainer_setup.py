"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

from loguru import logger
from transformers import TrainingArguments
from unsloth.tokenizer_utils import load_correct_tokenizer, _load_correct_tokenizer


from typing import List, Optional
from pydantic import BaseModel


class DataConfig(BaseModel):
    dataset: str = "data/cod_1k.json"
    test_ratio: float = 0.05
    dataset_num_proc: int = 4
    instruction_part: str = "Instruction:"
    response_part: str = "Response:"


class TrainingConfig(BaseModel):
    gpus: List[int] = [0]
    loss_type: str = "target_only"  # "target_only" from your config
    packing: bool = False


class FastModelArgs(BaseModel):
    model_name: str = "unsloth/gemma-3-4b-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    # If you want to allow for a 'token' field:
    token: Optional[str] = None


class LoraArgs(BaseModel):
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: int = 0
    bias: str = "none"
    random_state: int = 3407


class HyperConfig(BaseModel):
    grad_dir: str = "/dev/shm/hypersloth"
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    fast_model_args: FastModelArgs = FastModelArgs()
    lora_args: LoraArgs = LoraArgs()


def setup_model_and_training(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArguments,
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

    gpu_ith = hyper_config.training.gpus[gpu]

    # Initialize model and tokenizer
    model, tokenizer = FastModel.from_pretrained(**hyper_config.fast_model_args)

    # Load dataset
    ds_train = get_chat_dataset(hyper_config.data.dataset, tokenizer=tokenizer)
    if hyper_config.data.test_ratio > 0:
        ds = ds_train.train_test_split(test_size=hyper_config.data.test_ratio, shuffle=True)
        ds_train, ds_test = ds["train"], ds["test"]
        logger.info(f"Train size: {len(ds_train)}, Test size: {len(ds_test)}")
    else:
        ds_train, ds_test = ds_train, None

    # Apply PEFT model
    model = FastModel.get_peft_model(model, **hyper_config.lora_args)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test if gpu_ith == 0 else None,
        dataset_text_field="text",
        max_seq_length=hyper_config.fast_model_args.max_seq_length,
        dataset_num_proc=hyper_config.data.dataset_num_proc,
        packing=hyper_config.training.packing,
        args=hf_train_args,
    )

    # Adjust dataset for multi-GPU training
    max_len_ds = len(hyper_config.training.gpus) * (
        len(trainer.train_dataset) // len(hyper_config.training.gpus)
    )
    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(hyper_config.training.gpus), index=gpu_ith
    )

    # Handle specific training loss type
    if hyper_config.training.loss_type == "target_only":
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

    _debug_dataloader(trainer)
    return trainer


def _debug_dataloader(trainer):
    """
    Debug function to log the first batch of the training dataloader.
    """
    dl = trainer.get_train_dataloader()
    g = iter(dl)
    batch = next(g)
    input_ids = batch["input_ids"]
    from copy import deepcopy

    tokenizer = deepcopy(trainer.tokenizer)
    text = tokenizer.decode(input_ids.cpu()[0])
    labels = batch["labels"]
    labels[labels < 0] = 0  # Fill < 0 with 0
    label_text = tokenizer.decode(labels.cpu()[0])
    logger.info(f"=====\nI: {text}\n-----\nL: {label_text}\n=====")
    assert (batch["labels"] > 0).sum() != 0, "NO LABELS???"
