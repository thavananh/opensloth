"""
Lazy dataloader version of init_modules for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
Uses on-the-fly tokenization without dataset caching.
"""

import os
from typing import Any, Dict, List

from .hypersloth_config import HyperConfig, TrainingArgsConfig
from .logging_config import get_hypersloth_logger

gpu_id = os.environ.get("HYPERSLOTH_LOCAL_RANK", "0")
enhanced_logger = get_hypersloth_logger(gpu_id=gpu_id)
logger = get_hypersloth_logger()


def init_model_and_tokenizer(hyper_config: HyperConfig):
    """Initialize and optionally set up LoRA for the model."""
    from unsloth import FastLanguageModel

    enhanced_logger.start_timing("model_loading")

    if hyper_config.pretrained_lora:
        logger.info(
            f"Loading model from {hyper_config.pretrained_lora} with LoRA weights"
        )
        hyper_config.fast_model_args.model_name = hyper_config.pretrained_lora

    from HyperSloth.nccl_grad_sync import setup_nccl_for_hypersloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    enhanced_logger.finish_timing("model_loading")

    logger.info(f"Model created at {os.environ['CUDA_VISIBLE_DEVICES']}")

    enhanced_logger.start_timing("nccl_setup")
    setup_nccl_for_hypersloth(
        gpu=int(os.environ["HYPERSLOTH_LOCAL_RANK"]), gpus=hyper_config.training.gpus
    )
    enhanced_logger.finish_timing("nccl_setup")

    model_device = model.device
    logger.info(
        f"Model loaded on device {model_device}, "
        f"tokenizer: {tokenizer.__class__.__name__}"
    )

    if (
        not hyper_config.fast_model_args.full_finetuning
        and not hyper_config.pretrained_lora
    ):
        enhanced_logger.start_timing("lora_setup")
        model = FastLanguageModel.get_peft_model(
            model, **hyper_config.lora_args.model_dump()
        )
        enhanced_logger.finish_timing("lora_setup")

    # Allow custom chat templates
    if (
        hasattr(hyper_config.training, "chat_template")
        and hyper_config.training.chat_template is not None
    ):
        from transformers import AutoTokenizer

        new_template = AutoTokenizer.from_pretrained(
            hyper_config.training.chat_template
        ).chat_template
        tokenizer.chat_template = new_template
        logger.warning(f"Using chat template of {new_template}")

    return model, tokenizer


def create_trainer(
    tokenizer,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
    gpu_ith: int,
    model,
):
    """Create the SFTTrainer with lazy dataset loading."""
    from .logging_config import get_hypersloth_logger

    enhanced_logger = get_hypersloth_logger(gpu_id=str(gpu_ith))

    enhanced_logger.start_timing("trainer_setup")
    trainer = _create_lazy_trainer(
        tokenizer,
        hyper_config,
        hf_train_args,
        gpu_ith,
        model,
    )
    enhanced_logger.finish_timing("trainer_setup")

    from HyperSloth._patch_inner_training_loop import patch_inner_training_loop

    patch_inner_training_loop(trainer)
    return trainer


def _load_json_dataset(data_path: str):
    """Load dataset from JSON file containing list of message dictionaries."""
    import json
    from datasets import Dataset

    logger.info(f"Loading dataset from {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to HuggingFace dataset format
    # Assuming data is a list of dicts with 'messages' key
    dataset_dict = {"messages": []}

    for item in data:
        if "messages" in item:
            dataset_dict["messages"].append(item["messages"])
        else:
            # If the item itself contains the messages directly
            # dataset_dict["conversations"].append(item)
            raise ValueError(
                f"Item {item} does not contain 'messages' key. "
                "Expected format is a list of dictionaries with 'messages' key."
            )

    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Loaded {len(dataset)} samples from {data_path}")

    return dataset


def _tokenize_on_the_fly(
    sample: Dict[str, Any], tokenizer, max_seq_length: int
) -> Dict[str, List[int]]:
    """Tokenize conversations on-the-fly during training."""
    assert "messages" in sample, "Sample must contain 'messages' key, got: " + str(
        sample
    )

    # Handle the case where messages is a list of lists or direct list
    messages = sample["messages"]
    if isinstance(messages, list) and len(messages) > 0:
        # Check if it's a list of message dicts or a nested list
        if isinstance(messages[0], dict):
            # Direct list of message dicts
            first_message = messages[0]
        elif isinstance(messages[0], list) and len(messages[0]) > 0:
            # Nested list - take the first conversation
            messages = messages[0]
            first_message = messages[0] if isinstance(messages[0], dict) else None
        else:
            raise ValueError(f"Unexpected message format: {messages}")
    else:
        raise ValueError(f"Messages must be a non-empty list, got: {messages}")

    if first_message and isinstance(first_message, dict):
        assert first_message.get("role") in [
            "system",
            "user",
        ], f"First message must be system or user role, got: {first_message}"

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # padding happens in the collator
        return_attention_mask=True,
    )
    # print("=========")
    # print(f"{encoded=}")
    # print("=========")
    return encoded


def _create_lazy_trainer(
    tokenizer,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
    gpu_ith: int,
    model,
):
    """Create SFTTrainer with lazy dataset loading and on-the-fly tokenization."""
    from trl import SFTTrainer
    from transformers import DataCollatorForLanguageModeling

    # # Load raw dataset from JSON
    # enhanced_logger.start_timing("dataset_loading")
    # raw_dataset = _load_json_dataset(hyper_config.data.dataset_name_or_path)
    # enhanced_logger.finish_timing("dataset_loading")

    # # Create tokenization transform function
    # def tokenize_transform(sample):
    #     return _tokenize_on_the_fly(
    #         sample, tokenizer, hyper_config.fast_model_args.max_seq_length
    #     )

    # # Set the transform for on-the-fly tokenization
    # enhanced_logger.start_timing("dataset_transform_setup")
    # raw_dataset.set_transform(tokenize_transform)
    # enhanced_logger.finish_timing("dataset_transform_setup")

    # # Create data collator
    # collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt

    raw_chat = load_dataset("mlabonne/FineTome-100k", split="train")
    chat_std = standardize_sharegpt(raw_chat).remove_columns(["source", "score"])

    def tokenize_on_the_fly(sample: Dict[str, str]) -> Dict[str, List[int]]:
        text = tokenizer.apply_chat_template(sample["conversations"], tokenize=False)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,  # padding happens in the collator
            return_attention_mask=True,
        )

        return encoded

    chat_std.set_transform(tokenize_on_the_fly)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # -------------------------------------------------------------------
†
    # Configure training arguments for lazy loading
    hf_train_args.dataset_kwargs = {"skip_prepare_dataset": True}
    # if LOCAL_RANK != 0:
    hf_train_args.eval_strategy = "no"
    # hf_train_args.dataset_batch_size = hf_train_args.per_device_train_batch_size
    hf_train_args.remove_unused_columns = False

    logger.info(f"GPU {gpu_ith}: Creating trainer with lazy dataset loading")
    enhanced_logger.start_timing("trainer_creation")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=chat_std,
        eval_dataset=None,
        data_collator=collator,
        args=hf_train_args,
    )
    next_batch = next())

    enhanced_logger.finish_timing("trainer_creation")

    # Apply response-only training if configured
    # trainer = _maybe_train_on_responses_only(trainer, hyper_config)

    logger.info(f"GPU {gpu_ith}: Trainer created with lazy loading")
    return trainer


# def _maybe_train_on_responses_only(trainer, hyper_config: HyperConfig):
#     """Use a specialized approach if 'response_only' loss is desired."""
#     if hyper_config.training.loss_type == "response_only":
#         from unsloth.chat_templates import train_on_responses_only

#         # Get a sample to verify the format
#         sample_text = trainer.tokenizer.apply_chat_template(
#             trainer.train_dataset[0]["messages"],
#             tokenize=False,
#             add_generation_prompt=False,
#         )

#         instruction_part = hyper_config.data.instruction_part
#         response_part = hyper_config.data.response_part

#         assert (
#             instruction_part in sample_text
#         ), f"{instruction_part} not in {sample_text}"
#         assert response_part in sample_text, f"{response_part} not in {sample_text}"

#         logger.info("Applying response-only training")
#         trainer = train_on_responses_only(
#             trainer,
#             instruction_part=instruction_part,
#             response_part=response_part,
#             num_proc=hyper_config.data.dataset_num_proc,
#         )
#     return trainer


def configure_batch_size(hf_train_args, gpu_ith: int, num_gpus: int):
    """Configure batch size for multi-GPU training."""
    if num_gpus != 1:
        logger.info(
            f"Hypersloth will change the batch size to "
            f"{hf_train_args.per_device_train_batch_size * num_gpus} "
            f"so each gpu will have {hf_train_args.per_device_train_batch_size} "
            f"x {hf_train_args.gradient_accumulation_steps} per update step."
        )
        # This is the total batch size loaded by dataloader,
        # the trainer later will choose the correct batch size for each GPU
        hf_train_args.per_device_train_batch_size *= num_gpus

    if gpu_ith != 0:
        # disable reporting for all GPUs except the first one
        hf_train_args.report_to = "none"
        # disable evaluation for all GPUs except the first one
        hf_train_args.do_eval = False


__all__ = [
    "configure_batch_size",
    "init_model_and_tokenizer",
    "create_trainer",
]
