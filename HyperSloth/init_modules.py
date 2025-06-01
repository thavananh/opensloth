"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from trl.trainer.sft_trainer import SFTTrainer

from HyperSloth.dataset_utils import get_chat_dataset

from .hypersloth_config import (
    DataConfig,
    DataConfigHF,
    DataConfigShareGPT,
    HyperConfig,
    TrainingArgsConfig,
)

# Get enhanced logger for timing
from .logging_config import get_hypersloth_logger

# Setup logger with proper GPU ID
gpu_id = os.environ.get("HYPERSLOTH_LOCAL_RANK", "0")
hp_logger = get_hypersloth_logger(log_level="INFO")
logger = hp_logger  # Use the same enhanced logger instance


def init_model_and_tokenizer(hyper_config: HyperConfig):
    """Initialize and optionally set up LoRA for the model."""
    from unsloth import FastModel

    hp_logger.start_timing("model_loading")

    if hyper_config.pretrained_lora:
        logger.info(
            f"Loading model from {hyper_config.pretrained_lora} with LoRA weights"
        )
        hyper_config.fast_model_args.model_name = hyper_config.pretrained_lora
    from HyperSloth.nccl_grad_sync import setup_nccl_for_hypersloth

    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    hp_logger.finish_timing("model_loading")

    logger.info(f"Model created at {os.environ['CUDA_VISIBLE_DEVICES']}")

    hp_logger.start_timing("nccl_setup")
    setup_nccl_for_hypersloth(
        gpu=int(os.environ["HYPERSLOTH_LOCAL_RANK"]), gpus=hyper_config.training.gpus
    )
    hp_logger.finish_timing("nccl_setup")

    model_device = model.device
    logger.info(
        f"Model loaded on device {model_device}, tokenizer: {tokenizer.__class__.__name__}"
    )

    if (
        not hyper_config.fast_model_args.full_finetuning
        and not hyper_config.pretrained_lora
    ):
        hp_logger.start_timing("lora_setup")
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())
        hp_logger.finish_timing("lora_setup")

    # Allow custom chat templates
    if (
        hasattr(hyper_config.training, "chat_template")
        and hyper_config.training.chat_template is not None
    ):
        from transformers import AutoTokenizer  # type: ignore

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
    """Load or prepare the dataset and create the SFTTrainer."""

    # Get enhanced logger for timing
    from .logging_config import get_hypersloth_logger

    hp_logger = get_hypersloth_logger(log_level="INFO")

    dataset_cache_path = _identify_dataset_name(tokenizer, hyper_config, hf_train_args)
    dataset_cache_exists = os.path.exists(dataset_cache_path)

    # CASE 1: Dataset cache already exists, just load it
    hp_logger.start_timing("trainer_setup")
    trainer = _get_trainer(
        tokenizer,
        hyper_config,
        hf_train_args,
        gpu_ith,
        model,
        dataset_cache_path,
        dataset_cache_exists,
    )
    hp_logger.finish_timing("trainer_setup")

    from HyperSloth.patching.inner_training_loop import patch_inner_training_loop

    hp_logger.start_timing("training_loop_patch")
    patch_inner_training_loop(trainer)
    hp_logger.finish_timing("training_loop_patch")

    # DEBUG: change the sampler to sequential sampler for debugging
    from .patching.patch_sampler import apply_patch_sampler

    trainer = apply_patch_sampler(trainer)

    # from torch.utils.data import DataLoader, SequentialSampler  # type: ignore
    # from torch.utils.data.distributed import DistributedSampler  # type: ignore

    # def _get_sampler(self, dataset):
    #     """Override to use SequentialSampler for debugging."""
    #     return SequentialSampler(dataset)

    # trainer._get_train_sampler = _get_sampler
    return trainer


def _identify_dataset_name(tokenizer, hyper_config, hf_train_args):
    from speedy_utils import identify

    tokenizer_name = identify(str(tokenizer))
    # hash the dataset name and max_seq_length to create a unique cache name
    dataset_name = identify(
        [
            hyper_config.data.model_dump(),
            hyper_config.fast_model_args.max_seq_length,
        ]
    )
    dataset_cache_name = "dataset_" + tokenizer_name + "_" + dataset_name
    dataset_cache_path = os.path.join(".cache/", dataset_cache_name)
    return dataset_cache_path


def parse_data(data):
    if isinstance(data, DataConfigShareGPT):
        from HyperSloth.scripts.build_dataset import build_sharegpt_dataset

        dataset_name = build_sharegpt_dataset(
            dataset_path=data.dataset_path,
            tokenizer_name=data.tokenizer_name,
            num_samples=data.num_samples,
            seed=data.seed,
            instruction_part=data.instruction_part,
            response_part=data.response_part,
            print_samples=data.print_samples,
            use_cache=data.use_cache,
            name=data.name,
        )
        data = DataConfig.from_dataset_name(
            dataset_name=dataset_name,
        )
    elif isinstance(data, DataConfigHF):
        from HyperSloth.scripts.build_dataset import build_hf_dataset

        dataset_name = build_hf_dataset(
            dataset_name=data.dataset_name,
            tokenizer_name=data.tokenizer_name,
            num_samples=data.num_samples,
            seed=data.seed,
            split=data.split,
            instruction_part=data.instruction_part,
            response_part=data.response_part,
            name=data.name,
        )
        data = DataConfig.from_dataset_name(
            dataset_name=dataset_name,
        )
    elif isinstance(data, DataConfig):
        # Already a DataConfig, return as-is
        pass
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected DataConfig, DataConfigHF, or DataConfigShareGPT."
        )
    assert isinstance(data, DataConfig), f"DataConfig expected, got {type(data)}"
    return data


def _get_trainer(
    tokenizer,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
    gpu_ith,
    model,
    dataset_cache_path,
    dataset_cache_exists,
) -> SFTTrainer:
    """
    Returns an SFTTrainer instance. If a cached dataset exists, load from disk.
    If not, GPU 0 will create and save it, and other GPUs will wait for GPU 0
    to finish.
    """
    import os
    import time

    import filelock
    from datasets import load_from_disk
    from trl import SFTTrainer

    # Get enhanced logger for timing
    from .logging_config import get_hypersloth_logger

    hp_logger = get_hypersloth_logger(log_level="INFO")

    # Start timing for the overall dataset loading process
    hp_logger.start_timing("dataset_loading_total")

    LOCAL_RANK = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    lock = dataset_cache_path + ".lock"

    def _create_trainer(train_dataset, skip_prepare=True):
        """Helper to build an SFTTrainer from given train/eval datasets."""
        # We use a dataset_kwargs override so that, once the dataset is prepared,
        # it won't attempt the same "prepare" logic again.
        hf_train_args.dataset_kwargs = {"skip_prepare_dataset": skip_prepare}
        hf_train_args.dataset_batch_size = hf_train_args.per_device_train_batch_size
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            dataset_num_proc=hyper_config.data.dataset_num_proc,
            args=hf_train_args,
        )

    hyper_config.data = parse_data(hyper_config.data)

    # Ensure path_tokenized is set and valid
    if not hyper_config.data.path_tokenized:
        raise ValueError(
            f"Dataset path_tokenized is None. "
            f"Please ensure the dataset was properly built and registered."
        )

    if not os.path.exists(hyper_config.data.path_tokenized):
        raise FileNotFoundError(
            f"Tokenized dataset not found at {hyper_config.data.path_tokenized}. "
            f"Please rebuild the dataset or check the path."
        )

    dataset = load_from_disk(hyper_config.data.path_tokenized)
    trainer = _create_trainer(dataset, skip_prepare=True)

    _maybe_train_on_responses_only(trainer, hyper_config)
    hp_logger.finish_timing("dataset_loading_total")
    return trainer


def _maybe_train_on_responses_only(trainer, hyper_config: HyperConfig):
    """Use a specialized approach if 'response_only' loss is desired."""
    if hyper_config.training.loss_type == "response_only":
        from unsloth.chat_templates import train_on_responses_only

        first_text = trainer.train_dataset[0]["text"]
        instruction_part = hyper_config.data.instruction_part
        response_part = hyper_config.data.response_part
        assert instruction_part is not None, f"instruction_part is None"
        assert response_part is not None, f"response_part is None"
        assert instruction_part in first_text, f"{instruction_part} not in {first_text}"
        assert response_part in first_text, f"{response_part} not in {first_text}"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
            num_proc=hyper_config.data.dataset_num_proc,
        )
    return trainer


def configure_batch_size(hf_train_args, gpu_ith, num_gpus):
    if num_gpus != 1:
        hf_train_args.per_device_train_batch_size *= num_gpus  # This is the total batch size loaded by dataloader, the trainer later will chose the correct batch size for each GPU
        global_batch_size = (
            hf_train_args.per_device_train_batch_size
            * hf_train_args.gradient_accumulation_steps
            * num_gpus
        )
        logger.info(f"Global batch size: {global_batch_size} ")
    if not gpu_ith == 0:
        # disable reporting for all GPUs except the first one
        hf_train_args.report_to = "none"
        # disable evaluation for all GPUs except the first one
        hf_train_args.do_eval = False


__all__ = [
    "configure_batch_size",
    "init_model_and_tokenizer",
    "create_trainer",
]
