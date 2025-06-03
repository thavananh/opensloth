import fcntl
import time
from typing import Optional, Union

from speedy_utils import identify

from HyperSloth import HYPERSLOTH_DATA_DIR
from HyperSloth.hypersloth_config import HFDatasetConfig, PathDatasetConfig


DatasetConfig = Union[HFDatasetConfig, PathDatasetConfig]


def _get_tokenizer(tokenizer_name: str, chat_template: str):
    from transformers import AutoTokenizer
    from unsloth.chat_templates import get_chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )
    return tokenizer


def _get_std_chat_dataset_from_hf(dataset_name, split):
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_data_formats

    # Load and standardize dataset
    dataset = load_dataset(dataset_name, split=split)
    dataset = standardize_data_formats(dataset)
    return dataset


def _get_std_chat_dataset_from_path(path: str):
    """
    Load and standardize a chat dataset from a given path.
    """
    import datasets
    from speedy_utils import load_by_ext

    list_items = load_by_ext(path)
    dataset = datasets.Dataset.from_list(list_items)
    return dataset


def prepare_text_dataset(
    std_chat_dataset,
    tokenizer_name: str,
    chat_template: str,
    num_samples: Optional[int] = None,
):
    """
    Prepare a chat dataset by formatting conversations into text.
    """
    if "messages" in std_chat_dataset.column_names:
        std_chat_dataset = std_chat_dataset.rename_column("messages", "conversations")

    assert (
        "conversations" in std_chat_dataset.column_names
    ), f"Dataset must contain 'conversations' column, found: {std_chat_dataset.column_names}"

    if num_samples is not None:
        std_chat_dataset = std_chat_dataset.select(
            range(min(num_samples, len(std_chat_dataset)))
        )

    tokenizer = _get_tokenizer(tokenizer_name, chat_template)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    text_dataset = std_chat_dataset.map(formatting_prompts_func, batched=True)
    return text_dataset


def _get_cached_dataset(cache_id: str, prepare_func):
    """Generic function to handle dataset caching and locking."""
    output_path = HYPERSLOTH_DATA_DIR / "datasets" / cache_id
    lock_path = output_path.with_suffix(".lock")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .logging_config import get_hypersloth_logger

    logger = get_hypersloth_logger(log_level="INFO")
    if output_path.exists():
        from datasets import load_from_disk

        return load_from_disk(str(output_path))

    with open(lock_path, "w") as lock_file:
        try:

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            if output_path.exists():
                from datasets import load_from_disk

                logger.info(f"Dataset {cache_id} already exists, loading from disk.")
                return load_from_disk(str(output_path))
            logger.info(f"Preparing dataset {cache_id}...")
            text_dataset = prepare_func()
            text_dataset.save_to_disk(str(output_path))
            return text_dataset

        except BlockingIOError:
            printed = False
            while not output_path.exists() and lock_path.exists():
                if not printed:
                    printed = True
                    logger.info(
                        f"Dataset {cache_id} is being prepared by another process, waiting..."
                    )

                time.sleep(1)
            from datasets import load_from_disk

            for i in range(30):
                try:
                    return load_from_disk(str(output_path))
                except Exception as e:
                    err = str(e)[:100]
                    logger.warning(
                        f"Failed to load dataset {cache_id} from disk: {err}. "
                        "Retrying in 3 second..."
                    )
                    time.sleep(3)
        finally:
            if lock_path.exists():
                lock_path.unlink()


def get_text_dataset_hf(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    chat_template: str,
    num_samples: Optional[int] = None,
):
    """Load and prepare a text dataset from Hugging Face."""
    std_chat_dataset = _get_std_chat_dataset_from_hf(dataset_name, split=split)
    return prepare_text_dataset(
        std_chat_dataset=std_chat_dataset,
        tokenizer_name=tokenizer_name,
        chat_template=chat_template,
        num_samples=num_samples,
    )


def get_text_dataset_from_path(
    path: str,
    tokenizer_name: str,
    chat_template: str,
    num_samples: Optional[int] = None,
):
    """Load a text dataset from local path."""
    std_chat_dataset = _get_std_chat_dataset_from_path(path)
    return prepare_text_dataset(
        std_chat_dataset=std_chat_dataset,
        tokenizer_name=tokenizer_name,
        chat_template=chat_template,
        num_samples=num_samples,
    )


def get_text_dataset(config: DatasetConfig):
    """
    Load and prepare a text dataset from either HuggingFace or local path.

    Args:
        config: DatasetConfig object specifying the dataset source and parameters

    Returns:
        Dataset: Dataset with 'text' column containing formatted conversations
    """
    if config.source_type == "hf":
        return get_text_dataset_hf(
            dataset_name=config.dataset_name,
            split=config.split,
            tokenizer_name=config.tokenizer_name,
            chat_template=config.chat_template,
            num_samples=config.num_samples,
        )
    else:  # path
        return get_text_dataset_from_path(
            path=config.path,
            tokenizer_name=config.tokenizer_name,
            chat_template=config.chat_template,
            num_samples=config.num_samples,
        )


def _get_tokenized_dataset(
    config: DatasetConfig,
    model,
    tokenizer,
    hf_train_args,
):
    """
    Create a tokenized dataset using SFTTrainer and response-only training.

    Args:
        config: Dataset configuration
        model: The model to use for training
        tokenizer: The tokenizer to use
        hf_train_args: Training arguments

    Returns:
        The tokenized dataset ready for training
    """
    from trl import SFTTrainer
    from unsloth.chat_templates import train_on_responses_only
    from HyperSloth.logging_config import get_hypersloth_logger

    logger = get_hypersloth_logger(log_level="INFO")

    logger.info("Loading text dataset...")
    train_dataset = get_text_dataset(config)

    logger.info("Starting dataset tokenization...")
    logger.start_timing("dataset_tokenization")

    hf_train_args.skip_prepare_dataset = False
    tmp_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=hf_train_args.max_seq_len,
        args=hf_train_args,
    )

    logger.info(
        f"Applying response-only training with"
        f'"\nInstructionPart:{config.instruction_part}" and '
        f'"\nResponsePart:{config.response_part}"'
    )
    tmp_trainer = train_on_responses_only(
        tmp_trainer,
        instruction_part=config.instruction_part,
        response_part=config.response_part,
    )

    logger.finish_timing("dataset_tokenization")
    logger.info(
        f"Dataset tokenization completed. Dataset size: "
        f"{len(tmp_trainer.train_dataset)}"
    )
    return tmp_trainer.train_dataset


from .hypersloth_config import TrainingArgsConfig


def get_tokenized_dataset(
    data_config: DatasetConfig, model, tokenizer, hf_train_args: TrainingArgsConfig
):
    """
    Get a tokenized dataset ready for training, with caching support.

    Args:
        config: Dataset configuration
        model: The model to use for tokenization
        tokenizer: The tokenizer to use
        hf_train_args: Training arguments

    Returns:
        The tokenized dataset ready for training
    """
    from HyperSloth.logging_config import get_hypersloth_logger

    logger = get_hypersloth_logger(log_level="INFO")

    # Start timing for the overall dataset loading process
    logger.start_timing("dataset_loading_total")

    logger.info("Loading dataset... and tokenize")

    # Create cache ID for tokenized dataset
    cache_id = identify([data_config.model_dump_json(), hf_train_args.max_seq_len])

    def prepare_tokenized_func():
        return _get_tokenized_dataset(
            config=data_config,
            model=model,
            tokenizer=tokenizer,
            hf_train_args=hf_train_args,
        )

    logger.start_timing("dataset_caching")
    tokenized_dataset = _get_cached_dataset(cache_id, prepare_tokenized_func)
    logger.finish_timing("dataset_caching")
    logger.info(
        f"Final dataset loaded with {len(tokenized_dataset)} samples "
        f"(cache_id: {cache_id[:12]}...)"
    )

    logger.finish_timing("dataset_loading_total")
    return tokenized_dataset


__all__ = [
    "get_text_dataset",
    "get_tokenized_dataset",
]
