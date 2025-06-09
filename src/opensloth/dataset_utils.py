import fcntl
import time
from typing import Optional, Union, List, Tuple, Callable, Any

import datasets
from speedy_utils import identify, load_by_ext

from opensloth import HYPERSLOTH_DATA_DIR
from opensloth.logging_config import get_opensloth_logger
from opensloth.opensloth_config import HFDatasetConfig, PathDatasetConfig

DatasetConfig = Union[HFDatasetConfig, PathDatasetConfig]


def _get_tokenizer(
    tokenizer_name: str,
    chat_template: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Any:
    """Get tokenizer with optional chat template."""
    from transformers import AutoTokenizer
    from unsloth.chat_templates import get_chat_template

    if not tokenizer_name:
        raise ValueError("Tokenizer name must be provided and non-empty")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    if chat_template is not None:
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    return tokenizer


def _get_std_chat_dataset_from_hf(dataset_name: str, split: str) -> datasets.Dataset:
    """Load and standardize dataset from HuggingFace."""
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_data_formats

    dataset = load_dataset(dataset_name, split=split)
    return standardize_data_formats(dataset)


def _get_std_chat_dataset_from_path(path: str) -> datasets.Dataset:
    """Load dataset from local path."""
    return datasets.Dataset.from_list(load_by_ext(path))


def _compute_labels(
    input_ids: List[int], response_part: str, instruction_part: str, tokenizer: Any
) -> List[int]:
    """Compute labels for response-only training."""
    from unsloth_zoo.dataset_utils import _find_common_token_ids

    def get_token_pattern(text: str) -> Tuple[List[int], List[int], List[int]]:
        must, left, right = _find_common_token_ids(text, tokenizer, force_match=True)
        return must, left[::-1], right

    response_must, response_left_rev, response_right = get_token_pattern(response_part)
    instruction_must, instruction_left_rev, instruction_right = get_token_pattern(
        instruction_part
    )

    labels = [-100] * len(input_ids)
    n = len(input_ids)
    j = 0

    while j < n:
        if input_ids[j : j + len(response_must)] == response_must:
            assistant_start = j
            assistant_end = j + len(response_must)

            # Find response start boundary
            for token in response_left_rev:
                if assistant_start > 0 and input_ids[assistant_start - 1] == token:
                    assistant_start -= 1
                else:
                    break

            # Find response end boundary
            while assistant_end < n - 1 and input_ids[assistant_end] in response_right:
                assistant_end += 1

            # Find next instruction start
            next_instruction_start = n
            k = assistant_end
            while k < n:
                if input_ids[k : k + len(instruction_must)] == instruction_must:
                    next_instruction_start = k
                    for token in instruction_left_rev:
                        if (
                            next_instruction_start > 0
                            and input_ids[next_instruction_start - 1] == token
                        ):
                            next_instruction_start -= 1
                        else:
                            break

                    k = next_instruction_start + len(instruction_must)
                    while k < n - 1 and input_ids[k] in instruction_right:
                        k += 1
                    break
                k += 1

            # Set labels for response tokens
            labels[assistant_end:next_instruction_start] = input_ids[
                assistant_end:next_instruction_start
            ]
            j = next_instruction_start
        else:
            j += 1

    return labels


def _prepare_text_dataset(
    std_chat_dataset: datasets.Dataset,
    tokenizer_name: str,
    chat_template: Optional[str] = None,
    num_samples: Optional[int] = None,
    nproc: Optional[int] = None,
) -> datasets.Dataset:
    """Prepare text dataset by applying chat templates."""
    # Validate inputs
    if "text" in std_chat_dataset.column_names:
        return std_chat_dataset

    if not tokenizer_name:
        raise ValueError("Tokenizer name must be provided and non-empty")

    # Normalize column names
    if "messages" in std_chat_dataset.column_names:
        std_chat_dataset = std_chat_dataset.rename_column("messages", "conversations")

    if "conversations" not in std_chat_dataset.column_names:
        raise ValueError(
            f"Dataset must contain conversations column, "
            f"found: {std_chat_dataset.column_names}"
        )

    # Sample subset if requested
    if num_samples is not None and num_samples > 0:
        available_samples = len(std_chat_dataset)
        if num_samples > available_samples:
            raise ValueError(
                f"Requested {num_samples} samples but dataset only has "
                f"{available_samples} samples"
            )
        std_chat_dataset = std_chat_dataset.shuffle(seed=42).select(range(num_samples))

    tokenizer = _get_tokenizer(tokenizer_name, chat_template)

    def formatting_prompts_func(examples: dict) -> dict:
        return {
            "text": [
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                for convo in examples["conversations"]
            ]
        }

    return std_chat_dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=nproc,
        desc="Applying chat templates",
    )


def _get_cached_dataset(
    cache_id: str, prepare_func: Callable[[], datasets.Dataset]
) -> datasets.Dataset:
    """Get cached dataset or prepare and cache new one."""
    from datasets import load_from_disk

    if not cache_id or not cache_id.strip():
        raise ValueError("Cache ID must be provided and non-empty")

    output_path = HYPERSLOTH_DATA_DIR / "datasets" / cache_id
    lock_path = output_path.with_suffix(".lock")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_opensloth_logger(log_level="INFO")
    max_retries = 30
    retry_delay = 3

    # Return existing dataset if available
    if output_path.exists():
        try:
            return load_from_disk(str(output_path))
        except Exception as e:
            logger.warning(
                f"Failed to load existing dataset {cache_id}: {e}. " f"Will regenerate."
            )
            # Remove corrupted cache
            import shutil

            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)

    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Double-check after acquiring lock
        if output_path.exists():
            try:
                logger.info(f"Dataset {cache_id} already exists, loading.")
                return load_from_disk(str(output_path))
            except Exception as e:
                logger.warning(
                    f"Failed to load existing dataset {cache_id}: {e}. "
                    f"Will regenerate."
                )
                import shutil

                if output_path.exists():
                    shutil.rmtree(output_path, ignore_errors=True)

        logger.info(f"Preparing dataset {cache_id}...")
        dataset = prepare_func()

        if dataset is None:
            raise RuntimeError("prepare_func returned None")

        dataset.save_to_disk(str(output_path))
        return dataset

    except BlockingIOError:
        logger.info(f"Dataset {cache_id} being prepared by another process, waiting...")
        # Wait for other process to complete
        while not output_path.exists() and lock_path.exists():
            time.sleep(1)

        # Retry loading with backoff
        for attempt in range(max_retries):
            try:
                return load_from_disk(str(output_path))
            except Exception as e:
                logger.warning(
                    f"Failed to load dataset {cache_id}: {str(e)[:100]}. "
                    f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)

        raise RuntimeError(
            f"Failed to load dataset {cache_id} after {max_retries} retries"
        )
    finally:
        if lock_file is not None:
            try:
                lock_file.close()
            except Exception:
                pass  # Ignore errors during cleanup
        if lock_path.exists():
            try:
                lock_path.unlink()
            except Exception:
                pass  # Ignore errors during cleanup


def get_text_dataset(config: DatasetConfig) -> datasets.Dataset:
    """Get text dataset from config (HF or local path)."""
    # Validate config attributes explicitly
    if not hasattr(config, "source_type"):
        raise ValueError("Config must have source_type attribute")

    if config.source_type == "hf":
        # Validate HF-specific attributes
        if not hasattr(config, "dataset_name") or not config.dataset_name:
            raise ValueError("HF config must have non-empty dataset_name")
        if not hasattr(config, "split") or not config.split:
            raise ValueError("HF config must have non-empty split")

        std_chat_dataset = _get_std_chat_dataset_from_hf(
            config.dataset_name, config.split
        )
    elif config.source_type == "path":
        # Validate path-specific attributes
        if not hasattr(config, "path") or not config.path:
            raise ValueError("Path config must have non-empty path")

        std_chat_dataset = _get_std_chat_dataset_from_path(config.path)
    else:
        raise ValueError(
            f"Unknown source_type: {config.source_type}. " f'Must be "hf" or "path"'
        )

    # Validate required attributes for text processing
    if not hasattr(config, "tokenizer_name") or not config.tokenizer_name:
        raise ValueError("Config must have non-empty tokenizer_name")

    return _prepare_text_dataset(
        std_chat_dataset=std_chat_dataset,
        tokenizer_name=config.tokenizer_name,
        chat_template=config.chat_template,
        num_samples=config.num_samples,
        nproc=config.nproc,
    )


def get_tokenized_dataset(
    config: DatasetConfig,
    do_tokenize: bool = True,
    response_only: bool = True,
) -> datasets.Dataset:
    """Get tokenized dataset with optional response-only labeling."""
    # Validate config
    if not hasattr(config, "max_seq_length"):
        raise ValueError("Config must have max_seq_length attribute")

    if config.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    ignore_keys = ["nproc"]
    _config = config.model_dump()
    for key in ignore_keys:
        if key in _config:
            del _config[key]
    cache_id = identify(
        [
            _config,
            do_tokenize,
            response_only,
            config.max_seq_length,
        ]
    )

    def _prepare() -> datasets.Dataset:
        text_ds = get_text_dataset(config)

        if not do_tokenize:
            return text_ds

        # Validate tokenizer attributes
        if not hasattr(config, "tokenizer_name") or not config.tokenizer_name:
            raise ValueError(
                "Config must have non-empty tokenizer_name for tokenization"
            )

        tokenizer = _get_tokenizer(
            config.tokenizer_name, getattr(config, "chat_template", None)
        )

        def _pipeline(example: dict) -> dict:
            if "text" not in example:
                raise ValueError('Example must contain "text" field')

            encoded = tokenizer(
                example["text"], truncation=True, max_length=config.max_seq_length
            )

            if response_only:
                # Validate required attributes for response-only training
                if not hasattr(config, "response_part") or not config.response_part:
                    raise ValueError(
                        "Config must have non-empty response_part for response-only training"
                    )
                if (
                    not hasattr(config, "instruction_part")
                    or not config.instruction_part
                ):
                    raise ValueError(
                        "Config must have non-empty instruction_part for response-only training"
                    )

                encoded["labels"] = _compute_labels(
                    encoded["input_ids"],
                    response_part=config.response_part,
                    instruction_part=config.instruction_part,
                    tokenizer=tokenizer,
                )
            return encoded

        nproc = config.nproc or 1
        return text_ds.map(
            _pipeline,
            num_proc=nproc,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )

    return _get_cached_dataset(cache_id, _prepare)


__all__ = ["get_tokenized_dataset"]
