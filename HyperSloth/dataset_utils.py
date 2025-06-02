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
    if 'messages' in std_chat_dataset.column_names:
        std_chat_dataset = std_chat_dataset.rename_column('messages', 'conversations')

    assert 'conversations' in std_chat_dataset.column_names, \
        f"Dataset must contain 'conversations' column, found: {std_chat_dataset.column_names}"
    
    if num_samples is not None:
        std_chat_dataset = std_chat_dataset.select(range(min(num_samples, len(std_chat_dataset))))
    
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
    output_path = HYPERSLOTH_DATA_DIR / 'datasets' / cache_id
    lock_path = output_path.with_suffix('.lock')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        from datasets import load_from_disk
        return load_from_disk(str(output_path))
    
    with open(lock_path, 'w') as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            if output_path.exists():
                from datasets import load_from_disk
                return load_from_disk(str(output_path))
            
            text_dataset = prepare_func()
            text_dataset.save_to_disk(str(output_path))
            return text_dataset
                
        except BlockingIOError:
            while not output_path.exists():
                time.sleep(1)
            from datasets import load_from_disk
            return load_from_disk(str(output_path))
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
    cache_id = identify([dataset_name, split, tokenizer_name, chat_template, num_samples])
    
    def prepare_func():
        std_chat_dataset = _get_std_chat_dataset_from_hf(dataset_name, split=split)
        return prepare_text_dataset(
            std_chat_dataset=std_chat_dataset,
            tokenizer_name=tokenizer_name,
            chat_template=chat_template,
            num_samples=num_samples,
        )
    
    return _get_cached_dataset(cache_id, prepare_func)


def get_text_dataset_from_path(
    path: str,
    tokenizer_name: str,
    chat_template: str,
    num_samples: Optional[int] = None,
):
    """Load a text dataset from local path."""
    cache_id = identify([path, tokenizer_name, chat_template, num_samples])
    
    def prepare_func():
        std_chat_dataset = _get_std_chat_dataset_from_path(path)
        return prepare_text_dataset(
            std_chat_dataset=std_chat_dataset,
            tokenizer_name=tokenizer_name,
            chat_template=chat_template,
            num_samples=num_samples,
        )
    
    return _get_cached_dataset(cache_id, prepare_func)


def get_text_dataset(config: DatasetConfig):
    """
    Load and prepare a text dataset from either HuggingFace or local path.
    
    Args:
        config: DatasetConfig object specifying the dataset source and parameters
        
    Returns:
        Dataset: Dataset with 'text' column containing formatted conversations
    """
    if config.source_type == 'hf':
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


__all__ = [
    "get_text_dataset",
]