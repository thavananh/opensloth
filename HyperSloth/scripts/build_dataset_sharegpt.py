import datasets
from speedy_utils import load_by_ext

from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# from build_dataset import prepare_dataset

def get_std_chat_dataset_from_path(path: str):
    """
    Load and standardize a chat dataset from a given path.
    """

    path = '/mnt/data/sharegpt/selfeval_retranslate_2025_05_30.json'
    list_items = load_by_ext(path)
    dataset = datasets.Dataset.from_list(list_items)
    if 'messages' in dataset.column_names:
        dataset = dataset.rename_column('messages', 'conversations')
    assert 'conversations' in dataset.column_names, "Dataset must contain 'conversations' column"
    return dataset

    