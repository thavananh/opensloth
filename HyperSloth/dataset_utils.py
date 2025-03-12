from typing import Any, Dict, List, Tuple

from datasets import Dataset
from llm_utils import get_conversation_one_turn
from speedy_utils import load_by_ext

from .think_chat_template_tokenier_fix import fix_think_chat_template_tokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_sharegpt_dataset(file, tokenizer, test_ratio=0.052):
    # Load and shard dataset for this GPU
    files = file.split(",")
    dataset_raw = []
    for file in files:
        dataset_raw += load_by_ext(file)

    def format_chat_template(row: Dict[str, Any]) -> Dict[str, Any]:
        row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row

    dataset_raw = [format_chat_template(row) for row in dataset_raw]

    ds = Dataset.from_list(dataset_raw)

    ds = ds.train_test_split(test_size=test_ratio, seed=42)
    return ds["train"], ds["test"]


def get_alpaca(tokenizer, test_ratio=0.1):
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            messages = get_conversation_one_turn(
                instruction,
                input,
                output,
            )
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return {
            "text": texts,
        }

    pass

    from datasets import load_dataset

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    # split train val
    dataset = dataset.train_test_split(test_size=test_ratio, seed=42)
    train_ds = dataset["train"]
    # random shufle train_ds
    train_ds = train_ds.shuffle(seed=42)
    return train_ds, dataset["test"]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    train_ds, test_ds = get_alpaca(tokenizer, nsplits=2, split=0, test_ratio=0.1)
