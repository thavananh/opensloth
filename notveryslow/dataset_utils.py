from llm_utils import get_conversation_one_turn


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
