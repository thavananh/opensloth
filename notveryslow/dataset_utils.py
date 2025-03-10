def get_alpaca(tokenizer, nsplits=1, split=0, test_ratio=0.1):

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
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
    if nsplits>2:
        # split train_ds
        train_ds.shuffle(seed=42)
        ids = list(range(len(train_ds)))
        this_split_ids = ids[split::nsplits]
        train_ds = train_ds.select(this_split_ids)
    return train_ds, dataset["test"]


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test = get_alpaca(tokenizer, nsplits=2, split=0, test_ratio=0.1)