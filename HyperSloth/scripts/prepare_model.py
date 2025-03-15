from fastcore.all import *


@call_parse
def main(
    model_name: str,
    output_dir: str,
    tokenizer_name: str = None,
    chat_template: str = None,
    save_in_4bit: bool = True,
) -> None:
    """
    Prepare and save a model and tokenizer.

    Args:
        model_name (str): The name of the model to load.
        output_dir (str): The directory to save the model and tokenizer.
        tokenizer_name (str, optional): The name of the tokenizer to load. Defaults to None.
        chat_template (str, optional): The chat template to load. Defaults to None.
        save_in_4bit (bool, optional): Whether to save the model in 4-bit format. Defaults to True.
    """
    import os
    from transformers import AutoModel, AutoTokenizer
    import torch
    from unsloth import FastLanguageModel


    os.makedirs(output_dir, exist_ok=True)
    if tokenizer_name is None:
        tokenizer_name = model_name
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if chat_template is not None:
        print("Loading chat template")
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            chat_template, torch_dtype=torch.bfloat16
        ).chat_template

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if save_in_4bit:
        print("Saving model in 4bit")
        if output_dir.endswith("/"):
            output_dir = output_dir[:-1]
        output_dir_4bit = output_dir + "-bnb-4bit"
        os.makedirs(output_dir_4bit, exist_ok=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=output_dir,
        )
        model.save_pretrained(output_dir_4bit)
        tokenizer.save_pretrained(output_dir_4bit)
