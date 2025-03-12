"""
Some HF model does not have proper chat template, so we need to fix it and dump the model in 4bit
"""

from transformers import AutoModel, AutoTokenizer
import torch
from unsloth import FastLanguageModel
from speedy_utils.all import *


def prepare_model(
    model_name, output_dir, tokenizer_name=None, chat_template=None, save_in_4bit=True
):
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


model = "ModelSpace/GemmaX2-28-9B-v0.1"
tokenizer = None
chat_template = "google/gemma-2-9b-it"
output_dir = "/mnt/data/huggingface-models/ModelSpace/GemmaX2-28-9B-v0.1"
prepare_model(model, output_dir, tokenizer, chat_template, save_in_4bit=True)
