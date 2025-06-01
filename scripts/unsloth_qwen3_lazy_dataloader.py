from typing import Any, Dict, List

import torch
from datasets import load_dataset
from loguru import logger
from transformers import DataCollatorForLanguageModeling
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

RANDOM_SEED = 3407
logger.info("Loading raw splits")

# -------------------------------------------------------------------
# 0. Model / tokenizer (needed inside the converters)


# -------------------------------------------------------------------
# 1. Load datasets
# -------------------------------------------------------------------
if not "model" in dir():
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="model_store/unsloth/Qwen3-0.6B-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


raw_chat = load_dataset("mlabonne/FineTome-100k", split="train")
chat_std = standardize_sharegpt(raw_chat).remove_columns(["source", "score"])


def tokenize_on_the_fly(sample: Dict[str, str]) -> Dict[str, List[int]]:
    text = tokenizer.apply_chat_template(sample["conversations"], tokenize=False)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding=False,  # padding happens in the collator
        return_attention_mask=True,
    )

    return encoded


chat_std.set_transform(tokenize_on_the_fly)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# -------------------------------------------------------------------
# 2. Trainer
# -------------------------------------------------------------------
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=chat_std,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        max_steps=30,
        warmup_steps=5,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=RANDOM_SEED,
        logging_steps=1,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        # <-- critical line
        remove_unused_columns=False,
    ),
    data_collator=collator,
)
next_batch = next(iter(trainer.get_train_dataloader()))
# trainer.train()
