from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from loguru import logger
from transformers import DataCollatorForLanguageModeling

RANDOM_SEED = 3407


def load_model_and_tokenizer() -> Tuple[Any, Any]:
    """Load model and tokenizer with proper CUDA initialization."""
    # Import CUDA-related libraries inside function for proper GPU initialization
    import torch
    from unsloth import FastLanguageModel

    logger.info("Loading model and tokenizer")

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
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def main() -> None:
    """Main training function."""
    # Import CUDA-related libraries inside function
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    from unsloth.chat_templates import standardize_sharegpt

    logger.info("Loading raw splits")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Load dataset
    raw_chat = load_dataset("mlabonne/FineTome-100k", split="train")
    chat_std = standardize_sharegpt(raw_chat).remove_columns(["source", "score"])

    def tokenize_on_the_fly(sample: Dict[str, str]) -> Dict[str, List[int]]:
        """Tokenize sample on the fly."""
        text = tokenizer.apply_chat_template(sample["conversations"], tokenize=False)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,  # padding happens in the collator
            return_attention_mask=True,
        )
        return encoded

    chat_tokenized = chat_std.map(tokenize_on_the_fly, batched=False)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create trainer
    trainer = UnslothTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=chat_tokenized,  # type: ignore
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
            remove_unused_columns=False,
        ),
        data_collator=collator,
    )

    # Get next batch for testing
    next_batch = next(iter(trainer.get_train_dataloader()))
    logger.info(f"Successfully loaded batch with keys: {list(next_batch.keys())}")


if __name__ == "__main__":
    main()
