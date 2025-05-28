# -*- coding: utf-8 -*-
"""Minimal Qwen3 training script using Unsloth."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train_qwen3_model():
    """Train Qwen3 model with minimal setup."""
    # Import CUDA-related libraries inside function for proper GPU initialization
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset, Dataset
    from unsloth.chat_templates import standardize_sharegpt
    from trl import SFTTrainer, SFTConfig
    import pandas as pd

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

    # Load datasets
    reasoning_dataset = load_dataset(
        "unsloth/OpenMathReasoning-mini", split="cot"
    ).select(range(1000))
    non_reasoning_dataset = load_dataset(
        "mlabonne/FineTome-100k", split="train"
    ).select(range(1000))

    # Convert reasoning dataset to conversational format
    def generate_conversation(examples):
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append(
                [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            )
        return {"conversations": conversations}

    reasoning_conversations = tokenizer.apply_chat_template(
        reasoning_dataset.map(generate_conversation, batched=True)["conversations"],
        tokenize=False,
    )

    # Convert non-reasoning dataset
    dataset = standardize_sharegpt(non_reasoning_dataset)
    non_reasoning_conversations = tokenizer.apply_chat_template(
        dataset["conversations"],
        tokenize=False,
    )

    # Mix datasets (25% reasoning, 75% chat)
    chat_percentage = 0.75
    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations) * (1.0 - chat_percentage)),
        random_state=2407,
    )

    # Combine datasets
    data = pd.concat(
        [pd.Series(reasoning_conversations), pd.Series(non_reasoning_subset)]
    )
    data.name = "text"
    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=3407)

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train the model
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_qwen3_model()
    print("Training completed successfully!")
