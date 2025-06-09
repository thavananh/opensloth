# Full fine-tuning configuration with HuggingFace dataset
# To run: opensloth-train example_hf_dataset_full_finetuning.py
from opensloth.opensloth_config import *

# Configuration for full fine-tuning (no LoRA)
opensloth_config = OpenSlothConfig(
    data=HFDatasetConfig(
        dataset_name="mlabonne/FineTome-100k",  # Popular instruction dataset
        tokenizer_name="Qwen/Qwen3-8B",  # Tokenizer from same model family
        num_samples=100,  # Small subset for quick testing
        split="train",  # Dataset split to use
        name="finetome-1k",  # Local name for dataset caching
        # columns=["conversations"],  # Column containing conversation data
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        chat_template="qwen3",
    ),
    training=TrainingConfig(
        gpus=[0],  # Multi-GPU for full fine-tuning
        loss_type="response_only",
    ),
    fast_model_args=FastModelArgs(
        model_name="Qwen/Qwen3-0.6B",  # You need non-quantized model for full FT
        max_seq_length=8000,
        full_finetuning=True,  # Enable full fine-tuning instead of LoRA
    ),
)

# Training arguments for full fine-tuning
training_config = TrainingArgsConfig(
    output_dir="outputs/qwen3-0.6b-openthoughts-full-ft/",
    per_device_train_batch_size=1,  # Very small batch for full FT memory usage
    gradient_accumulation_steps=4,  # Large accumulation for effective batch size
    learning_rate=1e-5,  # Much lower LR for full fine-tuning
    logging_steps=1,
    num_train_epochs=1,  # Fewer epochs for full FT
    lr_scheduler_type="cosine",  # Cosine annealing for full FT
    warmup_steps=100,
    save_total_limit=1,
    weight_decay=0.1,  # Higher weight decay for full FT
    optim="adamw_8bit",
    seed=3407,
    report_to="tensorboard",  # Use wandb for experiment tracking
    max_seq_len=8000,  # Match model's max sequence length
)
