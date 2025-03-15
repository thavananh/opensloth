hyper_config = dict(
    grad_dir="/dev/shm/hypersloth",
    data=dict(
        dataset="data/cod_1k.json",
        test_ratio=0.05,
        dataset_num_proc=4,
    ),
    training=dict(
        gpus=[0],
        loss_type="all",  # Fixed argument reference
        packing=False,
    ),
    # =====
    fast_model_args=dict(
        model_name="unsloth/gemma-3-4b-it",
        max_seq_length=2048,  # Choose any for long context!
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    ),
    lora_args=dict(
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=8,  # Larger = higher accuracy, but might overfit
        lora_alpha=8,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
    ),
    
)

# MUST NOT INITIALIZE DEVICE BEFORE threaded.run() IN HyperSloth/scripts/hypersloth.py
training_config = dict(
    output_dir="model_training_outputs/debug",
    per_device_train_batch_size=8,
    learning_rate=0.0002,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=2,
    eval_steps=100,
    logging_steps=1,
    report_to="tensorboard",
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    seed=42,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",
    weight_decay=0.01,
)
