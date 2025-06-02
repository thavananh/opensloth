import unsloth # for it to patchSFTTrainer
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from fastcore.all import dict2obj
import torch


tokenizer_name='unsloth/Qwen3-0.6B-bnb-4bit'
model, tokenizer = unsloth.FastModel.from_pretrained(
    tokenizer_name)

model = unsloth.FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
def preproc_compltion(model, tokenizer, dataset, max_seq_length, instruction_part, response_part):
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part = response_part,
    )
    return trainer.train_dataset


from datasets import load_from_disk
dataset = load_from_disk('~/.cache/hypersloth/finetome-1k_tokenized/')

output_dataset = preproc_compltion(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    max_seq_length=1024,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)