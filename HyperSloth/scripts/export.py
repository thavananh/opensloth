from fastcore.all import call_parse
from speedy_utils.all import load_by_ext, logger


@call_parse
def merge_and_save_lora(
    lora_path: str,
    base_model_name_or_path: str = None,
    output_path: str = None,
    force_use_unsloth: bool = False,
) -> None:
    """
    Merges a LoRA model with its base model and saves the result.

    Args:
        base_model_name: Name of the base model on HuggingFace Hub
        lora_path: Local path to the LoRA adapter weights
        output_path: Where to save the merged model (defaults to lora_path + "-merged")
    """
    import os

    assert os.path.exists(lora_path), f"LoRA model not found at {lora_path}"
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from unsloth import FastModel

    # /adapter_config.json
    config = load_by_ext(lora_path + "/adapter_config.json")
    base_model_name_or_path = (
        base_model_name_or_path or config["base_model_name_or_path"]
    )
    if base_model_name_or_path.endswith("-bnb-4bit"):
        base_model_name_or_path = base_model_name_or_path.split("-bnb-4bit")[0]

    if "qwen" in base_model_name_or_path.lower():
        # replace unsloth/ with Qwen/
        base_model_name_or_path = base_model_name_or_path.replace("unsloth/", "Qwen/")
    if "gemma" in base_model_name_or_path.lower():
        # replace unsloth/ with Gemma/
        base_model_name_or_path = base_model_name_or_path.replace("unsloth/", "Google/")
    logger.info(f"Base model: {base_model_name_or_path}")

    if output_path is None:
        output_path = f"{lora_path}-merged"

    # Load the LoRA model
    if "gemma-3" in base_model_name_or_path.lower() or force_use_unsloth:
        logger.info("Using FastModel for LoRA")
        model, tokenizer = FastModel.from_pretrained(lora_path, load_in_4bit=False)
        model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")

    else:
        from transformers import AutoModelForCausalLM

        logger.info("Using PeftModel for LoRA")
        # model = AutoPeftModelForCausalLM.from_pretrained(
        #     lora_path, device_map="auto", trust_remote_code=True
        # ).eval()
        model = AutoModelForCausalLM.from_pretrained(
            lora_path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        from peft.peft_model import PeftModel

        lora_model = PeftModel.from_pretrained(model, lora_path)

        # Merge the LoRA weights with the base model
        merged_model = lora_model.merge_and_unload()

        # Save the merged model
        merged_model.save_pretrained(
            output_path, max_shard_size="2048MB", safe_serialization=True
        )

        # Save the tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")


def main():
    merge_and_save_lora()
