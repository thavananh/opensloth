from fastcore.all import call_parse

@call_parse
def merge_and_save_lora(
    base_model_name: str, lora_path: str, output_path: str = None
) -> None:
    """
    Merges a LoRA model with its base model and saves the result.

    Args:
        base_model_name: Name of the base model on HuggingFace Hub
        lora_path: Local path to the LoRA adapter weights
        output_path: Where to save the merged model (defaults to lora_path + "-merged")
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    if output_path is None:
        output_path = f"{lora_path}-merged"

    # Load the LoRA model
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path, device_map="auto", trust_remote_code=True
    ).eval()

    # Merge the LoRA weights with the base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained(
        output_path, max_shard_size="2048MB", safe_serialization=True
    )

    # Save the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

def main():
    merge_and_save_lora()