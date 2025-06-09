import argparse
from speedy_utils.all import load_by_ext, logger
import torch
import peft
from transformers import AutoTokenizer


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

    # /adapter_config.json
    config = load_by_ext(lora_path + "/adapter_config.json")
    base_model_name_or_path = (
        base_model_name_or_path or config["base_model_name_or_path"]
    )
    if base_model_name_or_path.endswith("-bnb-4bit"):
        base_model_name_or_path = base_model_name_or_path.split("-bnb-4bit")[0]

    if "qwen" in base_model_name_or_path.lower():
        # replace unsloth/ with Qwen/
        base_model_name_or_path = base_model_name_or_path.replace("unsloth/", "qwen/")
    if "gemma" in base_model_name_or_path.lower():
        # replace unsloth/ with Gemma/
        base_model_name_or_path = base_model_name_or_path.replace("unsloth/", "google/")
    logger.info(f"Base model: {base_model_name_or_path}")

    if output_path is None:
        output_path = f"{lora_path}/merged"
    logger.info(f"Output path: {output_path}")
    # Load the LoRA model
    if "gemma-3" in base_model_name_or_path.lower() or force_use_unsloth:
        logger.info("Using FastModel for LoRA")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(output_path)
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            base_model_name_or_path, device_map="cpu", torch_dtype=torch.bfloat16
        ).eval()
        # model, tokenizer = FastModel.from_pretrained(lora_path, load_in_4bit=False, dtype=torch.bfloat16)
        peft_model = peft.PeftModel.from_pretrained(model, lora_path)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(
            output_path, max_shard_size="5048MB", safe_serialization=True
        )

        file = "https://huggingface.co/unsloth/gemma-3-12b-it/raw/main/preprocessor_config.json"
        # download the file and put to the lora dir
        cmd = "wget --no-check-certificate -O {}/preprocessor_config.json {}".format(
            lora_path, file
        )
        os.system(cmd)
        # response = requests.get(file)
        # with open(os.path.join(lora_path, "preprocessor_config.json"), "wb") as f:
        #     f.write(response.content)

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
        merged_model = merged_model.to(torch.bfloat16)
        merged_model.save_pretrained(
            output_path, max_shard_size="2048MB", safe_serialization=True
        )
        # to bf16

        # Save the tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA model with base model and save result"
    )
    parser.add_argument(
        "lora_path", type=str, help="Local path to the LoRA adapter weights"
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="Name of the base model on HuggingFace Hub",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help='Where to save the merged model (defaults to lora_path + "/merged")',
    )
    parser.add_argument(
        "--force_use_unsloth",
        action="store_true",
        help="Force use of unsloth FastModel",
    )

    args = parser.parse_args()
    merge_and_save_lora(
        lora_path=args.lora_path,
        base_model_name_or_path=args.base_model_name_or_path,
        output_path=args.output_path,
        force_use_unsloth=args.force_use_unsloth,
    )


if __name__ == "__main__":
    main()
