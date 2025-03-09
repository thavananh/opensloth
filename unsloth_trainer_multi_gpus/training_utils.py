import os
import time
from transformers import (TrainerCallback,
                        TrainerState)
from collections import defaultdict
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from loguru import logger
from speedy_utils.all import load_by_ext
# 
# 


def create_weight(all_gpus_weights: list[str], target_weights: str) -> None:
    """Merge weights from multiple GPUs and save to a target file."""

    weights = defaultdict(list)

    # First check if all files exist before proceeding
    for weight in all_gpus_weights:
        wait_start = time.time()
        while not os.path.exists(weight):
            if time.time() - wait_start > 1800:  # 30 minute timeout
                logger.warning(f"Timeout waiting for {weight}")
                raise FileNotFoundError(f"Timeout waiting for {weight}")
            logger.debug(f"Waiting for {weight} to be created")
            time.sleep(1)
        # wait for the lock to release
        lock_file = f"{weight}.lock"
        wait_start = time.time()
        while os.path.exists(lock_file):
            if time.time() - wait_start > 60:
                logger.warning(f"Timeout waiting for lock to be released on {weight}")
                raise FileExistsError(f"Timeout waiting for lock to be released on {weight}")
            logger.debug(f"Waiting for lock to be released on {weight}")
            time.sleep(1)
        state_dict = torch.load(weight)
        for k, v in state_dict.items():
            weights[k].append(v)

    # Merge the weights
    merged_weights = {}
    for k, v in weights.items():
        merged_weights[k] = torch.stack(v).mean(0)

    # Create lock file then save weights
    lock_file = f"{target_weights}.lock"
    with open(lock_file, "w") as f:
        f.write("lock")

    # Save the weights
    torch.save(merged_weights, target_weights)

    # Remove lock file when done
    os.remove(lock_file)
    logger.debug(f"Saved {target_weights}")


def wait_and_load(model, target_weights: str) -> None:
    """Wait for target weights file to be fully written and available."""

    lock_file = f"{target_weights}.lock"

    # Wait for file to be created
    wait_start = time.time()

    while not os.path.exists(target_weights):
        if time.time() - wait_start > 1800:  # 30 minute timeout
            logger.warning(f"Timeout waiting for {target_weights}")
            return
        logger.debug(f"Waiting for {target_weights} to be created")
        time.sleep(1)

    # # Wait for lock file to be removed (indicating write is complete)
    wait_start = time.time()
    while os.path.exists(lock_file):
        if time.time() - wait_start > 60:  # 1 minute timeout
            logger.warning(
                f"Timeout waiting for lock to be released on {target_weights}"
            )
            return
        logger.debug(f"Waiting for lock to be released on {target_weights}")
        time.sleep(1)
    logger.debug(f"Loading weights from {target_weights} to {model.device}")
    # import ipdb; ipdb.set_trace()
    weight = torch.load(target_weights, map_location=f"cuda:{model.device.index}")
    ret = model.load_state_dict(weight, strict=False)
    num_loaded = len(ret[0])
    num_missing = len(ret[1])
    logger.debug(f"Loaded {num_loaded} weights, {num_missing} missing")


def prepare_input(gpu_id:str, all_gpus:str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
        # model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=16_000,
        dtype=None,
        # device_map={"": gpu_id},
    )

    # Importing the dataset
    dataset_raw = load_by_ext("./data/cod_6k5.json")[:100]
    idx = all_gpus.index(gpu_id)
    dataset_raw = dataset_raw[::len(all_gpus)]

    template_no_remove_think_tags = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt =
    false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false,
    is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if
    message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%-
    endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in
    messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false
    -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] ==
    'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%-
    for tool in message['tool_calls']%}{%- if not ns.is_first
    %}{{'<｜Assistant｜><|tool_calls_begin|><|tool_call_begin|>' + tool['type'] +
    '<|tool_sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' +
    tool['function']['arguments'] + '\n' + '```' + '<|tool_call_end|>'}}{%- set
    ns.is_first = true -%}{%- else %}{{'\n' + '<|tool_call_begin|>' + tool['type'] +
    '<|tool_sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' +
    tool['function']['arguments'] + '\n' + '```' +
    '<|tool_call_end|>'}}{{'<|tool_calls_end|><|end_of_sentence|>'}}{%- endif %}{%-
    endfor %}{%- endif %}{%- if message['role'] == 'assistant' and
    message['content'] is not none %}{%- if ns.is_tool %}{{'<|tool_outputs_end|>' +
    message['content'] + '<|end_of_sentence|>'}}{%- set ns.is_tool = false -%}{%-
    else %}{{'<｜Assistant｜>' + message['content'] + '<|end_of_sentence|>'}}{%- endif %}{%- endif %}{%- if message['role']
    == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first
    %}{{'<|tool_outputs_begin|><|tool_output_begin|>' + message['content'] +
    '<|tool_output_end|>'}}{%- set ns.is_output_first = false %}{%- else
    %}{{'\n<|tool_output_begin|>' + message['content'] + '<|tool_output_end|>'}}{%-
    endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool
    %}{{'<|tool_outputs_end|>'}}{% endif %}{% if add_generation_prompt and not
    ns.is_tool %}{{'<｜Assistant｜><think>
    '}}{% endif %}"""

    tokenizer.chat_template = template_no_remove_think_tags

    def format_chat_template(row):
        row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row

    dataset_raw = [format_chat_template(row) for row in dataset_raw]

    dataset = Dataset.from_list(dataset_raw)

    dataset = dataset.train_test_split(test_size=0.1)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    from transformers import DataCollatorForSeq2Seq, TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=16_000,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        callbacks=[SaveEvery10Steps(gpu_id, all_gpus)],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            # per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=0.2,
            warmup_steps=5,
            num_train_epochs=4,  # Uncomment for 1 full training run.
            # max_steps=400,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="model_training_outputs",
            report_to="none",
        ),
    )

    instruct_part = "<｜begin▁of▁sentence｜><｜User｜>"
    response_part = "<｜Assistant｜>"
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruct_part,
        response_part=response_part,
    )
    # trainer.state.gpu_id = gpu_id
    # trainer.state.all_gpus = all_gpus
    # logger.debug(f'SET GPU ID: {gpu_id}| ALL GPUS: {all_gpus} to trainer state')
    # import ipdb; ipdb.set_trace()
    return model, tokenizer, dataset, trainer


class SaveEvery10Steps(TrainerCallback):
    def __init__(self, gpu_id, all_gpus):
        self.gpu_id = gpu_id
        self.all_gpus = all_gpus
        
    def on_optimizer_step(self, args, state: TrainerState, control, **kwargs):
        output_dir = args.output_dir
        model = state.model
        local_rank = self.gpu_id
        this_gpu_weights = os.path.join(
            output_dir, f"checkpoint-{state.global_step}.gpu.{local_rank}.pt"
        )
        target_weights = os.path.join(output_dir, f"checkpoint-{state.global_step}.pt")

        if state.global_step % 10 == 0:
            logger.debug(
                f"Step {state.global_step}: Saving weights for GPU {local_rank}"
            )
            state_dict = model.state_dict()
            trainable_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}

            # Create lock file before saving weights
            lock_file = f"{this_gpu_weights}.lock"
            with open(lock_file, "w") as f:
                f.write("lock")
            logger.debug(f"Created lock file {lock_file}")

            # Save the weights
            torch.save(trainable_state_dict, this_gpu_weights)
            logger.debug(f"Saved weights to {this_gpu_weights}")
            os.remove(lock_file)
            logger.debug(f"Removed lock file {lock_file}")
            if int(local_rank) == 0:
                all_gpus_weights = [
                    os.path.join(
                        output_dir, f"checkpoint-{state.global_step}.gpu.{i}.pt"
                    )
                    for i in self.all_gpus.split(",")
                ]
                logger.debug(
                    f"GPU 0: Merging weights from all GPUs: {all_gpus_weights}"
                )
                create_weight(all_gpus_weights, target_weights)

            # Wait for the target weights file to be fully written and available
            logger.debug(
                f"GPU {local_rank}: Waiting for target weights {target_weights} to be available"
            )
            wait_and_load(state.model, target_weights)
            logger.debug(f"GPU {local_rank}: Loaded weights from {target_weights}")
