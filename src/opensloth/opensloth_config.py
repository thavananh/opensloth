from multiprocessing import cpu_count
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

WORKERS = max(1, cpu_count() // 2)
# In [3]: from unsloth.chat_templates import CHAT_TEMPLATES
# In [4]: CHAT_TEMPLATES.keys()

KNOWN_CHAT_TEMPLATES = [
    "unsloth",
    "zephyr",
    "chatml",
    "mistral",
    "llama",
    "vicuna",
    "vicuna_old",
    "vicuna old",
    "alpaca",
    "gemma",
    "gemma_chatml",
    "gemma2",
    "gemma2_chatml",
    "llama-3",
    "llama3",
    "phi-3",
    "phi-35",
    "phi-3.5",
    "llama-3.1",
    "llama-31",
    "llama-3.2",
    "llama-3.3",
    "llama-32",
    "llama-33",
    "qwen-2.5",
    "qwen-25",
    "qwen25",
    "qwen2.5",
    "phi-4",
    "gemma-3",
    "gemma3",
    "qwen-3",
    "qwen3",
]


class DatasetConfigBase(BaseModel):
    tokenizer_name: Optional[str] = Field(
        default=None,
        description="Name of the tokenizer to use for text processing",
    )
    chat_template: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Chat template for formatting input data",
    )
    instruction_part: Optional[str] = Field(
        default="<|im_start|>user\n",
        description="Part of the input that contains the instruction",
    )
    response_only: Optional[bool] = Field(
        default=True,
        description="If True, only the response part is used for training",
    )
    response_part: Optional[str] = Field(
        default="<|im_start|>assistant\n",
        description="Part of the input that contains the response",
    )
    num_samples: Optional[int] = None
    nproc: int = Field(
        default=WORKERS,
        description="Number of processes to use for dataset preparation",
    )

    max_seq_length: Optional[int] = Field(
        default=32_000,
        description="Maximum sequence length for tokenization",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate chat template after model initialization."""
        if self.chat_template is not None:
            templates = (
                [self.chat_template]
                if isinstance(self.chat_template, str)
                else self.chat_template
            )
            for template in templates:
                if template not in KNOWN_CHAT_TEMPLATES:
                    raise ValueError(
                        f"Unknown chat template '{template}'. "
                        f"Must be one of: {KNOWN_CHAT_TEMPLATES}"
                    )


class HFDatasetConfig(DatasetConfigBase):
    source_type: Literal["hf"] = "hf"
    dataset_name: str
    split: str


class PathDatasetConfig(DatasetConfigBase):
    source_type: Literal["path"] = "path"
    path: str


DatasetConfig = Union[HFDatasetConfig, PathDatasetConfig]


class FastModelArgs(BaseModel):
    """Configuration for Unsloth's FastModel initialization."""

    model_name: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


def _default_target_modules() -> List[str]:
    """Default target modules for LoRA application."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class LoraArgs(BaseModel):
    """Configuration for LoRA parameters in PEFT."""

    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    target_modules: List[str] = Field(
        default_factory=_default_target_modules,
        description="List of target modules for LoRA application",
    )
    use_rslora: bool = False

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class OpenSlothConfig(BaseModel):
    """Main configuration class combining all sub-configurations."""

    data: DatasetConfig = Field(
        default_factory=HFDatasetConfig,
        description="Dataset configuration for training",
    )
    devices: List[int] = Field(default=[0], description="List of GPU indices to use")
    fast_model_args: FastModelArgs = Field(default_factory=FastModelArgs)
    lora_args: Optional[LoraArgs] = Field(default_factory=LoraArgs)
    pretrained_lora: Optional[str] = Field(
        default=None,
        description="Path to pretrained LoRA model for continous lora training",
    )
    disable_packing: bool = Field(
        default=False,
        description="Disable packing of sequences for training",
    )

    log_level: Literal["info", "debug"] = Field(
        default="info",
        description="Logging level for the training process",
    )

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class TrainingArgsConfig(BaseModel):
    """Configuration for Hugging Face TrainingArguments."""

    output_dir: str = "saves/loras/"
    per_device_train_batch_size: int = 8
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 16
    logging_steps: int = 1
    num_train_epochs: int = 1
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 5
    save_total_limit: int = 2
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    save_only_model: bool = True

    seed: int = 42
    report_to: Literal["tensorboard", "wandb", "none"] = "tensorboard"
    eval_strategy: str = "no"  # must be no, when using multigpus
    dataset_num_proc: int = WORKERS

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TrainingArguments initialization."""
        return self.model_dump()
