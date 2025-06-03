from multiprocessing import cpu_count
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

WORKERS = max(1, cpu_count() // 2)


class DatasetConfigBase(BaseModel):
    tokenizer_name: str
    chat_template: str
    instruction_part: str
    response_part: str
    num_samples: Optional[int] = None
    # dataset_num_proc: int = CPU_COUNT


class HFDatasetConfig(DatasetConfigBase):
    source_type: Literal["hf"] = "hf"
    dataset_name: str
    split: str


class PathDatasetConfig(DatasetConfigBase):
    source_type: Literal["path"] = "path"
    path: str


DatasetConfig = Union[HFDatasetConfig, PathDatasetConfig]


class TrainingConfig(BaseModel):
    """Configuration for training setup and parameters."""

    gpus: List[int] = Field(default=[0], description="List of GPU indices to use")
    loss_type: Literal["all", "response_only"] = Field(
        default="response_only",
        description="Loss calculation type: 'all' or 'response_only'",
    )
    chat_template: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Chat template for formatting input data",
    )

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class FastModelArgs(BaseModel):
    """Configuration for Unsloth's FastModel initialization."""

    model_name: str = "unsloth/gemma-3-4b-it"
    max_seq_length: Optional[int] = None
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


class HyperConfig(BaseModel):
    """Main configuration class combining all sub-configurations."""

    data: DatasetConfig = Field(
        default_factory=HFDatasetConfig,
        description="Dataset configuration for training",
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fast_model_args: FastModelArgs = Field(default_factory=FastModelArgs)
    lora_args: Optional[LoraArgs] = Field(default_factory=LoraArgs)
    pretrained_lora: Optional[str] = Field(
        default=None,
        description="Path to pretrained LoRA model for continous lora training",
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
    max_seq_len: int = 32_000

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TrainingArguments initialization."""
        return self.model_dump()
