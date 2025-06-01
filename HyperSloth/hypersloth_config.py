import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

CPU_COUNT = min(1, cpu_count() - 2)


class DataConfig(BaseModel):
    """Configuration for dataset handling and processing."""

    path_to_text_dataset: str = Field(
        default="data/finetome_1000_samples",
        description="Path to the dataset directory",
    )
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"
    dataset_num_proc: int = Field(
        default=max(1, CPU_COUNT),
        description="Number of processes to use for dataset processing",
    )

    @classmethod
    def from_dataset_name(cls, dataset_name: str) -> "DataConfig":
        """Create DataConfig from dataset name using the registry.

        Args:
            dataset_name: Name of the dataset in the registry

        Returns:
            DataConfig instance with loaded configuration

        Raises:
            FileNotFoundError: If data_config.json is not found
            ValueError: If dataset_name is not found in registry
        """
        # Try to find data_config.json relative to current working directory
        registry_path = Path("data/data_config.json")
        if not registry_path.exists():
            # Try relative to this file's directory
            config_dir = Path(__file__).parent.parent
            registry_path = config_dir / "data" / "data_config.json"

        if not registry_path.exists():
            raise FileNotFoundError(
                f"Dataset registry not found at {registry_path}. "
                "Please run build_dataset.py to create datasets first."
            )

        with open(registry_path, "r") as f:
            registry = json.load(f)

        # Find dataset in registry
        dataset_config = None
        for config in registry:
            if config.get("name") == dataset_name:
                dataset_config = config
                break

        if dataset_config is None:
            available_names = [cfg.get("name", "unnamed") for cfg in registry]
            raise ValueError(
                f'Dataset "{dataset_name}" not found in registry. '
                f"Available datasets: {available_names}"
            )

        # Build full path
        if "path" in dataset_config:
            dataset_path = dataset_config["path"]
            if not dataset_path.startswith("/"):
                # Make relative paths absolute relative to data directory
                data_dir = registry_path.parent
                dataset_path = str(data_dir / dataset_path)
        else:
            raise ValueError(f'Dataset "{dataset_name}" missing path in registry')

        # Create DataConfig with loaded settings
        return cls(
            path_to_text_dataset=dataset_path,
            instruction_part=dataset_config.get(
                "instruction_part", "<|im_start|>user\n"
            ),
            response_part=dataset_config.get(
                "response_part", "<|im_start|>assistant\n"
            ),
        )


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
    shuffle_mode: Literal["on_dataset", "on_global_bz"] = Field(
        default="on_dataset",
        description="Shuffle mode for data sampling: 'on_dataset' or 'on_global_bz'",
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
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="List of target modules for LoRA application",
    )
    use_rslora: bool = False

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class HyperConfig(BaseModel):
    """Main configuration class combining all sub-configurations."""

    grad_dir: str = "/dev/shm/hypersloth"
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fast_model_args: FastModelArgs = Field(default_factory=FastModelArgs)
    lora_args: LoraArgs = Field(default_factory=LoraArgs)
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
    # bf16: bool = True
    # fp16: bool = False
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    save_only_model: bool = True

    seed: int = 42
    report_to: str = "tensorboard"
    eval_strategy: str = "no"  # must be no, when using multigpus
    max_steps: Optional[int] = None

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TrainingArguments initialization."""
        return self.model_dump()
