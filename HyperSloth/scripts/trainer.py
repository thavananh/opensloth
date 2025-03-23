import argparse
import time
from fastcore.all import threaded
from loguru import logger
from typing import Union, Dict, Any

from HyperSloth.hypersloth_config import TrainingArgsConfig, HyperConfig


def run(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
    run_id=None,
):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from HyperSloth.transformer_trainer_setup import setup_model_and_training
    from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback

    trainer, model, tokenizer = setup_model_and_training(
        gpu=gpu,
        hyper_config=hyper_config,
        hf_train_args=hf_train_args,
    )

    if len(hyper_config.training.gpus) > 1:
        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=f"/dev/shm/hypersloth/{run_id}",
            gpu=gpu,
            gpus=hyper_config.training.gpus,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()
    if gpu == hyper_config.training.gpus[0]:
        logger.info(f"Save model to {hf_train_args.output_dir}")
        model.save_pretrained(hf_train_args.output_dir)
        tokenizer.save_pretrained(hf_train_args.output_dir)


run_in_process = threaded(process=True)(run)

import importlib.util


def load_config_from_path(config_path: str):
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


from fastcore.all import call_parse


@call_parse
def train(config_file: str):
    import os

    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    config_module = load_config_from_path(config_file)
    import tabulate
    from speedy_utils import setup_logger
    setup_logger('D')

    # Get configurations from the module
    from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig

    # Use Pydantic models directly or create from dictionaries if needed
    if hasattr(config_module, "hyper_config_model"):
        hyper_config = config_module.hyper_config_model
    elif hasattr(config_module, "hyper_config"):
        hyper_config = HyperConfig(**config_module.hyper_config)
    else:
        hyper_config = HyperConfig()

    if hasattr(config_module, "training_config_model"):
        training_config = config_module.training_config_model
    elif hasattr(config_module, "training_config"):
        training_config = TrainingArgsConfig(**config_module.training_config)
    else:
        training_config = TrainingArgsConfig()

    # Display configuration
    combined_config = {**hyper_config.model_dump(), **training_config.model_dump()}
    config_table = tabulate.tabulate(combined_config.items(), headers=["Key", "Value"])
    logger.info("\n" + config_table)

    # Run training
    if len(hyper_config.training.gpus) > 1:
        from speedy_utils import identify

        run_id = identify(combined_config)
        for gpu_index in hyper_config.training.gpus:
            logger.debug(f"Running on GPU {gpu_index} with run_id {run_id}")
            run_in_process(
                gpu_index,
                hyper_config=hyper_config,
                hf_train_args=training_config,
                run_id=run_id,
            )
            time.sleep(0.2)
    else:
        run(
            gpu=hyper_config.training.gpus[0],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )
