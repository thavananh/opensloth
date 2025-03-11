import fire
from fastcore.all import threaded
from loguru import logger
from transformers.training_args import TrainingArguments
from HyperSloth.app_config import HyperSlothConfig


@threaded(process=True)
def run(
    gpu,
    hyper_config: HyperSlothConfig,
    train_args: TrainingArguments,
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from HyperSloth.transformer_trainer_setup import setup_model_and_training
    from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback

    trainer = setup_model_and_training(
        gpu=gpu,
        hyper_config=hyper_config,
        hf_train_args=train_args,
    )

    if len(hyper_config.gpus) > 1:
        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=hyper_config.grad_dir,
            gpu=gpu,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()


def main(
    config_py="configs/hypersloth_config_example.py",
):
    config_module = __import__(
        config_py.replace("/", ".").replace(".py", ""), fromlist=[""]
    )
    hyper_config: HyperSlothConfig = config_module.hyper_config
    training_config: TrainingArguments = config_module.training_config

    for gpu_index in hyper_config.gpus:
        logger.debug(f"Running on GPU {gpu_index}")
        run(
            gpu_index,
            hyper_config=hyper_config,
            train_args=training_config,
        )


if __name__ == "__main__":
    fire.Fire(main)
