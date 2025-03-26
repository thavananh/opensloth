import os
import time

from fastcore.all import threaded
from loguru import logger

from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig

if not "HYPERSLOTH_CACHE_DIR" in os.environ:
    os.environ["HYPERSLOTH_CACHE_DIR"] = "/dev/shm/hypersloth/"

# turn off user warnings
# Too verbose -> turn off
import warnings
warnings.filterwarnings("ignore")
os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"



def _get_run_dir(run_id):
    grad_dir = os.path.join(os.environ["HYPERSLOTH_CACHE_DIR"], f"run_{run_id}")
    #mnkdir
    os.makedirs(grad_dir, exist_ok=True)
    return grad_dir


def _setup_loger(gpu_id):
    # create a file logger for this specific gpu store at /dev/shm/hypersloth/log_gpu{gpu}.log
    from loguru import logger

    file = f".log/process_{gpu_id}.log"
    if os.path.exists(file):
        os.remove(file)
    logger.add(file)
    print(f"Logging to {file}")


def _train(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    _setup_loger(f"{gpu}")
    import os


    os.environ["HYPERSLOTH_LOCAL_RANK"] = str(hyper_config.training.gpus.index(gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from HyperSloth.transformer_trainer_setup import setup_model_and_training

    # from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback

    trainer, model, tokenizer = setup_model_and_training(
        gpu=gpu,
        hyper_config=hyper_config,
        hf_train_args=hf_train_args,
    )

    if len(hyper_config.training.gpus) > 0 and hyper_config.hps_version is not None:
        if hyper_config.hps_version == 2:
            from HyperSloth.mmap_gradient_sync_v2 import MmapGradSyncCallback

            logger.info("Using gradient sync callback v2")
        else:
            logger.info("Using gradient sync callback v1")
            from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback
        
        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=os.environ["HYPERSLOTH_RUN_DIR"],
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


# run_in_process = threaded(process=True)(_train)
@threaded(process=True)
def run_in_process(*args, **kwargs):
    # for i in range(5):
    _train(*args, **kwargs)


import importlib.util


def load_config_from_path(config_path: str):
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


from fastcore.all import call_parse


@call_parse
def train(config_file: str, rank: int = None, world_size: int = None):
    import os

    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    config_module = load_config_from_path(config_file)
    import tabulate
    from speedy_utils import setup_logger

    setup_logger(os.environ.get("HYPERSLOTH_LOG_LEVEL", "INFO"))
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
        raise ValueError("No training configuration found")

    # Display configuration
    combined_config = {**hyper_config.model_dump(), **training_config.model_dump()}
    config_table = tabulate.tabulate(combined_config.items(), headers=["Key", "Value"])
    logger.info("\n" + config_table)

    # Run training
    from speedy_utils import identify

    ## setup params

    os.environ["HYPERSLOTH_NUM_GPUS"] = str(len(hyper_config.training.gpus))
    
    
    # DEBUG MODE TMUX
    if rank is not None and world_size is not None:
        logger.warning(f"Running on rank {rank} with world size {world_size}")
        hyper_config.training.gpus = range(world_size)
        run_id = identify(combined_config)
        os.environ["HYPERSLOTH_RUN_DIR"] = _get_run_dir(run_id)
        os.environ["HYPERSLOTH_NUM_GPUS"] = str(len(hyper_config.training.gpus))
        _train(
            gpu=hyper_config.training.gpus[rank],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )
    # NORMAL MULTI PROCESS MODE
    else:
        os.environ["HYPERSLOTH_NUM_GPUS"] = str(len(hyper_config.training.gpus))
        if len(hyper_config.training.gpus) > 1:

            run_id = identify(combined_config)
            os.environ["HYPERSLOTH_RUN_DIR"] = _get_run_dir(run_id)

            processes = []
            for gpu_index in hyper_config.training.gpus:
                logger.debug(f"Running on GPU {gpu_index} with run_id {run_id}")

                p = run_in_process(
                    gpu_index,
                    hyper_config=hyper_config,
                    hf_train_args=training_config,
                )
                processes.append(p)

            # check proc if exit code == 1 then .terminate all
            while True:
                for proc in processes:
                    if not proc.is_alive():
                        if proc.exitcode == 1:
                            for p in processes:
                                p.terminate()
                            logger.error(
                                "Error in training, now terminating all processes"
                            )
                            raise Exception("Error in training")
                        else:
                            processes.remove(proc)
                            break
                if not processes:
                    logger.success("All processes finished")
                    break

        else:
            run_id = "single_gpu"
            os.environ["HYPERSLOTH_RUN_DIR"] = _get_run_dir(run_id)
            _train(
                gpu=hyper_config.training.gpus[0],
                hyper_config=hyper_config,
                hf_train_args=training_config,
            )
