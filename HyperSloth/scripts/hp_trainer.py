import os
import sys
import time
import warnings
import importlib.util
from fastcore.all import threaded, call_parse
from loguru import logger
import tabulate

from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig
from speedy_utils import setup_logger

if not "HYPERSLOTH_CACHE_DIR" in os.environ:
    os.environ["HYPERSLOTH_CACHE_DIR"] = "/dev/shm/hypersloth/"

warnings.filterwarnings("ignore")
os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"

def get_run_id(hyper_config_model, training_config_model):
    model_name = hyper_config_model.fast_model_args.model_name.split("/")[-1]
    dataset = hyper_config_model.data.dataset_name_or_path.split("/")[-1]
    loss_type = hyper_config_model.training.loss_type
    lora_r = hyper_config_model.lora_args.r
    lora_alpha = hyper_config_model.lora_args.lora_alpha
    seq_len = hyper_config_model.fast_model_args.max_seq_length
    lr = training_config_model.learning_rate
    batch_size = training_config_model.per_device_train_batch_size
    accum_steps = training_config_model.gradient_accumulation_steps
    epochs = training_config_model.num_train_epochs
    seed = training_config_model.seed

    run_id = (
        f"{model_name}-{dataset}-loss-{loss_type}-lora{lora_r}a{lora_alpha}"
        f"-seq{seq_len}-lr{lr}-bs{batch_size}x{accum_steps}-e{epochs}-seed{seed}"
    )
    return run_id

def _get_run_dir(run_id):
    grad_dir = os.path.join(os.environ["HYPERSLOTH_CACHE_DIR"], f"run_{run_id}")
    os.makedirs(grad_dir, exist_ok=True)
    return grad_dir

def _setup_logger(gpu_id):
    setup_logger(os.environ.get("HYPERSLOTH_LOG_LEVEL", "INFO"))
    file = f".log/process_{gpu_id}.log"
    if os.path.exists(file):
        os.remove(file)
    logger.add(file)
    print(f"Logging to {file}")

def _train(gpu: int, hyper_config: HyperConfig, hf_train_args: TrainingArgsConfig):
    _setup_logger(f"{gpu}")

    from HyperSloth.transformer_trainer_setup import setup_model_and_training  # avoid circular import
    os.environ["HYPERSLOTH_LOCAL_RANK"] = str(hyper_config.training.gpus.index(gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    trainer, model, tokenizer = setup_model_and_training(
        hyper_config=hyper_config,
        hf_train_args=hf_train_args,
    )

    if len(hyper_config.training.gpus) > 0 and hyper_config.hps_version is not None:
        from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback
        if hyper_config.hps_version == 2:
            logger.info("Using gradient sync callback v2")
        else:
            logger.info("Using gradient sync callback v1")

        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=os.environ["HYPERSLOTH_RUN_DIR"],
            gpu=gpu,
            gpus=hyper_config.training.gpus,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()

    # Save once from rank=0
    if gpu == hyper_config.training.gpus[0]:
        logger.info(f"Save model to {hf_train_args.output_dir}")
        model.save_pretrained(hf_train_args.output_dir)
        tokenizer.save_pretrained(hf_train_args.output_dir)

@threaded(process=True)
def run_in_process(*args, **kwargs):
    """Runs _train() in a separate Python process."""
    _train(*args, **kwargs)

def load_config_from_path(config_path: str):
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

# We'll just detect if the user wants a tmux script:


def build_tmux_script(
    session_name: str,
    script_path: str,
    run_id: str,
    config_file: str,
    gpus: list
):
    """
    Build a script that:
    1. Kills any existing tmux session with `session_name`
    2. Creates a new session for the first GPU
    3. Creates new windows for the remaining GPUs
    4. Sends the appropriate commands to each window
    Saves the final script to `script_path`.
    """
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("")
    lines.append("""# Create a new session with first GPU = 0
tmux new-session -d -s train_hp -n MAIN

tmux new-window -t train_hp -n nvtop
# tmux send-keys -t train_hp:MAIN 'source /home/anhvth5/miniconda3/etc/profile.d/conda.sh' Enter
# tmux send-keys -t train_hp:MAIN 'ECHO HELLOWORLD' Enter
echo =======================================""")

    # First GPU
    # check tmux session command, if yes, ask user enter "y" to kill the session
    # check_if_session_exists_then_ask_to_kill = f"tmux has-session -t {session_name} && read -p 'Session exists, kill it? (y/n): ' kill_session && [ $kill_session == 'y' ] && tmux kill-session -t {session_name}"
    # lines.append(check_if_session_exists_then_ask_to_kill)
    # Remaining GPUs
    for local_rank, gpu_index in enumerate(gpus):
        cmd = (
            f"USE_TMUX=0 "
            f"python {sys.argv[0]} "
            f"{config_file} "
            f"--rank {local_rank} "
            f"--world_size {len(gpus)}"
        )
        lines.append(f"tmux new-window -t {session_name} -n gpu_{gpu_index}")
        lines.append(
            f"tmux send-keys -t {session_name}:gpu_{gpu_index} '{cmd}' Enter"
        )
        lines.append("")

    lines.append(f'echo "Attach to this session via: tmux attach -t {session_name}"')

    # Write out the script
    script_body = "\n".join(lines)
    with open(script_path, "w") as f:
        f.write(script_body)
    os.chmod(script_path, 0o755)
    
    is_session_exists = os.system(f"tmux has-session -t {session_name}")
    if is_session_exists == 0:
        logger.warning(f"Session {session_name} exists, please kill it before running the script")
    else:
        os.system(f"bash {script_path}")
        logger.info(f"Script started with session name {session_name}")
        
        

@call_parse
def train(config_file: str, rank: int = None, world_size: int = None, use_tmux: bool = False):
    config_file, hyper_config, training_config = initialize_training_config(config_file, use_tmux)

    # CASE 1: Child process => single GPU
    if rank is not None and world_size is not None:
        logger.info(f"[CASE 1] Running on rank {rank} with world size {world_size}")
        hyper_config.training.gpus = list(range(world_size))

        # Prepare run_dir
        run_id = get_run_id(hyper_config, training_config)
        os.environ["HYPERSLOTH_RUN_DIR"] = _get_run_dir(run_id)

        _train(
            gpu=hyper_config.training.gpus[rank],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )
        return

    # CASE 2: Top-level process => spawn multi-GPU or single GPU
    run_id = get_run_id(hyper_config, training_config)
    os.environ["HYPERSLOTH_RUN_DIR"] = _get_run_dir(run_id)
    gpus = hyper_config.training.gpus

    # If multiple GPUs:
    if len(gpus) > 1:
        if os.environ.get("USE_TMUX", "0") == "1" or use_tmux:
            # Build a tmux script that the user can run manually
            session_name = f"train_hp"
            script_path = "/tmp/hp_train.sh"
            build_tmux_script(session_name, script_path, run_id, config_file, gpus)

            return
        else:
            # Launch via multi-processing (no tmux).
            processes = []
            for gpu_index in gpus:
                p = run_in_process(
                    gpu_index,
                    hyper_config=hyper_config,
                    hf_train_args=training_config,
                )
                processes.append(p)

            # Wait for processes; if one errors, kill them all
            while processes:
                for proc in processes:
                    if not proc.is_alive():
                        if proc.exitcode != 0:
                            for p in processes:
                                p.terminate()
                            logger.error("Error in training, terminating all processes")
                            raise Exception("Error in training")
                        else:
                            processes.remove(proc)
                            break
                time.sleep(1)
            logger.success("All processes finished")

    else:
        # Single GPU
        _train(
            gpu=gpus[0],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )

def initialize_training_config(config_file, use_tmux):
    # global USE_TMUX
    # USE_TMUX = USE_TMUX or use_tmux
    """Train entry-point. If rank/world_size are provided, we assume this is 
    a child process that trains on a single GPU. Otherwise, 
    we spawn multi-gpu runs either by generating a tmux script or by multi-process.
    """
    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    config_module = load_config_from_path(config_file)

    # Retrieve configs from the module
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

    # Display combined config
    combined_config = {**hyper_config.model_dump(), **training_config.model_dump()}
    config_table = tabulate.tabulate(combined_config.items(), headers=["Key", "Value"])
    logger.info("\n" + config_table)

    # # of GPUs
    os.environ["HYPERSLOTH_NUM_GPUS"] = str(len(hyper_config.training.gpus))
    return config_file,hyper_config,training_config
