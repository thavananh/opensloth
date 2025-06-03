import os
import os
import sys
import time
import warnings
import importlib.util
from fastcore.all import threaded, call_parse

from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig
from HyperSloth.logging_config import HyperSlothLogger


warnings.filterwarnings("ignore")


def get_current_python_path():
    """
    Return output of which python
    """
    import subprocess

    try:
        result = subprocess.run(
            ["which", "python"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting Python path: {e}")
        return None


def train_on_single_gpu(
    gpu: int, hyper_config: HyperConfig, hf_train_args: TrainingArgsConfig
):
    from HyperSloth.nccl_grad_sync import NCCLGradSyncCallback
    from HyperSloth.hp_trainer_setup import setup_model_and_training

    os.environ["HYPERSLOTH_LOCAL_RANK"] = str(hyper_config.training.gpus.index(gpu))

    # Setup enhanced logger
    logger = HyperSlothLogger()

    # Use enhanced logging
    logger.log_gpu_info(
        gpu=gpu,
        world_size=len(hyper_config.training.gpus),
        model_name=hyper_config.fast_model_args.model_name,
    )

    logger.info(f"Training on GPU {gpu} with output_dir {hf_train_args.output_dir}")

    # Start total training timer
    logger.start_total_training_timer()

    # setup_nccl_for_hypersloth(gpu, hyper_config.training.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.start_timing("model_and_training_setup")
    trainer, model, tokenizer = setup_model_and_training(
        hyper_config=hyper_config,
        hf_train_args=hf_train_args,
    )
    logger.finish_timing("model_and_training_setup")

    logger.start_timing("callback_setup")
    assert trainer.model is not None, "Trainer model is None"
    grad_sync_cb = NCCLGradSyncCallback(
        model=trainer.model,
        gpu=gpu,
        gpus=hyper_config.training.gpus,
    )
    logger.info(f"Using gradient sync callback for GPU {gpu}")
    trainer.add_callback(grad_sync_cb)
    logger.finish_timing("callback_setup")

    logger.start_timing("actual_training")
    trainer.train()
    logger.finish_timing("actual_training")

    # Save once from rank=0
    if gpu == hyper_config.training.gpus[0]:
        logger.start_timing("model_saving")
        logger.info(f"Save model to {hf_train_args.output_dir}")
        model.save_pretrained(hf_train_args.output_dir)
        tokenizer.save_pretrained(hf_train_args.output_dir)
        logger.finish_timing("model_saving")

        # Log training summary
        logger.log_training_summary()


def load_config_from_path(config_path: str):
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(config_module)  # type: ignore
    # return config_module
    # Retrieve configs from the module
    if hasattr(config_module, "hyper_config_model"):
        hyper_config = config_module.hyper_config_model
    elif hasattr(config_module, "hyper_config"):
        hyper_config = HyperConfig(**config_module.hyper_config)
    else:
        hyper_config = HyperConfig()
    # return hyper_config,
    if hasattr(config_module, "training_config_model"):
        training_config = config_module.training_config_model
    elif hasattr(config_module, "training_config"):
        training_config = TrainingArgsConfig(**config_module.training_config)
    else:
        raise ValueError("No training configuration found")
    return hyper_config, training_config


# We'll just detect if the user wants a tmux script:


def build_tmux_script(
    session_name: str,
    script_path: str,
    output_dir: str,
    config_file: str,
    gpus: list,
    auto_kill: bool = False,
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
    # remove grad_dir
    # lines.append(f"rm -rf {_get_hp_grad_dir(output_dir)}")
    lines.append(
        f"""# Create a new session with first GPU = 0
tmux new-session -d -s {session_name} -n MAIN"""
    )

    # First GPU
    # check tmux session command, if yes, ask user enter "y" to kill the session
    # check_if_session_exists_then_ask_to_kill = f"tmux has-session -t {session_name}
    # && read -p 'Session exists, kill it? (y/n): ' kill_session &&
    #  [ $kill_session == 'y' ] && tmux kill-session -t {session_name}"
    # lines.append(check_if_session_exists_then_ask_to_kill)
    # Remaining GPUs
    for local_rank, gpu_index in enumerate(gpus):
        cmd = (
            f"USE_TMUX=0 "
            f"{get_current_python_path()} {sys.argv[0]} "
            f"{config_file} "
            f"--rank {local_rank} "
            f"--world_size {len(gpus)}"
        )
        lines.append(f"tmux new-window -t {session_name} -n gpu_{gpu_index}")
        lines.append(f"tmux send-keys -t {session_name}:gpu_{gpu_index} '{cmd}' Enter")
        lines.append("")

    lines.append(f'echo "Automatically attaching to session {session_name}..."')
    lines.append(f"tmux attach -t {session_name}")

    # Write out the script
    script_body = "\n".join(lines)
    with open(script_path, "w") as f:
        f.write(script_body)
    os.chmod(script_path, 0o755)

    is_session_exists = os.system(f"tmux has-session -t {session_name}")
    if is_session_exists == 0:
        if auto_kill:
            print(f"Auto-killing existing session {session_name}")
            os.system(f"tmux kill-session -t {session_name}")
        else:
            (f"Session {session_name} exists, please kill it before running the script")
            # ask user if they want to kill the session
            user_input = input(
                f"Session {session_name} exists, do you want to kill it? (y/n): "
            )
            if user_input.lower() == "y":
                os.system(f"tmux kill-session -t {session_name}")
                print(f"Session {session_name} killed")
            else:
                return
    os.system(f"bash {script_path}")
    print(f"Training sessions started and attached to session {session_name}")


def run_tmux_training(
    session_name: str,
    config_file: str,
    training_config: TrainingArgsConfig,
    gpus: list,
    auto_kill: bool = False,
):
    """Handle multi-GPU training using tmux sessions."""
    script_path = "/tmp/hp_train.sh"
    build_tmux_script(
        session_name,
        script_path,
        training_config.output_dir,
        config_file,
        gpus,
        auto_kill=auto_kill,
    )


def run_multiprocess_training(
    gpus: list,
    hyper_config: HyperConfig,
    training_config: TrainingArgsConfig,
):
    """Handle multi-GPU training using multi-processing."""
    print(f"[MP] Running on {len(gpus)} GPUs")
    processes = []
    assert len(gpus) > 1, "Cannot use multi-processing with a single GPU"

    @threaded(process=True)
    def run_in_process(*args, **kwargs):
        """Runs train_on_single_gpu() in a separate Python process."""
        train_on_single_gpu(*args, **kwargs)

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
                    print("Error in training, terminating all processes")
                    raise Exception("Error in training")
                else:
                    processes.remove(proc)
                    break
        time.sleep(1)
    print("All processes finished")


@call_parse
def train(
    config_file: str,
    rank: int = None,
    world_size: int = None,
    tmux: str = None,
    y: bool = False,
):
    hyper_config, training_config = initialize_training_config(config_file)

    # CASE 1: Child process => single GPU
    if rank is not None and world_size is not None:
        print(f"[CASE 1] Running on rank {rank} with world size {world_size}")
        train_on_single_gpu(
            gpu=hyper_config.training.gpus[rank],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )
        return

    # CASE 2: Top-level process => spawn multi-GPU or single GPU

    # If multiple GPUs:
    if len(hyper_config.training.gpus) > 1:
        if os.environ.get("USE_TMUX", "0") == "1" or tmux is not None:
            session_name = tmux if tmux is not None else "train_hp"
            run_tmux_training(
                session_name=session_name,
                config_file=config_file,
                training_config=training_config,
                gpus=gpus,
                auto_kill=y,
            )
        else:
            run_multiprocess_training(
                gpus=hyper_config.training.gpus,
                hyper_config=hyper_config,
                training_config=training_config,
            )
    else:
        # Single GPU
        assert tmux is None, "Cannot use tmux with a single GPU"
        train_on_single_gpu(
            gpu=hyper_config.training.gpus[0],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )


def initialize_training_config(config_file):
    # global USE_TMUX
    # USE_TMUX = USE_TMUX or use_tmux
    """Train entry-point. If rank/world_size are provided, we assume this is
    a child process that trains on a single GPU. Otherwise,
    we spawn multi-gpu runs either by generating a tmux script or by multi-process.
    """

    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    hyper_config, training_config = load_config_from_path(config_file)
    from HyperSloth.logging_config import format_config_display, get_hypersloth_logger

    temp_logger = get_hypersloth_logger(log_level="INFO", allow_unknown_gpu=True)
    combined_config = format_config_display(hyper_config, training_config)
    temp_logger.log_config_table(
        combined_config, "🔧 HyperSloth Training Configuration"
    )
    setup_envs(hyper_config, training_config)
    return hyper_config, training_config
    # Display combined config with enhanced formatting


def setup_envs(hyper_config, training_config):
    os.environ["HYPERSLOTH_WORLD_SIZE"] = str(len(hyper_config.training.gpus))
    os.environ["HYPERSLOTH_FORWARD_BZ"] = str(
        training_config.per_device_train_batch_size
        # * training_config.gradient_accumulation_steps
        * len(hyper_config.training.gpus)
    )
    os.environ["HYPERSLOTH_GLOBAL_BZ"] = str(
        training_config.per_device_train_batch_size
        * training_config.gradient_accumulation_steps
        * len(hyper_config.training.gpus)
    )

    print(f"Global batch size: {os.environ['HYPERSLOTH_GLOBAL_BZ']}")
    os.environ["HYPERSLOTH_ACCUMULATION_STEPS"] = str(
        training_config.gradient_accumulation_steps
    )
    os.environ["HYPERSLOTH_PER_DEVICE_TRAIN_BZ"] = str(
        training_config.per_device_train_batch_size
    )
