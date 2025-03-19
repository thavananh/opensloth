import os
import subprocess
import time
from typing import List, Optional
from fastcore.script import call_parse
from ray import logger
import argparse
import requests
from sqlalchemy import desc


def kill_existing_vllm(vllm_binary: Optional[str] = None) -> None:
    """Kill selected vLLM processes using fzf."""
    if not vllm_binary:
        vllm_binary = get_vllm()

    # List running vLLM processes
    result = subprocess.run(
        f"ps aux | grep {vllm_binary} | grep -v grep",
        shell=True,
        capture_output=True,
        text=True,
    )
    processes = result.stdout.strip().split("\n")

    if not processes or processes == [""]:
        print("No running vLLM processes found.")
        return

    # Use fzf to select processes to kill
    fzf = subprocess.Popen(
        ["fzf", "--multi"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    selected, _ = fzf.communicate("\n".join(processes))

    if not selected:
        print("No processes selected.")
        return

    # Extract PIDs and kill selected processes
    pids = [line.split()[1] for line in selected.strip().split("\n")]
    for pid in pids:
        subprocess.run(
            f"kill -9 {pid}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Killed processes: {', '.join(pids)}")


def add_lora(
    lora_name: str,
    lora_path: str,
    port=None,
    url: str = "http://localhost:8152/v1/load_lora_adapter",
) -> dict:
    if port:
        url = f"http://localhost:{port}/v1/load_lora_adapter"
    assert url.startswith("http"), "URL must start with 'http'"
    headers = {"Content-Type": "application/json"}
    data = {"lora_name": lora_name, "lora_path": lora_path}
    logger.info(f"Adding LoRA adapter: {lora_name} from {lora_path}")

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        # Handle potential non-JSON responses
        try:
            return response.json()
        except ValueError:
            return {
                "status": "success",
                "message": (
                    response.text
                    if response.text.strip()
                    else "Request completed with empty response"
                ),
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}


def serve(
    model: str,
    gpu_groups: str,
    served_model_name: Optional[str] = None,
    port_start: int = 8155,
    gpu_memory_utilization: float = 0.93 ,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    enable_lora: bool = False,
    is_bnb: bool = False,
    not_verbose=True,
    extra_args: Optional[List[str]] = [],
):
    """Main function to start or kill vLLM containers."""

    """Start vLLM containers with dynamic args."""
    print("Starting vLLM containers...,")
    gpu_groups_arr = gpu_groups.split(",")
    VLLM_BINARY = get_vllm()
    if enable_lora:
        VLLM_BINARY = "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True " + VLLM_BINARY

    # Auto-detect quantization based on model name if not explicitly set
    if not is_bnb and model and ("bnb" in model.lower() or "4bit" in model.lower()):
        is_bnb = True
        print(f"Auto-detected quantization for model: {model}")

    # Set environment variables for LoRA if needed
    if enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        print("Enabled runtime LoRA updating")

    for i, gpu_group in enumerate(gpu_groups_arr):
        port = port_start + i
        gpu_group = ",".join([str(x) for x in gpu_group])
        tensor_parallel = len(gpu_group.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_group}",
            VLLM_BINARY,
            "serve",
            model,
            "--port",
            str(port),
            "--tensor-parallel",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--dtype",
            dtype,
            "--max-model-len",
            str(max_model_len),
            "--disable-log-requests",
        ]
        if not_verbose:
            cmd += ["--uvicorn-log-level critical", "--enable-prefix-caching"]

        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])

        if is_bnb:
            cmd.extend(
                ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
            )

        if enable_lora:
            cmd.extend(["--fully-sharded-loras", "--enable-lora"])
        # add kwargs
        if extra_args:
            for name_param in extra_args:
                name, param = name_param.split("=")
                cmd.extend([f"{name}", param])
        final_cmd = " ".join(cmd)
        log_file = f"/tmp/vllm_{port}.txt"
        final_cmd_with_log = f'"{final_cmd} 2>&1 | tee {log_file}"'
        run_in_tmux = (
            f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"
        )

        print(final_cmd)
        print("Logging to", log_file)
        os.system(run_in_tmux)


def get_vllm():
    VLLM_BINARY = subprocess.check_output("which vllm", shell=True, text=True).strip()
    VLLM_BINARY = os.getenv("VLLM_BINARY", VLLM_BINARY)
    logger.info(f"vLLM binary: {VLLM_BINARY}")
    assert os.path.exists(
        VLLM_BINARY
    ), f"vLLM binary not found at {VLLM_BINARY}, please set VLLM_BINARY env variable"
    return VLLM_BINARY


def get_args():
    """Parse command line arguments."""
    example_args = ['svllm add_lora --model localization_pro:./saves/loras/250312/LC_EN_VI_TH_27B_233k/checkpoint-1736/:8155',
                    'svllm add_lora lora_name@path:port',
                    'svllm kill']
    
    parser = argparse.ArgumentParser(description="vLLM Serve Script", epilog="Example: " + " || ".join(example_args))
    parser.add_argument(
        "mode", choices=["serve", "kill", "add_lora"], help="Mode to run the script in"
    )
    parser.add_argument("--model", "-m", type=str, help="Model to serve")
    parser.add_argument(
        "--gpu_groups", "-g", type=str, help="Comma-separated list of GPU groups"
    )
    parser.add_argument(
        "--served_model_name", type=str, help="Name of the served model"
    )
    parser.add_argument(
        "--port_start", type=int, default=8155, help="Starting port number"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument(
        "--max_model_len", type=int, default=8192, help="Maximum model length"
    )
    # parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA")
    parser.add_argument(
        "--disable_lora",
        dest="enable_lora",
        action="store_false",
        help="Enable LoRA",
        default=True,
    )
    parser.add_argument("--bnb", action="store_true", help="Enable quantization")
    parser.add_argument(
        "--not_verbose", action="store_true", help="Disable verbose logging"
    )
    parser.add_argument("--vllm_binary", type=str, help="Path to the vLLM binary")
    parser.add_argument("--lora_name", type=str, help="Name of the LoRA adapter")
    parser.add_argument(
        "--extra_args", nargs=argparse.REMAINDER, help="Additional arguments for the serve command"
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""

    args = get_args()
    # if help
    if args.mode == "serve":
        serve(
            args.model,
            args.gpu_groups,
            args.served_model_name,
            args.port_start,
            args.gpu_memory_utilization,
            args.dtype,
            args.max_model_len,
            args.enable_lora,
            args.bnb,
            args.not_verbose,
            args.extra_args,
        )
    elif args.mode == "kill":
        kill_existing_vllm(args.vllm_binary)
    elif args.mode == "add_lora":
        # split by :
        assert (
            ":" in args.model
        ), "Invalid format for add_lora, should be lora_name@path:port"
        name, path, port = args.model.split(":")
        path = os.path.abspath(path)
        add_lora(name, path, port)


if __name__ == "__main__":
    main()
