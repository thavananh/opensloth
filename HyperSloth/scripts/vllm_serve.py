import os
import subprocess
import time
from typing import List, Optional
from fastcore.script import call_parse


def kill_existing_vllm(vllm_binary=None) -> None:
    """Kill existing vLLM processes silently."""
    if not vllm_binary:
        vllm_binary = get_vllm()
    subprocess.run(
        "tmux kill-session -a -t vllm",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        f"ps aux | grep {vllm_binary} | grep -v grep | awk '{{print $2}}' | xargs -r kill -9",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep()


def start_vllm_containers(
    model: str,
    gpu_groups: str,
    served_model_name: Optional[str],
    port_start: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int,
    # extra_args: List[str],
    vllm_binary: str,
) -> None:
    """Start vLLM containers with dynamic args."""
    gpu_groups_arr = gpu_groups.split(",")

    for i, gpu_group in enumerate(gpu_groups_arr):
        port = port_start + i
        gpus = ",".join(gpu_group)
        tensor_parallel = len(gpus.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpus}",
            vllm_binary,
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
            "--uvicorn-log-level critical",
            "--enable-prefix-caching",
            "--enforce-eager",
            "--disable-log-requests",
        ]

        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])

        # if extra_args:
        #     cmd.extend(extra_args)

        final_cmd = " ".join(cmd)
        log_file = f"/tmp/vllm_{port}.txt"
        final_cmd_with_log = f'"{final_cmd} 2>&1 | tee {log_file}"'
        run_in_tmux = f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"

        print(final_cmd)
        print("Logging to", log_file)
        os.system(run_in_tmux)


@call_parse
def main(
    model: str,
    gpu_groups: str,
    served_model_name: Optional[str] = None,
    port_start: int = 8170,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "half",
    max_model_len: int = 8192,
    extra_args: List[str] = [],
):
    """Main function to start or kill vLLM containers."""
    VLLM_BINARY = get_vllm()

    if not gpu_groups:
        print("Usage: serve_vllm.py [--model path_to_model] GPU_GROUP_1[,GPU_GROUP_2,...]")
        exit(1)

    start_vllm_containers(
        model,
        gpu_groups,
        served_model_name,
        port_start,
        gpu_memory_utilization,
        dtype,
        max_model_len,
        extra_args,
        VLLM_BINARY,
    )

def get_vllm():
    VLLM_BINARY = "/home/ubuntu/.conda/envs/vllm/bin/vllm"
    VLLM_BINARY = os.getenv("VLLM_BINARY", VLLM_BINARY)
    assert os.path.exists(VLLM_BINARY), f"vLLM binary not found at {VLLM_BINARY}, please set VLLM_BINARY env variable"
    return VLLM_BINARY
