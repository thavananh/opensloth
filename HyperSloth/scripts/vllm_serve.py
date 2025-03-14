import os
import argparse
import subprocess
import time
def main():
    VLLM_BINARY = "/home/ubuntu/.conda/envs/vllm/bin/vllm"
    VLLM_BINARY = os.getenv("VLLM_BINARY", VLLM_BINARY)

    # Argument parser
    parser = argparse.ArgumentParser(description="Start vLLM containers with dynamic args.")
    parser.add_argument("model", type=str, help="Path to custom model")
    parser.add_argument("gpu_groups", type=str, help="GPU groups (comma-separated)")
    parser.add_argument(
        "--served-model-name",
        "-s",
        type=str,
        default=None,
        help="Name of the served model (default: vllm)",
    )
    parser.add_argument(
        "--port-start",
        type=int,
        default=8170,
        help="Starting port number (default: 8170)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        help="Data type (default: half)",
    )
    parser.add_argument(
        "--max-model-len",
        "-l",
        type=int,
        default=8192,
        help="Maximum model length (default: 8192)",
    )
    
    parser.add_argument(
        "--extra-args",
        type=str,
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arbitrary arguments for vLLM",
    )
    parser.add_argument(
        "--kill", "-k", action="store_true", help="Only kill existing vLLM processes"
    )

    args = parser.parse_args()


    # Kill existing VLLM processes
    def kill_existing_vllm() -> None:
        """Kill existing vLLM processes silently."""
        subprocess.run(
            "tmux kill-session -a -t vllm",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            f"ps aux | grep {VLLM_BINARY} | grep -v grep | awk '{{print $2}}' | xargs -r kill -9",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)


    if args.kill:
        kill_existing_vllm()
        exit(0)

    # Validate input
    if not args.gpu_groups:
        print("Usage: serve_vllm.py [--model path_to_model] GPU_GROUP_1[,GPU_GROUP_2,...]")
        exit(1)

    # Parse GPU groups from input
    gpu_groups_arr = args.gpu_groups.split(",")

    # Loop through each GPU group and start a container
    for i, gpu_group in enumerate(gpu_groups_arr):
        port = args.port_start + i

        # Fix GPU formatting: Convert "01" -> "0,1"
        gpus = ",".join(gpu_group)

        # Calculate tensor parallel size (number of GPUs in the group)
        tensor_parallel = len(gpus.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpus}",
            VLLM_BINARY,
            "serve",
            args.model,
            "--port",
            str(port),
            "--tensor-parallel",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--dtype",
            args.dtype,
            "--max-model-len",
            str(args.max_model_len),
            "--uvicorn-log-level critical",
            "--enable-prefix-caching",
            "--enforce-eager",
            "--disable-log-requests"
        ]

        if args.served_model_name:
            cmd.extend(["--served-model-name", args.served_model_name])

        # Add extra arbitrary arguments
        if args.extra_args:
            cmd.extend(args.extra_args)

        final_cmd = " ".join(cmd)
        log_file = '/tmp/vllm_{}.txt'.format(port)
        final_cmd_with_log = '"{} 2>&1 | tee {}"'.format(final_cmd, log_file)
        run_in_tmux = f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"

        print(final_cmd)
        print('Logging to', log_file)
        os.system(run_in_tmux)
