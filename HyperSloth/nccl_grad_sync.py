import torch
import torch.distributed as dist
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from typing import List
from HyperSloth.logging_config import get_hypersloth_logger


class NCCLGradSyncCallback(TrainerCallback):
    """NCCL-based gradient synchronization callback for Transformers trainer.

    This callback provides the same interface as MmapGradSyncCallback but uses
    NCCL for gradient synchronization instead of memory-mapped files.
    """

    def __init__(
        self,
        model,
        gpu: int,
        gpus: List[int],
    ):
        self.model = model
        self.gpu = gpu
        self.gpus = gpus
        self.local_rank = gpus.index(gpu)
        self.world_size = len(gpus)
        self.logger = get_hypersloth_logger(log_level="DEBUG")

        # Ensure distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "NCCL distributed training not initialized. "
                "Call torch.distributed.init_process_group() first."
            )

        # Verify world size matches
        if dist.get_world_size() != self.world_size:
            raise ValueError(
                f"Expected world size {self.world_size} but got "
                f"{dist.get_world_size()}"
            )

        # Verify rank matches
        if dist.get_rank() != self.local_rank:
            raise ValueError(
                f"Expected rank {self.local_rank} but got {dist.get_rank()}"
            )

        self.logger.info(
            f"[GPU={self.gpu}] NCCLGradSyncCallback initialized for "
            f"rank {self.local_rank}/{self.world_size}"
        )

    def _sync_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Synchronize gradients across all ranks using NCCL all-reduce."""
        params = 0

        for _, param in model.named_parameters():
            if param.grad is None:
                continue

            params += param.grad.numel()

            # All-reduce gradient across all ranks
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            # Average by dividing by world size
            param.grad.div_(self.world_size)

        self.logger.debug(
            f"[GPU={self.gpu}] Gradient sync step {step}: "
            f"{params / 1e6:.2f}M params"
        )

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """Called before optimizer step - synchronize gradients."""
        step = state.global_step

        self.logger.debug(
            f"[GPU={self.gpu}] Pre-optimizer step {step} - "
            f"starting gradient synchronization"
        )

        # Synchronize gradients across all ranks
        self._sync_gradients(self.model, step)

        self.logger.debug(
            f"[GPU={self.gpu}] Pre-optimizer step {step} - "
            f"gradient synchronization complete"
        )

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """Called after optimizer step - cleanup if needed."""
        step = state.global_step

        self.logger.debug(
            f"[GPU={self.gpu}] Post-optimizer step {step} - cleanup complete"
        )

        # No cleanup needed for NCCL-based sync
        # This method is kept for interface compatibility


# Add this integration code for HyperSloth at the end of the file


def setup_nccl_for_hypersloth(gpu: int, gpus: list) -> None:
    """Setup NCCL environment variables for HyperSloth integration."""
    import os
    import time
    import torch.distributed as dist

    # Map HyperSloth parameters to NCCL environment variables
    rank = gpus.index(gpu)  # Local rank based on position in GPU list
    world_size = len(gpus)  # Total number of GPUs

    # Set required NCCL environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Localhost for single machine
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"  # Use fixed port
    logger = get_hypersloth_logger(log_level="DEBUG")
    # Log all set environment variables
    logger.info(
        f'[GPU={gpu}] NCCL env: RANK={os.environ["RANK"]}, WORLD_SIZE={os.environ["WORLD_SIZE"]}, MASTER_ADDR={os.environ["MASTER_ADDR"]}, MASTER_PORT={os.environ["MASTER_PORT"]}'
    )

    # Assert the port is free
    # import socket

    # if rank == 0:
    #     try:
    #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    #             sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #             sock.bind((os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])))
    #     except OSError as e:
    #         raise RuntimeError(
    #             f"[GPU={gpu}] Port {os.environ['MASTER_PORT']} is not available: {e}"
    #         )

    # Set the current CUDA device to the specific GPU

    logger.info(
        f"[GPU={gpu}] Setting current CUDA device to:0, {os.environ['CUDA_VISIBLE_DEVICES']=}"
    )

    torch.cuda.set_device(0)

    # Retry logic for NCCL initialization
    max_retries = 100
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            # Initialize NCCL process group
            dist.init_process_group(
                backend="nccl", init_method="env://", rank=rank, world_size=world_size
            )

            logger.info(
                f"[GPU={gpu}] NCCL setup complete: "
                f"rank={rank}, world_size={world_size}, attempt={attempt + 1}"
            )
            return

        except Exception as e:
            logger.info(
                f"[GPU={gpu}] NCCL init attempt {attempt + 1}/{max_retries} "
                f"failed: {e}"
            )

            if attempt < max_retries - 1:
                logger.info(f"[GPU={gpu}] Retrying NCCL init in {retry_delay}s...")
                time.sleep(retry_delay)

                # Clean up any partial initialization
                if dist.is_initialized():
                    try:
                        dist.destroy_process_group()
                    except:
                        pass
            else:
                logger.info(
                    f"[GPU={gpu}] Failed to initialize NCCL after "
                    f"{max_retries} attempts: {e}"
                )
                raise
