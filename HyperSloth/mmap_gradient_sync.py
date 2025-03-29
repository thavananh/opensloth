import os
import time
import numpy as np
import torch
from filelock import FileLock
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from loguru import logger

TIME_OUT = 120
SLEEP_TIME = 0.05


class MmapGradientSync:
    """
    Single-file gradient sync that also stores a 'global_step' in the file,
    so that each GPU only waits on flags belonging to the *current* iteration.

    Refactored to enforce:
      0) All GPUs start at global_step=0 in sync.
      1) Each GPU writes its local gradient => set write=1
      2) Wait for all writes => read/average => set read=1
      3) Optim step => set iteration_done=1 => main increments global_step => others wait
    """

    def __init__(
        self,
        model: torch.nn.Module,
        gpu: int,
        gpus: list,
        grad_dir: str = "/dev/shm/hypersloth/",
    ):
        # Basic info
        self.gpu = gpu
        self.gpus = gpus
        self.local_rank = gpus.index(gpu)
        self.num_gpus = len(gpus)
        self.is_main = (self.local_rank == 0)

        os.makedirs(grad_dir, exist_ok=True)
        self.buffer_path = os.path.join(grad_dir, "grad_buffer_v2.dat")
        self.lock_path = self.buffer_path + ".lock"
        self.lock = FileLock(self.lock_path)

        # ------------------------------------------------
        # Build param info: each param => param.numel * num_gpus
        # ------------------------------------------------
        self.param_info = []
        current_offset = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            shape = param.shape
            self.param_info.append(
                {
                    "name": name,
                    "offset": current_offset,
                    "numel": numel,
                    "shape": shape,
                }
            )
            current_offset += numel * self.num_gpus

        # ------------------------------------------------
        # We'll store 3 sets of flags: write, read, iteration_done
        #   write flags:       [0 .. (num_gpus-1)]
        #   read flags:        [num_gpus .. (2*num_gpus-1)]
        #   iteration_done:    [2*num_gpus .. (3*num_gpus-1)]
        # => total_size_for_flags = 3 * num_gpus
        # ------------------------------------------------
        self.flags_offset = current_offset
        self.flags_size = 3 * self.num_gpus
        total_size = self.flags_offset + self.flags_size

        # ------------------------------------------------
        # Add 1 slot for "global_step" (float). total_size += 1
        # ------------------------------------------------
        self.step_offset = total_size
        total_size += 1  # just 1 float for the global step

        # ------------------------------------------------
        # Only the main GPU creates/truncates the file. Others open r+
        # ------------------------------------------------
        if self.is_main:
            # Create/truncate the file to the required size in bytes (float32=4 bytes)
            with open(self.buffer_path, "wb") as f:
                f.truncate(total_size * 4)
        else:
            # Wait until the file is created by the main GPU
            while not os.path.exists(self.buffer_path):
                time.sleep(SLEEP_TIME)
            time.sleep(5.0)  # Give it a few seconds to be ready
        # Open memmap in read/write mode (r+). It's now guaranteed to exist.
        self.mem = np.memmap(
            self.buffer_path,
            dtype="float32",
            mode="r+",
            shape=(total_size,),
        )

        # Initialize global_step=0 if main GPU
        if self.is_main:
            self._init_values()

        logger.info(
            f"[Init GPU={self.gpu}] Memmap opened: total_size={total_size}, "
            f"local_rank={self.local_rank}, gpus={self.gpus}, is_main={self.is_main}, "
            f"starting global_step={self._get_current_step()}"
        )

    def _init_values(self):
        """Force global_step=0 at the beginning."""
        with self.lock:
            self.mem[:] = 0.0
            self.mem.flush()

    def _get_current_step(self) -> int:
        with self.lock:
            step_val = self.mem[self.step_offset]
        return int(step_val)

    def _set_current_step(self, step: int):
        # Only the main GPU writes the step
        assert self.is_main, "Only main GPU can set global_step"
        with self.lock:
            self.mem[self.step_offset] = float(step)
            self.mem.flush()

    def _increment_global_step(self):
        """main GPU increments the global_step by +1"""
        old = self._get_current_step()
        new = old + 1
        self._set_current_step(new)
        logger.debug(f"[GPU={self.gpu}] incremented global_step from {old} to {new}")

    def _check_step_matches_local(self, local_step: int):
        """Check that the file's global_step matches local_step."""
        current_file_step = self._get_current_step()
        return current_file_step == local_step

    # ==================================================
    # Stage-based flags (write/read/iteration_done)
    # ==================================================
    def _set_flag(self, stage: str):
        if stage == "write":
            idx = self.local_rank
        elif stage == "read":
            idx = self.local_rank + self.num_gpus
        elif stage == "iteration_done":
            idx = self.local_rank + 2 * self.num_gpus
        else:
            raise ValueError(f"Unknown stage={stage}")

        with self.lock:
            self.mem[self.flags_offset + idx] = 1.0
            self.mem.flush()

        logger.debug(f"[GPU={self.gpu}] set {stage}=1 at index={idx}")

    def _get_flags_slice(self, stage: str) -> np.ndarray:
        """
        Return the slice of the flags array for the given stage,
        as a *copy* so that we don't hold the lock while analyzing it.
        """
        with self.lock:
            flags_slice = self.mem[self.flags_offset : self.flags_offset + self.flags_size]
        if stage == "write":
            return flags_slice[0 : self.num_gpus]
        elif stage == "read":
            return flags_slice[self.num_gpus : 2 * self.num_gpus]
        elif stage == "iteration_done":
            return flags_slice[2 * self.num_gpus : 3 * self.num_gpus]
        else:
            raise ValueError(f"Unknown stage={stage}")

    def _wait_for_all(self, stage: str, local_step: int, timeout=TIME_OUT):
        """
        Wait until all GPUs have set stage=1,
        but ONLY if the global_step in the file matches local_step.
        If the file step changes, keep waiting or eventually fail on timeout.
        """
        start_time = time.time()
        printed = False

        while True:
            # If the file's global_step changed, we keep waiting
            # until it matches local_step again, or time out.
            if not self._check_step_matches_local(local_step):
                # Not matching yet, keep sleeping
                time.sleep(SLEEP_TIME)
                if (time.time() - start_time) > timeout:
                    raise RuntimeError(
                        f"[GPU={self.gpu}] Timeout: global_step changed or did not match local_step={local_step}"
                    )
                continue

            done_slice = self._get_flags_slice(stage)

            if np.all(done_slice == 1.0):
                logger.opt(depth=1).debug(
                    f"[GPU={self.gpu}] all GPUs done with {stage} at step={local_step}."
                )
                break
            else:
                elapsed = time.time() - start_time
                if (not printed) and (elapsed > 5.0):
                    logger.opt(depth=1).debug(
                        f"[GPU={self.gpu}] waiting for {stage}, step={local_step}, flags={done_slice.tolist()}"
                    )
                    printed = True

                if elapsed > timeout:
                    raise RuntimeError(
                        f"[GPU={self.gpu}] Timeout waiting for {stage} at local_step={local_step}"
                    )
                time.sleep(SLEEP_TIME)

        logger.debug(f"[GPU={self.gpu}] done waiting for {stage} at step={local_step}")

    def _reset_all_flags(self):
        """main GPU zeros out the 3 sets of flags"""
        logger.debug(
            f"[GPU={self.gpu}] main GPU resetting all flags => [write, read, iteration_done]"
        )
        with self.lock:
            self.mem[self.flags_offset : self.flags_offset + self.flags_size] = 0.0
            self.mem.flush()

    # ==================================================
    # Public Methods: Writing Grad, Reading Grad, Zero, Iteration
    # ==================================================
    def accumulate_local_grad(self, model: torch.nn.Module, local_step: int):
        """
        Step 1: Each GPU writes its local gradient into the memmap => set write=1
        """
        # Make sure the file global_step matches local_step
        if not self._check_step_matches_local(local_step):
            raise RuntimeError(
                f"[GPU={self.gpu}] accumulate_local_grad mismatch: "
                f"local_step={local_step}, file_step={self._get_current_step()}"
            )

        named_params = dict(model.named_parameters())
        wrote_grad = False
        with self.lock:
            for info in self.param_info:
                name = info["name"]
                offset = info["offset"]
                numel = info["numel"]
                param = named_params[name]
                if param.grad is not None:
                    sub_offset = offset + self.local_rank * numel
                    flat = param.grad.detach().cpu().numpy().ravel()
                    self.mem[sub_offset : sub_offset + numel] = flat
                    wrote_grad = True

            self.mem.flush()

        logger.debug(
            f"[GPU={self.gpu}] accumulate_local_grad done, wrote_any_grad={wrote_grad}, step={local_step}"
        )
        self._set_flag("write")

    def read_final_grad_into_model(
        self, model: torch.nn.Module, local_step: int, average=True
    ):
        """
        Step 2: Wait for all writes => sum (optionally average) => set read=1
        """
        logger.debug(
            f"[GPU={self.gpu}] read_final_grad => waiting for write, step={local_step}"
        )
        self._wait_for_all("write", local_step)

        named_params = dict(model.named_parameters())
        # Summation has to be done under lock to ensure consistent reads
        with self.lock:
            for info in self.param_info:
                name = info["name"]
                offset = info["offset"]
                numel = info["numel"]
                shape = info["shape"]
                param = named_params[name]
                if not param.requires_grad:
                    continue

                # Summation from all ranks
                acc = np.zeros(numel, dtype=np.float32)
                for rank_id in range(self.num_gpus):
                    sub_offset = offset + rank_id * numel
                    acc += self.mem[sub_offset : sub_offset + numel]

                if average:
                    acc /= self.num_gpus

                param.grad = torch.from_numpy(acc.reshape(shape)).to(param.device)

        logger.debug(f"[GPU={self.gpu}] done summing => set read=1, step={local_step}")
        self._set_flag("read")

    def zero_mmaps(self, local_step: int):
        """
        Called right after the optimizer step on each GPU:
        Wait for all read=1 => main GPU zeroes the gradient region => others wait
        """
        logger.debug(
            f"[GPU={self.gpu}] zero_mmaps => waiting for read, step={local_step}"
        )
        self._wait_for_all("read", local_step)

        # Only main GPU zeros out param region
        if self.is_main:
            with self.lock:
                logger.debug(
                    f"[GPU={self.gpu}] main => zero param region [0..{self.flags_offset}), step={local_step}"
                )
                self.mem[0 : self.flags_offset] = 0.0
                self.mem.flush()

        logger.debug(f"[GPU={self.gpu}] zero_mmaps done, step={local_step}")
        self._set_flag("iteration_done")

    def iteration_done(self, local_step: int):
        """
        Mark iteration_done=1 => wait => main increments global_step => others wait
        This completes Step 3 (post-optim sync).
        """
        logger.debug(
            f"[GPU={self.gpu}] iteration_done => set iteration_done=1 at step={local_step}"
        )
        self._set_flag("iteration_done")

        if self.is_main:
            self._wait_for_all("iteration_done", local_step)

            logger.debug("[GPU={}] main GPU increments global_step and resets flags".format(self.gpu))
            # Reset flags and increment global step
            self._reset_all_flags()
            self._increment_global_step()
        else:
            # Wait until main increments global_step
            while True:
                if self._get_current_step() > local_step:
                    logger.debug(f"[GPU={self.gpu}] main GPU incremented global_step")
                    break
                time.sleep(SLEEP_TIME)

        logger.debug(
            f"[GPU={self.gpu}] iteration_done complete => next iteration (old step={local_step})"
        )


# =======================================================================
# Example Callback using the global-step approach
# =======================================================================
class MmapGradSyncCallback(TrainerCallback):
    """
    A TrainerCallback that uses 'MmapGradientSync' with a global step approach,
    matching the requested 3-step flow. The HuggingFace Trainer calls:

      - on_pre_optimizer_step(...)   -> we do Step 1 & Step 2
      - **the trainer does the optimizer step**
      - on_optimizer_step(...)       -> we do the post-step syncing + global_step increment
    """

    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu = gpu
        self.local_rank = gpus.index(gpu)
        self.gpus = gpus

        self.grad_sync = MmapGradientSync(model, gpu, gpus, grad_dir)

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        HF calls this right before the optimizer step.
         - Step 1: Each GPU accumulates/writes its gradient => set 'write=1'
         - Step 2: Wait for all writes, read & average => set 'read=1'
        """
        logger.debug("=" * 80)

        local_step = state.global_step  # Our "iteration" index
        logger.debug(
            f"[GPU={self.gpu}] on_pre_optimizer_step => Step1 accumulate_local_grad"
        )
        self.grad_sync.accumulate_local_grad(self.model, local_step)

        logger.debug(f"[GPU={self.gpu}] on_pre_optimizer_step => Step2 read_final_grad")
        self.grad_sync.read_final_grad_into_model(self.model, local_step, average=True)

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        HF calls this right after the optimizer step is done.
         - Zero memmaps => wait
         - Mark iteration_done => main GPU increments global_step => others wait
        """
        local_step = state.global_step
        logger.debug(
            f"[GPU={self.gpu}] on_optimizer_step => zero memmaps, step={local_step}"
        )
        self.grad_sync.zero_mmaps(local_step)

        logger.debug(
            f"[GPU={self.gpu}] on_optimizer_step => iteration_done, step={local_step}"
        )
        # This is effectively a barrier
        self.grad_sync.iteration_done(local_step)

        logger.debug(
            f"[GPU={self.gpu}] on_optimizer_step => complete => next step will be {local_step + 1}"
        )
