import os
import time
import numpy as np
import torch
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from loguru import logger

TIME_OUT = 120
SLEEP_TIME = 0.05


class MmapGradientSyncV2:
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
        grad_dir: str = "/dev/shm/hypersloth_v2/",
    ):
        # Basic info
        self.gpu = gpu
        self.gpus = gpus
        self.local_rank = gpus.index(gpu)
        self.num_gpus = len(gpus)
        self.is_main = self.local_rank == 0

        os.makedirs(grad_dir, exist_ok=True)
        self.buffer_path = os.path.join(grad_dir, "grad_buffer_v2.dat")

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
        # Main GPU creates/truncates the file; open memmap
        # ------------------------------------------------
        self.mem = np.memmap(
            self.buffer_path,
            dtype="float32",
            mode="w+",
            shape=(total_size,),
        )

        logger.info(f"[Init GPU={self.gpu}] Memmap opened: total_size={total_size}")

        # If main GPU, initialize global_step=0
        if self.is_main:
            self._init_global_step_area()

        # Ensure all GPUs see global_step=0 before proceeding (Step 0).
        self._barrier_wait_for_step0()

        logger.info(
            f"[Init GPU={self.gpu}] local_rank={self.local_rank}, gpus={self.gpus}, is_main={self.is_main}, starting global_step={self._get_current_step()}"
        )

        self.current_state = None
        
        # log_current_state(self)

    # ==================================================
    # Global step management
    # ==================================================
    def _init_global_step_area(self):
        """Force global_step=0 at the beginning."""
        self.mem[self.step_offset] = 0.0
        self.mem.flush()

    def _barrier_wait_for_step0(self):
        """
        Ensure that all GPUs see the global_step=0 before continuing.
        The main GPU sets it to 0 if not already; others wait.
        """
        start_time = time.time()
        while True:
            if self.is_main:
                # Just ensure it's written as 0
                self._set_current_step(0)

            step_val = self._get_current_step()
            if step_val == 0:
                break

            if (time.time() - start_time) > TIME_OUT:
                raise RuntimeError(
                    f"[GPU={self.gpu}] Timeout waiting for global_step=0"
                )
            time.sleep(SLEEP_TIME)

        logger.info(f"[GPU={self.gpu}] Synced on global_step=0")

    def _get_current_step(self) -> int:
        return int(self.mem[self.step_offset])

    def _set_current_step(self, step: int):
        # Only the main GPU writes the step
        assert self.is_main, "Only main GPU can set global_step"
        self.mem[self.step_offset] = float(step)
        self.mem.flush()

    def _increment_global_step(self):
        """main GPU increments the global_step by +1"""
        old = self._get_current_step()
        new = old + 1
        self._set_current_step(new)
        logger.debug(f"[GPU={self.gpu}] incremented global_step from {old} to {new}")

    def _check_step_matches_local(self, local_step: int):
        """
        Before waiting on flags, confirm the file's global_step
        matches our local iteration step (what we believe we're on).
        """
        file_step = self._get_current_step()
        return file_step == local_step

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

        self.mem[self.flags_offset + idx] = 1.0
        self.mem.flush()
        logger.debug(f"[GPU={self.gpu}] set {stage}=1 at index={idx}")

    def _wait_for_all(self, stage: str, local_step: int, timeout=TIME_OUT):
        """
        Wait until all GPUs have set stage=1,
        but ONLY if the global_step in the file matches local_step.
        If the file step changes, keep checking or eventually fail on timeout.
        """
        start_time = time.time()
        while True:
            # If the file's global_step changed, we keep waiting
            # until it matches local_step again, or time out.
            if not self._check_step_matches_local(local_step):
                # Not matching yet, keep sleeping
                time.sleep(SLEEP_TIME)
                continue

            flags_slice = self.mem[
                self.flags_offset : self.flags_offset + self.flags_size
            ].copy()

            if stage == "write":
                done_slice = flags_slice[0 : self.num_gpus]
            elif stage == "read":
                done_slice = flags_slice[self.num_gpus : 2 * self.num_gpus]
            elif stage == "iteration_done":
                done_slice = flags_slice[2 * self.num_gpus : 3 * self.num_gpus]
            else:
                raise ValueError(f"Unknown stage={stage}")

            if np.all(done_slice == 1.0):
                logger.opt(depth=1).debug(
                    f"[GPU={self.gpu}] all GPUs done with {stage} at step={local_step}."
                )
                break
            else:
                elapsed = time.time() - start_time
                printed = {}
                t = int(elapsed) % 5
                if t == 0 and not printed.get(t, False):
                    logger.opt(depth=1).debug(
                        f"[GPU={self.gpu}] waiting for {stage}, step={local_step}, flags={done_slice.tolist()}"
                    )
                    printed[t] = True

                if (time.time() - start_time) > timeout:
                    raise RuntimeError(
                        f"[GPU={self.gpu}] Timeout waiting for {stage} at local_step={local_step}"
                    )
                time.sleep(SLEEP_TIME)

        logger.debug(
            f"[GPU={self.gpu}] done waiting for {stage} at step={local_step}"
        )

    def _reset_all_flags(self):
        """main GPU zeros out the 3 sets of flags"""
        logger.debug(
            f"[GPU={self.gpu}] main GPU resetting all flags => [write, read, iteration_done]"
        )
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
        self.current_state = "accumulate_local_grad"
        if not self._check_step_matches_local(local_step):
            raise RuntimeError(
                f"[GPU={self.gpu}] accumulate_local_grad mismatch: "
                f"local_step={local_step}, file_step={self._get_current_step()}"
            )

        named_params = dict(model.named_parameters())
        wrote_grad = False
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
        self.current_state = "done accumulate_local_grad"

    def read_final_grad_into_model(
        self, model: torch.nn.Module, local_step: int, average=True
    ):
        """
        Step 2: Wait for all writes => sum (optionally average) => set read=1
        """

        logger.debug(
            f"[GPU={self.gpu}] read_final_grad => waiting for write, step={local_step}"
        )
        self.current_state = "waiting for write"
        self._wait_for_all("write", local_step)
        self.current_state = "summing"

        named_params = dict(model.named_parameters())
        for info in self.param_info:
            name = info["name"]
            offset = info["offset"]
            numel = info["numel"]
            shape = info["shape"]
            param = named_params[name]
            if not param.requires_grad:
                continue

            acc = np.zeros(numel, dtype=np.float32)
            for rank_id in range(self.num_gpus):
                sub_offset = offset + rank_id * numel
                acc += self.mem[sub_offset : sub_offset + numel]

            if average:
                acc /= self.num_gpus

            param.grad = torch.from_numpy(acc.reshape(shape)).to(param.device)

        logger.debug(f"[GPU={self.gpu}] done summing => set read=1, step={local_step}")
        self._set_flag("read")
        self.current_state = "done summing"

    def zero_mmaps(self, local_step: int):
        """
        Called right after the optimizer step on each GPU:
        Wait for all read=1 => main GPU zeroes the gradient region => others wait
        """
        logger.debug(
            f"[GPU={self.gpu}] zero_mmaps => waiting for read, step={local_step}"
        )
        self.current_state = "waiting for read"
        self._wait_for_all("read", local_step)

        # Only main GPU zeros out param region
        
        if self.is_main:
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
            
            logger.debug('Main GPU increments global_step and resets flags')
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
    A TrainerCallback that uses 'MmapGradientSyncV2' with a global step approach,
    matching the requested 3-step flow. The HuggingFace Trainer calls:

      - on_pre_optimizer_step(...)   -> we do Step 1 & Step 2
      - **the trainer does the optimizer step**
      - on_optimizer_step(...)       -> we do the post-step syncing + global_step increment
    """

    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu_index = gpu
        self.gpus = gpus
        self.grad_sync = MmapGradientSyncV2(model, gpu, gpus, grad_dir)

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        HF calls this right before the optimizer step.
         - Step 1: Each GPU accumulates/writes its gradient => set 'write=1'
         - Step 2: Wait for all writes, read & average => set 'read=1'
        """
        logger.info("=" * 80)
        local_step = state.global_step  # Our "iteration" index
        logger.debug(
            f"[GPU={self.gpu_index}] on_pre_optimizer_step => Step1 accumulate_local_grad"
        )
        self.grad_sync.accumulate_local_grad(self.model, local_step)

        logger.debug(
            f"[GPU={self.gpu_index}] on_pre_optimizer_step => Step2 read_final_grad"
        )
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
            f"[GPU={self.gpu_index}] on_optimizer_step => zero memmaps, step={local_step}"
        )
        self.grad_sync.zero_mmaps(local_step)

        logger.debug(
            f"[GPU={self.gpu_index}] on_optimizer_step => iteration_done, step={local_step}"
        )
        self.grad_sync.iteration_done(local_step)

        logger.debug(
            f"[GPU={self.gpu_index}] on_optimizer_step => complete => next step will be {local_step + 1}"
        )


# from fastcore.all import threaded


# @threaded
# def log_current_state(mg: MmapGradientSyncV2):
#     t = time.time()
#     while True:
#         elapse_sec = int(time.time() - t)
#         if elapse_sec % 5 == 0 and mg.current_state is not None:
#             logger.info(f"[GPU={mg.gpu}] Current state: {mg.current_state}")
