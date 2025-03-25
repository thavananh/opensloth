import os
import time
import numpy as np
import torch
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from loguru import logger

TIME_OUT = 120
SLEEP_TIME = 0.1

class MmapGradientSyncV2:
    """
    Enhanced single-file gradient sync with an extra "iteration_done" barrier
    so that all GPUs move in lockstep across iterations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        gpu: int,
        gpus: list,
        grad_dir: str = "/dev/shm/hypersloth_v2/",
    ):
        # Basic info
        self.gpu = gpu  # e.g. actual device ID
        self.gpus = gpus  # e.g. [0,1] or [4,5]
        self.local_rank = gpus.index(gpu)  # 0..(num_gpus-1)
        self.num_gpus = len(gpus)
        self.is_main = (self.local_rank == 0)

        os.makedirs(grad_dir, exist_ok=True)
        self.buffer_path = os.path.join(grad_dir, "grad_buffer_v2.dat")

        logger.info(f"[GPU {self.gpu}] local_rank={self.local_rank}, gpus={self.gpus}, is_main={self.is_main}")

        # ------------------------------------------------
        # Build param info: each param uses param.numel * num_gpus
        # for storing local grads from each rank
        # ------------------------------------------------
        self.param_info = []
        current_offset = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            shape = param.shape
            self.param_info.append({
                "name": name,
                "offset": current_offset,
                "numel": numel,
                "shape": shape,
            })
            current_offset += numel * self.num_gpus

        # ------------------------------------------------
        # We'll store 3 sets of flags:
        #   1) write flags   [0..(num_gpus-1)]
        #   2) read flags    [num_gpus..(2*num_gpus-1)]
        #   3) iteration_done [2*num_gpus..(3*num_gpus-1)]
        # So total size = param_slices + 3*num_gpus
        # ------------------------------------------------
        self.flags_offset = current_offset
        self.flags_size = 3 * self.num_gpus  # write + read + iteration_done
        total_size = self.flags_offset + self.flags_size

        # We'll put a "magic header" at index 0, shifting data by +1.
        self.magic_offset = 0
        self.data_offset = 1
        total_size_with_header = total_size + self.data_offset

        # ------------------------------------------------
        # Main GPU creates or recreates the file
        # Others wait
        # ------------------------------------------------
        if self.is_main:
            if os.path.exists(self.buffer_path):
                logger.info(f"[GPU {self.gpu}] Removing stale {self.buffer_path}")
                os.remove(self.buffer_path)

            logger.info(f"[GPU {self.gpu}] Creating {self.buffer_path} size={total_size_with_header}")
            tmp = np.memmap(
                self.buffer_path,
                dtype="float32",
                mode="w+",
                shape=(total_size_with_header,),
            )
            tmp[:] = 0.0
            # Magic header
            tmp[self.magic_offset] = 1234.0
            tmp.flush()
            del tmp
        else:
            # Wait for main GPU to create
            start = time.time()
            while True:
                if os.path.exists(self.buffer_path):
                    sz = os.path.getsize(self.buffer_path)
                    needed = total_size_with_header * 4
                    if sz >= needed:
                        test_map = np.memmap(self.buffer_path, dtype="float32", mode="r", shape=(total_size_with_header,))
                        if test_map[self.magic_offset] == 1234.0:
                            del test_map
                            break
                        del test_map
                if (time.time() - start) > TIME_OUT:
                    raise RuntimeError(f"[GPU {self.gpu}] Timed out waiting for {self.buffer_path}")
                time.sleep(SLEEP_TIME)

        # ------------------------------------------------
        # Open for read+write
        # ------------------------------------------------
        self.mem = np.memmap(
            self.buffer_path,
            dtype="float32",
            mode="r+",
            shape=(total_size_with_header,),
        )
        logger.info(f"[GPU {self.gpu}] Mmap opened total_size_with_header={total_size_with_header}")

        # Shift offsets by +1 for the magic header
        self.total_size = total_size
        self.flags_offset += self.data_offset
        for info in self.param_info:
            info["offset"] += self.data_offset

    # ------------------------------------------------
    # Helper to set a particular stage's flag
    # stage: "write", "read", or "iteration_done"
    # ------------------------------------------------
    def _set_flag(self, stage: str):
        if stage == "write":
            # Index = local_rank
            idx = self.local_rank
        elif stage == "read":
            # Index = local_rank + num_gpus
            idx = self.local_rank + self.num_gpus
        elif stage == "iteration_done":
            # Index = local_rank + 2*num_gpus
            idx = self.local_rank + 2 * self.num_gpus
        else:
            raise ValueError(f"Unknown stage={stage}")

        self.mem[self.flags_offset + idx] = 1.0
        self.mem.flush()
        logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) set {stage} flag @ {idx}")

    # ------------------------------------------------
    # Wait for all GPUs to set "stage" flag
    # "stage" can be "write", "read", "iteration_done"
    # ------------------------------------------------
    def _wait_for_all(self, stage: str, timeout=TIME_OUT):
        start = time.time()
        while True:
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
                logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) all done with {stage}")
                break

            if (time.time() - start) > timeout:
                raise RuntimeError(
                    f"[GPU {self.gpu}] Timeout waiting for all GPUs to finish {stage} stage"
                )
            time.sleep(SLEEP_TIME)

    def _reset_flags(self, offset: int, length: int):
        """
        Zero out a slice of the flags array (for a given stage).
        """
        self.mem[offset : offset + length] = 0.0
        self.mem.flush()

    # ------------------------------------------------
    # PUBLIC: set / wait / reset iteration_done
    # ------------------------------------------------
    def set_iteration_done(self):
        self._set_flag("iteration_done")

    def wait_for_all_iteration_done(self):
        self._wait_for_all("iteration_done")

    def reset_iteration_done_flags(self):
        """
        Only main GPU resets them to zero
        (range [2*num_gpus..3*num_gpus])
        """
        if self.is_main:
            self._reset_flags(
                self.flags_offset + 2 * self.num_gpus,
                self.num_gpus,
            )
            logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) reset iteration_done flags")

    # ------------------------------------------------
    # Main synchronization steps (write/read/zero)
    # ------------------------------------------------
    def accumulate_local_grad(self, model: torch.nn.Module):
        wrote_any_grad = False
        named_params = dict(model.named_parameters())

        for info in self.param_info:
            name = info["name"]
            offset = info["offset"]
            numel = info["numel"]
            param = named_params[name]
            if param.grad is not None:
                sub_offset = offset + self.local_rank * numel
                flat_grad = param.grad.detach().cpu().numpy().ravel()
                self.mem[sub_offset : sub_offset + numel] = flat_grad
                wrote_any_grad = True

        self.mem.flush()
        logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) accumulate_local_grad: wrote_any_grad={wrote_any_grad}")
        # Even if we didn't write anything, we must set write=1
        self._set_flag("write")

    def read_final_grad_into_model(self, model: torch.nn.Module, average=True):
        """
        Wait for all writes, sum up slices, set read=1
        """
        self._wait_for_all("write")
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

        self._set_flag("read")
        logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) read_final_grad_into_model done.")

    def zero_mmaps(self):
        """
        Wait for all read. Main GPU zeros out param region & read flags.
        Others wait for that. Then they are all free to call iteration_done.
        """
        self._wait_for_all("read")

        if self.is_main:
            # Zero out param slices [0.. flags_offset)
            logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) zero_mmaps: main GPU clearing param region.")
            self.mem[0 : self.flags_offset] = 0.0
            self.mem.flush()

            # Reset the write+read flags ( [0..2*num_gpus] slice )
            self._reset_flags(self.flags_offset, 2 * self.num_gpus)
            logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) zero_mmaps: main done clearing write/read flags")
        else:
            # Wait for main GPU to wipe the param region + reset write/read flags
            start = time.time()
            while True:
                flags_slice = self.mem[self.flags_offset : self.flags_offset + 2 * self.num_gpus].copy()
                # if they are all 0.0, then main finished
                if np.all(flags_slice == 0.0):
                    break
                if (time.time() - start) > TIME_OUT:
                    raise RuntimeError(f"[GPU {self.gpu}] Timeout waiting for zero_mmaps by main GPU")
                time.sleep(SLEEP_TIME)

        logger.debug(f"[GPU {self.gpu}] (lr={self.local_rank}) zero_mmaps done.")

#
# A sample callback that uses the extra iteration barrier
#
class MmapGradSyncCallback(TrainerCallback):
    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu_index = gpu
        self.gpus = gpus
        self.is_main = (gpu == gpus[0])
        self.grad_sync = MmapGradientSyncV2(model, gpu, gpus, grad_dir)

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        1) local_grad -> memmap (write=1)
        2) wait all -> final sum => param.grad (read=1)
        """
        logger.debug(f"[GPU {self.gpu_index}] Step1: accumulate_local_grad")
        self.grad_sync.accumulate_local_grad(self.model)

        logger.debug(f"[GPU {self.gpu_index}] Step2: read_final_grad_into_model")
        self.grad_sync.read_final_grad_into_model(self.model, average=True)
        logger.debug(f"[GPU {self.gpu_index}] Completed read_final_grad_into_model")

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        3) zero + wait => iteration done
        4) wait all iteration done => main resets iteration_done => next iteration
        """
        logger.debug(f"[GPU {self.gpu_index}] Step3: zero_mmaps")
        self.grad_sync.zero_mmaps()
        logger.debug(f"[GPU {self.gpu_index}] zero_mmaps done. Step4: iteration barrier")

        # Mark iteration done
        self.grad_sync.set_iteration_done()
        self.grad_sync.wait_for_all_iteration_done()
        # Main GPU resets iteration_done flags for next iteration
        self.grad_sync.reset_iteration_done_flags()
        logger.debug(f"[GPU {self.gpu_index}] Completed iteration barrier")
