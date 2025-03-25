import os
import sys
import time
from functools import partial
from typing import List, Tuple

import filelock
import numpy as np
import torch
from loguru import logger
from speedy_utils import Clock
from speedy_utils.all import multi_thread
from transformers.trainer_callback import (TrainerCallback, TrainerControl,
                                           TrainerState)

multi_thread = partial(multi_thread, report=False, progress=False, workers=64)




# Just a small constant for "polling" in the barrier.

TIME_OUT = 120
SLEEP_TIME = 0.1


class Memap:
    """
    Demonstration of a single-file memory map that:
      - Stores parameter tensors contiguously via offset
      - Optionally includes read/write "flags" in the same file for barriers
    """

    def __init__(self, path: str, param_metadata: dict, num_gpus: int, gpu_id: int, mode="r+"):
        """
        Args:
          path: Path to the single .dat file (will create if needed).
          param_metadata: dict of the form:
             {
               "layer1.weight":  (offset=0, numel=1024, shape=(256,4)),
               "layer1.bias":    (offset=1024, numel=4, shape=(4,)),
               ...
             }
          num_gpus: total number of processes/gpus participating
          gpu_id: integer ID of the local process (0..num_gpus-1)
          mode: 'w+' to create/overwrite, 'r+' to open existing, etc.
        """
        self.path = path
        self.num_gpus = num_gpus
        self.gpu_id = gpu_id
        self.param_metadata = param_metadata

        # Compute total size needed.
        # You could also store "flags" in extra space at the end.
        max_offset = 0
        for meta in param_metadata.values():
            offset, numel, _ = meta
            max_offset = max(max_offset, offset + numel)

        # Decide how many slots to reserve for flags. For example,
        # we store a single int per GPU for "write done" plus "read done".
        # That might be 2 * num_gpus ints. Let's store them *after* the params.
        self.flag_offset = max_offset
        self.flag_length = 2 * num_gpus  # 2 states per GPU: [write_done, read_done]
        total_size = self.flag_offset + self.flag_length

        # Create the memmap array.
        # mode='w+' means create or overwrite a file, clearing it to zeros.
        # mode='r+' means update an existing file in place.
        if mode.startswith("w"):
            # If we want to start fresh, let's create with zeros.
            # Then open again in read+write mode.
            tmp = np.memmap(path, dtype="float32", mode="w+", shape=(total_size,))
            tmp[:] = 0.0
            tmp.flush()
            del tmp

        self.mem = np.memmap(path, dtype="float32", mode="r+", shape=(total_size,))

    def update_param(self, param_name: str, value: np.ndarray):
        """
        Write a local gradient (or param) `value` into the buffer at the offset
        for `param_name`.
        """
        offset, numel, _ = self.param_metadata[param_name]
        # Flatten in case 'value' is multi-dimensional.
        flat = value.astype(np.float32).ravel()
        assert flat.size == numel, f"Size mismatch: expected {numel}, got {flat.size}"

        # If each GPU updates a separate region (like param_name already
        # subdivided by GPU), no lock is needed. Otherwise, you might add one.
        self.mem[offset : offset + numel] = flat
        self.mem.flush()

    def read_param(self, param_name: str) -> np.ndarray:
        """Return a copy of the param array from memmap."""
        offset, numel, shape = self.param_metadata[param_name]
        arr = self.mem[offset : offset + numel].copy()
        return arr.reshape(shape)

    def all_reduce(self, param_name: str, method="sum"):
        """
        A naive "all-reduce" that just sums in-place across GPU 'slots'
        if you stored them in separate slices. 
        Otherwise, for a single-slice approach, you'd need actual locks
        or a separate aggregator step.
        
        For a demonstration, let's assume each GPU has a *distinct* slice
        and we are the "main" GPU that sums them up. Then we broadcast back.
        """
        if self.gpu_id != 0:
            # Non-main GPU: do nothing
            return

        # (Pseudo-code) If we had param slices for each GPU, we would:
        #   1) read each slice
        #   2) sum them up
        #   3) store the final sum in slice 0
        #   4) optionally divide if method='mean'
        #
        # For now, let's show a single-slice approach: we read it, do something, write it back
        arr = self.read_param(param_name)
        if method == "sum":
            # In a single-slice scenario, not much to do. We'll just do arr = arr
            pass
        elif method == "mean":
            arr /= self.num_gpus
        # Write back
        self.update_param(param_name, arr)

    # -------------------------------------------------
    # Barrier logic: store each GPU's "flag" in memmap
    # -------------------------------------------------
    def set_flag(self, stage: str, value: float = 1.0):
        """
        stage can be 'write' or 'read'
        Let's store:
          index = self.gpu_id if stage == 'write'
          index = self.gpu_id + self.num_gpus if stage == 'read'
        """
        if stage == "write":
            idx = self.gpu_id
        elif stage == "read":
            idx = self.gpu_id + self.num_gpus
        else:
            raise ValueError("stage must be 'write' or 'read'")
        self.mem[self.flag_offset + idx] = value
        self.mem.flush()

    def wait_for_all(self, stage: str, timeout=30):
        """
        Wait until all GPUs have set their flags for `stage`.
        """
        start = time.time()
        while True:
            flags = self.mem[self.flag_offset : self.flag_offset + self.flag_length].copy()
            if stage == "write":
                # check if [0..num_gpus-1] are all == 1.0
                if np.all(flags[0 : self.num_gpus] == 1.0):
                    break
            else:  # stage == "read"
                # check if [num_gpus..2*num_gpus-1] are all == 1.0
                if np.all(flags[self.num_gpus : 2 * self.num_gpus] == 1.0):
                    break
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for stage={stage}")
            time.sleep(SLEEP_TIME)

    def reset_flags(self):
        """
        Zero out all flags. Typically called by main GPU after a step completes.
        """
        self.mem[self.flag_offset : self.flag_offset + self.flag_length] = 0.0
        self.mem.flush()





class MmapGradientSyncV2:
    def __init__(
        self,
        model: torch.nn.Module,
        gpu: int,
        gpus: List[int],
        grad_dir: str = "/dev/shm/hypersloth_v2/",
    ):
        # The actual GPU ID used by the system:
        self.gpu = gpu
        # The list of all GPU IDs (could be [4,5,6,7])
        self.gpus = gpus
        # The local rank (0..num_gpus-1) in that list:
        self.local_rank = gpus.index(gpu)
        self.num_gpus = len(gpus)
        # "Main" GPU is the one with local_rank == 0
        self.is_main = (self.local_rank == 0)

        os.makedirs(grad_dir, exist_ok=True)
        self.buffer_path = os.path.join(grad_dir, "grad_buffer_v2.dat")

        # Build param info and figure out total_size
        self.param_info = []
        current_offset = 0
        named_params = list(model.named_parameters())
        for name, param in named_params:
            if not param.requires_grad:
                continue
            numel = param.numel()
            shape = param.shape
            self.param_info.append({
                "name": name,
                "offset": current_offset,
                "numel": numel,
                "shape": shape
            })
            current_offset += numel * self.num_gpus

        # Reserve space for flags
        self.flags_offset = current_offset
        self.flags_size = 2 * self.num_gpus
        total_size = self.flags_offset + self.flags_size

        # We'll store a "magic header" in the very first float (index=0),
        # then put the parameter data after that. So let's shift everything by +1.
        #    [0] = magic header
        #    [1.. total_size] = param slices + flags
        self.magic_offset = 0
        self.data_offset = 1  # the actual start for param slices
        total_size_with_header = total_size + self.data_offset

        if self.is_main:
            # 1) Remove any stale file
            if os.path.exists(self.buffer_path):
                logger.info(f"[GPU {self.gpu}] Removing stale {self.buffer_path}")
                os.remove(self.buffer_path)

            # 2) Create with zeros
            logger.info(f"[GPU {self.gpu}] Creating {self.buffer_path} w/ size={total_size_with_header}")
            tmp = np.memmap(self.buffer_path, dtype="float32", mode="w+", shape=(total_size_with_header,))
            tmp[:] = 0.0

            # 3) Write a "magic" marker into the first element to indicate "fresh file"
            tmp[self.magic_offset] = 1234.0  # any distinct float
            tmp.flush()
            del tmp

        else:
            # Wait until the main GPU finishes creating the file properly
            start_time = time.time()
            while True:
                if os.path.exists(self.buffer_path):
                    # Also check that the file is big enough
                    # and that the magic header is set
                    sz = os.path.getsize(self.buffer_path)
                    # size in bytes, so compare to total_size_with_header * sizeof(float)
                    needed_bytes = total_size_with_header * 4
                    if sz >= needed_bytes:
                        # Now open in read mode quickly and check the header
                        test_map = np.memmap(self.buffer_path, dtype="float32", mode="r", shape=(total_size_with_header,))
                        if test_map[self.magic_offset] == 1234.0:
                            # Good. The main GPU must have created it for THIS run
                            del test_map
                            break
                        del test_map
                if (time.time() - start_time) > TIME_OUT:
                    raise RuntimeError(f"[GPU {self.gpu}] Timeout waiting for {self.buffer_path} to be created with magic header.")
                time.sleep(SLEEP_TIME)
        # Everyone now opens in r+ mode for reading/writing
        self.mem = np.memmap(self.buffer_path, dtype="float32", mode="r+", shape=(total_size_with_header,))
        logger.info(f"[GPU {self.gpu}] Mmap opened: total_size_with_header={total_size_with_header}")

        # Adjust offsets: param data + flags is after the header
        self.total_size = total_size
        # The param slices, flags_offset, etc. need to be shifted by self.data_offset:
        self.flags_offset += self.data_offset
        for info in self.param_info:
            info["offset"] += self.data_offset

        # Now you can do the rest of your logic as before
        # (accumulate_local_grad, read_final_grad_into_model, etc.)
    # -----------------------------------------------------------------
    # Helper methods for barrier flags
    # -----------------------------------------------------------------

    def _set_flag(self, stage: str):
        if stage == "write":
            idx = self.local_rank
        elif stage == "read":
            idx = self.local_rank + self.num_gpus
        else:
            raise ValueError(...)
        self.mem[self.flags_offset + idx] = 1.0
        self.mem.flush()


    def _wait_for_all(self, stage: str, timeout=TIME_OUT):
        """
        Wait until all GPUs have set their flags for `stage`.
        Polling approach. If stage='write', we wait for mem[flags_offset.. flags_offset+num_gpus] == 1.0
        If stage='read', we wait for mem[flags_offset+num_gpus.. flags_offset+2*num_gpus] == 1.0
        """
        start_time = time.time()
        while True:
            flags_slice = self.mem[self.flags_offset : self.flags_offset + self.flags_size].copy()

            if stage == "write":
                # check first num_gpus elements
                done_slice = flags_slice[0 : self.num_gpus]
            else:  # stage == "read"
                done_slice = flags_slice[self.num_gpus : 2*self.num_gpus]

            if np.all(done_slice == 1.0):
                break

            if time.time() - start_time > timeout:
                raise RuntimeError(f"Timeout waiting for all GPUs to finish {stage} stage")
            time.sleep(SLEEP_TIME)

    def _reset_flags(self):
        """
        Zero out the entire flag range. 
        Called by main GPU after finishing the iteration.
        """
        self.mem[self.flags_offset : self.flags_offset + self.flags_size] = 0.0
        self.mem.flush()

    # -----------------------------------------------------------------
    # Main public methods: accumulate, read+reduce, zero
    # -----------------------------------------------------------------

    def accumulate_local_grad(self, model: torch.nn.Module):
        for info in self.param_info:
            name = info["name"]
            offset = info["offset"]
            numel = info["numel"]
            param = dict(model.named_parameters())[name]
            if param.grad is None:
                continue

            # Write local grad into the slice for local_rank
            sub_offset = offset + self.local_rank * numel
            flat_grad = param.grad.detach().cpu().numpy().ravel()
            self.mem[sub_offset : sub_offset + numel] = flat_grad

        self.mem.flush()
        self._set_flag("write")
        
    def read_final_grad_into_model(self, model: torch.nn.Module, average: bool = True):
        """
        Wait for all GPUs to finish writing, then sum across each GPU slice for every param.
        Assign the final gradient to model.param.grad. (Optionally average.)
        Then set this GPU's "read" flag (meaning: I've read final grads).
        """
        # 1) Wait for all writes
        self._wait_for_all("write")

        # 2) Compute final grad for each param by summing across sub-slices
        #    Then store into param.grad
        named_params = dict(model.named_parameters())
        for info in self.param_info:
            name = info["name"]
            offset = info["offset"]
            numel = info["numel"]
            shape = info["shape"]

            param = named_params[name]
            if not param.requires_grad:
                continue

            # sum all GPUs' slices
            acc = np.zeros(numel, dtype=np.float32)
            for rank_id in range(self.num_gpus):
                sub_offset = offset + rank_id * numel
                acc += self.mem[sub_offset : sub_offset + numel]

            if average:
                acc /= self.num_gpus

            # put into param.grad
            param.grad = torch.from_numpy(acc.reshape(shape)).to(param.device)

        # 3) Set this GPU's "read" flag
        self._set_flag("read")

    def zero_mmaps(self):
        """
        Wait for all GPUs to finish reading, then the main GPU zeros the entire
        param region. The main GPU also resets flags to 0. Others wait until that finishes.
        """
        # 1) Wait for all reads
        self._wait_for_all("read")

        # 2) Main GPU zeros out the param region
        if self.is_main:
            # Zero out [0 .. flags_offset], i.e. all param slices
            self.mem[0 : self.flags_offset] = 0.0
            self.mem.flush()
            # Also reset all flags to 0
            self._reset_flags()
        else:
            # Non-main GPU: wait until flags are reset to 0 by main
            start_time = time.time()
            while True:
                flags_slice = self.mem[self.flags_offset : self.flags_offset + self.flags_size].copy()
                if np.all(flags_slice == 0.0):
                    break
                if time.time() - start_time > TIME_OUT:
                    raise RuntimeError("Timeout waiting for main GPU to reset flags.")
                time.sleep(SLEEP_TIME)


class MmapGradSyncCallback(TrainerCallback):
    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu_index = gpu
        self.gpus = gpus
        self.is_main = gpu == gpus[0]
        self.grad_sync = MmapGradientSyncV2(
            model,
            gpu,
            gpus,
            grad_dir,
        )
        self.loss_file = np.memmap(
            os.path.join(self.grad_dir, "loss.mmap"),
            dtype="float32",
            mode="w+",
            shape=(len(self.gpus),),
        )
        # self.clock = Clock()

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called before optimizer step.
        """
        logger.debug(
            f"[GPU {self.gpu_index}] >>>> Step 1: Accumulating local gradients.. Next step: read_final_grad_into_model"
        )
        self.grad_sync.accumulate_local_grad(self.model)
        logger.debug(
            f"[GPU {self.gpu_index}] >>>> Step 2: Reading final gradients.. Next step: zero_mmaps"
        )
        self.grad_sync.read_final_grad_into_model(self.model, average=True)
        logger.debug(
            f"[GPU {self.gpu_index}] <<<< Step 2: Completed reading final gradients.. Next step: zero_mmaps"
        )

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called after optimizer step.
        """
        logger.debug(
            f"[GPU {self.gpu_index}] >>>> Step 3: Zeroing memmaps.. Next step: accumulate_local_grad"
        )
        self.grad_sync.zero_mmaps()
        logger.debug(
            f"[GPU {self.gpu_index}] <<<< Step 3: Completed zeroing memmaps.. Next step: accumulate_local_grad"
        )
