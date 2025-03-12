import os
import sys
import time
from functools import partial
from typing import List, Tuple

import filelock
import numpy as np
import torch
from loguru import logger
from speedy_utils.all import multi_thread
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

# Disable "report" and "verbose" in multi_thread calls
multi_thread = partial(multi_thread, report=False, verbose=False)

logger.remove()
logger.add("mmap_gradient_sync.log", level="DEBUG")
logger.add(sys.stdout, level="INFO")
SLEEP_TIME = 0.1


# Transformers / Trainer imports
class MmapGradientSync:
    """
    A class that uses memory-mapped files (one per parameter) to accumulate
    gradients across multiple processes. Uses multi_thread(...) to parallelize
    I/O across parameters.

    Typical usage in each training iteration:
      1. loss.backward()
      2. grad_sync.accumulate_local_grad(model)
      3. grad_sync.read_final_grad_into_model(model, average=True)
      4. optimizer.step()
      5. grad_sync.zero_mmaps()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        grad_dir: str,
        gpu: int,
        gpus: List[int],
        lock_dir: str = "/dev/shm",
    ):
        """
        Args:
          model (nn.Module): The model whose parameters we'll synchronize.
          grad_dir (str): Directory for memmap files.
          gpu_id (int): The local GPU index for this process.
          gpus (List[int]): List of ALL GPU indices in the job (for counting).
          lock_dir (str, optional): Directory for .lock files (defaults to grad_dir).
        """
        self.is_main = gpu == gpus[0]

        self.grad_dir = grad_dir
        self.gpu = gpu
        self.gpus = gpus  # store the entire list for synchronization checks
        self.lock_dir = lock_dir if lock_dir is not None else grad_dir

        # Create directories if needed
        os.makedirs(self.grad_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)

        # Gather info for each parameter
        self.param_info = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace(".", "_")
                filename = os.path.join(self.grad_dir, f"{safe_name}.dat")
                numel = param.numel()

                # Initialize memmap file if missing (zeros)
                if not os.path.exists(filename):
                    np.zeros(numel, dtype=np.float32).tofile(filename)

                self.param_info.append(
                    {
                        "name": name,
                        "shape": param.shape,
                        "numel": numel,
                        "filename": filename,
                    }
                )

        logger.debug(
            "Initialized MmapGradientSync with {} parameters.", len(self.param_info)
        )

    def _lockfile(self, filename: str) -> str:
        """Return the path to a .lock file corresponding to filename."""
        basename = os.path.basename(filename)
        return os.path.join(self.lock_dir, basename + ".lock")

    # -------------------------------------------------------
    # Internal single-parameter operations for multi_thread
    # -------------------------------------------------------

    class UniversalLocker:
        """
        A context manager for handling file locks.
        """

        def __init__(self, lockfile_path: str):
            self.lockfile_path = lockfile_path
            self.lock = filelock.FileLock(lockfile_path)

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release()

    def _accumulate_one_param(self, task: Tuple[str, int, np.ndarray]):
        """
        Merges local_grad into the `memmap` for one parameter.
        'task' is a tuple: (filename, numel, local_grad).
        """
        filename, numel, local_grad = task
        lockfile_path = self._lockfile(filename)

        with self.UniversalLocker(lockfile_path):
            mm = np.memmap(filename, dtype=np.float32, mode="r+", shape=(numel,))
            mm[:] += local_grad[:]
            mm.flush()
            del mm

    def _read_one_param(self, task: Tuple[str, int]) -> np.ndarray:
        """
        Reads the final sum from memmap into a NumPy array, returns it.
        'task' is a tuple: (filename, numel).
        """
        filename, numel = task
        lockfile_path = self._lockfile(filename)

        with self.UniversalLocker(lockfile_path):
            mm = np.memmap(filename, dtype=np.float32, mode="r", shape=(numel,))
            arr = np.copy(mm[:])  # copy out
            del mm

        return arr

    def _zero_one_param(self, task: Tuple[str, int]):
        """
        Zeros out memmap for one parameter.
        'task' is a tuple: (filename, numel).
        """
        filename, numel = task
        lockfile_path = self._lockfile(filename)

        with self.UniversalLocker(lockfile_path):
            mm = np.memmap(filename, dtype=np.float32, mode="r+", shape=(numel,))
            mm[:] = 0.0
            mm.flush()
            del mm

    # -------------------------------------------------------
    # Public synchronization methods
    # -------------------------------------------------------

    def accumulate_local_grad(self, model: torch.nn.Module):
        """
        For each parameter, add its local gradient to the memmap.
        Must be called *after* loss.backward().
        """
        tasks = []
        named_params = dict(model.named_parameters())

        for info in self.param_info:
            name = info["name"]
            filename = info["filename"]
            numel = info["numel"]
            param = named_params[name]
            if param.grad is not None:
                local_grad = param.grad.detach().cpu().numpy().reshape(-1)
                tasks.append((filename, numel, local_grad))

        # Parallelize across parameters
        multi_thread(self._accumulate_one_param, tasks)

        logger.debug("[GPU {}] Accumulated local gradients into memmaps.", self.gpu)

        # Write a "done writing" file for this GPU
        write_file_path = f"{self.grad_dir}/count_write_gpu{self.gpu}.txt"
        with self.UniversalLocker(write_file_path + ".lock"):
            with open(write_file_path, "w") as f:
                f.write("1")

    def _wait_for_all_write(self):
        """
        Wait until all GPUs have accumulated gradients
        (presence of count_write_gpu{i}.txt for each i in self.gpus).
        """
        logger.debug(
            "[GPU {}] Waiting for all GPUs to accumulate gradients..", self.gpu
        )
        while True:
            count = 0
            for i in self.gpus:
                cf = f"{self.grad_dir}/count_write_gpu{i}.txt"
                if os.path.exists(cf):
                    count += 1
                # else:
                #     logger.debug("[GPU {}] Missing file: {}", self.gpu_index, cf)
            if count == len(self.gpus):
                break
            time.sleep(SLEEP_TIME)  # reduce busy waiting

        logger.debug("[GPU {}] All GPUs have accumulated gradients.", self.gpu)

    def _wait_for_all_read(self):
        """
        Wait until all GPUs have read gradients
        (presence of count_read_gpu{i}.txt for each i in self.gpus).
        """
        logger.debug("[GPU {}] Waiting for all GPUs to read gradients..", self.gpu)
        while True:
            count = 0
            for gpu_id in self.gpus:
                cf = f"{self.grad_dir}/count_read_gpu{gpu_id}.txt"
                if os.path.exists(cf):
                    count += 1
                # else:
                # logger.debug("[GPU {}] Missing file: {}", self.gpu_index, cf)
            if count == len(self.gpus):
                break
            time.sleep(SLEEP_TIME)  # reduce busy waiting

        logger.debug("[GPU {}] All GPUs have read gradients.", self.gpu)

    def read_final_grad_into_model(self, model: torch.nn.Module, average: bool = True):
        """
        Reads the final gradient from memmaps into param.grad.
        Optionally divides by the number of GPUs in self.gpus.
        """
        # Wait for *all* ranks to finish writing first
        self._wait_for_all_write()

        tasks = []
        named_params = dict(model.named_parameters())

        for info in self.param_info:
            name = info["name"]
            filename = info["filename"]
            numel = info["numel"]
            # Only read if param.requires_grad
            if named_params[name].requires_grad:
                tasks.append((filename, numel))

        results = multi_thread(self._read_one_param, tasks)

        idx = 0
        for info in self.param_info:
            name = info["name"]
            if not named_params[name].requires_grad:
                continue

            arr = results[idx]
            idx += 1
            if average:
                arr /= len(self.gpus)

            param = named_params[name]
            shape = info["shape"]
            param.grad = torch.from_numpy(arr).view(shape).to(param.device)

        logger.debug("[GPU {}] Read final gradients from memmaps into model.", self.gpu)

        # Write a "done reading" file for this GPU
        read_file_path = f"{self.grad_dir}/count_read_gpu{self.gpu}.txt"
        with self.UniversalLocker(read_file_path + ".lock"):
            with open(read_file_path, "w") as f:
                f.write("1")

    def zero_mmaps(self):
        """
        Zeros out all memmap files so each iteration starts fresh.
        Also removes the count files to reset synchronization state.
        """
        self._wait_for_all_read()
        # only perform zeroing on the main GPU
        if self.is_main:
            self._clean()

        else:
            # wait for all count files to be removed
            while True:
                count = 0
                for gpu in self.gpus:
                    wfile = f"{self.grad_dir}/count_write_gpu{gpu}.txt"
                    rfile = f"{self.grad_dir}/count_read_gpu{gpu}.txt"
                    if not os.path.exists(wfile) and not os.path.exists(rfile):
                        count += 1
                if count == len(self.gpus):
                    break
                time.sleep(SLEEP_TIME)

    def _clean(self):
        with self.UniversalLocker(os.path.join(self.lock_dir, "zero.lock")):
            logger.debug(f"[GPU {self.gpu}] Zeroing all memmap files..")
            tasks = []
            for info in self.param_info:
                filename = info["filename"]
                numel = info["numel"]
                tasks.append((filename, numel))

            multi_thread(self._zero_one_param, tasks)

            # Clean up count files (both write and read signals)
            for gpu in self.gpus:
                wfile = f"{self.grad_dir}/count_write_gpu{gpu}.txt"
                with self.UniversalLocker(wfile + ".lock"):
                    if os.path.exists(wfile):
                        os.remove(wfile)
                rfile = f"{self.grad_dir}/count_read_gpu{gpu}.txt"
                with self.UniversalLocker(rfile + ".lock"):
                    if os.path.exists(rfile):
                        os.remove(rfile)


class MmapGradSyncCallback(TrainerCallback):
    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu_index = gpu
        self.gpus = gpus
        self.is_main = gpu == gpus[0]

        self.grad_sync = MmapGradientSync(
            model,
            grad_dir,
            gpu,
            gpus,
        )
        os.makedirs(self.grad_dir, exist_ok=True)
        self.loss_file = np.memmap(
            os.path.join(self.grad_dir, "loss.mmap"),
            dtype="float32",
            mode="w+",
            shape=(len(self.gpus),),
        )

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called before optimizer step.
        """
        self.grad_sync.accumulate_local_grad(self.model)
        self.grad_sync.read_final_grad_into_model(self.model, average=True)

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called after optimizer step.
        """

        self.grad_sync.zero_mmaps()

    def on_log(self, args, state, control, **kwargs):
        if "loss" in state.log_history[-1]:
            gputh = self.gpus.index(self.gpu_index)
            self.loss_file[gputh] = np.float32(state.log_history[-1]["loss"])
            t = time.time()
            if self.is_main:
                # if main gpu, then read all the losses
                # wait for all the losses to be written
                while any(self.loss_file == 0):
                    time.sleep(0.01)
                losses = self.loss_file[:]
                state.log_history[-1]["mean_loss"] = np.mean(losses)
                logger.info(f"Mean loss: {state.log_history[-1]['mean_loss']}")
                # if all losses are not zero, then reset all the losses
                if np.all(losses != 0):
                    self.loss_file[:] = 0
            else:
                # if not main gpu, then wait for the main gpu to reset the losses
                while True:
                    losses = self.loss_file[:]
                    if np.all(losses == 0):
                        break
                    time.sleep(0.01)
            t = time.time() - t
