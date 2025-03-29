import os
import time
import numpy as np
import torch
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from loguru import logger
from filelock import FileLock, Timeout # Import FileLock and Timeout

TIME_OUT = 120
SLEEP_TIME = 0.05
LOCK_TIMEOUT = 30 # Timeout for acquiring the lock in seconds
LOCK_FILE_SUFFIX = ".lock" # Suffix for the lock file

class MmapGradientSync:
    """
    Single-file gradient sync that also stores a 'global_step' in the file,
    so that each GPU only waits on flags belonging to the *current* iteration.

    Uses file locking via `filelock` to prevent race conditions when multiple
    processes write to the memory-mapped file concurrently.

    Refactored to enforce:
      0) All GPUs start at global_step=0 in sync.
      1) Each GPU writes its local gradient => set write=1 (under lock)
      2) Wait for all writes => read/average => set read=1 (under lock)
      3) Optim step => zero grads (under lock by main) => set iteration_done=1 (under lock)
         => main increments global_step (under lock) => others wait
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
        self.is_main = self.local_rank == 0

        os.makedirs(grad_dir, exist_ok=True)
        self.buffer_path = os.path.join(grad_dir, "grad_buffer_v2.dat")
        self.lock_path = self.buffer_path + LOCK_FILE_SUFFIX # Define lock file path
        self.lock_timeout = LOCK_TIMEOUT

        # Initialize the file lock
        # Each process gets its own FileLock object, but they all point to the same lock file
        self.file_lock = FileLock(self.lock_path, timeout=1) # Short timeout for basic check

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
        # Flags layout (total 3 * num_gpus)
        # ------------------------------------------------
        self.flags_offset = current_offset
        self.flags_size = 3 * self.num_gpus
        total_size = self.flags_offset + self.flags_size

        # ------------------------------------------------
        # Global step slot (1 float)
        # ------------------------------------------------
        self.step_offset = total_size
        total_size += 1

        # ------------------------------------------------
        # Create/open memmap. Initialization happens next.
        # ------------------------------------------------
        # No lock needed here yet, as mode='w+' truncates/creates.
        # Subsequent writes MUST be locked.
        self.mem = np.memmap(
            self.buffer_path,
            dtype="float32",
            mode="w+",
            shape=(total_size,),
        )

        # If main GPU, initialize values under lock
        if self.is_main:
            self._init_values() # This method now handles locking

        # Wait briefly for main process to potentially initialize
        # This isn't strictly necessary if using locks correctly everywhere else,
        # but can prevent non-main processes from reading garbage briefly if they start super fast.
        if not self.is_main:
            time.sleep(0.5) # Small delay for safety

        logger.info(
            f"[Init GPU={self.gpu}] Memmap opened: total_size={total_size}, "
            f"local_rank={self.local_rank}, gpus={self.gpus}, is_main={self.is_main}, "
            f"starting global_step={self._get_current_step()}" # Reading step doesn't strictly need lock if writes are locked
        )


    def _acquire_lock(self, operation_name: str):
        """Acquires the file lock with timeout and warning."""
        try:
            # Use the actual lock timeout defined during init
            lock_handle = self.file_lock.acquire(timeout=self.lock_timeout)
            logger.trace(f"[GPU={self.gpu}] Acquired lock for {operation_name}")
            return lock_handle
        except Timeout:
            logger.warning(
                f"[GPU={self.gpu}] Waited > {self.lock_timeout}s to acquire lock for {operation_name}. "
                f"Possible contention or deadlock."
            )
            # Continue trying to acquire, but now potentially block indefinitely
            # Or raise an error: raise TimeoutError(f"Could not acquire lock for {operation_name} within {self.lock_timeout}s")
            # For this use case, let's try acquiring again without timeout after warning.
            return self.file_lock.acquire()


    def _init_values(self):
        """Force global_step=0 and zero flags at the beginning (main GPU only)."""
        assert self.is_main, "Only main GPU initializes values"
        logger.debug(f"[GPU={self.gpu}] Attempting to acquire lock for _init_values")
        with self._acquire_lock("_init_values"):
            logger.debug(f"[GPU={self.gpu}] Lock acquired. Initializing memmap buffer.")
            self.mem[:] = 0.0
            self.mem.flush()
            logger.debug(f"[GPU={self.gpu}] Initialization complete. Releasing lock.")


    def _get_current_step(self) -> int:
        # Reading a single float is likely atomic, but locking writes guarantees
        # we won't read while a write is in progress. Reading without lock is
        # generally okay if performance is critical and atomicity is assumed.
        # For safety during development/debugging, you could lock reads too,
        # but we'll omit it here based on typical mmap behavior for single values.
        return int(self.mem[self.step_offset])

    def _set_current_step(self, step: int):
        # Only the main GPU writes the step
        assert self.is_main, "Only main GPU can set global_step"
        logger.trace(f"[GPU={self.gpu}] Attempting to acquire lock for _set_current_step")
        with self._acquire_lock("_set_current_step"):
            self.mem[self.step_offset] = float(step)
            self.mem.flush()
            logger.trace(f"[GPU={self.gpu}] Set step to {step}. Releasing lock.")


    def _increment_global_step(self):
        """main GPU increments the global_step by +1 (atomic read-modify-write)."""
        assert self.is_main, "Only main GPU increments step"
        logger.debug(f"[GPU={self.gpu}] Attempting to acquire lock for _increment_global_step")
        with self._acquire_lock("_increment_global_step"):
            old = int(self.mem[self.step_offset]) # Read inside lock
            new = old + 1
            self.mem[self.step_offset] = float(new) # Write inside lock
            self.mem.flush()
            logger.debug(f"[GPU={self.gpu}] Incremented global_step from {old} to {new}. Releasing lock.")
        # No need to call _set_current_step as it's done atomically here.


    def _check_step_matches_local(self, local_step: int):
        """Check if file's global step matches local expectation."""
        file_step = self._get_current_step()
        return file_step == local_step


    def _set_flag(self, stage: str):
        """Sets the appropriate flag for the current GPU for the given stage."""
        if stage == "write":
            idx = self.local_rank
        elif stage == "read":
            idx = self.local_rank + self.num_gpus
        elif stage == "iteration_done":
            idx = self.local_rank + 2 * self.num_gpus
        else:
            raise ValueError(f"Unknown stage={stage}")

        flag_index = self.flags_offset + idx
        logger.trace(f"[GPU={self.gpu}] Attempting to acquire lock to set flag '{stage}' at index {flag_index}")
        with self._acquire_lock(f"set_flag_{stage}"):
            self.mem[flag_index] = 1.0
            self.mem.flush()
            logger.debug(f"[GPU={self.gpu}] Set {stage}=1 at index={flag_index}. Releasing lock.")


    def _wait_for_all(self, stage: str, local_step: int, timeout=TIME_OUT):
        """Wait until all GPUs have set stage=1 for the current local_step."""
        start_time = time.time()
        logged_waiting = False # Track if waiting message has been logged for this call

        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed > timeout:
                 raise RuntimeError(
                    f"[GPU={self.gpu}] Timeout ({timeout}s) waiting for {stage} at local_step={local_step}. "
                    f"File step: {self._get_current_step()}"
                 )

            # Check if global step in file matches our local step expectation
            if not self._check_step_matches_local(local_step):
                if not logged_waiting: # Log only once per wait cycle if step mismatches
                     logger.debug(
                        f"[GPU={self.gpu}] Waiting for global step sync. "
                        f"local_step={local_step}, file_step={self._get_current_step()}. "
                        f"Waiting for stage '{stage}'. Elapsed: {elapsed:.1f}s"
                     )
                     logged_waiting = True
                time.sleep(SLEEP_TIME)
                continue # Keep waiting for step sync

            # Step matches, now check the flags
            # Reading flags without lock - okay since writes are locked and we wait for consistency.
            flags_slice = self.mem[
                self.flags_offset : self.flags_offset + self.flags_size
            ].copy() # Copy to avoid race conditions during check

            if stage == "write":
                target_slice = flags_slice[0 : self.num_gpus]
                flag_start_idx = self.flags_offset
            elif stage == "read":
                target_slice = flags_slice[self.num_gpus : 2 * self.num_gpus]
                flag_start_idx = self.flags_offset + self.num_gpus
            elif stage == "iteration_done":
                target_slice = flags_slice[2 * self.num_gpus : 3 * self.num_gpus]
                flag_start_idx = self.flags_offset + 2 * self.num_gpus
            else:
                raise ValueError(f"Unknown stage={stage}")

            if np.all(target_slice == 1.0):
                logger.debug(
                    f"[GPU={self.gpu}] All GPUs done with {stage} at step={local_step}. Wait complete."
                )
                break # Success
            else:
                 # Log detailed status periodically if waiting long
                if not logged_waiting or (elapsed > 5 and int(elapsed) % 5 == 0): # Log every 5s after initial 5s
                    incomplete_ranks = [i for i, flag in enumerate(target_slice) if flag != 1.0]
                    logger.debug(
                        f"[GPU={self.gpu}] Waiting for stage '{stage}' at step={local_step}. "
                        f"File step matches. Ranks not ready: {incomplete_ranks}. "
                        f"Flags: {target_slice.tolist()}. Elapsed: {elapsed:.1f}s"
                    )
                    logged_waiting = True # Ensure we log at least once if waiting

                time.sleep(SLEEP_TIME)


    def _reset_all_flags(self):
        """main GPU zeros out the 3 sets of flags (under lock)."""
        assert self.is_main, "Only main GPU resets flags"
        logger.debug(f"[GPU={self.gpu}] Attempting to acquire lock for _reset_all_flags")
        with self._acquire_lock("_reset_all_flags"):
            logger.debug(f"[GPU={self.gpu}] Lock acquired. Resetting all flags.")
            flags_start = self.flags_offset
            flags_end = self.flags_offset + self.flags_size
            self.mem[flags_start:flags_end] = 0.0
            self.mem.flush()
            logger.debug(f"[GPU={self.gpu}] Flags reset. Releasing lock.")


    # ==================================================
    # Public Methods: Writing Grad, Reading Grad, Zero, Iteration
    # ==================================================
    def accumulate_local_grad(self, model: torch.nn.Module, local_step: int):
        """
        Step 1: Each GPU writes its local gradient into the memmap => set write=1
        (Gradient writing and flag setting happen under lock).
        """
        # Check step match *before* acquiring lock to avoid unnecessary waiting
        if not self._check_step_matches_local(local_step):
            raise RuntimeError(
                f"[GPU={self.gpu}] accumulate_local_grad step mismatch before lock: "
                f"local_step={local_step}, file_step={self._get_current_step()}"
            )

        logger.debug(f"[GPU={self.gpu}] Attempting lock for accumulate_local_grad step={local_step}")
        with self._acquire_lock(f"accumulate_grad_step_{local_step}"):
            logger.debug(f"[GPU={self.gpu}] Lock acquired for accumulate_local_grad step={local_step}")

            # Double-check step match *inside* lock for safety, though less likely to mismatch now
            if not self._check_step_matches_local(local_step):
                 # Release lock before raising
                self.file_lock.release()
                raise RuntimeError(
                    f"[GPU={self.gpu}] accumulate_local_grad step mismatch *inside* lock: "
                    f"local_step={local_step}, file_step={self._get_current_step()}"
                )

            named_params = dict(model.named_parameters())
            wrote_grad = False
            for info in self.param_info:
                name = info["name"]
                offset = info["offset"]
                numel = info["numel"]
                param = named_params[name]

                if param.grad is not None and param.requires_grad: # Ensure grad exists and requires grad
                    sub_offset = offset + self.local_rank * numel
                    try:
                        flat = param.grad.detach().cpu().numpy().ravel()
                        # Ensure shapes match exactly
                        if flat.size != numel:
                             raise ValueError(f"Gradient size mismatch for {name}: expected {numel}, got {flat.size}")
                        self.mem[sub_offset : sub_offset + numel] = flat
                        wrote_grad = True
                    except Exception as e:
                        logger.error(f"[GPU={self.gpu}] Error writing grad for {name}: {e}")
                        # Release lock before raising
                        self.file_lock.release()
                        raise

            self.mem.flush() # Flush all gradient writes
            logger.debug(
                f"[GPU={self.gpu}] Gradient writes done (wrote_any={wrote_grad}), step={local_step}. Releasing lock."
            )
        # End of lock scope for gradient writing

        # Set the write flag *after* gradient writing is complete and lock is released
        # _set_flag handles its own locking internally
        self._set_flag("write")


    def read_final_grad_into_model(
        self, model: torch.nn.Module, local_step: int, average=True
    ):
        """
        Step 2: Wait for all writes => sum (optionally average) => set read=1
        (Reading gradients doesn't need lock, setting read flag does).
        """
        logger.debug(f"[GPU={self.gpu}] Waiting for 'write' flag step={local_step}")
        self._wait_for_all("write", local_step)
        logger.debug(f"[GPU={self.gpu}] All writes done. Reading/Averaging grads step={local_step}")

        named_params = dict(model.named_parameters())
        # Perform read and averaging - no lock needed as writes for this step are complete
        for info in self.param_info:
            name = info["name"]
            offset = info["offset"]
            numel = info["numel"]
            shape = info["shape"]
            param = named_params[name]
            if not param.requires_grad:
                continue

            # Use float64 for accumulation to potentially improve precision, then cast back
            acc = np.zeros(numel, dtype=np.float64)
            try:
                for rank_id in range(self.num_gpus):
                    sub_offset = offset + rank_id * numel
                    # Read segment corresponding to this rank's contribution
                    grad_segment = self.mem[sub_offset : sub_offset + numel].astype(np.float64)
                    acc += grad_segment

                if average:
                    acc /= self.num_gpus

                # Cast back to float32 before converting to Tensor
                final_grad = acc.astype(np.float32).reshape(shape)
                # Ensure param.grad is None or set correctly
                param.grad = torch.from_numpy(final_grad).to(param.device)
            except Exception as e:
                 logger.error(f"[GPU={self.gpu}] Error reading/averaging grad for {name}: {e}")
                 raise

        logger.debug(f"[GPU={self.gpu}] Done reading/averaging grads step={local_step}.")

        # Set the read flag (this acquires/releases lock internally)
        self._set_flag("read")


    def zero_mmaps(self, local_step: int):
        """
        Called right after the optimizer step.
        Wait for all read=1 => main GPU zeroes the gradient region (under lock)
        => All GPUs set iteration_done=1 (under lock).
        """
        logger.debug(f"[GPU={self.gpu}] Waiting for 'read' flag step={local_step}")
        self._wait_for_all("read", local_step)
        logger.debug(f"[GPU={self.gpu}] All reads done step={local_step}.")

        # Only main GPU zeros out param region, under lock
        if self.is_main:
            logger.debug(f"[GPU={self.gpu}] Attempting lock to zero gradient region step={local_step}")
            with self._acquire_lock(f"zero_grads_step_{local_step}"):
                 logger.debug(f"[GPU={self.gpu}] Lock acquired. Zeroing param region [0..{self.flags_offset}) step={local_step}")
                 self.mem[0 : self.flags_offset] = 0.0
                 self.mem.flush()
                 logger.debug(f"[GPU={self.gpu}] Gradient region zeroed. Releasing lock.")
            # Lock released automatically by 'with'

        # All GPUs now set the iteration_done flag (acquires/releases lock internally)
        # This happens *after* main has zeroed (if applicable), ensuring zeroing is done first.
        self._set_flag("iteration_done")
        logger.debug(f"[GPU={self.gpu}] Set 'iteration_done' flag step={local_step}.")


    def iteration_done(self, local_step: int):
        """
        Wait for all iteration_done=1 => main increments global_step & resets flags (under lock).
        This completes Step 3 (post-optim sync).
        """
        # No need to explicitly set iteration_done flag here, it's done in zero_mmaps

        logger.debug(f"[GPU={self.gpu}] Waiting for 'iteration_done' flag step={local_step}")

        if self.is_main:
             # Main waits for all others to set iteration_done=1
            self._wait_for_all("iteration_done", local_step)
            logger.debug(f"[GPU={self.gpu}] All iteration_done received. Main GPU resetting flags and incrementing step.")

            # Reset flags (acquires lock internally)
            self._reset_all_flags()

            # Increment global step (acquires lock internally, performs atomic RMW)
            self._increment_global_step()

            logger.debug(f"[GPU={self.gpu}] Main GPU finished post-iteration sync for step {local_step}.")

        else:
            # Non-main GPUs wait until the main GPU increments the global step
            start_wait_step = time.time()
            while True:
                current_step = self._get_current_step()
                if current_step > local_step:
                    logger.debug(f"[GPU={self.gpu}] Detected global step incremented to {current_step} (was {local_step}). Iteration sync done.")
                    break

                # Add a timeout for waiting on the step increment to prevent infinite loops
                if time.time() - start_wait_step > TIME_OUT:
                     raise RuntimeError(
                        f"[GPU={self.gpu}] Timeout waiting for main GPU to increment global step "
                        f"from {local_step}. Current file step: {current_step}."
                     )

                # Log if waiting long
                if time.time() - start_wait_step > 5 and int(time.time() - start_wait_step) % 5 == 0 :
                     logger.debug(f"[GPU={self.gpu}] Waiting for main GPU to increment step from {local_step}...")

                time.sleep(SLEEP_TIME)

        logger.debug(f"[GPU={self.gpu}] iteration_done complete for step {local_step}")


class MmapGradSyncCallback(TrainerCallback):
    """
    A TrainerCallback that uses 'MmapGradientSync' with a global step approach.
    The locking mechanism is handled entirely within MmapGradientSync.
    """

    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu = gpu
        self.local_rank = gpus.index(gpu)
        self.gpus = gpus

        # Instantiate the gradient syncer with locking enabled
        self.grad_sync = MmapGradientSync(model, gpu, gpus, grad_dir)

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        HF calls this right before the optimizer step.
         - Step 1: Accumulate/write local gradient (locked write) => set 'write=1' (locked write)
         - Step 2: Wait for all writes, read & average (read is unlocked) => set 'read=1' (locked write)
        """
        logger.debug("=" * 80)
        local_step = state.global_step
        logger.info(
            f"[GPU={self.gpu} RANK={self.local_rank} STEP={local_step}] === PRE OPTIMIZER STEP ==="
        )

        logger.debug(
            f"[GPU={self.gpu} STEP={local_step}] Calling accumulate_local_grad"
        )
        self.grad_sync.accumulate_local_grad(self.model, local_step)

        logger.debug(
             f"[GPU={self.gpu} STEP={local_step}] Calling read_final_grad_into_model"
        )
        self.grad_sync.read_final_grad_into_model(self.model, local_step, average=True)
        logger.info(
            f"[GPU={self.gpu} STEP={local_step}] === PRE OPTIMIZER STEP Complete ==="
        )


    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        HF calls this right after the optimizer step is done.
         - Wait for reads => Main zeroes memmaps (locked write) => All set 'iteration_done=1' (locked write)
         - Wait for iteration_done => Main increments global_step & resets flags (locked writes) => Others wait for step change
        """
        local_step = state.global_step
        logger.info(
            f"[GPU={self.gpu} RANK={self.local_rank} STEP={local_step}] === POST OPTIMIZER STEP ==="
        )

        logger.debug(
            f"[GPU={self.gpu} STEP={local_step}] Calling zero_mmaps"
        )
        # zero_mmaps includes waiting for reads, zeroing (main only, locked), and setting iteration_done (all, locked)
        self.grad_sync.zero_mmaps(local_step)

        logger.debug(
            f"[GPU={self.gpu} STEP={local_step}] Calling iteration_done (barrier)"
        )
        # iteration_done includes waiting for iteration_done flags, then main resets flags/increments step (locked), others wait for step change
        self.grad_sync.iteration_done(local_step)

        logger.info(
            f"[GPU={self.gpu} STEP={local_step}] === POST OPTIMIZER STEP Complete === (Next step: {local_step + 1})"
        )
        logger.debug("=" * 80)