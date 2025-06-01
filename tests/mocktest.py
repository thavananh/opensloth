import os
import time
import random
import torch
import multiprocessing
from loguru import logger

# Import your MmapGradientSync, or paste its code here

from HyperSloth.mmap_gradient_sync import MmapGradientSync
R = random.Random(42)  # For reproducibility
# ------------------------------------------------------------------------
# Paste your existing MmapGradientSync and MmapGradSyncCallback code here,
# or do "from my_sync_module import MmapGradientSync" if you have it in a separate file.
#
# For brevity, let's assume something like:
#
# from my_sync_module import MmapGradientSync
#
# We'll just pretend MmapGradientSync is available.
# ------------------------------------------------------------------------


class DummyModel(torch.nn.Module):
    """
    A tiny dummy model for testing. We'll just store one linear layer.
    """
    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def mock_gpu_process(local_rank, gpus, grad_dir, steps=5):
    """
    Each GPU process:
      1. Creates a DummyModel.
      2. Creates MmapGradientSync with that model.
      3. Repeats a few "training steps", each with random sleeps to stress concurrency.
    """
    # Fake "GPU id" from the gpus list
    gpu_id = gpus[local_rank]

    # Log messages with GPU ID
    logger.info(f"[GPU={gpu_id}] Process started")

    # Create model on CPU for simplicity here
    model = DummyModel()
    model.train()

    # Create the sync object
    sync = MmapGradientSync(model, gpu_id, gpus, grad_dir=grad_dir)

    # We'll do a simple "training loop"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(steps):

        # Simulate forward + backward
        inputs = torch.randn(2, 4)  # shape (batch_size=2, in_features=4)
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()

        # Sleep randomly to force concurrency
        time.sleep(R.uniform(00.1, 1))

        # Step 1 & 2: accumulate + read final grad
        sync.accumulate_local_grad(model, step)
        sync.read_final_grad_into_model(model, step, average=True)

        # Actual optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Sleep randomly
        time.sleep(R.uniform(0.1, 1))

        # zero_mmaps
        sync.zero_mmaps(step)

        # Sleep randomly
        time.sleep(R.uniform(0.1, 1))

        # iteration_done
        sync.iteration_done(step)

        logger.info(f"[GPU={gpu_id}] Finished step {step}")

        # Sleep again
        time.sleep(R.uniform(0.1, 1))


    logger.info(f"[GPU={gpu_id}] Process done. Exiting.")


def main_test(n_gpus=2, grad_dir="/tmp/mmap_sync_test", steps=5):
    """
    Spawns n_gpus processes, each simulating a "GPU" using our sync logic.
    """
    # E.g., gpus = [0,1,2,3] for 4 "GPUs"
    gpus = list(range(n_gpus))

    # Clean up any leftover memmap file if you want a fresh start
    buffer_file = os.path.join(grad_dir, "grad_buffer_v2.dat")
    lock_file = buffer_file + ".lock"
    for f in [buffer_file, lock_file]:
        if os.path.exists(f):
            os.remove(f)

    # Launch processes
    processes = []
    for local_rank in range(n_gpus):
        p = multiprocessing.Process(
            target=mock_gpu_process,
            args=(local_rank, gpus, grad_dir, steps),
        )
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()

    logger.info("All GPU processes joined. Test complete.")


if __name__ == "__main__":
    logger.info("Starting mock test with multiple 'GPU' processes...")
    main_test(n_gpus=8, grad_dir="/tmp/mmap_sync_test", steps=1000)
    logger.info("Mock test finished successfully.")
