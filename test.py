import numpy as np
from speedy_utils import multi_process
from multiprocessing import Value, Lock

# Shared value and lock for synchronization
x = Value('d', 0.0)
lock = Lock()

def f(_a: int) -> None:
    """Function to increment the shared value."""
    with lock:
        x.value += 1

multi_process(f, range(10000))
print(x.value)