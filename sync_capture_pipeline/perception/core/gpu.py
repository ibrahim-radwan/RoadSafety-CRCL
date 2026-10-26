"""
gpu.py - shared GPU state for the threads of one process.

GPU_LOCK: CUDA graph capture in this torch (2.1, global capture mode) fails if another thread
issues CUDA work while a graph is being captured. The radar detector captures under this lock
(once per configuration) and the other GPU users that can run in a parallel thread (the LiDAR
detector, in the replay) hold it while they work. Uncontended, it costs microseconds.
"""
from __future__ import annotations

import threading

GPU_LOCK = threading.RLock()
