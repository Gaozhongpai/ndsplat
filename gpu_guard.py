"""Fail fast if the container did not get a usable GPU.

`docker run --gpus device=N` remaps the chosen device to CUDA index 0 inside the
container, so a mis-quoted --gpus silently lands every job on the SAME physical
GPU -- which shows up only as a confusing OOM much later.

The free-memory bar is a floor, not the full working set: a 40-preset dataset
needs ~33 GiB of train frames, while a single-preset specialist dataset needs
~1/40 of that. Set FACTORSPLAT_MIN_FREE_GIB per job type; the default is
deliberately low so co-tenancy during another job's render/metrics phase is
allowed, and a genuinely full card still fails immediately.
"""
import os
import sys

import torch

need = float(os.environ.get("FACTORSPLAT_MIN_FREE_GIB", "12"))
n = torch.cuda.device_count()
free_gb = torch.cuda.mem_get_info(0)[0] / 2**30
name = torch.cuda.get_device_name(0)
print(f"[gpu_guard] {n} visible GPU(s); dev0={name}; free={free_gb:.1f} GiB "
      f"(need >= {need:.0f})", flush=True)
if n != 1:
    sys.exit(f"[gpu_guard] expected exactly 1 visible GPU, got {n}")
if free_gb < need:
    sys.exit(f"[gpu_guard] only {free_gb:.1f} GiB free, need {need:.0f} -- "
             "another job is using this GPU")
