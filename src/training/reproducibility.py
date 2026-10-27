from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import torchvision


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set random seed across Python, NumPy, PyTorch CPU and CUDA.

    Args:
        seed: Random seed integer (e.g. 42, 43, 44, 45, 46).
        deterministic: If True, configures CuDNN and PyTorch deterministic modes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Avoid CUBLAS nondeterminism warnings/errors
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # warn_only=True ensures operations that have no deterministic implementation
            # log a warning rather than crashing if encountered on specific hardware
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader workers.

    Ensures distinct yet reproducible random state for each worker process.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_generator(seed: int) -> torch.Generator:
    """Create and return a deterministic PyTorch Generator for DataLoader samplers."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def collect_environment_info() -> Dict[str, Any]:
    """Collect runtime environment metadata for provenance and reproducibility logs.

    Returns:
        Dictionary containing OS, Python, PyTorch, Torchvision, CUDA, and GPU hardware details.
    """
    env_info = {
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_devices": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            env_info["gpu_devices"].append(
                {
                    "device_index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_mb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**2), 2
                    ),
                }
            )

    return env_info

