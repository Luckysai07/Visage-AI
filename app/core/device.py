"""Device management — auto-detect CUDA/CPU for all models."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Returns the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} "
                    f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS backend.")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected — using CPU. Inference will be slower.")
    return device


def get_device_info() -> dict:
    """Return a dict of device information for API health checks."""
    info = {"device": str(get_device())}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory // 1024**2
        info["gpu_memory_free_mb"] = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) // 1024**2
    return info


DEVICE = get_device()
