"""
Hardware Device Management
=============================
Detects GPU availability, selects the best compute device, and provides
utilities for moving data/models between devices.

GPU is only relevant for stages 3 (prepare) and 4 (train).
Stages 1 (acquire) and 2 (process) are I/O-bound and always use CPU.

Usage:
    from common.device import get_device, print_device_info

    device = get_device()              # Auto-detect best device
    device = get_device(prefer_gpu=False)  # Force CPU
    print_device_info(device)

CLI integration (add to argparse in training scripts):
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu-index", type=int, default=0)
"""

import logging
from typing import Optional

__all__ = [
    "get_device",
    "print_device_info",
    "move_batch_to_device",
    "auto_batch_size",
    "supports_mixed_precision",
    "setup_deterministic",
    "add_device_args",
    "device_from_args",
]

logger = logging.getLogger(__name__)



#  LAZY TORCH IMPORT

#
# Torch is heavy (~2s import). We import lazily so that scripts
# which don't need GPU (stages 1-2) never pay this cost.

_torch = None


def _get_torch():
    """Lazily import torch, raising a clear error if unavailable."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            print("\n  [ERROR] PyTorch is required for GPU/training operations.")
            print("          Run: python setup.py --install")
            print("          Or:  https://pytorch.org/get-started/locally/")
            raise
    return _torch


def _get_gpu_mem_bytes(props) -> int:
    """Get total GPU memory in bytes from device properties.

    Handles both old PyTorch (total_mem) and new PyTorch 2.10+ (total_memory).
    """
    return getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)



#  DEVICE SELECTION


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> "torch.device":
    """Select the best available compute device.

    Priority order (when prefer_gpu=True):
      1. CUDA (NVIDIA GPU)
      2. MPS  (Apple Silicon)
      3. CPU  (always available)

    Args:
        prefer_gpu: Set False to force CPU even when GPU is available.
        gpu_index:  Which CUDA GPU to use if multiple are present.

    Returns:
        torch.device
    """
    torch = _get_torch()

    if prefer_gpu and torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            logger.warning(
                f"Requested GPU {gpu_index} but only "
                f"{torch.cuda.device_count()} available. Using GPU 0."
            )
            gpu_index = 0

        # Verify the GPU can actually run kernels - PyTorch may report
        # CUDA as available but fail at runtime if the GPU's compute
        # capability is too old for the installed CUDA toolkit
        try:
            test_tensor = torch.zeros(1, device=f"cuda:{gpu_index}")
            del test_tensor
            device = torch.device(f"cuda:{gpu_index}")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(gpu_index)}")
            return device
        except (RuntimeError, Exception) as e:
            gpu_name = torch.cuda.get_device_name(gpu_index)
            cc = torch.cuda.get_device_capability(gpu_index)
            logger.warning(
                f"GPU {gpu_name} (SM {cc[0]}.{cc[1]}) is incompatible with "
                f"this PyTorch build - falling back to CPU"
            )
            print(f"\n  [WARN] {gpu_name} (compute {cc[0]}.{cc[1]}) cannot run "
                  f"CUDA kernels with this PyTorch version.")
            print(f"         Training will use CPU instead. This is slower but works fine.")

    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
        return device

    if prefer_gpu:
        logger.info("No GPU detected - falling back to CPU")
    else:
        logger.info("GPU disabled by user - using CPU")

    return torch.device("cpu")



#  DEVICE INFO


def print_device_info(device: "torch.device") -> None:
    """Print diagnostic information about the selected device."""
    torch = _get_torch()

    print(f"\n  Compute device:  {device}")

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        mem_total = _get_gpu_mem_bytes(props) / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        mem_free = mem_total - mem_reserved

        print(f"  GPU name:        {props.name}")
        print(f"  GPU memory:      {mem_total:.1f} GB total, {mem_free:.1f} GB free")
        print(f"  Compute cap.:    {props.major}.{props.minor}")
        print(f"  CUDA version:    {torch.version.cuda}")

        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            try:
                print(f"  cuDNN version:   {torch.backends.cudnn.version()}")
            except RuntimeError:
                print(f"  cuDNN version:   incompatible with this GPU")

        if torch.cuda.device_count() > 1:
            print(f"  GPUs available:  {torch.cuda.device_count()}")

    elif device.type == "mps":
        print(f"  Backend:         Apple Metal Performance Shaders")

    else:
        print(f"  Backend:         CPU ({torch.get_num_threads()} threads)")

    print()



#  DATA MOVEMENT


def move_batch_to_device(batch: dict, device: "torch.device") -> dict:
    """Move a dict of tensors to the target device.

    Handles mixed-type batches where some values may not be tensors
    (e.g., string labels, metadata). Non-tensor values are passed through.

    Args:
        batch:  Dict mapping keys to tensors (or other values).
        device: Target device.

    Returns:
        New dict with tensors moved to device, non-tensors unchanged.
    """
    torch = _get_torch()
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved



#  BATCH SIZE ESTIMATION


def auto_batch_size(
    device: "torch.device",
    model_size_mb: float = 100.0,
    sample_size_mb: float = 0.1,
    memory_fraction: float = 0.7,
    default_cpu: int = 64,
    max_batch: int = 512,
) -> int:
    """Estimate a reasonable batch size based on available memory.

    Args:
        device:          Compute device.
        model_size_mb:   Approximate model size in MB.
        sample_size_mb:  Approximate size of one training sample in MB.
        memory_fraction: Fraction of free GPU memory to use (leave headroom).
        default_cpu:     Default batch size for CPU.
        max_batch:       Maximum batch size cap.

    Returns:
        Suggested batch size (minimum 1, capped at max_batch).
    """
    torch = _get_torch()

    if device.type == "cuda":
        idx = device.index or 0
        mem_total = _get_gpu_mem_bytes(torch.cuda.get_device_properties(idx)) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
        mem_free = (mem_total - mem_reserved) * memory_fraction
        available = mem_free - model_size_mb

        if available <= 0:
            logger.warning("Very limited GPU memory - using batch size 1")
            return 1

        batch_size = int(available / sample_size_mb)
        result = max(1, min(batch_size, max_batch))
        logger.info(f"Auto batch size: {result} (based on {mem_free:.0f} MB free GPU memory)")
        return result

    elif device.type == "mps":
        # MPS doesn't expose memory info easily; use a conservative default
        return min(128, max_batch)

    else:
        return min(default_cpu, max_batch)



#  MIXED PRECISION


def supports_mixed_precision(device: "torch.device") -> bool:
    """Check if the device supports efficient mixed-precision training.

    Mixed precision (float16/bfloat16) gives ~2x speedup on:
      - NVIDIA Ampere+ GPUs (compute capability >= 7.0)
      - Apple Silicon MPS

    Falls back to float32 on older hardware or CPU.
    """
    torch = _get_torch()

    if device.type == "cuda":
        idx = device.index or 0
        cap = torch.cuda.get_device_properties(idx)
        # Volta (7.0) and above support tensor cores
        supported = cap.major >= 7
        if supported:
            logger.info(f"Mixed precision supported (compute capability {cap.major}.{cap.minor})")
        else:
            logger.info(f"Mixed precision not efficient on this GPU "
                        f"(compute capability {cap.major}.{cap.minor})")
        return supported

    elif device.type == "mps":
        # MPS supports float16 operations
        return True

    return False



#  DETERMINISTIC MODE


def setup_deterministic(seed: int = 42) -> None:
    """Set all random seeds for reproducible results.

    Configures:
      - Python's random module
      - NumPy random generator
      - PyTorch CPU and all CUDA generators
      - cuDNN deterministic mode

    Note: Deterministic mode is slightly slower on GPU due to
    disabling cuDNN auto-tuner and enforcing deterministic algorithms.

    Args:
        seed: Random seed value.
    """
    import random
    torch = _get_torch()

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (slower but reproducible)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except RuntimeError:
        pass  # cuDNN incompatible with GPU - not needed for CPU training

    # PyTorch 1.8+: warn on non-deterministic operations
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch without warn_only parameter
            torch.use_deterministic_algorithms(True)

    logger.info(f"Deterministic mode enabled (seed={seed})")



#  ARGPARSE INTEGRATION


def add_device_args(parser) -> None:
    """Add --device, --gpu-index, and --seed arguments to an argparse parser.

    Usage:
        parser = argparse.ArgumentParser()
        add_device_args(parser)
        args = parser.parse_args()
        device = device_from_args(args)
    """
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto-detect best available)"
    )
    parser.add_argument(
        "--gpu-index", type=int, default=0,
        help="Which GPU to use if multiple are available (default: 0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (enables deterministic mode)"
    )


def device_from_args(args) -> "torch.device":
    """Create a torch.device from parsed argparse args.

    Also sets up deterministic mode if --seed is provided.

    Args:
        args: Parsed argparse namespace with device, gpu_index, and seed fields.

    Returns:
        torch.device
    """
    if args.seed is not None:
        setup_deterministic(args.seed)

    if args.device == "auto":
        return get_device(prefer_gpu=True, gpu_index=args.gpu_index)
    elif args.device == "cpu":
        return get_device(prefer_gpu=False)
    elif args.device == "cuda":
        torch = _get_torch()
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available - falling back to CPU")
            return get_device(prefer_gpu=False)
        return get_device(prefer_gpu=True, gpu_index=args.gpu_index)
    elif args.device == "mps":
        torch = _get_torch()
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available - falling back to CPU")
            return get_device(prefer_gpu=False)
        return torch.device("mps")

    return get_device(prefer_gpu=True, gpu_index=args.gpu_index)
