"""Device management: pick the best available compute device."""

import torch


def get_device() -> torch.device:
    """Automatically select the best available device.

    Priority:
    1. CUDA GPU
    2. Apple Silicon MPS
    3. CPU

    Returns
    -------
    torch.device
        Selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_device_info() -> None:
    """Print information about the selected device."""
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    if device.type == "mps":
        print("Apple Silicon MPS is available.")
