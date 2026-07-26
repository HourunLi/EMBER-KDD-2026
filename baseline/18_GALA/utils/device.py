"""Execution-device selection shared by all experiment entry points."""

import torch


def resolve_device(device_spec="auto", gpu_id=0):
    """Resolve CPU/CUDA configuration and validate the requested GPU index.

    Supported forms are ``auto``, ``cpu``, ``cuda`` and ``cuda:N``.  With
    ``auto`` or ``cuda``, ``gpu_id`` selects the zero-based CUDA device.
    ``cuda:N`` takes precedence over ``gpu_id``.
    """

    spec = str(device_spec).strip().lower()
    selected_gpu = int(gpu_id)

    if spec == "cpu":
        return torch.device("cpu")

    if spec.startswith("cuda:"):
        suffix = spec.split(":", 1)[1]
        if suffix == "" or not suffix.isdigit():
            raise ValueError(
                f"Invalid CUDA device {device_spec!r}. Use cuda:N, for example cuda:1."
            )
        selected_gpu = int(suffix)
        spec = "cuda"

    if spec not in {"auto", "cuda"}:
        raise ValueError(
            f"Unsupported device {device_spec!r}. Use auto, cpu, cuda, or cuda:N."
        )

    if spec == "auto" and not torch.cuda.is_available():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available. "
            "Use --device cpu or --device auto."
        )

    device_count = torch.cuda.device_count()
    if selected_gpu < 0 or selected_gpu >= device_count:
        raise ValueError(
            f"GPU index {selected_gpu} is invalid: {device_count} CUDA device(s) "
            f"are visible, so valid indices are 0 through {device_count - 1}."
        )

    torch.cuda.set_device(selected_gpu)
    return torch.device(f"cuda:{selected_gpu}")
