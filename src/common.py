"""Shared utilities: seeding, device, logging, checkpoints, metrics."""
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Fix every RNG that matters (python / numpy / torch / cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_report() -> Dict[str, object]:
    """Environment fingerprint written into every result file."""
    import platform
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:  # pragma: no cover
        tv = "n/a"
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": tv,
        "cuda_available": bool(torch.cuda.is_available() and torch.cuda.device_count() > 0),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        prop = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": prop.name,
                "gpu_total_mem_gb": round(prop.total_memory / 1024**3, 2),
                "gpu_capability": f"{prop.major}.{prop.minor}",
            }
        )
    return info


# --------------------------------------------------------------------------- #
# logging / io
# --------------------------------------------------------------------------- #
class CSVLogger:
    """Append-only CSV logger so an interrupted run keeps its history."""

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(fieldnames) + "\n")

    def log(self, row: Dict[str, object]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(str(row.get(k, "")) for k in self.fieldnames) + "\n")


def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, model: nn.Module, meta: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, model: nn.Module, strict: bool = True) -> Dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=strict)
    return ckpt.get("meta", {})


# --------------------------------------------------------------------------- #
# training helpers
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def cosine_lr(optimizer, epoch: int, epochs: int, base_lr: float, warmup: int = 0) -> None:
    """Cosine decay with optional linear warm-up (applied per epoch)."""
    if warmup > 0 and epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, epochs - warmup)
        lr = 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr


def gpu_peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024**3, 3)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
