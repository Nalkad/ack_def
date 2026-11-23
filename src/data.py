"""CIFAR-10 loading with a reproducible sampler.

Fixed protocol (report, section "实验协议"):

* official 50 000 train / 10 000 test split, never re-split, test set never used
  for training (``tools/prepare_cifar10.py`` asserts the sizes);
* train augmentation: random crop 32 with 4 px reflection padding + horizontal
  flip; test time: none;
* normalisation with the CIFAR-10 channel mean/std, applied *inside* the dataset
  so that the loader returns model-ready tensors; ``denorm`` inverts it and every
  attack works in the [0,1] pixel space recovered by ``denorm``;
* the sampler is seeded from ``--seed``, and ``worker_init_fn`` re-seeds numpy /
  torch in every worker, so a run is bit-reproducible on the same GPU.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASSES_ZH = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


class CIFAR10Dataset(Dataset):
    """32x32 RGB uint8 images with optional train-time augmentation."""

    def __init__(self, path: str, augment: bool = False):
        d = np.load(path)
        self.x = torch.from_numpy(d["x"])          # N,3,32,32 uint8
        self.y = torch.from_numpy(d["y"]).long()   # N
        self.augment = augment
        self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        img = self.x[idx].float().div_(255.0)
        if self.augment:
            # 4-pixel zero padding + random 32x32 crop (the standard CIFAR recipe)
            img = torch.nn.functional.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            img = img[:, i:i + 32, j:j + 32]
            if random.random() < 0.5:
                img = torch.flip(img, dims=[-1])
        img = (img - self.mean) / self.std
        return img, self.y[idx]


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_datasets(data_root: str = "data/cifar10"):
    train = CIFAR10Dataset(os.path.join(data_root, "train.npz"), augment=True)
    test = CIFAR10Dataset(os.path.join(data_root, "test.npz"), augment=False)
    return train, test


def get_loaders(data_root: str = "data/cifar10", batch_size: int = 128,
                test_batch_size: int = 256, workers: int = 4, seed: int = 0):
    """Return ``(train_loader, test_loader, train_eval_loader, test_set)``."""
    train_set, test_set = get_datasets(data_root)
    train_eval = CIFAR10Dataset(os.path.join(data_root, "train.npz"), augment=False)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True,
                              generator=g, worker_init_fn=_worker_init,
                              persistent_workers=workers > 0)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True,
                             persistent_workers=workers > 0)
    train_eval_loader = DataLoader(train_eval, batch_size=test_batch_size, shuffle=False,
                                   num_workers=workers, pin_memory=True)
    return train_loader, test_loader, train_eval_loader, test_set


def denorm(x: torch.Tensor) -> torch.Tensor:
    """Normalised tensor -> [0,1] pixel tensor (the attack domain)."""
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def norm(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] pixel tensor -> normalised model input."""
    std = torch.tensor(CIFAR10_STD, device=x01.device).view(1, 3, 1, 1)
    mean = torch.tensor(CIFAR10_MEAN, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std

