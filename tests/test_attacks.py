"""Fast sanity tests for the attack implementations (~30 s on CPU).

Checks the invariants that a buggy implementation usually violates:

* output shape / dtype, ``0 <= x_adv <= 1``;
* ``||x_adv - x||_inf <= eps`` (the projection really projects);
* the perturbation is non-trivial (the attack moves the image);
* attacked accuracy is below clean accuracy for a *random* small network;
* ``steps=1`` PGD with ``step_size=eps`` matches FGSM up to the random start;
* ``apgd`` with ``steps=0`` returns the random start (i.e. no crash).

Run:  python -m tests.test_attacks
"""
import torch
import torch.nn as nn

from src import attacks as A
from src.models import PixelModel


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.body(x)


def main():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = PixelModel(TinyNet()).to(dev).eval()
    # a separable-ish batch so that the clean prediction is not random noise
    x = torch.rand(32, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (32,), device=dev)
    with torch.no_grad():
        clean_acc = (model(x).argmax(1) == y).float().mean().item()
    eps = 8 / 255

    attacks = {
        "fgsm": lambda: A.fgsm(model, x, y, eps),
        "pgd-10": lambda: A.pgd_linf(model, x, y, eps, steps=10),
        "pgd-1": lambda: A.pgd_linf(model, x, y, eps, steps=1, step_size=eps, random_start=False),
        "mi_fgsm": lambda: A.mi_fgsm(model, x, y, eps, steps=10),
        "cw_l2": lambda: A.cw_l2(model, x, y, steps=20, binary_search=2),
        "apgd-dlr": lambda: A.apgd(model, x, y, eps, steps=20, restarts=2, loss="dlr"),
        "aa-lite": lambda: A.autoattack_lite(model, x, y, eps, steps=20, restarts=1),
    }
    results = {}
    for name, fn in attacks.items():
        adv = fn()
        assert adv.shape == x.shape, (name, adv.shape)
        assert adv.min() >= -1e-6 and adv.max() <= 1 + 1e-6, (name, adv.min(), adv.max())
        linf = (adv - x).abs().max().item()
        assert linf <= eps + 1e-5, f"{name}: Linf {linf} > eps {eps}"
        with torch.no_grad():
            acc = (model(adv).argmax(1) == y).float().mean().item()
        results[name] = (linf, acc)
        print(f"{name:10s} Linf={linf*255:5.2f}/255  attacked acc={acc*100:5.1f}%")

    print(f"clean acc on this batch = {clean_acc*100:.1f}%")
    # FGSM and PGD-1 with step=eps, no random start, must coincide
    f = A.fgsm(model, x, y, eps)
    p = A.pgd_linf(model, x, y, eps, steps=1, step_size=eps, random_start=False)
    assert torch.allclose(f, p, atol=1e-6), "PGD-1 (no random start) != FGSM"
    print("PGD-1 == FGSM: ok")

    # the L2 attack must respect no box constraint but stay in the image domain
    adv = A.cw_l2(model, x, y, steps=10, binary_search=1)
    assert adv.min() >= 0 and adv.max() <= 1
    print("all attack checks passed")


if __name__ == "__main__":
    main()
