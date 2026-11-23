"""Cross-check the self-implemented PGD against torchattacks on the same samples.

Used only as an external reference: the assignment asks that, if the difference
in success rate exceeds 3 percentage points, the report must explain why.

    python -m src.crosscheck --ckpt checkpoints/std_r18.pt --eps 8
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import get_device, save_json, set_seed
from .data import CIFAR10_MEAN, CIFAR10_STD
from .evaluate import load_model
from .run_experiments import EvalSet


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/std_r18.pt")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torchattacks

    set_seed(args.seed)
    dev = get_device()
    ev = EvalSet(args.n_eval, dev)
    model = load_model(args.ckpt, dev)
    eps = args.eps / 255.0

    # torchattacks works on normalised inputs; we feed it normalised tensors and
    # set eps accordingly in *its* space (eps / std is channel dependent), so to
    # make the comparison fair we wrap our PixelModel: torchattacks then perturbs
    # [0,1] images and our wrapper normalises inside -- exactly like our attacks.
    # torchattacks perturbs its input directly (default normalisation is identity)
    # and our PixelModel maps [0,1] -> logits, so the two are directly comparable.
    ref = torchattacks.PGD(model, eps=eps, alpha=2.5 * eps / args.steps,
                           steps=args.steps, random_start=True)

    with torch.no_grad():
        clean_ok = model(ev.x01).argmax(1) == ev.y
    n_ok = int(clean_ok.sum().item())

    # ours
    from . import attacks as A
    adv = A.pgd_linf(model, ev.x01, ev.y, eps=eps, steps=args.steps, random_start=True)
    with torch.no_grad():
        wrong_ours = model(adv).argmax(1) != ev.y
    asr_ours = int((wrong_ours & clean_ok).sum().item()) / max(n_ok, 1)

    # torchattacks (it internally loops over samples; keep it on the same subset)
    adv_ta = ref(ev.x01, ev.y)
    with torch.no_grad():
        wrong_ta = model(adv_ta).argmax(1) != ev.y
    asr_ta = int((wrong_ta & clean_ok).sum().item()) / max(n_ok, 1)

    out = {
        "ckpt": args.ckpt, "eps": args.eps, "steps": args.steps, "n": ev.N,
        "asr_self": asr_ours, "asr_torchattacks": asr_ta,
        "diff_pp": round((asr_ours - asr_ta) * 100, 2),
        "torchattacks_version": torchattacks.__version__,
    }
    save_json("results/crosscheck_pgd.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
