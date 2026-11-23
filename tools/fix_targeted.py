"""Recompute the targeted-attack row of a ``results/defense_*.csv`` file.

The targeted PGD row is cheap and was affected by a sign bug in an earlier
version of ``margin_loss`` (fixed in src/attacks.py).  This script recomputes it
in place so the CSV stays consistent with the code.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.common import get_device, set_seed
from src.evaluate import load_model
from src.run_experiments import EvalSet, _targeted_attack, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", nargs="+", required=True)
    p.add_argument("--eps", type=float, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=1000)
    args = p.parse_args()
    dev = get_device()
    set_seed(0)
    ev = EvalSet(args.n_eval, dev)

    for tag in args.tag:
        ckpt = f"checkpoints/{tag}.pt"
        model = load_model(ckpt, dev)
        with torch.no_grad():
            clean_pred = model(ev.x01).argmax(1)
            clean = float((clean_pred == ev.y).float().mean().item())
        ok = clean_pred == ev.y
        adv_parts = []
        for x, y in ev.batches():
            adv_parts.append(_targeted_attack(model, x, y, args.eps / 255.0, steps=args.steps).detach())
        adv = torch.cat(adv_parts)
        with torch.no_grad():
            wrong = model(adv).argmax(1) != ev.y
        new_row = {
            "ckpt": tag, "attack": "pgd20_targeted", "eps": args.eps, "n": ev.N,
            "clean_acc": clean, "robust_acc": float(1 - wrong.float().mean().item()),
            "asr": float((wrong & ok).sum().item() / max(ok.sum().item(), 1)),
            "uncond_err": float(wrong.float().mean().item()),
            "desc": f"目标化 PGD-{args.steps}（最不可能类）",
        }
        path = f"results/defense_{tag}.csv"
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        replaced = False
        for i, r in enumerate(rows):
            if r.get("attack") == "pgd20_targeted":
                rows[i] = {**r, **new_row}
                replaced = True
        if not replaced:
            rows.append(new_row)
        write_csv(path, rows)
        print(f"[{tag}] targeted PGD-{args.steps}: RA={new_row['robust_acc']*100:.2f}% "
              f"ASR={new_row['asr']*100:.2f}%")


if __name__ == "__main__":
    main()
