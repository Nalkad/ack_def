"""Run one extra attack for a model and upsert the row in results/defense_<tag>.csv.

Used to fill the APGD-40 column of the step-ablation table for the models that
were evaluated with ``--minimal``.
"""
import argparse

import pandas as pd
import torch

from src.common import get_device, set_seed
from src.evaluate import load_model
from src.run_experiments import EvalSet, attack_spec, run_one_attack, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", nargs="+", required=True)
    p.add_argument("--attack", default="apgd")
    p.add_argument("--key", default="apgd40_dlr")
    p.add_argument("--desc", default="APGD-40 DLR")
    p.add_argument("--eps", type=float, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--loss", default="dlr")
    p.add_argument("--n-eval", type=int, default=1000)
    args = p.parse_args()

    dev = get_device()
    set_seed(0)
    ev = EvalSet(args.n_eval, dev)
    for tag in args.tag:
        ckpt = f"checkpoints/{tag}.pt"
        if not __import__("os").path.exists(ckpt):
            print("skip", ckpt)
            continue
        model = load_model(ckpt, dev)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        m, *_ = run_one_attack(
            model, ev,
            attack_spec(args.attack, args.eps / 255.0, 0, steps=args.steps,
                        restarts=args.restarts, loss=args.loss),
            args.eps / 255.0)
        row = {"ckpt": tag, "attack": args.key, "eps": args.eps, "desc": args.desc,
               **m}
        path = f"results/defense_{tag}.csv"
        rows = pd.read_csv(path).to_dict("records")
        replaced = False
        for i, r in enumerate(rows):
            if r.get("attack") == args.key:
                rows[i] = {**r, **row}
                replaced = True
        if not replaced:
            rows.append(row)
        write_csv(path, rows)
        print(f"[{tag}] {args.key}: RA={m['robust_acc']*100:.2f}% ASR={m['asr']*100:.2f}%")


if __name__ == "__main__":
    main()
