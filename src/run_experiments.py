"""Orchestrator: runs every experiment reported in the paper-style report.

Because the machine used for this assignment is a single 2 GB GT 1030, the
evaluation set is a **fixed stratified subset of the CIFAR-10 test set**
(``--n-eval 1000``, 100 images per class, indices from a fixed seed and stored in
``data/cifar10/eval_subset.npy``).  Every attack, every model and every table in
the report uses exactly this subset, so all comparisons are apples-to-apples.
The subset is never used for training or model selection.

Commands
--------
    python -m src.run_experiments attack            # 作业一: ε grid, cost, transfer
    python -m src.run_experiments defense --ckpt ... --tag ...
    python -m src.run_experiments all
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import attacks as A
from .common import device_report, get_device, gpu_peak_mem_gb, save_json, set_seed
from .data import CIFAR10_MEAN, CIFAR10_STD, CLASSES, denorm
from .evaluate import load_model
from .models import PixelModel, build_model

EVAL_NPY = "data/cifar10/eval_subset.npy"
EPS_LIST = [1, 2, 4, 8]


# --------------------------------------------------------------------------- #
# fixed evaluation subset
# --------------------------------------------------------------------------- #
def build_eval_subset(n=1000, seed=42):
    """Stratified test subset: n/10 images per class, indices fixed by ``seed``."""
    path = EVAL_NPY if n == 1000 else f"data/cifar10/eval_subset_{n}.npy"
    if os.path.exists(path):
        return np.load(path), path
    d = np.load("data/cifar10/test.npz")
    y = d["y"]
    rng = np.random.RandomState(seed)
    idx = []
    per = n // 10
    for c in range(10):
        pool = np.where(y == c)[0]
        idx.append(rng.choice(pool, size=per, replace=False))
    idx = np.sort(np.concatenate(idx))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, idx)
    return idx, path


class EvalSet:
    """Tensor bundle for the fixed evaluation subset (kept on the GPU)."""

    def __init__(self, n=1000, dev=None):
        idx, self.path = build_eval_subset(n)
        d = np.load("data/cifar10/test.npz")
        self.idx = idx
        x = torch.from_numpy(d["x"][idx]).float().div_(255.0)
        self.y_cpu = torch.from_numpy(d["y"][idx]).long()
        self.dev = dev or get_device()
        # normalised copy for the model, [0,1] copy for the attacks
        self.x01 = x.to(self.dev)
        mean = torch.tensor(CIFAR10_MEAN, device=self.dev).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, device=self.dev).view(1, 3, 1, 1)
        self.x_norm = (self.x01 - mean) / std
        self.y = self.y_cpu.to(self.dev)
        self.N = len(idx)

    def batches(self, bs=128):
        for i in range(0, self.N, bs):
            yield self.x01[i:i + bs], self.y[i:i + bs]


# --------------------------------------------------------------------------- #
# attack helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def clean_pred(model, ev):
    return model(ev.x01).argmax(1)


def run_one_attack(model, ev, attack, eps_px, seed=0, **kw):
    """Attack the whole evaluation subset, return metrics + per-sample vectors."""
    model.eval()
    adv_parts = []
    t0 = time.perf_counter()
    for x, y in ev.batches():
        adv_parts.append(attack(model, x, y).detach())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    adv = torch.cat(adv_parts)
    with torch.no_grad():
        logits_adv = model(adv)
    pred_adv = logits_adv.argmax(1)
    with torch.no_grad():
        logits_clean = model(ev.x01)
    pred_clean = logits_clean.argmax(1)
    ok = pred_clean == ev.y
    wrong = pred_adv != ev.y
    l2 = (adv - ev.x01).flatten(1).norm(p=2, dim=1)
    linf = (adv - ev.x01).flatten(1).abs().max(dim=1).values
    succ = wrong & ok
    metrics = {
        "n": ev.N,
        "clean_acc": float(ok.float().mean().item()),
        "asr": float(succ.sum().item() / max(ok.sum().item(), 1)),
        "uncond_err": float(wrong.float().mean().item()),
        "robust_acc": float(1.0 - wrong.float().mean().item()),
        "mean_l2_succ": float(l2[succ].mean().item()) if succ.any() else 0.0,
        "mean_linf_succ": float(linf[succ].mean().item()) if succ.any() else 0.0,
        "mean_l2_all": float(l2.mean().item()),
        "mean_linf_all": float(linf.mean().item()),
        "time_ms_per_img": float(dt / ev.N * 1000.0),
        "peak_mem_gb": gpu_peak_mem_gb(),
    }
    return metrics, adv, succ, l2, linf


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def attack_spec(name, eps_px, seed=0, **kw):
    """Build a callable + a human-readable spec string."""
    return A.build_attack(name, eps=eps_px, seed=seed, **kw)


GRID_ATTACKS = [
    ("fgsm", {}, "1 步"),
    ("pgd", {"steps": 20}, "20 步，随机起点"),
    ("mi_fgsm", {"steps": 10}, "10 步，动量 μ=1.0"),
    ("apgd", {"steps": 40, "restarts": 1, "loss": "dlr"}, "40 步，DLR，1 次重启"),
]


# --------------------------------------------------------------------------- #
# 作业一：攻击实验
# --------------------------------------------------------------------------- #
def exp_attack(args):
    dev = get_device()
    set_seed(args.seed)
    ev = EvalSet(args.n_eval, dev)
    print(f"[eval] subset {ev.N} images -> {ev.path}")
    rows, step_rows, cw_rows = [], [], []

    for tag, ckpt in [("std_r18", "checkpoints/std_r18.pt")]:
        if not os.path.exists(ckpt):
            print("skip", ckpt, "(not trained yet)")
            continue
        model = load_model(ckpt, dev)
        for eps in EPS_LIST:
            eps_px = eps / 255.0
            for name, kw, desc in GRID_ATTACKS:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                m, adv, succ, l2, linf = run_one_attack(
                    model, ev, attack_spec(name, eps_px, args.seed, **kw), eps_px)
                m.update({"model": tag, "attack": name, "eps": eps, "desc": desc,
                          "steps": kw.get("steps", 1), "restarts": kw.get("restarts", 1),
                          "sub": f"{ev.N} 张"})
                rows.append(m)
                print(f"  [{tag}] eps={eps} {name:9s} ASR={m['asr']*100:6.2f}% "
                      f"Linf={m['mean_linf_succ']*255:5.2f}/255 "
                      f"{m['time_ms_per_img']:7.1f} ms/img", flush=True)
        # step-vs-strength at eps = 8/255 (single to multi step)
        for steps in (1, 2, 5, 10, 20):
            name = "fgsm" if steps == 1 else "pgd"
            kw = {} if steps == 1 else {"steps": steps}
            m, *_ = run_one_attack(model, ev, attack_spec(name, 8 / 255, args.seed, **kw), 8 / 255)
            m.update({"model": tag, "attack": name, "steps": steps, "eps": 8})
            step_rows.append(m)
            print(f"  [step] T={steps:2d} ASR={m['asr']*100:6.2f}% "
                  f"{m['time_ms_per_img']:7.1f} ms/img", flush=True)
        # C&W L2 (unconstrained L∞, minimal L2)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        m, adv, succ, l2, linf = run_one_attack(
            model, ev, A.build_attack("cw_l2", steps=25, binary_search=2), 8 / 255)
        m.update({"model": tag, "attack": "cw_l2", "eps": "", "steps": 25})
        cw_rows.append(m)
        print(f"  [cw] ASR={m['asr']*100:6.2f}% L2={m['mean_l2_succ']:.2f} "
              f"{m['time_ms_per_img']:7.1f} ms/img", flush=True)

    if rows:
        write_csv("results/attack_grid_std_r18.csv", rows)
    if step_rows:
        write_csv("results/attack_steps.csv", step_rows)
    if cw_rows:
        write_csv("results/cw_l2.csv", cw_rows)


def exp_transfer(args):
    """3x3 transfer matrix with PGD-20 (eps=8/255) and MI-FGSM (10 steps)."""
    dev = get_device()
    set_seed(args.seed)
    ev = EvalSet(args.n_eval, dev)
    models = {}
    for tag, ckpt in [("r18", "checkpoints/std_r18.pt"),
                      ("vgg16", "checkpoints/std_vgg16.pt"),
                      ("at_r18", "checkpoints/at_r18_eps8_s7.pt")]:
        if os.path.exists(ckpt):
            models[tag] = load_model(ckpt, dev)
    if len(models) < 2:
        print("need at least 2 trained models for the transfer experiment")
        return
    tags = list(models)
    ok = {t: (clean_pred(models[t], ev) == ev.y) for t in tags}
    for attack_name, steps, fname in (("pgd", 20, "results/transfer_pgd8.csv"),
                                      ("mi_fgsm", 10, "results/transfer_mifgsm8.csv")):
        rows = []
        for src in tags:
            attack = A.build_attack(attack_name, eps=8 / 255, steps=steps, seed=args.seed)
            adv_parts = []
            for x, y in ev.batches():
                adv_parts.append(attack(models[src], x, y).detach())
            adv = torch.cat(adv_parts)
            for tgt in tags:
                with torch.no_grad():
                    pred = models[tgt](adv).argmax(1)
                asr = ((pred != ev.y) & ok[tgt]).sum().item() / max(ok[tgt].sum().item(), 1)
                rows.append({"source": src, "target": tgt, "attack": attack_name,
                             "eps": 8, "steps": steps, "asr": float(asr),
                             "target_clean_acc": float(ok[tgt].float().mean().item())})
                print(f"  transfer {src:6s} -> {tgt:6s} ASR={asr*100:6.2f}%", flush=True)
        write_csv(fname, rows)


# --------------------------------------------------------------------------- #
# 作业二：防御评估
# --------------------------------------------------------------------------- #
ADAPTIVE_ATTACKS = [
    ("pgd", dict(steps=20), "pgd20", "PGD-20（CE 损失）"),
    ("pgd", dict(steps=20, loss="dlr", restarts=3), "pgd20_dlr_r3", "PGD-20 + DLR + 3 次重启"),
    ("pgd", dict(steps=20, loss="ce", restarts=1, target="least"), "pgd20_targeted",
     "目标化 PGD-20（最不可能类）"),
    ("apgd", dict(steps=40, restarts=1, loss="dlr"), "apgd40_dlr", "APGD-40 DLR"),
]


def _targeted_attack(model, x, y, eps_px, steps=20, seed=0):
    """Targeted PGD towards the least-likely class of the clean input."""
    with torch.no_grad():
        logits = model(x)
    target = logits.masked_fill(F.one_hot(y, 10).bool(), 1e9).argmin(1)
    return A.pgd_linf(model, x, y, eps=eps_px, steps=steps, loss="dlr", target=target,
                      random_start=True)


def exp_defense(args):
    dev = get_device()
    set_seed(args.seed)
    ev = EvalSet(args.n_eval, dev)
    tag, ckpt = args.tag, args.ckpt
    if not os.path.exists(ckpt):
        print("skip", ckpt)
        return
    model = load_model(ckpt, dev)
    with torch.no_grad():
        clean = float((model(ev.x01).argmax(1) == ev.y).float().mean().item())
    rows = []

    # (a) accuracy vs epsilon
    for eps in (2, 4, 8):
        eps_px = eps / 255.0
        for name, kw, key, desc in ADAPTIVE_ATTACKS[:1]:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            m, *_ = run_one_attack(model, ev, attack_spec(name, eps_px, args.seed, **kw), eps_px)
            m.update({"ckpt": tag, "attack": key, "eps": eps, "desc": desc, "clean_acc": clean})
            rows.append(m)
            print(f"  [{tag}] eps={eps} {key:16s} RA={m['robust_acc']*100:6.2f}%", flush=True)

    # (b) attack suite at eps = 8/255
    if getattr(args, "minimal", False):
        write_csv(f"results/defense_{tag}.csv", rows)
        return
    for name, kw, key, desc in ADAPTIVE_ATTACKS[1:]:
        eps_px = 8 / 255.0
        if key == "pgd20_targeted":
            t0 = time.perf_counter()
            adv_parts = []
            for x, y in ev.batches():
                adv_parts.append(_targeted_attack(model, x, y, eps_px, steps=20).detach())
            adv = torch.cat(adv_parts)
            with torch.no_grad():
                pred = model(adv).argmax(1)
            wrong = pred != ev.y
            m = {"n": ev.N, "asr": float(((wrong) & (clean_pred(model, ev) == ev.y)).sum().item() /
                                        max((clean_pred(model, ev) == ev.y).sum().item(), 1)),
                 "robust_acc": float(1 - wrong.float().mean().item()),
                 "uncond_err": float(wrong.float().mean().item()),
                 "clean_acc": clean, "time_ms_per_img": (time.perf_counter() - t0) / ev.N * 1000,
                 "peak_mem_gb": gpu_peak_mem_gb()}
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            m, *_ = run_one_attack(model, ev, attack_spec(name, eps_px, args.seed, **kw), eps_px)
            m["clean_acc"] = clean
        m.update({"ckpt": tag, "attack": key, "eps": 8, "desc": desc})
        rows.append(m)
        print(f"  [{tag}] eps=8 {key:16s} RA={m['robust_acc']*100:6.2f}%", flush=True)

    write_csv(f"results/defense_{tag}.csv", rows)

    # (c) convergence: robust accuracy vs number of PGD steps
    if getattr(args, "light", False):
        return
    conv = []
    for steps in (1, 2, 3, 5, 8, 10, 15, 20):
        m, *_ = run_one_attack(model, ev, attack_spec("pgd", 8 / 255, args.seed, steps=steps), 8 / 255)
        conv.append({"steps": steps, "robust_acc": m["robust_acc"], "asr": m["asr"],
                     "attack": "pgd", "eps": 8, "ckpt": tag})
        print(f"  [conv] T={steps:2d} RA={m['robust_acc']*100:6.2f}%", flush=True)
    write_csv(f"results/convergence_{tag}.csv", conv)

    # (d) AutoAttack-lite (strongest evaluation, smaller subset for time)
    if getattr(args, "skip_aa", False):
        return
    n_small = min(500, ev.N)
    ev_small = EvalSet(n_small, dev)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    adv_parts = []
    for x, y in ev_small.batches():
        adv_parts.append(A.autoattack_lite(model, x, y, 8 / 255, steps=40, restarts=1,
                                           seed=args.seed))
    adv = torch.cat(adv_parts)
    with torch.no_grad():
        pred = model(adv).argmax(1)
        ok = model(ev_small.x01).argmax(1) == ev_small.y
    wrong = pred != ev_small.y
    row = {"ckpt": tag, "attack": "autoattack_lite", "eps": 8, "n": ev_small.N,
           "clean_acc": float(ok.float().mean().item()),
           "asr": float((wrong & ok).sum().item() / max(ok.sum().item(), 1)),
           "robust_acc": float(1 - wrong.float().mean().item()),
           "uncond_err": float(wrong.float().mean().item()),
           "time_ms_per_img": (time.perf_counter() - t0) / ev_small.N * 1000,
           "desc": "AutoAttack-lite（APGD-CE + APGD-DLR + 目标化），40 步，500 张"}
    rows.append(row)
    write_csv(f"results/defense_{tag}.csv", rows)
    print(f"  [{tag}] autoattack-lite RA={row['robust_acc']*100:6.2f}%", flush=True)


def exp_clean_only(args):
    dev = get_device()
    ev = EvalSet(args.n_eval, dev)
    model = load_model(args.ckpt, dev)
    with torch.no_grad():
        logits = model(ev.x01)
    ok = logits.argmax(1) == ev.y
    out = {"ckpt": args.tag, "clean_acc": float(ok.float().mean().item()), "n": ev.N,
           "per_class": {CLASSES[c]: float(ok[ev.y == c].float().mean().item()) for c in range(10)},
           "env": device_report()}
    save_json(f"results/clean_{args.tag}.json", out)
    print(f"[{args.tag}] clean acc on {ev.N} eval images = {out['clean_acc']*100:.2f}%")


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("attack")
    sp.add_argument("--n-eval", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=exp_attack)

    sp = sub.add_parser("transfer")
    sp.add_argument("--n-eval", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=exp_transfer)

    sp = sub.add_parser("defense")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--tag", required=True)
    sp.add_argument("--n-eval", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--light", action="store_true", help="skip the convergence curve")
    sp.add_argument("--skip-aa", action="store_true", help="skip AutoAttack-lite")
    sp.add_argument("--minimal", action="store_true", help="only the epsilon grid")
    sp.set_defaults(func=exp_defense)

    sp = sub.add_parser("clean")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--tag", required=True)
    sp.add_argument("--n-eval", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=exp_clean_only)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
