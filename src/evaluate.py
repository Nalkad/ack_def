"""Evaluation harness for the attack experiments (作业一) and the defence
evaluation (作业二).

Metric definitions used everywhere in the report (see "评估协议"):

* **ASR (attack success rate)** = #{x : clean-correct(x) and f(x_adv) != y} / #{x : clean-correct(x)}.
  Restricting to clean-correct samples separates "attack effectiveness" from
  "the model is simply bad".
* **unconditional error rate** = #{x : f(x_adv) != y} / N.  This is the quantity
  that a deployment cares about; both are written to the CSV.
* **robust accuracy (RA)** = #{x : f(x_adv) == y} / N over the *whole* test set.
  RA = 1 - unconditional error rate.
* Per-image cost is wall-clock attack time divided by the number of images, with
  the model in ``eval()`` mode and ``cuda.synchronize`` around the timed region.

Sub-commands
------------
    python -m src.evaluate clean      --ckpt checkpoints/std_r18.pt
    python -m src.evaluate grid       --ckpt checkpoints/std_r18.pt --attacks fgsm,pgd,...
    python -m src.evaluate transfer   --sources r18=...,vgg16=... --targets ...
    python -m src.evaluate defense    --ckpt checkpoints/at_r18_eps8_s7.pt --adaptive
    python -m src.evaluate convergence --ckpt ... --max-steps 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import attacks as A
from .common import device_report, get_device, gpu_peak_mem_gb, save_json, set_seed
from .data import CLASSES, denorm, get_datasets, get_loaders
from .models import PixelModel, build_model

EPS_LIST = [1, 2, 4, 8]


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_model(ckpt: str, dev, model_name: str = None) -> PixelModel:
    ckpt_val = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = ckpt_val.get("meta", {})
    name = model_name or meta.get("args", {}).get("model", "resnet18")
    net = build_model(name)
    net.load_state_dict(ckpt_val["state_dict"])
    net.eval()
    return PixelModel(net).to(dev).eval()


@torch.no_grad()
def clean_predictions(model, loader, dev):
    """Return (logits, labels) for the whole loader."""
    logits_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits_all.append(model(denorm(x)).float().cpu())
        labels_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


# --------------------------------------------------------------------------- #
def attack_stats(model, loader, attack_name, eps_px, dev, seed=0, **kw):
    """Attack the whole loader and return a dict of metrics."""
    attack = A.build_attack(attack_name, eps=eps_px, seed=seed, **kw)
    model.eval()
    adv_all, y_all, x_all = [], [], []
    t_total, n_total = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        x01 = denorm(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_adv = attack(model, x01, y)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        n_total += y.numel()
        adv_all.append(x_adv.detach().float().cpu())
        y_all.append(y.cpu())
        x_all.append(x01.detach().float().cpu())
    adv = torch.cat(adv_all)
    x01 = torch.cat(x_all)
    y = torch.cat(y_all)

    with torch.no_grad():
        logits_clean = model(x01.to(dev)).float().cpu()
        pred_adv = model(adv.to(dev)).argmax(1).cpu()
    clean_pred = logits_clean.argmax(1)
    ok = clean_pred == y
    wrong = pred_adv != y
    asr = ((wrong & ok).sum().item() / max(ok.sum().item(), 1))
    uncond = wrong.float().mean().item()

    l2 = (adv - x01).flatten(1).norm(p=2, dim=1)
    linf = (adv - x01).flatten(1).abs().max(dim=1).values
    # norm of the *successful* perturbations is the meaningful "cost" figure
    succ = wrong & ok
    stats = {
        "n": int(y.numel()),
        "asr": float(asr),
        "uncond_err": float(uncond),
        "robust_acc": float(1.0 - uncond),
        "clean_acc": float(ok.float().mean().item()),
        "mean_l2_all": float(l2.mean().item()),
        "mean_linf_all": float(linf.mean().item()),
        "mean_l2_succ": float(l2[succ].mean().item()) if succ.any() else 0.0,
        "mean_linf_succ": float(linf[succ].mean().item()) if succ.any() else 0.0,
        "time_ms_per_img": float(t_total / n_total * 1000.0),
        "peak_mem_gb": gpu_peak_mem_gb(),
    }
    return stats


# --------------------------------------------------------------------------- #
# sub-commands
# --------------------------------------------------------------------------- #
def cmd_clean(args):
    dev = get_device()
    set_seed(args.seed)
    model = load_model(args.ckpt, dev, args.model)
    _, test_loader, _, _ = get_loaders(args.data, test_batch_size=args.bs,
                                       workers=args.workers, seed=args.seed)
    logits, y = clean_predictions(model, test_loader, dev)
    acc = (logits.argmax(1) == y).float().mean().item()
    per_class = []
    for c, name in enumerate(CLASSES):
        m = y == c
        per_class.append({"class": name, "acc": float((logits.argmax(1)[m] == y[m]).float().mean())})
    out = {"ckpt": args.ckpt, "clean_acc": acc, "per_class": per_class,
           "env": device_report()}
    print(f"clean accuracy = {acc*100:.2f}%")
    if args.out:
        save_json(args.out, out)
    return out


def cmd_grid(args):
    dev = get_device()
    set_seed(args.seed)
    model = load_model(args.ckpt, dev, args.model)
    _, test_loader, _, _ = get_loaders(args.data, test_batch_size=args.bs,
                                       workers=args.workers, seed=args.seed)
    rows = []
    for eps in args.eps:
        eps_px = eps / 255.0
        for name in args.attacks.split(","):
            kw = {}
            if name in ("pgd", "apgd", "autoattack_lite"):
                kw["steps"] = args.steps
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if name == "pgd" and args.restarts > 1:
                kw["restarts"] = args.restarts
            if name == "apgd" and args.restarts > 1:
                kw["restarts"] = args.restarts
            st = attack_stats(model, test_loader, name, eps_px, dev, seed=args.seed, **kw)
            st.update({"attack": name, "eps": eps, "eps_px": eps_px,
                       "ckpt": args.ckpt, "steps": kw.get("steps", ""),
                       "restarts": kw.get("restarts", 1)})
            rows.append(st)
            print(f"eps={eps:>2} [{name:16s}] ASR={st['asr']*100:6.2f}% "
                  f"RA={st['robust_acc']*100:6.2f}% |L2|={st['mean_l2_succ']:6.2f} "
                  f"|Linf|={st['mean_linf_succ']*255:5.2f}/255 "
                  f"{st['time_ms_per_img']:7.1f} ms/img", flush=True)
    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)
    return rows


def cmd_transfer(args):
    """Cross-model transfer matrix.

    ``--sources name=ckpt,name=ckpt`` and ``--targets name=ckpt,name=ckpt``; the
    adversary is generated on the source (white box) and applied to the target.
    """
    dev = get_device()
    set_seed(args.seed)
    _, test_loader, _, _ = get_loaders(args.data, test_batch_size=args.bs,
                                       workers=args.workers, seed=args.seed)
    pairs = dict(p.split("=", 1) for p in (args.sources + "," + args.targets).split(",") if p)
    models = {k: load_model(v, dev) for k, v in pairs.items()}
    src_names = [p.split("=")[0] for p in args.sources.split(",")]
    tgt_names = [p.split("=")[0] for p in args.targets.split(",")]
    eps_px = args.eps / 255.0

    # clean predictions of every target, so that the ASR is measured on the
    # subset each target classifies correctly (otherwise a weak target inflates
    # the apparent transferability)
    target_ok = {}
    y_ref = None
    for t in tgt_names:
        logits_t, y_ref = clean_predictions(models[t], test_loader, dev)
        target_ok[t] = logits_t.argmax(1) == y_ref

    rows = []
    for s in src_names:
        attack = A.build_attack(args.attack, eps=eps_px, steps=args.steps,
                                seed=args.seed, restarts=args.restarts)
        adv_batches, y_all = [], []
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            adv_batches.append(attack(models[s], denorm(x), y).cpu())
            y_all.append(y.cpu())
        adv = torch.cat(adv_batches)
        y = torch.cat(y_all)
        for t in tgt_names:
            with torch.no_grad():
                pred = models[t](adv.to(dev)).argmax(1).cpu()
            ok = target_ok[t]
            asr = ((pred != y) & ok).sum().item() / max(ok.sum().item(), 1)
            rows.append({"source": s, "target": t, "attack": args.attack, "eps": args.eps,
                         "asr": float(asr), "robust_acc": float(1 - asr),
                         "target_clean_acc": float(ok.float().mean().item()),
                         "same_model": s == t})
            print(f"transfer {s:7s} -> {t:7s} eps={args.eps} ASR={asr*100:6.2f}%"
                  f" (target clean {ok.float().mean().item()*100:.2f}%)", flush=True)
    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)
    return rows


def cmd_defense(args):
    dev = get_device()
    set_seed(args.seed)
    model = load_model(args.ckpt, dev, args.model)
    _, test_loader, _, _ = get_loaders(args.data, test_batch_size=args.bs,
                                       workers=args.workers, seed=args.seed)
    rows = []
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    for eps in args.eps:
        eps_px = eps / 255.0
        for spec in args.attacks.split(","):
            name, _, extra = spec.partition(":")
            kw = {}
            for kv in extra.split("+") if extra else []:
                k, _, v = kv.partition("=")
                kw[k] = int(v) if v.isdigit() else float(v)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            st = attack_stats(model, test_loader, name, eps_px, dev, seed=args.seed, **kw)
            st.update({"attack": name, "spec": spec, "eps": eps, "ckpt": args.ckpt})
            rows.append(st)
            print(f"eps={eps:>2} [{spec:32s}] RA={st['robust_acc']*100:6.2f}% "
                  f"ASR={st['asr']*100:6.2f}% {st['time_ms_per_img']:7.1f} ms/img",
                  flush=True)
    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)
    return rows


def cmd_convergence(args):
    """Robust accuracy as a function of the number of PGD steps (convergence test)."""
    dev = get_device()
    set_seed(args.seed)
    model = load_model(args.ckpt, dev, args.model)
    _, test_loader, _, _ = get_loaders(args.data, test_batch_size=args.bs,
                                       workers=args.workers, seed=args.seed)
    eps_px = args.eps / 255.0
    rows = []
    for steps in range(1, args.max_steps + 1):
        st = attack_stats(model, test_loader, args.attack, eps_px, dev, seed=args.seed,
                          steps=steps)
        rows.append({"steps": steps, "robust_acc": st["robust_acc"], "asr": st["asr"],
                     "attack": args.attack, "eps": args.eps})
        print(f"steps={steps:2d} RA={st['robust_acc']*100:6.2f}% ASR={st['asr']*100:6.2f}%",
              flush=True)
    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)
    return rows


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", default="data/cifar10")
        sp.add_argument("--bs", type=int, default=256)
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--model", default=None)
        sp.add_argument("--out", default=None)

    sp = sub.add_parser("clean"); common(sp); sp.add_argument("--ckpt", required=True)
    sp.set_defaults(func=cmd_clean)

    sp = sub.add_parser("grid"); common(sp)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--attacks", default="fgsm,pgd,mi_fgsm,cw_l2,apgd")
    sp.add_argument("--eps", type=float, nargs="+", default=EPS_LIST)
    sp.add_argument("--steps", type=int, default=20)
    sp.add_argument("--restarts", type=int, default=1)
    sp.set_defaults(func=cmd_grid)

    sp = sub.add_parser("transfer"); common(sp)
    sp.add_argument("--sources", required=True)
    sp.add_argument("--targets", required=True)
    sp.add_argument("--attack", default="pgd")
    sp.add_argument("--eps", type=float, default=8)
    sp.add_argument("--steps", type=int, default=20)
    sp.add_argument("--restarts", type=int, default=1)
    sp.set_defaults(func=cmd_transfer)

    sp = sub.add_parser("defense"); common(sp)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--attacks", default="pgd:steps=20")
    sp.add_argument("--eps", type=float, nargs="+", default=[8])
    sp.set_defaults(func=cmd_defense)

    sp = sub.add_parser("convergence"); common(sp)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--attack", default="pgd")
    sp.add_argument("--eps", type=float, default=8)
    sp.add_argument("--max-steps", type=int, default=20)
    sp.set_defaults(func=cmd_convergence)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

