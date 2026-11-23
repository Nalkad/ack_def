"""Training entry point: standard training, PGD adversarial training, TRADES.

The whole pipeline works in **[0,1] pixel space**: the loader returns normalised
tensors (because the transforms normalise), and ``denorm`` maps them back to
[0,1] once per batch.  From that point on the model used everywhere is
``PixelModel`` (which normalises internally), so the inner attack and the outer
loss see the same tensor, and ``eps`` is in pixel units.

Usage
-----
    python -m src.train --mode std    --model resnet18 --epochs 40 --tag std_r18
    python -m src.train --mode pgd_at --model resnet18 --epochs 30 --eps 8 --steps 7
    python -m src.train --mode trades --model resnet18 --epochs 30 --beta 6
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attacks
from .common import (AverageMeter, CSVLogger, accuracy, cosine_lr, device_report,
                     get_device, gpu_peak_mem_gb, save_checkpoint, save_json, set_seed)
from .data import denorm, get_loaders
from .models import PixelModel, build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="std", choices=["std", "pgd_at", "trades", "fast_at"])
    p.add_argument("--model", default="resnet18")
    p.add_argument("--data", default="data/cifar10")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resdir", default="results")
    # adversarial-training knobs; eps / alpha are given in units of 1/255
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--alpha", type=float, default=None, help="default: 2.5*eps/steps")
    p.add_argument("--beta", type=float, default=6.0, help="TRADES coefficient")
    p.add_argument("--trades_steps", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--init", default=None,
                   help="warm-start weights (used for the AT runs: fine-tune the clean model)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--deterministic", action="store_true",
                   help="bitwise-deterministic cudnn (slower); default uses the faster autotuned path")
    p.add_argument("--log-every", type=int, default=0,
                   help="print a progress line every N iterations (0 = off)")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="also write checkpoints/<tag>_e{N}.pt (used for the step ablation)")
    p.add_argument("--eval-subset", type=int, default=512,
                   help="how many fixed test images are used for the per-epoch robust-acc probe")
    return p.parse_args(argv)


def train_step(model, x01, y, args, eps_px):
    """One optimisation step. Returns (loss, logits_for_accuracy)."""
    if args.mode == "std":
        logits = model(x01)
        return F.cross_entropy(logits, y), logits

    if args.mode == "fast_at":
        # single-step variant, only used as an ablation
        x_adv = attacks.pgd_linf(model, x01, y, eps=eps_px, steps=1, random_start=True)
        logits = model(x_adv)
        return F.cross_entropy(logits, y), logits

    if args.mode == "pgd_at":
        alpha = (args.alpha / 255.0) if args.alpha is not None else None
        x_adv = attacks.pgd_linf(model, x01, y, eps=eps_px, steps=args.steps,
                                 step_size=alpha, random_start=True, loss="ce")
        logits = model(x_adv)
        return F.cross_entropy(logits, y), logits

    if args.mode == "trades":
        x_adv = trades_inner(model, x01, y, eps_px, args.trades_steps)
        logits = model(x01)
        logits_adv = model(x_adv)
        loss_ce = F.cross_entropy(logits, y)
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),
                           F.softmax(logits.detach(), dim=1),
                           reduction="batchmean")
        return loss_ce + args.beta * loss_kl, logits

    raise ValueError(args.mode)


def trades_inner(model, x01, y, eps_px, steps):
    """Inner maximisation of KL(p_clean || p_adv) with an L_inf PGD loop."""
    alpha = 2.5 * eps_px / steps
    x_adv = (x01 + torch.empty_like(x01).uniform_(-eps_px, eps_px)).clamp(0, 1).detach()
    p_clean = F.softmax(model(x01), dim=1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_clean, reduction="batchmean")
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = (x_adv.detach() + alpha * grad.sign()).clamp(x01 - eps_px, x01 + eps_px).clamp(0, 1)
    return x_adv.detach()


@torch.no_grad()
def evaluate(model, loader, dev):
    """Clean accuracy on a loader whose transform is *test* (no augmentation)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        logits = model(denorm(x))
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def probe_batch(data_root: str, n: int, dev):
    """First ``n`` test images (fixed order) used as an AT progress probe."""
    import numpy as np
    from .data import CIFAR10_MEAN as _M, CIFAR10_STD as _S
    d = np.load(os.path.join(data_root, "test.npz"))
    x = torch.from_numpy(d["x"][:n]).float().div_(255.0)
    y = torch.from_numpy(d["y"][:n]).long()
    return x.to(dev), y.to(dev)


@torch.no_grad()
def robust_probe(model, x01, y, dev, eps=8 / 255, steps=5):
    """Cheap PGD-5 accuracy on the probe subset (progress monitoring only)."""
    model.eval()
    x_adv = attacks.pgd_linf(model, x01, y, eps=eps, steps=steps, random_start=True)
    acc = (model(x_adv).argmax(1) == y).float().mean().item()
    model.train()
    return acc


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed, deterministic=args.deterministic)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        dev = torch.device("cpu")
    elif args.device == "cuda":
        dev = torch.device("cuda")
    else:
        dev = get_device()
    eps_px = args.eps / 255.0
    tag = args.tag or f"{args.mode}_{args.model}_eps{args.eps:g}_s{args.steps}"
    out_ckpt = os.path.join(args.out, tag + ".pt")
    out_csv = os.path.join(args.resdir, "logs", tag + ".csv")
    out_json = os.path.join(args.resdir, "runs", tag + ".json")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.threads:
        torch.set_num_threads(args.threads)
    train_loader, test_loader, _, _ = get_loaders(
        args.data, batch_size=args.bs, workers=args.workers, seed=args.seed)

    net = build_model(args.model).to(dev)
    if args.init:
        from .common import load_checkpoint
        load_checkpoint(args.init, net)
        print(f"[{tag}] warm start from {args.init}", flush=True)
    model = PixelModel(net).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")

    logger = CSVLogger(out_csv, ["epoch", "lr", "train_loss", "train_acc",
                                 "test_acc", "epoch_time_s", "peak_mem_gb",
                                 "robust_probe_acc"])

    # a fixed probe subset of the test set, used only to watch the adversarial
    # training progress (never used for any reported number -- see the report)
    probe_x = probe_y = None
    if args.eval_subset:
        probe_x, probe_y = probe_batch(args.data, args.eval_subset, dev)

    total_t0 = time.perf_counter()
    for epoch in range(args.epochs):
        cosine_lr(opt, epoch, args.epochs, args.lr, args.warmup)
        lr = opt.param_groups[0]["lr"]
        model.train()
        t0 = time.perf_counter()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for it, (x, y) in enumerate(train_loader):
            if args.log_every and it % args.log_every == 0 and it > 0:
                print(f"    epoch {epoch+1} iter {it}/{len(train_loader)} "
                      f"{1000*(time.perf_counter()-t0)/it:.0f} ms/iter", flush=True)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            x01 = denorm(x)
            opt.zero_grad(set_to_none=True)
            if args.amp:
                with torch.cuda.amp.autocast():
                    loss, logits = train_step(model, x01, y, args, eps_px)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss, logits = train_step(model, x01, y, args, eps_px)
                loss.backward()
                opt.step()
            with torch.no_grad():
                loss_m.update(loss.item(), x.size(0))
                acc_m.update(accuracy(logits.detach(), y), x.size(0))
        test_acc = evaluate(model, test_loader, dev)
        probe_acc = ""
        if probe_x is not None:
            probe_acc = robust_probe(model, probe_x, probe_y, dev)
        dt = time.perf_counter() - t0
        if args.save_every_epoch:
            save_checkpoint(os.path.join(args.out, f"{tag}_e{epoch+1}.pt"), net,
                            {"tag": f"{tag}_e{epoch+1}", "args": vars(args)})
        logger.log({"epoch": epoch + 1, "lr": f"{lr:.6g}",
                    "train_loss": f"{loss_m.avg:.4f}", "train_acc": f"{acc_m.avg:.4f}",
                    "test_acc": f"{test_acc:.4f}", "epoch_time_s": f"{dt:.2f}",
                    "peak_mem_gb": gpu_peak_mem_gb(),
                    "robust_probe_acc": f"{probe_acc:.4f}" if probe_acc != "" else ""})
        print(f"[{tag}] epoch {epoch+1:3d}/{args.epochs} lr={lr:.4g} "
              f"loss={loss_m.avg:.3f} train={acc_m.avg*100:.2f}% test={test_acc*100:.2f}% "
              f"attacked(probe)={probe_acc*100 if probe_acc != '' else float('nan'):.2f}% "
              f"{dt:.1f}s", flush=True)

    total = time.perf_counter() - total_t0
    save_checkpoint(out_ckpt, net, {"tag": tag, "args": vars(args), "test_acc": test_acc})
    save_json(out_json, {
        "tag": tag, "args": vars(args), "final_test_acc": test_acc,
        "total_train_time_s": round(total, 1),
        "epoch_time_s": round(total / args.epochs, 2),
        "peak_mem_gb": gpu_peak_mem_gb(),
        "env": device_report(),
    })
    print(f"[{tag}] done. clean test acc = {test_acc*100:.2f}%  "
          f"total {total/60:.1f} min  peak {gpu_peak_mem_gb():.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

