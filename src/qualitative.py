"""Model-dependent figures: qualitative adversarial examples, confidence shifts,
and the loss-surface / decision-boundary slice (standard vs adversarially trained).

    python -m src.qualitative --std checkpoints/std_r18.pt --at checkpoints/at_r18_eps8_s7.pt

Produces
    figures/fig09_triptych.pdf     干净 / 对抗 / 扰动x10 三联图（8 类）
    figures/fig10_confidence.pdf   干净 vs 对抗输入的 top-3 置信度
    figures/fig11_loss_landscape.pdf  损失曲面与沿对抗方向的切片
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from . import attacks as A
from .data import CLASSES, CLASSES_ZH, denorm, get_datasets
from .evaluate import load_model
from .plot_style import PALETTE, save, setup


# --------------------------------------------------------------------------- #
# 图 9  三联对照图
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _select_one_per_class(model, n_class, dev, eps=8 / 255, seed=0):
    _, test_set = get_datasets()
    picked = {}
    idx = np.random.RandomState(seed).permutation(len(test_set))
    for i in idx:
        x, y = test_set[int(i)]
        y = int(y)                      # the dataset returns a 0-dim tensor; cast so
        x01 = denorm(x.unsqueeze(0).to(dev))  # dict keys are by value, not by object id
        if model(x01).argmax(1).item() == y and y not in picked:
            picked[y] = (x01, y)
        if len(picked) >= n_class:
            break
    ys = sorted(picked)
    x = torch.cat([picked[y][0] for y in ys])
    y = torch.tensor([picked[y][1] for y in ys], device=dev)
    return x, y, ys


def fig_triptych(ckpt, dev, outdir="figures", eps=8 / 255, seed=0):
    model = load_model(ckpt, dev)
    x, y, ys = _select_one_per_class(model, 8, dev, eps, seed)
    x_adv = A.pgd_linf(model, x, y, eps=eps, steps=20, step_size=2 / 255)
    with torch.no_grad():
        pred_adv = model(x_adv).argmax(1)
        conf_adv = model(x_adv).softmax(1).max(1).values
        pred_clean = model(x).argmax(1)
    n = len(ys)
    fig, axes = plt.subplots(3, n, figsize=(1.75 * n, 6.2))
    for j in range(n):
        img = x[j].detach().cpu().permute(1, 2, 0).numpy()
        adv = x_adv[j].detach().cpu().permute(1, 2, 0).numpy()
        diff = (adv - img) * 10.0
        panels = [(img, f"原图\n{CLASSES_ZH[ys[j]]}"),
                  (adv, f"对抗样本\n→{CLASSES_ZH[pred_adv[j].item()]}"),
                  (np.abs(diff), f"扰动×10\n‖δ‖∞={np.abs(adv-img).max()*255:.0f}/255")]
        for i, (im, title) in enumerate(panels):
            ax = axes[i, j]
            if i < 2:
                ax.imshow(np.clip(im, 0, 1))
            else:
                ax.imshow(np.clip(im, 0, 1))
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("图9  干净图像 / 对抗样本 / 放大 10 倍的扰动（PGD-20，ε=8/255，ResNet-18）",
                 y=0.99, fontsize=11)
    fig.tight_layout()
    save(fig, "fig09_triptych", outdir)
    return x, x_adv, y


# --------------------------------------------------------------------------- #
# 图 10  置信度变化
# --------------------------------------------------------------------------- #
def fig_confidence(ckpt, x, x_adv, y, dev, outdir="figures"):
    model = load_model(ckpt, dev)
    with torch.no_grad():
        p_clean = model(x).softmax(1).cpu()
        p_adv = model(x_adv).softmax(1).cpu()
    n_show = min(4, x.size(0))
    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 3.9))
    if n_show == 1:
        axes = [axes]
    for k in range(n_show):
        ax = axes[k]
        top = torch.topk(torch.maximum(p_clean[k], p_adv[k]), 3).indices.tolist()
        top = sorted(top)
        xs = np.arange(len(top))
        ax.bar(xs - 0.19, p_clean[k][top].numpy() * 100, 0.38, label="干净输入",
               color=PALETTE["sky"], edgecolor="k", lw=0.4)
        ax.bar(xs + 0.19, p_adv[k][top].numpy() * 100, 0.38, label="对抗输入",
               color=PALETTE["vermillion"], edgecolor="k", lw=0.4, hatch="//")
        ax.set_xticks(xs)
        ax.set_xticklabels([CLASSES_ZH[c] for c in top], fontsize=8.5)
        ax.set_ylim(0, 105)
        ax.set_ylabel("softmax 置信度 / %" if k == 0 else "")
        ax.set_title(f"真类：{CLASSES_ZH[y[k].item()]}", fontsize=9.5)
        for xi, c in zip(xs, top):
            ax.text(xi - 0.19, p_clean[k][c].item() * 100 + 1.5,
                    f"{p_clean[k][c].item()*100:.1f}", ha="center", fontsize=7, rotation=90)
            ax.text(xi + 0.19, p_adv[k][c].item() * 100 + 1.5,
                    f"{p_adv[k][c].item()*100:.1f}", ha="center", fontsize=7, rotation=90)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("图10  同一张图在干净与对抗输入下的 top-3 类别置信度", y=1.02)
    fig.tight_layout()
    save(fig, "fig10_confidence", outdir)


# --------------------------------------------------------------------------- #
# 图 11  损失曲面 / 决策边界切片
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _loss_grid(model, x, y, d1, d2, alphas, betas):
    """CE loss on  x + a*d1 + b*d2  (clamped to [0,1]), averaged over the batch."""
    H = np.zeros((len(betas), len(alphas)))
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            z = (x + a * d1 + b * d2).clamp(0, 1)
            H[i, j] = F.cross_entropy(model(z), y).item()
    return H


@torch.no_grad()
def _loss_slice(model, x, y, d, ts):
    out = []
    for t in ts:
        z = (x + t * d).clamp(0, 1)
        out.append(F.cross_entropy(model(z), y).item())
    return np.array(out)


def fig_loss_landscape(std_ckpt, at_ckpt, dev, outdir="figures", eps=8 / 255, seed=0):
    std = load_model(std_ckpt, dev)
    at = load_model(at_ckpt, dev) if at_ckpt and os.path.exists(at_ckpt) else None
    x, y, _ = _select_one_per_class(std, 16, dev, eps, seed)
    x_adv = A.pgd_linf(std, x, y, eps=eps, steps=20, step_size=2 / 255)
    d1 = (x_adv - x)
    # a second, random direction orthogonal-ish to d1 (same L_inf scale)
    g = torch.randn_like(x)
    g = g - (g * d1).sum() / (d1.pow(2).sum() + 1e-12) * d1
    d2 = g / g.abs().max() * eps

    # alphas / betas are in units of eps: d1 and d2 already have L_inf = eps,
    # so alpha = 1 lands exactly on the adversarial example.
    alphas = np.linspace(-1.2, 1.2, 25)
    betas = np.linspace(-1.2, 1.2, 25)
    models = [("标准训练", std)] + ([("对抗训练（AT-7，ε=8/255）", at)] if at is not None else [])
    cmap = LinearSegmentedColormap.from_list("loss", ["#f7fbff", "#6baed6", "#08306b"])
    fig = plt.figure(figsize=(4.4 * len(models) + 4.4, 4.4))
    for i, (name, m) in enumerate(models):
        ax = fig.add_subplot(1, len(models) + 1, i + 1)
        H = _loss_grid(m, x, y, d1, d2, alphas, betas)
        im = ax.imshow(H, origin="lower", cmap=cmap, aspect="auto",
                       extent=[alphas[0], alphas[-1], betas[0], betas[-1]])
        cs = ax.contour(alphas, betas, H, levels=6,
                        colors="k", linewidths=0.5, alpha=0.6)
        ax.plot(0, 0, marker="*", ms=13, color=PALETTE["orange"], mec="k",
                label="干净样本")
        ax.plot(1, 0, marker="o", ms=7, color=PALETTE["vermillion"], mec="k",
                label="对抗样本（沿 α 方向）")
        ax.set_title(name)
        ax.set_xlabel(r"$\alpha$（× ε，沿对抗方向）")
        ax.set_ylabel(r"$\beta$（× ε，沿随机正交方向）")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="交叉熵损失")
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper right")
    # 1D slice along the adversarial direction
    ax1 = fig.add_subplot(1, len(models) + 1, len(models) + 1)
    ts = np.linspace(-0.5, 2.0, 60)
    styles = [("-", PALETTE["grey"]), ("-", PALETTE["green"])]
    for (name, m), (ls, c) in zip(models, styles):
        ax1.plot(ts, _loss_slice(m, x, y, d1, ts), ls=ls, color=c, lw=1.6, label=name)
    ax1.axvline(0, color="k", lw=0.7, ls=":")
    ax1.axvline(1, color=PALETTE["vermillion"], lw=0.9, ls="--")
    ax1.set_xlabel(r"$t$：$x+t\cdot\delta_{\mathrm{adv}}$")
    ax1.set_ylabel("交叉熵损失")
    ax1.set_title("沿对抗方向的损失切片")
    ax1.legend(fontsize=8)
    fig.suptitle("图11  损失曲面：标准训练 vs 对抗训练（二维投影与一维切片）", y=1.02)
    fig.tight_layout()
    save(fig, "fig11_loss_landscape", outdir)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--std", default="checkpoints/std_r18.pt")
    p.add_argument("--at", default="checkpoints/at_r18_eps8_s7.pt")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    setup()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, x_adv, y = fig_triptych(args.std, dev, args.outdir, seed=args.seed)
    fig_confidence(args.std, x, x_adv, y, dev, args.outdir)
    fig_loss_landscape(args.std, args.at, dev, args.outdir, seed=args.seed)


if __name__ == "__main__":
    main()
