"""Build every data-driven figure of the report from the CSV files in results/.

Run after the experiments:

    python -m src.figures

Each figure is written as PDF (vector, required by the assignment) and PNG
(300 dpi, embedded in the report).  If a result file is missing the figure is
skipped with a message instead of crashing, so the script can be re-run while
the experiments are still going on.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import PALETTE, SERIES_STYLE, save, setup


def _read(path):
    if not os.path.exists(path):
        print("  [skip] missing", path)
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _main_model(df, model="resnet18"):
    return df


# --------------------------------------------------------------------------- #
# 图 1  成功率 - epsilon
# --------------------------------------------------------------------------- #
def fig_asr_vs_eps(grid_csv="results/attack_grid_std_r18.csv", outdir="figures"):
    df = _read(grid_csv)
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, metric, title in ((axes[0], "asr", "(a) 攻击成功率 ASR"),
                              (axes[1], "robust_acc", "(b) 被攻击后的准确率")):
        for name in ("fgsm", "pgd", "mi_fgsm", "apgd"):
            d = df[df.attack == name].sort_values("eps")
            if not len(d):
                continue
            st = SERIES_STYLE[name]
            ax.plot(d.eps, d[metric] * 100, color=st["color"], marker=st["marker"],
                    ls=st["ls"], ms=5, lw=1.6, label=st["label"], clip_on=False)
        ax.set_xlabel(r"扰动预算 $\epsilon$ （单位 1/255，$L_\infty$）")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_title(title)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("比例 / %")
    # the unconditional error rate of the clean model is a useful reference
    axes[0].axhline(df[df.attack == "fgsm"].asr.max() * 0 + (1 - df.clean_acc.iloc[0]) * 100,
                    color="k", ls=":", lw=1, alpha=0.6)
    axes[0].text(1.05, (1 - df.clean_acc.iloc[0]) * 100 + 2, "干净误分类率",
                 fontsize=8, alpha=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("图1  四种攻击在 ResNet-18（标准训练）上的成功率–ε 曲线", y=1.02)
    fig.tight_layout()
    save(fig, "fig01_asr_vs_eps", outdir)


# --------------------------------------------------------------------------- #
# 图 2  扰动范数 - epsilon
# --------------------------------------------------------------------------- #
def fig_norm_vs_eps(grid_csv="results/attack_grid_std_r18.csv",
                    cw_csv="results/cw_l2.csv", outdir="figures"):
    df = _read(grid_csv)
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, col, ylab, scale in (
            (axes[0], "mean_linf_succ", r"平均 $L_\infty$ 扰动（/255）", 255.0),
            (axes[1], "mean_l2_succ", r"平均 $L_2$ 扰动", 1.0)):
        for name in ("fgsm", "pgd", "mi_fgsm", "apgd"):
            d = df[df.attack == name].sort_values("eps")
            st = SERIES_STYLE[name]
            ax.plot(d.eps, d[col] * scale, color=st["color"], marker=st["marker"],
                    ls=st["ls"], ms=5, lw=1.6, label=st["label"])
        ax.set_xlabel(r"扰动预算 $\epsilon$ （/255）")
        ax.set_ylabel(ylab)
        ax.set_xticks([1, 2, 4, 8])
    axes[0].plot([1, 8], [1, 8], color="k", lw=0.8, ls="--", alpha=0.5)
    axes[0].text(6.4, 5.2, r"$L_\infty=\epsilon$", fontsize=8, alpha=0.7, rotation=28)
    cw = _read(cw_csv) if cw_csv else None
    if cw is not None:
        l2 = float(cw.mean_l2_succ.iloc[0])
        axes[1].axhline(l2, color=PALETTE["purple"], ls="-.", lw=1.4)
        axes[1].text(1.1, l2 + 0.10, f"C&W-L2（无 ε 约束，仅最小化 L2）={l2:.2f}",
                     fontsize=8, color=PALETTE["purple"],
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    axes[0].set_title(r"(a) $L_\infty$ 范数（成功样本）")
    axes[1].set_title(r"(b) $L_2$ 范数（成功样本）")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("图2  三种约束型攻击的扰动代价随 ε 的变化", y=1.02)
    fig.tight_layout()
    save(fig, "fig02_perturbation_vs_eps", outdir)


# --------------------------------------------------------------------------- #
# 图 3  代价对比
# --------------------------------------------------------------------------- #
def fig_cost(grid_csv="results/attack_grid_std_r18.csv", outdir="figures"):
    """(a) per-image cost at eps=8/255, (b) success rate vs cost at eps=2/255."""
    df = _read(grid_csv)
    if df is None:
        return
    d = df[df.eps == 8].copy()
    order = [n for n in ("fgsm", "pgd", "mi_fgsm", "apgd") if n in set(d.attack)]
    d = d.set_index("attack").loc[order]
    base = d.time_ms_per_img.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    x = np.arange(len(d))
    axes[0].bar(x, d.time_ms_per_img, color=[SERIES_STYLE[n]["color"] for n in d.index],
                width=0.6, edgecolor="k", linewidth=0.5)
    axes[0].set_yscale("log")
    for i, (n, v) in enumerate(d.time_ms_per_img.items()):
        axes[0].text(i, v * 1.18, f"{v:.1f} ms\n({v/base:.1f}×)", ha="center",
                     va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["FGSM", "PGD-20", "MI-FGSM-10", "APGD-40"])
    axes[0].set_ylabel("单张图生成耗时 / ms（对数坐标）")
    axes[0].set_ylim(0.05, d.time_ms_per_img.max() * 6)
    axes[0].set_title("(a) 计算代价（ε=8/255）")

    e2 = df[df.eps == 2].set_index("attack")
    for n in order:
        st = SERIES_STYLE[n]
        xi = e2.loc[n, "time_ms_per_img"]
        yi = e2.loc[n, "asr"] * 100
        axes[1].scatter(xi, yi, s=90, color=st["color"], marker=st["marker"],
                        edgecolor="k", lw=0.6, zorder=3)
        axes[1].annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                         xytext=(6, 6), fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("单张图生成耗时 / ms（对数坐标）")
    axes[1].set_ylabel("攻击成功率 / %")
    axes[1].set_title("(b) 成功率与代价（ε=2/255）")
    lo = min(e2.loc[n, "asr"] for n in order) * 100
    axes[1].set_ylim(max(0, lo - 15), 102)
    fig.suptitle("图3  攻击的计算代价，以及“代价–收益”关系", y=1.02)
    fig.tight_layout()
    save(fig, "fig03_cost", outdir)


# --------------------------------------------------------------------------- #
# 图 4  迁移矩阵
# --------------------------------------------------------------------------- #
def fig_transfer(csv="results/transfer_pgd8.csv", outdir="figures"):
    df = _read(csv)
    if df is None:
        return
    srcs = list(dict.fromkeys(df.source))
    tgts = list(dict.fromkeys(df.target))
    mat = np.zeros((len(srcs), len(tgts)))
    for i, s in enumerate(srcs):
        for j, t in enumerate(tgts):
            row = df[(df.source == s) & (df.target == t)]
            mat[i, j] = row.asr.iloc[0] * 100 if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(100, np.nanmax(mat)))
    for i in range(len(srcs)):
        for j in range(len(tgts)):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center",
                    color="white" if mat[i, j] > 55 else "black", fontsize=11)
    ax.set_xticks(range(len(tgts)))
    ax.set_xticklabels([t.upper() for t in tgts])
    ax.set_yticks(range(len(srcs)))
    ax.set_yticklabels([s.upper() for s in srcs])
    ax.set_xlabel("目标模型（攻击时未知）")
    ax.set_ylabel("替代模型（白盒生成对抗样本）")
    ax.set_title(f"图4  迁移攻击成功率（PGD-20，ε={df.eps.iloc[0]:g}/255）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ASR / %")
    fig.tight_layout()
    save(fig, "fig04_transfer_matrix", outdir)


# --------------------------------------------------------------------------- #
# 图 5  不同 epsilon 下的干净 / 鲁棒准确率
# --------------------------------------------------------------------------- #
def fig_acc_vs_eps(std_csv="results/attack_grid_std_r18.csv",
                   at_csv="results/defense_at_r18_eps8_s7.csv", outdir="figures"):
    std = _read(std_csv)
    at = _read(at_csv)
    if std is None or at is None:
        return
    s = std[(std.attack == "pgd")].sort_values("eps")
    atk_col = "pgd20" if (at.attack == "pgd20").any() else "pgd"
    a = at[(at.attack == atk_col)].sort_values("eps")
    eps = sorted(set(s.eps) & set(a.eps))
    s = s[s.eps.isin(eps)]
    a = a[a.eps.isin(eps)]
    x = np.arange(len(eps))
    w = 0.19
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - 1.5 * w, s.clean_acc * 100, w, label="标准训练（干净）", color="#BBBBBB",
           edgecolor="k", lw=0.4)
    ax.bar(x - 0.5 * w, a.clean_acc * 100, w, label="对抗训练（干净）",
           color=PALETTE["sky"], edgecolor="k", lw=0.4)
    ax.bar(x + 0.5 * w, s.robust_acc * 100, w, label="标准训练（受攻击）",
           color=PALETTE["vermillion"], edgecolor="k", lw=0.4, hatch="//")
    ax.bar(x + 1.5 * w, a.robust_acc * 100, w, label="对抗训练（受攻击，PGD-20）",
           color=PALETTE["green"], edgecolor="k", lw=0.4, hatch="//")
    for i in range(len(eps)):
        for off, val in ((-1.5, s.clean_acc.iloc[i]), (-0.5, a.clean_acc.iloc[i]),
                         (0.5, s.robust_acc.iloc[i]), (1.5, a.robust_acc.iloc[i])):
            ax.text(i + off * w, val * 100 + 1, f"{val*100:.1f}", ha="center",
                    fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e:g}/255" for e in eps])
    ax.set_ylabel("准确率 / %")
    ax.set_ylim(0, 108)
    ax.set_title("图5  标准训练 vs PGD 对抗训练：干净与鲁棒准确率（ε 为评估时的攻击预算）")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False)
    fig.tight_layout()
    save(fig, "fig05_acc_vs_eps", outdir)


# --------------------------------------------------------------------------- #
# 图 6  鲁棒性 - 精度权衡
# --------------------------------------------------------------------------- #
def fig_tradeoff(points=None, outdir="figures"):
    """``points`` is a list of (label, clean_acc, robust_acc) tuples."""
    if points is None:
        points = collect_tradeoff_points()
    if not points:
        print("  [skip] no trade-off points")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label, ca, ra, color, marker in points:
        ax.scatter(ca * 100, ra * 100, s=70, color=color, marker=marker,
                   edgecolor="k", lw=0.6, zorder=3, label=label)
    # Pareto front: no other point has both higher clean and higher robust accuracy
    pts = sorted([(p[1], p[2]) for p in points], key=lambda t: -t[0])
    front, best = [], -1
    for ca, ra in pts:
        if ra > best:
            front.append((ca, ra))
            best = ra
    if len(front) > 1:
        ax.plot([p[0] * 100 for p in front], [p[1] * 100 for p in front],
                color=PALETTE["grey"], ls="--", lw=1.2, label="帕累托前沿")
    ax.set_xlabel("干净测试准确率 / %")
    ax.set_ylabel("鲁棒准确率 / %（PGD-20，ε=8/255）")
    ax.set_title("图6  鲁棒性–精度权衡：多组超参形成的帕累托前沿")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, "fig06_tradeoff", outdir)


def collect_tradeoff_points():
    """Read every trained defence model's (clean, robust) pair from results/."""
    spec = [
        ("results/defense_std_r18.csv", "标准训练", PALETTE["grey"], "o"),
        ("results/defense_at_r18_eps2_s7.csv", "AT-7 ε=2/255", PALETTE["sky"], "s"),
        ("results/defense_at_r18_eps4_s7.csv", "AT-7 ε=4/255", PALETTE["blue"], "^"),
        ("results/defense_at_r18_eps8_s7.csv", "AT-7 ε=8/255", PALETTE["green"], "D"),
        ("results/defense_at_r18_eps8_s10.csv", "AT-10 ε=8/255", PALETTE["orange"], "P"),
        ("results/defense_at_r18_eps8_s20.csv", "AT-20 ε=8/255", PALETTE["vermillion"], "X"),
        ("results/defense_trades_r18_b1.csv", "TRADES β=1", PALETTE["purple"], "*"),
        ("results/defense_trades_r18_b6.csv", "TRADES β=6", "#8B0000", "h"),
    ]
    out = []
    for path, label, color, marker in spec:
        df = _read(path)
        if df is None:
            continue
        atk_col = "pgd20" if (df.attack == "pgd20").any() else "pgd"
        d = df[(df.attack == atk_col) & (df.eps == 8)]
        if not len(d):
            continue
        out.append((label, d.clean_acc.iloc[0], d.robust_acc.iloc[0], color, marker))
    return out


# --------------------------------------------------------------------------- #
# 图 7  攻击强度（步数）与鲁棒准确率
# --------------------------------------------------------------------------- #
def fig_convergence(csvs=None, outdir="figures"):
    if csvs is None:
        csvs = sorted(glob.glob("results/convergence_*.csv"))
    if not csvs:
        print("  [skip] no convergence csv")
        return
    label_map = {
        "std_r18": ("标准训练", PALETTE["grey"], "o", ":"),
        "at_r18_eps8_s7": ("AT-7 ε=8/255", PALETTE["green"], "D", "-"),
        "at_r18_eps8_s10": ("AT-10 ε=8/255", PALETTE["orange"], "P", "--"),
        "at_r18_eps8_s20": ("AT-20 ε=8/255", PALETTE["vermillion"], "X", "-."),
        "at_r18_eps4_s7": ("AT-7 ε=4/255", PALETTE["blue"], "^", "--"),
        "at_r18_eps2_s7": ("AT-7 ε=2/255", PALETTE["sky"], "s", "--"),
        "trades_r18_b6": ("TRADES β=6", "#8B0000", "h", "-"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for path in csvs:
        df = _read(path)
        if df is None:
            continue
        tag = os.path.basename(path)[len("convergence_"):-4]
        label, color, marker, ls = label_map.get(tag, (tag, None, "o", "-"))
        ax.plot(df.steps, df.robust_acc * 100, marker=marker, color=color, ls=ls,
                lw=1.5, ms=5, label=label)
    ax.set_xlabel("PGD 迭代步数")
    ax.set_ylabel("鲁棒准确率 / %")
    ax.set_title("图7  攻击强度（PGD 步数）与鲁棒准确率：评估是否已收敛")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig07_convergence", outdir)


# --------------------------------------------------------------------------- #
# 图 8  对抗训练过程中的精度演化
# --------------------------------------------------------------------------- #
def R_clean(tag):
    import json
    p = os.path.join("results", "runs", tag + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("final_test_acc")


def R_robust(tag):
    p = os.path.join("results", f"defense_{tag}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    col = "pgd20" if (df.attack == "pgd20").any() else "pgd"
    d = df[(df.attack == col) & (df.eps == 8)]
    return float(d.robust_acc.iloc[0]) if len(d) else None

def fig_training_curves(outdir="figures"):
    """(a) clean-accuracy curves of the standard models, (b) where the AT models land."""
    std = _read("results/logs/std_r18.csv")
    vgg = _read("results/logs/std_vgg16.csv")
    if std is None:
        print("  [skip] no training logs")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].plot(std.epoch, std.test_acc * 100, marker="o", ms=5, lw=1.6,
                 color=PALETTE["blue"], label="ResNet-18 标准训练")
    if vgg is not None:
        axes[0].plot(vgg.epoch, vgg.test_acc * 100, marker="s", ms=5, lw=1.6,
                     color=PALETTE["orange"], label="VGG-16 标准训练")
    axes[0].set_xlabel("训练 epoch")
    axes[0].set_ylabel("干净测试准确率 / %")
    axes[0].set_title("(a) 标准训练的收敛过程")
    axes[0].legend(fontsize=8.5)
    axes[0].set_ylim(0, 95)

    # (b) clean vs robust accuracy of every trained defence model
    items = [("std_r18", "标准训练", PALETTE["grey"]),
             ("at_r18_eps4_s7", "AT-7 ε=4", PALETTE["blue"]),
             ("at_r18_eps8_s7", "AT-7 ε=8", PALETTE["green"]),
             ("at_r18_eps8_s10", "AT-10 ε=8", PALETTE["orange"]),
             ("at_r18_eps8_s20", "AT-20 ε=8", PALETTE["vermillion"])]
    labels, clean_vals, rob_vals, colors = [], [], [], []
    for tag, label, color in items:
        c = R_clean(tag)
        if c is None:
            continue
        labels.append(label)
        clean_vals.append(c * 100)
        rob_vals.append(100 * (R_robust(tag) or 0.0))
        colors.append(color)
    x = np.arange(len(labels))
    axes[1].bar(x - 0.19, clean_vals, 0.38, label="干净准确率", color="#BBBBBB",
                edgecolor="k", lw=0.4)
    axes[1].bar(x + 0.19, rob_vals, 0.38, label="鲁棒准确率 (PGD-20, ε=8/255)",
                color=PALETTE["green"], edgecolor="k", lw=0.4, hatch="//")
    for xi, (c, r) in enumerate(zip(clean_vals, rob_vals)):
        axes[1].text(xi - 0.19, c + 1, f"{c:.1f}", ha="center", fontsize=7.5)
        axes[1].text(xi + 0.19, r + 1, f"{r:.1f}", ha="center", fontsize=7.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylabel("准确率 / %")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) 一次短程对抗训练带来的精度迁移")
    axes[1].legend(fontsize=8.5)
    fig.suptitle("图8  训练过程与对抗训练的精度代价", y=1.02)
    fig.tight_layout()
    save(fig, "fig08_training_curves", outdir)


def fig_adaptive(csvs=None, outdir="figures"):
    """图 12: 五种攻击下的鲁棒准确率（自适应攻击评估）。"""
    if csvs is None:
        csvs = sorted(glob.glob("results/defense_*.csv"))
    if not csvs:
        print("  [skip] no defense csv")
        return
    labels = {"pgd20": "PGD-20\n(CE)", "pgd20_dlr_r3": "PGD-20\n+DLR+3重启",
              "pgd20_targeted": "目标化\nPGD-20", "apgd40_dlr": "APGD-40\n(DLR)",
              "autoattack_lite": "AutoAttack\n-lite"}
    order = list(labels)
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    tags = [os.path.basename(c)[len("defense_"):-4] for c in csvs]
    colors = [PALETTE["grey"], PALETTE["green"], PALETTE["blue"], PALETTE["orange"],
              PALETTE["purple"], PALETTE["vermillion"]]
    width = 0.8 / max(len(tags), 1)
    for k, (path, tag) in enumerate(zip(csvs, tags)):
        df = _read(path)
        if df is None:
            continue
        vals = []
        for key in order:
            d = df[(df.attack == key) & (df.eps == 8)] if "eps" in df else df[df.attack == key]
            vals.append(float(d.robust_acc.iloc[0]) * 100 if len(d) else np.nan)
        x = np.arange(len(order)) + (k - (len(tags) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, label=tag, color=colors[k % len(colors)],
               edgecolor="k", lw=0.4)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([labels[k] for k in order], fontsize=8.5)
    ax.set_ylabel("鲁棒准确率 / %")
    ax.set_ylim(0, 100)
    ax.set_title("图12  自适应攻击下的鲁棒准确率（ε=8/255）")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig12_adaptive", outdir)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures")
    args = p.parse_args(argv)
    setup()
    fig_asr_vs_eps(outdir=args.outdir)
    fig_norm_vs_eps(outdir=args.outdir)
    fig_cost(outdir=args.outdir)
    fig_transfer(outdir=args.outdir)
    fig_acc_vs_eps(outdir=args.outdir)
    fig_tradeoff(outdir=args.outdir)
    fig_convergence(outdir=args.outdir)
    fig_training_curves(outdir=args.outdir)
    fig_adaptive(outdir=args.outdir)


if __name__ == "__main__":
    main()
