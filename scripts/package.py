"""Build the two submission packages required by the assignment.

    学号_姓名_作业一.zip   作业二.zip
    ├── README.md          ├── README.md
    ├── requirements.txt   ├── requirements.txt
    ├── src/ ...           ├── src/ ...
    ├── tools/ tests/      ├── tools/ tests/
    ├── figures/           ├── figures/
    ├── results/           ├── results/
    ├── checkpoints/       ├── checkpoints/
    └── 报告.docx           └── 报告.docx

Usage:  python scripts/package.py --sid 2024123456 --name 张三
"""
import argparse
import os
import zipfile

CODE_DIRS = ["src", "tools", "tests", "scripts", "report"]
TOP_FILES = ["README.md", "requirements.txt"]

FIG_ATTACK = [
    "fig01_asr_vs_eps", "fig02_perturbation_vs_eps", "fig03_cost",
    "fig04_transfer_matrix", "fig09_triptych", "fig10_confidence",
]
FIG_DEFENCE = [
    "fig05_acc_vs_eps", "fig06_tradeoff", "fig07_convergence",
    "fig08_training_curves", "fig11_loss_landscape", "fig12_adaptive",
]


def add_path(zf, path, arc=None):
    if not os.path.exists(path):
        print("  skip missing", path)
        return
    arc = arc or path
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".pyc", ".pyo", ".docx", ".pdf", ".png")):
                    continue
                p = os.path.join(root, f)
                zf.write(p, os.path.relpath(p, path).replace(os.sep, "/")
                         if False else os.path.join(arc, os.path.relpath(p, path)))
    else:
        zf.write(path, arc)


def build(zip_name, root, figures, checkpoints, extra_results):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in TOP_FILES:
            add_path(zf, os.path.join(root, f), f)
        for d in CODE_DIRS:
            add_path(zf, os.path.join(root, d), d)
        for fig in figures:
            for ext in (".pdf", ".png"):
                p = os.path.join(root, "figures", fig + ext)
                add_path(zf, p, os.path.join("figures", fig + ext))
        for ck in checkpoints:
            p = os.path.join(root, "checkpoints", ck)
            add_path(zf, p, os.path.join("checkpoints", ck))
        res = os.listdir(os.path.join(root, "results"))
        for r in extra_results:
            if r in res:
                add_path(zf, os.path.join(root, "results", r), os.path.join("results", r))
        for f in sorted(os.listdir(os.path.join(root, "results", "logs"))):
            if f.endswith(".csv"):
                add_path(zf, os.path.join(root, "results", "logs", f),
                         os.path.join("results", "logs", f))
        for f in sorted(os.listdir(os.path.join(root, "results", "runs"))):
            if f.endswith(".json"):
                add_path(zf, os.path.join(root, "results", "runs", f),
                         os.path.join("results", "runs", f))
        report = [f for f in os.listdir(os.path.join(root, "report")) if f.endswith(".docx")]
        for r in report:
            add_path(zf, os.path.join(root, "report", r), r)
    print("wrote", zip_name, f"{os.path.getsize(zip_name)/1024**2:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sid", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--outdir", default="submission")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    root = args.root

    res_attack = ["attack_grid_std_r18.csv", "attack_steps.csv", "cw_l2.csv",
                  "transfer_pgd8.csv", "transfer_mifgsm8.csv", "crosscheck_pgd.json",
                  "clean_std_r18.json"]
    res_def = [f for f in os.listdir(os.path.join(root, "results"))
               if f.startswith(("defense_", "convergence_"))] + ["clean_std_r18.json"]

    build(os.path.join(args.outdir, f"{args.sid}_{args.name}_作业一.zip"), root,
          FIG_ATTACK, ["std_r18.pt", "std_vgg16.pt"], res_attack)
    build(os.path.join(args.outdir, f"{args.sid}_{args.name}_作业二.zip"), root,
          FIG_DEFENCE, ["std_r18.pt", "at_r18_eps8_s7.pt", "at_r18_eps4_s7.pt"],
          res_def)


if __name__ == "__main__":
    main()
