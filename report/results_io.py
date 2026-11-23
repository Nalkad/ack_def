"""Load every result file into a small, convenient object for the report builder."""
import glob
import json
import os

import numpy as np
import pandas as pd


def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


class Results:
    def __init__(self, root="results"):
        self.root = root
        self.runs = {}
        for p in glob.glob(os.path.join(root, "runs", "*.json")):
            tag = os.path.basename(p)[:-5]
            with open(p, encoding="utf-8") as f:
                self.runs[tag] = json.load(f)

        self.grid_r18 = _csv(os.path.join(root, "attack_grid_std_r18.csv"))
        self.grid_vgg = _csv(os.path.join(root, "attack_grid_std_vgg16.csv"))
        self.steps = _csv(os.path.join(root, "attack_steps.csv"))
        self.cw = _csv(os.path.join(root, "cw_l2.csv"))
        self.transfer = _csv(os.path.join(root, "transfer_pgd8.csv"))
        self.transfer_mi = _csv(os.path.join(root, "transfer_mifgsm8.csv"))
        self.crosscheck = None
        cc_path = os.path.join(root, "crosscheck_pgd.json")
        if os.path.exists(cc_path):
            with open(cc_path, encoding="utf-8") as f:
                self.crosscheck = json.load(f)
        self.defense = {}
        for p in glob.glob(os.path.join(root, "defense_*.csv")):
            tag = os.path.basename(p)[len("defense_"):-4]
            self.defense[tag] = pd.read_csv(p)
        self.convergence = {}
        for p in glob.glob(os.path.join(root, "convergence_*.csv")):
            tag = os.path.basename(p)[len("convergence_"):-4]
            self.convergence[tag] = pd.read_csv(p)
        self.logs = {}
        for p in glob.glob(os.path.join(root, "logs", "*.csv")):
            tag = os.path.basename(p)[:-4]
            self.logs[tag] = pd.read_csv(p)
        self.attack_meta = _csv(os.path.join(root, "attack_meta.csv"))

    # ---------------- convenience ----------------
    def run(self, tag, key, default=None):
        return self.runs.get(tag, {}).get(key, default)

    def clean_acc(self, tag):
        """Final clean test accuracy of a trained model."""
        if tag in self.runs:
            return self.runs[tag].get("final_test_acc")
        return None

    def grid_value(self, df, attack, eps, col):
        if df is None:
            return None
        d = df[(df.attack == attack) & (np.isclose(df.eps, eps))]
        return float(d[col].iloc[0]) if len(d) else None

    def defense_value(self, tag, attack, eps, col):
        df = self.defense.get(tag)
        if df is None:
            return None
        d = df[(df.attack == attack) & (np.isclose(df.eps, eps))]
        return float(d[col].iloc[0]) if len(d) else None

    def defense_row(self, tag, spec, eps, col):
        df = self.defense.get(tag)
        if df is None:
            return None
        d = df[(df.spec == spec) & (np.isclose(df.eps, eps))]
        return float(d[col].iloc[0]) if len(d) else None


def pct(x, nd=2):
    if x is None:
        return "--"
    return f"{x*100:.{nd}f}"


def num(x, nd=2):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def times(a, b):
    if a is None or b in (None, 0):
        return "--"
    return f"{a/b:.1f}"
