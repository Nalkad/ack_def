"""Rebuild ``results/runs/<tag>.json`` from the per-epoch CSV when a run was
interrupted after the last epoch (the CSV is written every epoch, the JSON only
at the very end, so this recovers the run metadata without retraining).
"""
import glob
import json
import os

import pandas as pd

from src.common import device_report

for csv_path in sorted(glob.glob("results/logs/*.csv")):
    tag = os.path.basename(csv_path)[:-4]
    json_path = os.path.join("results", "runs", tag + ".json")
    df = pd.read_csv(csv_path)
    if df.empty or "epoch_time_s" not in df.columns:
        continue
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("total_train_time_s"):
            continue
    total = float(df.epoch_time_s.sum())
    out = {
        "tag": tag,
        "reconstructed_from": csv_path,
        "epochs": int(df.epoch.max()),
        "final_test_acc": float(df.test_acc.iloc[-1]),
        "total_train_time_s": round(total, 1),
        "epoch_time_s": round(total / len(df), 2),
        "peak_mem_gb": float(df.peak_mem_gb.max()),
        "robust_probe_acc": (float(df.robust_probe_acc.iloc[-1])
                             if "robust_probe_acc" in df and pd.notna(df.robust_probe_acc.iloc[-1])
                             else None),
        "env": device_report(),
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote", json_path, out["final_test_acc"], out["epoch_time_s"])
