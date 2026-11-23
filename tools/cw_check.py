"""Quick check of the corrected C&W attack on a 200-image subset."""
import time

import torch

from src import attacks as A
from src.evaluate import load_model
from src.run_experiments import EvalSet, run_one_attack

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ev = EvalSet(200, dev)
model = load_model("checkpoints/std_r18.pt", dev)
for lr, steps, bs in ((0.01, 50, 3), (0.05, 50, 3)):
    t = time.time()
    m, *_ = run_one_attack(model, ev,
                           A.build_attack("cw_l2", steps=steps, lr=lr, binary_search=bs),
                           8 / 255)
    print("lr={} steps={} bs={}: ASR={:.1f}% L2={:.2f} {:.0f} ms/img ({:.0f}s)".format(
        lr, steps, bs, m["asr"] * 100, m["mean_l2_succ"], m["time_ms_per_img"], time.time() - t),
        flush=True)
