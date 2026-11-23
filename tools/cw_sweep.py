"""Quick sweep of C&W hyper-parameters on a small subset (the first attempt
converged too slowly to attack a trained model)."""
import time

import torch

from src import attacks as A
from src.evaluate import load_model
from src.run_experiments import EvalSet, run_one_attack

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ev = EvalSet(200, dev)
model = load_model("checkpoints/std_r18.pt", dev)
for lr, steps, bs in ((0.1, 60, 2), (0.2, 100, 2), (0.05, 100, 3), (0.2, 60, 3)):
    t0 = time.time()
    m, *_ = run_one_attack(model, ev, A.build_attack("cw_l2", steps=steps, lr=lr,
                                                     binary_search=bs), 8 / 255)
    print(f"lr={lr} steps={steps} bs={bs}: ASR={m['asr']*100:5.1f}% "
          f"L2={m['mean_l2_succ']:.2f} {m['time_ms_per_img']:.0f} ms/img "
          f"({time.time()-t0:.0f}s)", flush=True)
