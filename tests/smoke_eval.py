"""Smoke test of the evaluation harness on 100 images (fast)."""
import time

import torch

from src.models import PixelModel, build_model
from src.run_experiments import EvalSet, attack_spec, run_one_attack

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ev = EvalSet(100, dev)
print("eval subset", ev.N, tuple(ev.x01.shape), ev.path)
m = PixelModel(build_model("resnet18")).to(dev).eval()
for name, kw in [("fgsm", {}), ("pgd", {"steps": 20}), ("mi_fgsm", {"steps": 10}),
                 ("apgd", {"steps": 40, "restarts": 1, "loss": "dlr"}),
                 ("cw_l2", {"steps": 10, "binary_search": 2})]:
    t = time.time()
    met, adv, succ, l2, linf = run_one_attack(m, ev, attack_spec(name, 8 / 255, 0, **kw), 8 / 255)
    print(f"{name:9s} ASR={met['asr']*100:6.2f}% Linf={met['mean_linf_succ']*255:5.2f} "
          f"RA={met['robust_acc']*100:6.2f}% {met['time_ms_per_img']:6.1f} ms/img "
          f"wall={time.time()-t:.1f}s")
print("ok")
