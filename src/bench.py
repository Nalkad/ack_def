"""Quick speed benchmark: decides the epoch budget of the reported runs."""
import time

import torch

from src.data import denorm
from src.models import PixelModel, build_model

torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0))


def bench(model, bs=128, iters=12, adv_steps=0, eps=8 / 255):
    model.train()
    x = torch.rand(bs, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (bs,), device=dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # warm-up
    for _ in range(3):
        out = model(x)
        torch.nn.functional.cross_entropy(out, y).backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        xa = x
        if adv_steps:
            xa = x.clone()
            for _s in range(adv_steps):
                xa.requires_grad_(True)
                loss = torch.nn.functional.cross_entropy(model(xa), y)
                g, = torch.autograd.grad(loss, xa)
                xa = (xa.detach() + 2.5 * eps / adv_steps * g.sign()).clamp(x - eps, x + eps).clamp(0, 1)
        out = model(xa)
        torch.nn.functional.cross_entropy(out, y).backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt


for name in ("resnet18", "resnet34", "vgg16"):
    net = build_model(name).to(dev)
    model = PixelModel(net).to(dev)
    n = sum(p.numel() for p in model.parameters())
    t = bench(model)
    t7 = bench(model, adv_steps=7)
    t10 = bench(model, adv_steps=10)
    print(f"{name:9s} {n/1e6:5.2f}M  fwd+bwd {t*1000:6.1f} ms/iter | "
          f"PGD-7 {t7*1000:6.1f} | PGD-10 {t10*1000:6.1f} | "
          f"epoch@391it clean {t*391/60:5.1f} min  AT7 {t7*391/60:5.1f} min")
