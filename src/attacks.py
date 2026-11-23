"""Adversarial attacks, implemented from scratch with ``torch.autograd``.

Nothing in this file calls an attack library.  Every attack is written as an
explicit loop over (i) a forward pass, (ii) a backward pass, (iii) an explicit
update rule (gradient sign / Adam step / momentum), and (iv) an explicit
projection onto the feasible set ``{x' : ||x'-x||_inf <= eps} ∩ [0,1]^d``.

Input convention
----------------
Every attack operates on the **[0,1] pixel space** tensor ``x01`` and is given a
``PixelModel`` (see ``models.PixelModel``) whose forward normalises internally.
Consequently:

* ``eps`` is expressed in pixel units -- ``eps = 8/255`` is literally "at most 8
  grey levels per channel";
* the projection is ``clamp(x + d, 0, 1)`` after ``clamp(d, -eps, eps)``;
* ``L2``/``Linf`` norms reported downstream are computed in the same pixel space.

Reference:
  Goodfellow et al. 2015 (FGSM); Kurakin et al. 2017 (BIM); Madry et al. 2018
  (PGD); Dong et al. 2018 (MI-FGSM); Carlini & Wagner 2017 (C&W-L2);
  Croce & Hein 2020 (APGD / DLR, the backbone of AutoAttack).
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# losses (all in "maximise me" form)
# --------------------------------------------------------------------------- #
def ce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y, reduction="none")


def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Difference-of-Logits-Ratio loss (Croce & Hein 2020), per-sample.

        DLR = (max_{j != y} z_j - z_y) / (z_(1) - z_(3))

    which is scale invariant and therefore much better behaved than CE when the
    logit scale is arbitrary.  Maximising it *is* the attack.
    """
    n_cls = logits.size(1)
    z_sorted, idx = logits.sort(dim=1, descending=True)
    rank = (idx == y.view(-1, 1)).float().argmax(dim=1)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    z_max_other = torch.where(rank == 0, z_sorted[:, 1], z_sorted[:, 0])
    denom = z_sorted[:, 0] - z_sorted[:, min(3, n_cls - 1)] + 1e-12
    return (z_max_other - z_y) / denom


def margin_loss(logits: torch.Tensor, y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Targeted margin, already in "maximise me" form:  z_target - z_true.

    Maximising it pushes the sample towards ``target``.  (Using the opposite sign
    here -- as an earlier version of this file did -- makes a *targeted* attack
    behave like a very weak untargeted one, which is exactly the kind of silent
    sign error the unit tests in tests/test_attacks.py are meant to catch.)
    """
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    z_t = logits.gather(1, target.view(-1, 1)).squeeze(1)
    return z_t - z_y


LOSSES = {"ce": ce_loss, "dlr": dlr_loss}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
          loss_fn: Callable, target: Optional[torch.Tensor] = None,
          eot: int = 1) -> torch.Tensor:
    """d(loss)/dx for the *sum* over the batch, computed w.r.t. ``x``."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        if target is not None:
            loss = margin_loss(model(x), y, target).sum()
        else:
            loss = loss_fn(model(x), y).sum()
        grad, = torch.autograd.grad(loss, x)
    if eot > 1:
        # expectation over transformation: average gradients at random
        # transformations of the input, used only for stochastic defences
        for _ in range(eot - 1):
            x2 = x.detach().requires_grad_(True)
            if target is not None:
                l2 = margin_loss(model(x2), y, target).sum()
            else:
                l2 = loss_fn(model(x2), y).sum()
            g2, = torch.autograd.grad(l2, x2)
            grad = grad + g2
        grad = grad / eot
    return grad


def project_linf(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Projection onto the L_inf ball and the valid-image box, in one step."""
    return torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)


def project_l2_ball(x_adv: torch.Tensor, x: torch.Tensor, radius: float) -> torch.Tensor:
    d = x_adv - x
    flat = d.flatten(1)
    norm = flat.norm(p=2, dim=1).clamp_min(1e-12)
    factor = (radius / norm).clamp(max=1.0).view(-1, *([1] * (d.dim() - 1)))
    return (x + d * factor).clamp(0.0, 1.0)


def linf_norm(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return (x - x0).flatten(1).abs().max(dim=1).values


def l2_norm(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return (x - x0).flatten(1).norm(p=2, dim=1)


# --------------------------------------------------------------------------- #
# 1. FGSM -- baseline only, never used as "our method"
# --------------------------------------------------------------------------- #
def fgsm(model, x01, y, eps, loss="ce"):
    grad = _grad(model, x01, y, LOSSES[loss])
    return (x01 + eps * grad.sign()).clamp(0, 1).detach()


# --------------------------------------------------------------------------- #
# 2. PGD / BIM (L_inf) -- iterative sign attack with random start
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pgd_linf(model, x01, y, eps, steps=10, step_size=None, random_start=True,
             restarts=1, loss="ce", target=None, generator=None, eot=1):
    """Madry-style PGD (a.k.a. BIM with random start).

    x^{t+1} = Proj(x^t + alpha * sign(grad_x L(x^t, y)))
    Proj(v)  = clamp(clamp(v, x-eps, x+eps), 0, 1)
    """
    if step_size is None:
        step_size = 2.5 * eps / steps
    loss_fn = LOSSES[loss]
    best_adv, best_loss = x01.clone(), torch.full((x01.size(0),), -1e30, device=x01.device)
    for _ in range(restarts):
        if random_start:
            noise = torch.empty_like(x01).uniform_(-eps, eps, generator=generator)
            x_adv = (x01 + noise).clamp(0, 1)
        else:
            x_adv = x01.clone()
        for _step in range(steps):
            grad = _grad(model, x_adv, y, loss_fn, target=target, eot=eot)
            x_adv = project_linf(x_adv + step_size * grad.sign(), x01, eps)
        with torch.no_grad():
            if target is not None:
                cur = margin_loss(model(x_adv), y, target)
            else:
                cur = loss_fn(model(x_adv), y)
        # keep, per sample, the restart with the largest loss
        better = cur > best_loss
        best_loss = torch.where(better, cur, best_loss)
        best_adv[better] = x_adv[better]
    return best_adv.detach()


# --------------------------------------------------------------------------- #
# 3. MI-FGSM -- momentum, transfer-oriented
# --------------------------------------------------------------------------- #
def mi_fgsm(model, x01, y, eps, steps=10, decay=1.0, loss="ce"):
    """Dong et al. 2018.  g_{t+1} = mu*g_t + grad/||grad||_1 ; x += alpha*sign(g)."""
    alpha = eps / steps
    loss_fn = LOSSES[loss]
    x_adv = x01.clone()
    g = torch.zeros_like(x01)
    for _ in range(steps):
        grad = _grad(model, x_adv, y, loss_fn)
        g = decay * g + grad / (grad.flatten(1).abs().sum(dim=1).clamp_min(1e-12)
                                .view(-1, *([1] * (x01.dim() - 1))))
        x_adv = project_linf(x_adv + alpha * g.sign(), x01, eps)
    return x_adv.detach()


# --------------------------------------------------------------------------- #
# 4. C&W L2 -- optimisation-based, minimal-norm
# --------------------------------------------------------------------------- #
def cw_l2(model, x01, y, eps=None, c=(0.01, 1.0, 10.0), steps=40, lr=0.01,
          binary_search=4, num_classes=10, verbose=False):
    """Carlini & Wagner (2017) L2 attack, decoded with the tanh change of variable.

    We optimise  w  with  x_adv = 0.5*(tanh(w)+1)  so the box constraint holds by
    construction, minimising    ||x_adv - x||_2^2 + c * f(x_adv)
    where  f(x) = max(max_{j!=y} z_j - z_y, -kappa), kappa = 0.

    The confidence ``c`` is found per-sample by a coarse binary search on the
    geometric grid given by ``c``; the best (successful, smallest-distance)
    solution is kept.  No projection is needed because the parameterisation
    never leaves the box.
    """
    loss_fn = LOSSES["ce"]
    best_adv = x01.clone()
    best_dist = torch.full((x01.size(0),), 1e30, device=x01.device)
    done = torch.zeros(x01.size(0), dtype=torch.bool, device=x01.device)

    # coarse geometric grid for the confidence trade-off
    grid = torch.logspace(-2, 2, binary_search).tolist() if binary_search > 1 else list(c)

    for c_val in grid:
        x0 = x01.clone()
        w = torch.atanh((2 * x0 - 1).clamp(-0.999999, 0.999999)).clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            x_adv_c = 0.5 * (torch.tanh(w) + 1)
            logits = model(x_adv_c)
            z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
            other = logits.clone()
            other.scatter_(1, y.view(-1, 1), -1e9)
            z_max = other.max(dim=1).values
            # C&W margin term, in *minimisation* form:  f = max(z_y - z_max, -kappa)
            # with kappa = 0.  For a correctly classified sample this is positive and
            # the gradient pushes the margin down; for an already-successful sample it
            # clamps to 0 so that only the distance term keeps acting.
            # (Writing it as clamp(z_max - z_y, min=0) -- the opposite sign -- gives a
            # *zero* gradient on every correctly classified sample and the attack
            # silently degenerates into "return the clean image"; see the note in the
            # report's 2.5 节.)
            f = torch.clamp(z_y - z_max, min=0.0)
            dist = (x_adv_c - x01).flatten(1).pow(2).sum(dim=1)
            loss = (dist + c_val * f).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x_adv_c = 0.5 * (torch.tanh(w) + 1)
            pred = model(x_adv_c).argmax(1)
            success = pred != y
            dist = (x_adv_c - x01).flatten(1).norm(p=2, dim=1)
            take = success & (dist < best_dist) & (~done)
            best_dist = torch.where(take, dist, best_dist)
            best_adv[take] = x_adv_c[take]
            # once a sample succeeds we keep it (C&W alternates c to minimise dist)
            if verbose:
                print(f"  c={c_val:.3g} success={success.float().mean().item():.3f}")
    # samples for which every c failed are returned as the (unmodified) clean image,
    # so that they are counted as failures by the success criterion downstream
    return best_adv.detach()


# --------------------------------------------------------------------------- #
# 5. APGD -- the engine of AutoAttack (self-implemented)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def apgd(model, x01, y, eps, steps=100, restarts=5, loss="dlr", rho=0.75,
         seed=0, target=None, eot=1, initial_step=None):
    """Auto-PGD (Croce & Hein 2020), L_inf version.

    Three ingredients that separate it from PGD:
      * momentum on the normalised gradient:  g_t = grad + rho * g_{t-1};
      * per-sample adaptive step size: step is multiplied by 1.75 after an
        improvement of the objective and halved otherwise (bounded by eps);
      * random restarts, keeping the worst (best-for-the-attacker) point.
    """
    gen = torch.Generator(device=x01.device if x01.is_cuda else "cpu")
    gen.manual_seed(seed)
    loss_fn = LOSSES[loss]
    shape = x01.shape
    best_adv = x01.clone()
    best_loss = torch.full((x01.size(0),), -1e30, device=x01.device)

    for r in range(restarts):
        noise = torch.empty(shape, device=x01.device).uniform_(-eps, eps, generator=gen)
        x_adv = (x01 + noise).clamp(0, 1)
        step = torch.full((x01.size(0),), eps if initial_step is None else initial_step,
                          device=x01.device)
        g_prev = torch.zeros_like(x01)
        l_prev = torch.full((x01.size(0),), -1e30, device=x01.device)
        l_best_r = torch.full((x01.size(0),), -1e30, device=x01.device)
        x_best_r = x_adv.clone()
        for it in range(steps):
            grad = _grad(model, x_adv, y, loss_fn, target=target, eot=eot)
            # L2 normalise per sample (APGD uses the L2-normalised ascent direction)
            gnorm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
            grad = grad / gnorm.view(-1, *([1] * (x01.dim() - 1)))
            if it > 0:
                grad = grad + rho * g_prev
            g_prev = grad
            x_new = project_linf(x_adv + step.view(-1, 1, 1, 1) * grad.sign(), x01, eps)
            with torch.no_grad():
                if target is not None:
                    l_new = margin_loss(model(x_new), y, target)
                else:
                    l_new = loss_fn(model(x_new), y)
            improve = l_new > l_prev
            step = torch.where(improve, (step * 1.75).clamp(max=eps), step / 2.0)
            step = step.clamp_min(1e-8)
            l_prev = l_new
            x_adv = x_new
            better = l_new > l_best_r
            l_best_r = torch.where(better, l_new, l_best_r)
            x_best_r[better] = x_new[better]
        n_better = l_best_r > best_loss
        best_loss = torch.where(n_better, l_best_r, best_loss)
        best_adv[n_better] = x_best_r[n_better]
    return best_adv.detach()


# --------------------------------------------------------------------------- #
# 6. "AutoAttack-lite": the ensemble used for the final robustness number
# --------------------------------------------------------------------------- #
@torch.no_grad()
def autoattack_lite(model, x01, y, eps, steps=100, restarts=5, seed=0, verbose=False):
    """APGD-CE, APGD-DLR and a targeted APGD, union of successes.

    This is *our* re-implementation of the AutoAttack protocol (Croce & Hein 2020)
    on top of :func:`apgd`; it deliberately keeps the ``FAB`` and ``Square``
    components out (they are the expensive query-based parts) and instead adds a
    targeted APGD towards the least-likely class, which is cheap and covers the
    same failure mode for the models studied here.
    """
    adv_ce = apgd(model, x01, y, eps, steps=steps, restarts=restarts, loss="ce", seed=seed + 1)
    adv_dlr = apgd(model, x01, y, eps, steps=steps, restarts=restarts, loss="dlr", seed=seed + 2)

    with torch.no_grad():
        clean_logits = model(x01)
        least = clean_logits.clone()
        least.scatter_(1, y.view(-1, 1), 1e9)
        target = least.argmin(dim=1)
    adv_t = apgd(model, x01, y, eps, steps=max(20, steps // 2), restarts=max(1, restarts // 2),
                 loss="dlr", seed=seed + 3, target=target)

    out = x01.clone()
    with torch.no_grad():
        for cand in (adv_ce, adv_dlr, adv_t):
            wrong = model(cand).argmax(1) != y
            take = wrong & torch.eq(out, x01).flatten(1).all(dim=1)
            out[take] = cand[take]
        if verbose:
            ok = model(x01).argmax(1) == y
            asr = ((model(out).argmax(1) != y) & ok).float().sum() / ok.float().sum().clamp_min(1)
            print(f"  autoattack-lite ASR = {asr.item():.4f}")
    return out


# --------------------------------------------------------------------------- #
# registry used by the evaluation scripts
# --------------------------------------------------------------------------- #
def build_attack(name: str, **kw):
    """Return a callable ``f(model, x01, y) -> x_adv`` for a given attack name."""
    name = name.lower()

    def _fgsm(model, x, y):
        return fgsm(model, x, y, kw.get("eps", 8 / 255), loss=kw.get("loss", "ce"))

    def _pgd(model, x, y):
        return pgd_linf(model, x, y, kw.get("eps", 8 / 255), steps=kw.get("steps", 10),
                        step_size=kw.get("step_size"), random_start=kw.get("random_start", True),
                        restarts=kw.get("restarts", 1), loss=kw.get("loss", "ce"))

    def _mi(model, x, y):
        return mi_fgsm(model, x, y, kw.get("eps", 8 / 255), steps=kw.get("steps", 10),
                       decay=kw.get("decay", 1.0), loss=kw.get("loss", "ce"))

    def _cw(model, x, y):
        return cw_l2(model, x, y, steps=kw.get("steps", 40), lr=kw.get("lr", 0.01),
                     binary_search=kw.get("binary_search", 4))

    def _apgd(model, x, y):
        return apgd(model, x, y, kw.get("eps", 8 / 255), steps=kw.get("steps", 100),
                    restarts=kw.get("restarts", 5), loss=kw.get("loss", "dlr"),
                    seed=kw.get("seed", 0))

    def _aa(model, x, y):
        return autoattack_lite(model, x, y, kw.get("eps", 8 / 255), steps=kw.get("steps", 100),
                               restarts=kw.get("restarts", 5), seed=kw.get("seed", 0))

    table = {"fgsm": _fgsm, "pgd": _pgd, "mi_fgsm": _mi, "cw_l2": _cw,
             "apgd": _apgd, "autoattack_lite": _aa}
    if name not in table:
        raise ValueError(f"unknown attack {name}; available: {sorted(table)}")
    return table[name]
