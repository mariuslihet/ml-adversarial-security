import numpy as np
import torch
from torch import nn

from ..models import art_classifier, score


def generate(model, x, y, domain, cfg, method="pgd", eps=None, norm=np.inf, return_raw=False):
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, HopSkipJump
    eps = cfg["epsilon"] if eps is None else eps
    estimator = art_classifier(model)
    labels = np.eye(2, dtype=np.float32)[y]
    if method == "fgsm":
        attack = FastGradientMethod(estimator, norm=norm, eps=eps, batch_size=128)
        raw = attack.generate(x, y=labels)
    elif method == "cleverhans-fgsm":
        from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
        raw = fast_gradient_method(model, torch.from_numpy(x), eps, norm, y=torch.from_numpy(y)).detach().numpy()
    elif method == "pgd":
        best = x.copy()
        original_p = np.clip(score(model, x), 1e-7, 1 - 1e-7)
        best_loss = -(y * np.log(original_p) + (1 - y) * np.log(1 - original_p))
        for fraction in cfg["step_fractions"]:
            attack = ProjectedGradientDescent(estimator, norm=norm, eps=eps, eps_step=eps * fraction, max_iter=cfg["steps"], num_random_init=cfg["restarts"], batch_size=128, verbose=False)
            candidate = attack.generate(x, y=labels, mask=(~domain.immutable).astype(np.float32))
            candidate = domain.project(candidate, x, eps, norm)
            p = np.clip(score(model, candidate), 1e-7, 1 - 1e-7)
            losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            take = losses > best_loss
            best[take], best_loss[take] = candidate[take], losses[take]
        raw = best
    elif method == "cw":
        attack = CarliniL2Method(estimator, max_iter=cfg["cw_steps"], binary_search_steps=3, batch_size=32, verbose=False)
        raw = attack.generate(x, y=labels)
        norm = 2
    elif method == "hsj":
        # Decision-only ART attack supports tabular vectors; SquareAttack expects images.
        from art.estimators.classification import BlackBoxClassifier
        queries = [0]
        def decisions(a):
            queries[0] += len(a)
            p = score(model, a)
            return np.eye(2, dtype=np.float32)[(p >= 0.5).astype(int)]
        blackbox = BlackBoxClassifier(decisions, input_shape=(57,), nb_classes=2)
        attack = HopSkipJump(blackbox, norm=norm, max_iter=cfg["hsj_steps"], max_eval=100, init_eval=20, init_size=20, verbose=False)
        raw = attack.generate(x, y=labels, mask=np.broadcast_to(~domain.immutable, x.shape).astype(np.float32))
    else:
        raise ValueError(f"Unknown attack: {method}")
    budget = None if method == "cw" else eps
    projected = domain.project(raw, x, budget, norm)
    stats = {"method": method, "norm": "linf" if norm == np.inf else "l2", "epsilon": budget, "raw_valid_fraction": float(domain.valid(raw, x, budget, norm).mean()), "projected_valid_fraction": float(domain.valid(projected, x, budget, norm).mean()), "max_linf": float(np.abs(projected - x).max()), "mean_l2": float(np.linalg.norm(projected - x, axis=1).mean())}
    if method == "hsj":
        stats["decision_queries"] = queries[0]
    if method == "cw":
        eligible = score(model, x) >= 0.5
        success = eligible & (score(model, projected) < 0.5)
        stats["optimization_eligible"] = int(eligible.sum())
        stats["optimization_successes"] = int(success.sum())
        stats["successful_l2"] = np.linalg.norm(projected[success] - x[success], axis=1).tolist()
    return (projected, stats, raw) if return_raw else (projected, stats)


def bpda(model, x, y, domain, cfg, eps):
    """PGD through exact quantization forward with identity backward (STE/BPDA)."""
    original = torch.from_numpy(x)
    z = original.clone()
    for _ in range(cfg["steps"]):
        z.requires_grad_(True)
        squeezed = torch.round(z / cfg["quantum"]) * cfg["quantum"]
        low = torch.as_tensor(domain.low, dtype=z.dtype)
        high = torch.as_tensor(domain.high, dtype=z.dtype)
        squeezed = torch.maximum(torch.minimum(squeezed, high), low)
        forward = z + (squeezed - z).detach()
        loss = nn.functional.cross_entropy(model(forward), torch.from_numpy(y))
        grad = torch.autograd.grad(loss, z)[0]
        candidate = z.detach() + eps / cfg["steps"] * grad.sign()
        z = torch.from_numpy(domain.project(candidate.numpy(), x, eps))
    return z.numpy()
