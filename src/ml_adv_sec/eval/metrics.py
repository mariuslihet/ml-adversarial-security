import time

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score

from ..models import score


def metrics(model, x, y, threshold, transform=None):
    def predict(a):
        return score(model, transform(a) if transform else a)
    p = predict(x)
    predicted = p >= threshold
    negative, positive = y == 0, y == 1
    # Per-record wall-clock latency includes an optional inference defense.
    predict(x[:1])
    timings = []
    for a in x[: min(128, len(x))]:
        start = time.perf_counter_ns()
        predict(a[None])
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return {"accuracy": float(accuracy_score(y, predicted)), "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "f1": float(f1_score(y, predicted, zero_division=0)), "fpr": float(predicted[negative].mean()), "fnr": float((~predicted[positive]).mean()), "brier": float(brier_score_loss(y, p)), "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)), "n": len(y), "threshold": threshold}


def attack_metrics(model, original, adversarial, threshold, transform=None):
    clean = score(model, transform(original) if transform else original) >= threshold
    adv = score(model, transform(adversarial) if transform else adversarial) >= threshold
    eligible = int(clean.sum())
    return {"adversarial_recall": float(adv.mean()), "asr": float((~adv[clean]).mean()) if eligible else None, "eligible": eligible, "attack_n": len(original)}
