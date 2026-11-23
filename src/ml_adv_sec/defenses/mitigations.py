import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from ..models import art_classifier


def squeeze(x, domain, quantum):
    if quantum <= 0:
        raise ValueError("Quantization step must be positive")
    return np.clip(np.round(x / quantum) * quantum, domain.low, domain.high).astype(np.float32)


def sanitize(model, x, y):
    from art.defences.detector.poison import ActivationDefence
    defence = ActivationDefence(art_classifier(model), x, np.eye(2, dtype=np.float32)[y])
    report, clean = defence.detect_poison(nb_clusters=2, nb_dims=min(10, x.shape[1]), reduce="PCA", cluster_analysis="smaller")
    mask = np.asarray(clean, dtype=bool)
    if len(np.unique(y[mask])) != 2 or mask.sum() < 4:
        raise ValueError("Activation filter removed a class or too many records")
    return mask, report


class Detector:
    """Training-only Mahalanobis and input-gradient quantile detector."""

    @staticmethod
    def gradients(model, x):
        model.eval()
        z = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        grad = torch.autograd.grad(model(z)[:, 1].sum(), z)[0]
        return torch.linalg.vector_norm(grad, dim=1).detach().numpy()

    def fit(self, model, x):
        self.covariance = LedoitWolf().fit(x)
        self.distance_cutoff = float(np.quantile(self.covariance.mahalanobis(x), 0.99))
        self.gradient_cutoff = float(np.quantile(self.gradients(model, x), 0.99))
        return self

    def flags(self, model, x):
        return (self.covariance.mahalanobis(x) > self.distance_cutoff) | (self.gradients(model, x) > self.gradient_cutoff)

    def conservative(self, model, x, probabilities):
        result = probabilities.copy()
        result[self.flags(model, x)] = 1.0
        return result
