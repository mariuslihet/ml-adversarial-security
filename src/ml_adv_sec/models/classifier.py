import importlib.metadata
import subprocess
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from xgboost import XGBClassifier

from ..common import read_json, write_json, sha256, seed
from ..data import verify


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.1), nn.Linear(width, width))

    def forward(self, x):
        return torch.relu(x + self.block(x))


class MLP(nn.Module):
    def __init__(self, features=57, hidden=64):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), Residual(hidden))
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        logit = self.output(self.embedding(x))
        return torch.cat([torch.zeros_like(logit), logit], dim=1)


class BinaryLoss(nn.Module):
    def forward(self, logits, labels):
        target = labels[:, 1] if labels.ndim == 2 else labels
        return nn.functional.binary_cross_entropy_with_logits(logits[:, 1], target.float())


def art_classifier(model, clip_values=None):
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(model=model, loss=BinaryLoss(), input_shape=(57,), nb_classes=2, device_type="cpu", clip_values=clip_values)


def train_mlp(cfg, x, y, mode="clean", domain=None):
    seed(cfg["seed"], cfg["threads"])
    model = MLP(x.shape[1], cfg["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    x_tensor = torch.from_numpy(x.astype(np.float32))
    y_tensor = torch.from_numpy(y)
    for _ in range(cfg["epochs"]):
        # Avoid singleton BatchNorm batches without dropping real records.
        batches = list(torch.randperm(len(x)).split(cfg["batch_size"]))
        if len(batches) > 1 and len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = torch.cat([batches[-1], last])
        for ids in batches:
            clean, labels = x_tensor[ids], y_tensor[ids]
            if mode != "clean":
                model.eval()
                adv = clean.clone()
                for _ in range(cfg["steps"]):
                    adv.requires_grad_(True)
                    if mode == "trades":
                        stable = torch.softmax(model(clean).detach(), dim=1)
                        attack_loss = nn.functional.kl_div(torch.log_softmax(model(adv), 1), stable, reduction="batchmean")
                        # A clean initial point has zero KL gradient; use CE on the first step.
                        if torch.equal(adv.detach(), clean):
                            attack_loss = BinaryLoss()(model(adv), labels)
                    else:
                        attack_loss = BinaryLoss()(model(adv), labels)
                    grad = torch.autograd.grad(attack_loss, adv)[0]
                    candidate = adv.detach() + cfg["epsilon"] / max(cfg["steps"], 1) * grad.sign()
                    adv = torch.from_numpy(domain.project(candidate.numpy(), clean.numpy(), cfg["epsilon"]))
            model.train()
            optimizer.zero_grad()
            logits = model(clean)
            loss = BinaryLoss()(logits, labels)
            if mode == "pgd":
                loss = 0.5 * (loss + BinaryLoss()(model(adv), labels))
            elif mode == "trades":
                loss = loss + 3.0 * nn.functional.kl_div(torch.log_softmax(model(adv), 1), torch.softmax(logits, 1), reduction="batchmean")
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def train_tree(cfg, x, y):
    model = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, subsample=1, colsample_bytree=1, random_state=cfg["seed"], n_jobs=cfg["threads"], tree_method="hist", eval_metric="logloss")
    return model.fit(x, y)


def score(model, x):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32))[:, 1]).numpy()
    return model.predict_proba(x)[:, 1]


def calibrate(y, probabilities, max_fpr=0.01):
    negatives = probabilities[y == 0]
    if not len(negatives) or not (y == 1).any():
        raise ValueError("Calibration requires both classes")
    # Enumerate score boundaries, accounting for ties and >= decision semantics.
    candidates = np.unique(np.r_[0.0, negatives, np.nextafter(negatives, np.float32(np.inf)), 1.000001])
    for threshold in candidates:
        if np.mean(negatives >= threshold) <= max_fpr:
            return float(threshold)
    raise ValueError("No valid threshold")


def save(cfg, name, model, threshold, validation, extra=None):
    manifest = verify(cfg)
    directory = Path(cfg["root"]) / "models" / name
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".pt" if isinstance(model, nn.Module) else ".joblib"
    existing = list(directory.glob("v*.json"))
    version = f"v{max((int(p.stem[1:]) for p in existing), default=0) + 1:04d}"
    path = directory / (version + suffix)
    if suffix == ".pt":
        torch.save(model.state_dict(), path)
    else:
        joblib.dump(model, path)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    package = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))}
    metadata = {"name": name, "version": version, "weights": path.name, "weights_sha256": sha256(path), "threshold": threshold, "validation": validation, "config": cfg, "dataset_files": manifest["files"], "git_commit": commit.stdout.strip() if commit.returncode == 0 else None, "source_hashes": source_hashes, "packages": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "scikit-learn", "xgboost", "adversarial-robustness-toolbox", "cleverhans"]}, "extra": extra or {}}
    write_json(directory / (version + ".json"), metadata)
    write_json(directory / "latest.json", {"metadata": version + ".json"})
    return metadata


def restore(cfg, name, version=None):
    verify(cfg)
    directory = Path(cfg["root"]) / "models" / name
    metadata_name = version + ".json" if version else read_json(directory / "latest.json")["metadata"]
    if Path(metadata_name).name != metadata_name or not metadata_name.startswith("v"):
        raise ValueError("Invalid checkpoint version")
    metadata = read_json(directory / metadata_name)
    if metadata["config"] != cfg or metadata["dataset_files"] != verify(cfg)["files"]:
        raise ValueError("Checkpoint provenance does not match prepared data")
    path = directory / metadata["weights"]
    if sha256(path) != metadata["weights_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if path.suffix == ".pt":
        model = MLP(hidden=cfg["hidden"])
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        model.eval()
    else:
        model = joblib.load(path)
    return model, metadata["threshold"], metadata
