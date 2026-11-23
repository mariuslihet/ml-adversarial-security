import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"root", "data_path", "seed", "epochs", "epsilon", "steps"}
    if not isinstance(cfg, dict) or not required <= cfg.keys():
        raise ValueError("Incomplete configuration")
    if cfg["epochs"] < 1 or cfg["steps"] < 1 or cfg["epsilon"] <= 0:
        raise ValueError("epochs, steps and epsilon must be positive")
    if not 0 <= cfg["fpr"] < 1 or not all(0 < r < 0.5 for r in cfg["poison_ratios"]):
        raise ValueError("Invalid FPR or poisoning ratio")
    seed(cfg["seed"], cfg.get("threads", 1))
    return cfg


def seed(value, threads=1):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
