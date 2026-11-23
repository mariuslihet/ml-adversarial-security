import platform
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from ..common import read_json, write_json, sha256, seed
from ..data import load, Domain, verify
from ..models import train_mlp, train_tree, calibrate, score, save, restore
from ..attacks import generate, bpda, poison, trigger
from ..defenses import squeeze, sanitize, Detector
from .metrics import metrics, attack_metrics


class Calibrated(nn.Module):
    """Make attack argmax decisions match the validation-calibrated threshold."""

    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        if threshold > 1:
            self.offset = 1e30
        elif threshold <= 0:
            self.offset = -1e30
        else:
            # Match the actual float32 sigmoid decision, including scores rounded to 1.
            low, high = -100.0, 100.0
            for _ in range(64):
                middle = (low + high) / 2
                if float(torch.sigmoid(torch.tensor(middle, dtype=torch.float32))) >= threshold:
                    high = middle
                else:
                    low = middle
            boundary = np.float32(high)
            if float(torch.sigmoid(torch.tensor(boundary))) < threshold:
                boundary = np.nextafter(boundary, np.float32(np.inf))
            self.offset = float(np.nextafter(boundary, np.float32(-np.inf)))

    def forward(self, x):
        logits = self.model(x)
        return torch.stack([logits[:, 0], logits[:, 1] - self.offset], dim=1)


def register(cfg, name, model, data, extra=None):
    threshold = calibrate(data["y_val"], score(model, data["x_val"]), cfg["fpr"])
    validation = metrics(model, data["x_val"], data["y_val"], threshold)
    return save(cfg, name, model, threshold, validation, extra)


def train(cfg):
    data = load(cfg)
    for name, builder in [("clean", train_mlp), ("tree", train_tree)]:
        model = builder(cfg, data["x_train"], data["y_train"])
        register(cfg, name, model, data)


def attacked_slice(data, cfg):
    ids = np.flatnonzero(data["y_test"] == 1)[:cfg["attack_samples"]]
    if not len(ids):
        raise ValueError("No malicious test records")
    return data["x_test"][ids], data["y_test"][ids]


def save_arrays(cfg, name, arrays, details):
    directory = Path(cfg["root"]) / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".npz")
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix(".json"), {"sha256": sha256(path), "details": details, "prepared": verify(cfg)["files"]})


def load_arrays(cfg, name):
    path = Path(cfg["root"]) / "artifacts" / (name + ".npz")
    info = read_json(path.with_suffix(".json"))
    if sha256(path) != info["sha256"] or info["prepared"] != verify(cfg)["files"]:
        raise ValueError(f"Artifact integrity failure: {name}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, info["details"]


def evasion(cfg):
    data, domain = load(cfg), Domain(cfg)
    model, threshold, metadata = restore(cfg, "clean")
    calibrated = Calibrated(model, threshold).eval()
    x, y = attacked_slice(data, cfg)
    for method, norm in [("fgsm", np.inf), ("cleverhans-fgsm", np.inf), ("pgd", np.inf), ("pgd", 2), ("cw", 2), ("hsj", np.inf)]:
        seed(cfg["seed"], cfg["threads"])
        adversarial, details, raw = generate(calibrated, x, y, domain, cfg, method, norm=norm, return_raw=True)
        if not domain.valid(adversarial, x, None if method == "cw" else cfg["epsilon"], norm).all():
            raise ValueError("Projected attack violates domain constraints")
        details["source_checkpoint"] = metadata["weights_sha256"]
        ids = data["id_test"][data["y_test"] == 1][:len(x)]
        save_arrays(cfg, f"evasion-{method}-{'linf' if norm == np.inf else 'l2'}", {"original": x, "adversarial": adversarial, "raw": raw, "labels": y, "record_ids": ids}, details)


def poison_name(kind, ratio):
    # Preserve familiar names without merging distinct user-supplied fractions.
    value = f"{ratio:.3f}" if float(f"{ratio:.3f}") == ratio else repr(ratio)
    return f"{kind}-{value}"


def poisoning(cfg):
    data, domain = load(cfg), Domain(cfg)
    clean, _, _ = restore(cfg, "clean")
    for kind in ["label-flip", "collision", "backdoor"]:
        for ratio in cfg["poison_ratios"]:
            name = poison_name(kind, ratio)
            seed(cfg["seed"], cfg["threads"])
            x, y, ids = poison(clean, data["x_train"], data["y_train"], domain, cfg, kind, ratio)
            save_arrays(cfg, name, {"x": x, "y": y, "poison_ids": ids}, {"kind": kind, "requested_ratio": ratio, "actual_ratio": len(ids) / len(y)})
            model = train_mlp(cfg, x, y)
            register(cfg, name, model, data, {"poison_ids": ids.tolist(), "kind": kind, "ratio": ratio})


def defend(cfg):
    data, domain = load(cfg), Domain(cfg)
    for mode in ["pgd", "trades"]:
        model = train_mlp(cfg, data["x_train"], data["y_train"], mode, domain)
        register(cfg, mode, model, data)
    clean, _, _ = restore(cfg, "clean")
    detector = Detector().fit(clean, data["x_train"])
    joblib.dump(detector, Path(cfg["root"]) / "models/detector.joblib")
    for kind in ["label-flip", "collision", "backdoor"]:
        for ratio in cfg["poison_ratios"]:
            name = poison_name(kind, ratio)
            arrays, _ = load_arrays(cfg, name)
            model, _, _ = restore(cfg, name)
            mask, report = sanitize(model, arrays["x"], arrays["y"])
            sanitized = train_mlp(cfg, arrays["x"][mask], arrays["y"][mask])
            truth = np.zeros(len(mask), dtype=bool)
            truth[arrays["poison_ids"]] = True
            stats = {"quarantined": int((~mask).sum()), "poison_detected": int((~mask & truth).sum()), "clean_quarantined": int((~mask & ~truth).sum()), "activation_report": report}
            register(cfg, name + "-sanitized", sanitized, data, stats)
            save_arrays(cfg, name + "-quarantine", {"keep": mask, "quarantined_ids": np.flatnonzero(~mask), "raw_row_ids": data["id_train"][~mask]}, stats)


def evaluate(cfg):
    data, domain = load(cfg), Domain(cfg)
    names = ["clean", "tree", "pgd", "trades"] + [poison_name(kind, ratio) + suffix for kind in ["label-flip", "collision", "backdoor"] for ratio in cfg["poison_ratios"] for suffix in ["", "-sanitized"]]
    checkpoints = {name: restore(cfg, name)[2] for name in names}
    def fixed(name):
        return restore(cfg, name, checkpoints[name]["version"])
    clean, threshold, metadata = fixed("clean")
    tree, tree_threshold, _ = fixed("tree")
    predictions = {}
    rows, transfers, sweeps, poisoning_rows = [], [], [], []
    for name in ["clean", "tree", "pgd", "trades"]:
        model, t, _ = fixed(name)
        predictions[name] = score(model, data["x_test"])
        rows.append({"model": name, **metrics(model, data["x_test"], data["y_test"], t)})
    for artifact in sorted((Path(cfg["root"]) / "artifacts").glob("evasion-*.npz")):
        arrays, details = load_arrays(cfg, artifact.stem)
        if details["source_checkpoint"] != metadata["weights_sha256"]:
            raise ValueError("Evasion examples target a different clean checkpoint; rerun attack-evasion")
        for target, model, t in [("clean", clean, threshold), ("tree", tree, tree_threshold)]:
            predictions[artifact.stem + "-" + target + "-before"] = score(model, arrays["original"])
            predictions[artifact.stem + "-" + target + "-after"] = score(model, arrays["adversarial"])
            transfers.append({"source": "clean", "target": target, "attack": artifact.stem, **attack_metrics(model, arrays["original"], arrays["adversarial"], t), **details})
    x, y = attacked_slice(data, cfg)
    for name in ["clean", "pgd", "trades"]:
        model, t, _ = fixed(name)
        utility = metrics(model, data["x_test"], data["y_test"], t)
        for eps in cfg["epsilons"]:
            seed(cfg["seed"], cfg["threads"])
            adv, _ = generate(Calibrated(model, t).eval(), x, y, domain, cfg, "pgd", eps)
            save_arrays(cfg, f"sweep-{name}-{eps!r}", {"original": x, "adversarial": adv, "labels": y}, {"model": name, "epsilon": eps, "version": checkpoints[name]["version"], "threshold": t})
            sweeps.append({"model": name, "epsilon": eps, "clean_f1": utility["f1"], "p95_ms": utility["p95_ms"], **attack_metrics(model, x, adv, t)})
    transform = lambda a: squeeze(a, domain, cfg["quantum"])
    squeezed_t = calibrate(data["y_val"], score(clean, transform(data["x_val"])), cfg["fpr"])
    utility = metrics(clean, data["x_test"], data["y_test"], squeezed_t, transform)
    rows.append({"model": "squeezed", **utility})
    predictions["squeezed"] = score(clean, transform(data["x_test"]))
    for eps in cfg["epsilons"]:
        adaptive = bpda(Calibrated(clean, squeezed_t).eval(), x, y, domain, cfg, eps)
        save_arrays(cfg, f"sweep-squeezed-bpda-{eps!r}", {"original": x, "adversarial": adaptive, "labels": y}, {"model": "clean", "epsilon": eps, "version": metadata["version"], "threshold": squeezed_t, "transform": "squeeze"})
        sweeps.append({"model": "squeezed-bpda", "epsilon": eps, "clean_f1": utility["f1"], "p95_ms": utility["p95_ms"], **attack_metrics(clean, x, adaptive, squeezed_t, transform)})
    detector = Detector().fit(clean, data["x_train"])
    flags = detector.flags(clean, data["x_test"])
    conservative = (score(clean, data["x_test"]) >= threshold) | flags
    detector_report = {"flagged": int(flags.sum()), "flag_rate": float(flags.mean()), "fallback": "classify flagged samples as spam", "fpr": float(conservative[data["y_test"] == 0].mean()), "fnr": float((~conservative[data["y_test"] == 1]).mean())}
    original_p = score(clean, data["x_test"])
    base_metrics = metrics(clean, data["x_test"], data["y_test"], threshold)
    subgroups = {"spam-with-free": (data["y_test"] == 1) & (data["x_test"][:, 15] * domain.scale[15] + domain.center[15] > 0), "other-spam": (data["y_test"] == 1) & (data["x_test"][:, 15] * domain.scale[15] + domain.center[15] <= 0)}
    for kind in ["label-flip", "collision", "backdoor"]:
        for ratio in cfg["poison_ratios"]:
            for suffix in ["", "-sanitized"]:
                name = poison_name(kind, ratio) + suffix
                model, t, meta = fixed(name)
                result = metrics(model, data["x_test"], data["y_test"], t)
                probabilities = score(model, data["x_test"])
                predictions[name] = probabilities
                injected, injection_info = load_arrays(cfg, poison_name(kind, ratio))
                modified_count = int((np.any(injected["x"] != data["x_train"], axis=1) | (injected["y"] != data["y_train"])).sum())
                row = {"model": name, "kind": kind, "ratio": ratio, "selected_count": len(injected["poison_ids"]), "modified_count": modified_count, "actual_ratio": injection_info["actual_ratio"], "sanitized": bool(suffix), **result, "fnr_delta": result["fnr"] - base_metrics["fnr"], "brier_delta": result["brier"] - base_metrics["brier"], "decision_disagreement": float(((probabilities >= t) != (original_p >= threshold)).mean()), "mean_score_shift": float(np.mean(np.abs(probabilities - original_p))), "subgroup_fnr": {key: float((probabilities[mask] < t).mean()) if mask.any() else None for key, mask in subgroups.items()}, "filter": meta["extra"] if suffix else None}
                if kind == "backdoor":
                    row["trigger_asr"] = float((score(model, trigger(x, domain, cfg["epsilon"])) < t).mean())
                    row["clean_trigger_asr"] = float((score(clean, trigger(x, domain, cfg["epsilon"])) < threshold).mean())
                poisoning_rows.append(row)
    # Keep all measured points; mark those dominated in utility, recall, and latency.
    for a in sweeps:
        a["pareto"] = not any(b["clean_f1"] >= a["clean_f1"] and b["adversarial_recall"] >= a["adversarial_recall"] and b["p95_ms"] <= a["p95_ms"] and (b["clean_f1"] > a["clean_f1"] or b["adversarial_recall"] > a["adversarial_recall"] or b["p95_ms"] < a["p95_ms"]) for b in sweeps)
    save_arrays(cfg, "evaluation-scores", predictions, {"versions": {name: meta["version"] for name, meta in checkpoints.items()}})
    report = {"config": cfg, "platform": platform.platform(), "packages": metadata["packages"], "dataset": verify(cfg), "counts": {"features": data["x_train"].shape[1], "attack_samples": len(x), "selected_records": sum(len(data["y_" + split]) for split in ["train", "val", "test"])}, "clean": rows, "transferability": transfers, "pareto": sweeps, "poisoning": poisoning_rows, "detector": detector_report}
    directory = Path(cfg.get("report_dir", "reports" if cfg["root"] == "runs/default" else str(Path(cfg["root"]) / "reports")))
    write_json(directory / "benchmark_results.json", report)
    markdown = "# Measured local benchmark\n\nGenerated from real execution. Configuration and dataset hashes are in the adjacent JSON. Quick settings verify execution, not converged robustness.\n\n"
    markdown += table("Clean utility", rows, ["model", "n", "accuracy", "roc_auc", "pr_auc", "f1", "fpr", "fnr", "p50_ms", "p95_ms"])
    markdown += table("Evasion and transfer", transfers, ["source", "target", "attack", "attack_n", "eligible", "adversarial_recall", "asr", "raw_valid_fraction", "projected_valid_fraction"])
    markdown += table("Robustness and utility sweep", sweeps, ["model", "epsilon", "clean_f1", "adversarial_recall", "asr", "p95_ms", "pareto"])
    markdown += table("Poisoned and sanitized", poisoning_rows, ["model", "f1", "fpr", "fnr", "fnr_delta", "brier_delta", "decision_disagreement", "trigger_asr"])
    markdown += "## Conservative detector\n\n```json\n" + __import__("json").dumps(detector_report, indent=2) + "\n```\n"
    (directory / "benchmark_results.md").write_text(markdown, encoding="utf-8")
    evidence = dict(verify(cfg)["files"])
    prepared_manifest = Path(cfg["root"]) / "prepared/manifest.json"
    evidence[str(prepared_manifest.resolve())] = sha256(prepared_manifest)
    for name, meta in checkpoints.items():
        model_dir = Path(cfg["root"]) / "models" / name
        for p in [model_dir / (meta["version"] + ".json"), model_dir / meta["weights"]]:
            evidence[str(p.resolve())] = sha256(p)
    for p in (Path(cfg["root"]) / "artifacts").iterdir():
        if p.suffix in [".npz", ".json"]:
            evidence[str(p.resolve())] = sha256(p)
    report_path = directory / "benchmark_results.json"
    write_json(directory / "benchmark_results.manifest.json", {"report_sha256": sha256(report_path), "files": evidence, "checkpoints": {name: meta["version"] for name, meta in checkpoints.items()}})
    return directory / "benchmark_results.md"


def table(title, rows, columns):
    def format_value(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value) if value is not None else "n/a"
    text = f"## {title}\n\n| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in rows:
        text += "| " + " | ".join(format_value(row.get(c)) for c in columns) + " |\n"
    return text + "\n"
