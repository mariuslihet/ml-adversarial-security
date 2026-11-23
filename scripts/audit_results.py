"""Independently check a completed report against its real records and weights.

Run from the project directory, for example:
    python scripts/audit_results.py --config configs/default.yaml

This does not train models, generate new attacks, or rewrite measured results.
Historical timings are checked for valid values, not claimed to be reproduced.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

import ml_adv_sec
from ml_adv_sec.common import config, read_json, sha256, write_json
from ml_adv_sec.data import Domain, load, verify
from ml_adv_sec.data.pipeline import SHA256
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.models import calibrate, restore, score


class Auditor:
    absolute_tolerance = 1e-8
    relative_tolerance = 1e-7

    def __init__(self, cfg, report_path, require_current_source=False):
        self.cfg = cfg
        self.report_path = Path(report_path).resolve()
        self.root = Path(cfg["root"]).resolve()
        self.require_current_source = require_current_source
        self.report = read_json(self.report_path)
        self.manifest_path = self.report_path.with_name("benchmark_results.manifest.json")
        self.manifest = read_json(self.manifest_path)
        self.models = {}
        self.saved = {}
        self.recomputed_scores = {}
        self.primary = {}
        self.evidence = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_path": str(self.report_path),
            "report_sha256": sha256(self.report_path),
            "report_manifest_path": str(self.manifest_path),
            "report_manifest_sha256": sha256(self.manifest_path),
            "outcome": "running",
            "checks": 0,
            "checked_rows": {},
            "numeric_tolerance": {
                "absolute": self.absolute_tolerance,
                "relative": self.relative_tolerance,
                "saved_scores_absolute": 1e-7,
                "saved_scores_relative": 1e-6,
            },
            "maximum_metric_difference": 0.0,
            "limitations": [
                "Historical p50/p95 latency cannot be reproduced exactly. Only finite, nonnegative, ordered timing values are checked.",
                "Local hashes establish consistency with the recorded files, not a signed publisher attestation or protection against someone rewriting all local evidence.",
                "Recomputing predictions proves the recorded results match the real saved weights and records; it does not certify model robustness or raw-email realizability.",
            ],
        }

    def check(self, condition, label):
        self.evidence["checks"] += 1
        if not bool(condition):
            raise ValueError(label)

    def same(self, actual, expected, label):
        if actual is None or expected is None:
            self.check(actual is None and expected is None, label)
            return
        actual, expected = float(actual), float(expected)
        self.check(math.isfinite(actual) and math.isfinite(expected), label + ": non-finite value")
        difference = abs(actual - expected)
        self.evidence["maximum_metric_difference"] = max(self.evidence["maximum_metric_difference"], difference)
        self.check(math.isclose(actual, expected, rel_tol=self.relative_tolerance, abs_tol=self.absolute_tolerance), f"{label}: computed {actual}, recorded {expected}")

    def arrays_equal(self, actual, expected, label):
        self.check(np.asarray(actual).shape == np.asarray(expected).shape, label + ": shape")
        self.check(np.array_equal(actual, expected), label + ": values")

    def bound_file(self, path):
        path = Path(path).resolve()
        expected = self.manifest["files"].get(str(path))
        self.check(expected is not None, f"Evidence manifest omits {path}")
        self.check(sha256(path) == expected, f"Evidence file changed: {path}")

    def model(self, name):
        if name not in self.models:
            version = self.manifest["checkpoints"].get(name)
            self.check(version is not None, f"No pinned checkpoint for {name}")
            directory = self.root / "models" / name
            metadata_path = directory / (version + ".json")
            self.bound_file(metadata_path)
            metadata = read_json(metadata_path)
            self.bound_file(directory / metadata["weights"])
            model, threshold, loaded = restore(self.cfg, name, version)
            self.check(loaded == metadata, f"Checkpoint metadata mismatch for {name}")
            self.models[name] = (model, threshold, metadata)
        return self.models[name]

    def artifact(self, name):
        if name not in self.saved:
            path = self.root / "artifacts" / (name + ".npz")
            metadata_path = path.with_suffix(".json")
            self.bound_file(path)
            self.bound_file(metadata_path)
            metadata = read_json(metadata_path)
            self.check(metadata["sha256"] == sha256(path), f"Artifact hash mismatch: {name}")
            self.check(metadata["prepared"] == self.prepared["files"], f"Artifact uses another dataset: {name}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            self.saved[name] = arrays, metadata["details"]
        return self.saved[name]

    def metrics(self, y, probabilities, threshold):
        probabilities = np.asarray(probabilities)
        self.check(probabilities.shape == y.shape, "Prediction and label shapes")
        self.check(np.isfinite(probabilities).all(), "Predictions must be finite")
        self.check(((probabilities >= 0) & (probabilities <= 1)).all(), "Prediction probabilities must be in [0, 1]")
        predicted = probabilities >= threshold
        return {
            "accuracy": float(accuracy_score(y, predicted)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "pr_auc": float(average_precision_score(y, probabilities)),
            "brier": float(brier_score_loss(y, probabilities)),
            "fpr": float(predicted[y == 0].mean()),
            "fnr": float((~predicted[y == 1]).mean()),
            "n": len(y),
            "threshold": threshold,
        }

    def compare_metrics(self, row, measured, label):
        for key, value in measured.items():
            self.same(value, row[key], f"{label}.{key}")
        p50, p95 = row["p50_ms"], row["p95_ms"]
        self.check(math.isfinite(p50) and math.isfinite(p95) and 0 <= p50 <= p95, f"{label}: invalid historical latency")

    def attack_metrics(self, before, after, threshold):
        originally_caught = np.asarray(before) >= threshold
        still_caught = np.asarray(after) >= threshold
        eligible = int(originally_caught.sum())
        return {
            "adversarial_recall": float(still_caught.mean()),
            "asr": float((~still_caught[originally_caught]).mean()) if eligible else None,
            "eligible": eligible,
            "attack_n": len(before),
        }

    def constraints(self, candidate, original, epsilon, norm, label):
        self.check(candidate.shape == original.shape, f"{label}: candidate shape")
        self.check(np.isfinite(candidate).all(), f"{label}: non-finite candidate")
        self.arrays_equal(candidate[:, self.domain.immutable], original[:, self.domain.immutable], f"{label}: locked measurements")
        low = np.minimum(self.domain.low, original)
        high = np.maximum(self.domain.high, original)
        self.check(((candidate >= low - 2e-5) & (candidate <= high + 2e-5)).all(), f"{label}: bounds")
        if epsilon is not None:
            distance = np.linalg.norm(candidate - original, ord=norm, axis=1)
            self.check((distance <= epsilon + 2e-5).all(), f"{label}: perturbation limit")
        raw = candidate * self.domain.scale + self.domain.center
        integer_raw = raw[:, self.domain.integer]
        # Float32 transformed counts cannot always invert to exact decimal integers.
        tolerance = np.maximum(1e-3, np.abs(integer_raw) * np.finfo(np.float32).eps * 2)
        self.check((np.abs(integer_raw - np.rint(integer_raw)) <= tolerance).all(), f"{label}: whole-number measurements")
        self.check(self.domain.valid(candidate, original, epsilon, norm).all(), f"{label}: domain projector")

    def check_source_records(self):
        self.prepared = verify(self.cfg)
        self.check(self.report["config"] == self.cfg, "Report configuration changed")
        self.check(self.report["dataset"] == self.prepared, "Report dataset manifest differs")
        self.data = load(self.cfg)
        self.domain = Domain(self.cfg)
        self.schema = read_json(self.root / "prepared/schema.json")
        for path, expected in self.prepared["files"].items():
            self.bound_file(path)
            self.check(sha256(path) == expected, "Prepared evidence file hash")
        raw_path = Path(self.cfg["data_path"]).resolve()
        self.check(sha256(raw_path) == SHA256, "The source does not match pinned UCI file bytes")
        raw = np.loadtxt(raw_path, delimiter=",")
        self.check(raw.shape == (4601, 58), "Unexpected original UCI file shape")
        self.check(np.isfinite(raw).all(), "Original records must be finite")
        source_ids = set()
        raw_rows = set()
        for split in ["train", "val", "test"]:
            ids = self.data["id_" + split]
            self.check(np.issubdtype(ids.dtype, np.integer), f"{split}: row ids must be integers")
            self.check(((ids >= 0) & (ids < len(raw))).all(), f"{split}: row ids out of bounds")
            self.check(len(set(ids.tolist())) == len(ids), f"{split}: repeated source rows")
            self.check(not (source_ids & set(ids.tolist())), f"{split}: source records leaked across splits")
            rows = {tuple(row) for row in raw[ids, :-1]}
            self.check(not (raw_rows & rows), f"{split}: repeated feature vectors across splits")
            source_ids.update(ids.tolist())
            raw_rows.update(rows)
            expected = ((raw[ids, :-1] - self.domain.center) / self.domain.scale).astype(np.float32)
            self.arrays_equal(self.data["x_" + split], expected, f"{split}: transformed real source values")
            self.arrays_equal(self.data["y_" + split], raw[ids, -1].astype(np.int64), f"{split}: original source labels")
            self.same(len(ids), self.prepared["split_sizes"][split], f"{split}: recorded size")
        train_raw = raw[self.data["id_train"], :-1]
        self.check(np.allclose(np.median(train_raw, axis=0), self.domain.center, atol=1e-10, rtol=1e-10), "Scaler center must come only from training records")
        quantiles = np.percentile(train_raw, [25, 75], axis=0)
        expected_scale = quantiles[1] - quantiles[0]
        expected_scale[expected_scale < 10 * np.finfo(np.float64).eps] = 1
        self.check(np.allclose(expected_scale, self.domain.scale, atol=1e-10, rtol=1e-10), "Scaler IQR must come only from training records")
        self.check(np.array_equal(train_raw.min(0), self.schema["minimum"]), "Training feature minima")
        self.check(np.array_equal(train_raw.max(0), self.schema["maximum"]), "Training feature maxima")
        # Spambase declares 48 word percentages, six character percentages,
        # one average capital-run length, then two whole-number counts.
        # A percentage observed only as zero in a small training selection does
        # not become a count. Observed statistics must not change domain types.
        documented_integer = np.zeros(raw.shape[1] - 1, dtype=bool)
        documented_integer[-2:] = True
        self.arrays_equal(self.schema["integer"], documented_integer, "Documented Spambase count-field semantics")
        self.arrays_equal(self.schema["binary"], np.zeros(raw.shape[1] - 1, dtype=bool), "Spambase has no binary input measurements")
        self.arrays_equal(self.domain.integer, documented_integer, "Domain count mask must follow documented field types")
        self.check(self.schema["names"][-2:] == ["capital_run_length_longest", "capital_run_length_total"], "Documented whole-number measurement names")
        self.check(np.equal(raw[:, -3:-1], np.rint(raw[:, -3:-1])).all(), "Original UCI count columns contain actual whole numbers")
        observed_integer = np.all(np.isclose(train_raw, np.rint(train_raw), atol=1e-9, rtol=0), axis=0)
        observed_binary = observed_integer & np.isin(train_raw, [0, 1]).all(axis=0)
        self.arrays_equal(self.schema["observed_integer"], observed_integer, "Observed integer statistics must match real training values")
        self.arrays_equal(self.schema["observed_binary"], observed_binary, "Observed binary statistics must match real training values")
        self.evidence["schema_semantics"] = {
            "definition_source": "https://archive.ics.uci.edu/dataset/94/spambase",
            "continuous_measurements": 55,
            "whole_number_measurements": self.schema["names"][-2:],
            "binary_input_measurements": 0,
            "continuous_fields_observed_only_as_integers_in_training": [self.schema["names"][i] for i in np.flatnonzero(observed_integer & ~documented_integer)],
            "observed_statistics_separate_from_documented_types": True,
        }
        count = sum(len(self.data["y_" + split]) for split in ["train", "val", "test"])
        self.same(count, self.report["counts"]["selected_records"], "Selected record count")
        self.same(self.data["x_train"].shape[1], self.report["counts"]["features"], "Measurement count")
        self.attack_x = self.data["x_test"][self.data["y_test"] == 1][:self.cfg["attack_samples"]]
        self.attack_ids = self.data["id_test"][self.data["y_test"] == 1][:len(self.attack_x)]
        self.same(len(self.attack_x), self.report["counts"]["attack_samples"], "Actual attack sample count")
        self.evidence["source_records"] = {
            "path": str(raw_path), "sha256": SHA256, "original_rows": len(raw),
            "verified_selected_records": count, "split_sizes": self.prepared["split_sizes"],
            "attacked_source_row_ids": self.attack_ids.tolist(),
        }

    def check_checkpoints(self):
        package = Path(ml_adv_sec.__file__).resolve().parent
        records = {}
        for name in self.manifest["checkpoints"]:
            _, _, metadata = self.model(name)
            current = {}
            for relative, expected in metadata.get("source_hashes", {}).items():
                path = package / relative
                actual = sha256(path) if path.is_file() else None
                current[relative] = {"recorded_sha256": expected, "current_sha256": actual, "matches_current_source": expected == actual}
                if self.require_current_source:
                    self.check(expected == actual, f"Source changed since training: {name}/{relative}")
            self.check(bool(current), f"No recorded source hashes for {name}")
            records[name] = {
                "version": metadata["version"], "weights_sha256": metadata["weights_sha256"],
                "metadata_sha256": sha256(self.root / "models" / name / (metadata["version"] + ".json")),
                "source_files": current,
            }
        self.evidence["checkpoints"] = records
        self.evidence["checked_rows"]["pinned_checkpoints"] = len(records)

    def check_clean(self):
        for row in self.report["clean"]:
            name = row["model"]
            model, threshold, _ = self.model("clean" if name == "squeezed" else name)
            inputs = self.data["x_test"]
            if name == "squeezed":
                transform = lambda a: squeeze(a, self.domain, self.cfg["quantum"])
                threshold = calibrate(self.data["y_val"], score(model, transform(self.data["x_val"])), self.cfg["fpr"])
                inputs = transform(inputs)
            probabilities = score(model, inputs)
            measured = self.metrics(self.data["y_test"], probabilities, threshold)
            self.compare_metrics(row, measured, name)
            self.primary[name] = measured
            self.recomputed_scores[name] = probabilities
        self.evidence["checked_rows"]["clean_metrics"] = len(self.report["clean"])

    def check_transfers(self):
        raw_checked = set()
        clean_weights = self.model("clean")[2]["weights_sha256"]
        for row in self.report["transferability"]:
            arrays, details = self.artifact(row["attack"])
            self.check(details["source_checkpoint"] == clean_weights, "Attack source checkpoint differs")
            self.arrays_equal(arrays["original"], self.attack_x, row["attack"] + ": real source vectors")
            self.arrays_equal(arrays["record_ids"], self.attack_ids, row["attack"] + ": original source row ids")
            self.arrays_equal(arrays["labels"], np.ones(len(self.attack_x), dtype=np.int64), row["attack"] + ": original spam labels")
            model, threshold, _ = self.model(row["target"])
            before = score(model, arrays["original"])
            after = score(model, arrays["adversarial"])
            measured = self.attack_metrics(before, after, threshold)
            for key, value in measured.items():
                self.same(value, row[key], f"{row['attack']}/{row['target']}.{key}")
            prefix = row["attack"] + "-" + row["target"]
            self.recomputed_scores[prefix + "-before"] = before
            self.recomputed_scores[prefix + "-after"] = after
            norm = np.inf if details["norm"] == "linf" else 2
            self.constraints(arrays["adversarial"], arrays["original"], details["epsilon"], norm, row["attack"])
            raw_fraction = float(self.domain.valid(arrays["raw"], arrays["original"], details["epsilon"], norm).mean())
            self.same(raw_fraction, details["raw_valid_fraction"], row["attack"] + ": raw candidate validity")
            self.same(float(self.domain.valid(arrays["adversarial"], arrays["original"], details["epsilon"], norm).mean()), row["projected_valid_fraction"], row["attack"] + ": projected validity")
            self.same(float(np.abs(arrays["adversarial"] - arrays["original"]).max()), details["max_linf"], row["attack"] + ": measured maximum change")
            self.same(float(np.linalg.norm(arrays["adversarial"] - arrays["original"], axis=1).mean()), details["mean_l2"], row["attack"] + ": measured mean L2 change")
            raw_checked.add(row["attack"])
        self.evidence["checked_rows"]["transfer_metrics"] = len(self.report["transferability"])
        self.evidence["checked_rows"]["raw_attack_artifacts"] = len(raw_checked)

    def check_sweeps(self):
        for row in self.report["pareto"]:
            name = row["model"]
            arrays, details = self.artifact(f"sweep-{name}-{row['epsilon']!r}")
            model, _, metadata = self.model(details["model"])
            threshold = details["threshold"]
            self.check(details["version"] == metadata["version"], "Sweep target version differs")
            self.arrays_equal(arrays["original"], self.attack_x, "Sweep real source vectors")
            self.arrays_equal(arrays["labels"], np.ones(len(self.attack_x), dtype=np.int64), "Sweep original spam labels")
            self.constraints(arrays["adversarial"], arrays["original"], row["epsilon"], np.inf, "Sweep constraints")
            before, after = arrays["original"], arrays["adversarial"]
            utility_name = name
            if details.get("transform") == "squeeze":
                before = squeeze(before, self.domain, self.cfg["quantum"])
                after = squeeze(after, self.domain, self.cfg["quantum"])
                utility_name = "squeezed"
            measured = self.attack_metrics(score(model, before), score(model, after), threshold)
            for key, value in measured.items():
                self.same(value, row[key], f"sweep-{name}/{row['epsilon']}.{key}")
            self.same(self.primary[utility_name]["f1"], row["clean_f1"], "Sweep clean F1")
            self.check(math.isfinite(row["p95_ms"]) and row["p95_ms"] >= 0, "Sweep historical timing must be valid")
        # Pareto labels can be checked using the recorded timings, without claiming
        # those timings have been remeasured under today's machine load.
        for a in self.report["pareto"]:
            dominated = any(b["clean_f1"] >= a["clean_f1"] and b["adversarial_recall"] >= a["adversarial_recall"] and b["p95_ms"] <= a["p95_ms"] and (b["clean_f1"] > a["clean_f1"] or b["adversarial_recall"] > a["adversarial_recall"] or b["p95_ms"] < a["p95_ms"]) for b in self.report["pareto"])
            self.check(bool(a["pareto"]) == (not dominated), "Incorrect recorded Pareto label")
        self.evidence["checked_rows"]["saved_sweep_metrics"] = len(self.report["pareto"])

    def trigger_inputs(self):
        indices = np.flatnonzero(~self.domain.immutable & ~self.domain.integer)[:2]
        self.check(len(indices) == 2, "Pattern requires two editable continuous measurements")
        candidate = self.attack_x.copy()
        candidate[:, indices] += self.cfg["epsilon"] / 2
        return self.domain.project(candidate, self.attack_x, self.cfg["epsilon"])

    def check_training_changes(self):
        original_x, original_y = self.data["x_train"], self.data["y_train"]
        baseline_p = self.recomputed_scores["clean"]
        baseline_t = self.primary["clean"]["threshold"]
        baseline = self.primary["clean"]
        raw_frequency = self.data["x_test"][:, 15] * self.domain.scale[15] + self.domain.center[15]
        positive = self.data["y_test"] == 1
        subgroups = {"spam-with-free": positive & (raw_frequency > 0), "other-spam": positive & (raw_frequency <= 0)}
        for row in self.report["poisoning"]:
            name = row["model"]
            base = name.removesuffix("-sanitized")
            arrays, details = self.artifact(base)
            ids = arrays["poison_ids"]
            self.check(len(np.unique(ids)) == len(ids) and ((ids >= 0) & (ids < len(original_y))).all(), "Training change row ids")
            self.check((original_y[ids] == 1).all(), "Training change must select actual original spam records")
            self.check("selected_count" in row and "modified_count" in row, "Report omits actual selected/modified training-record counts")
            modified = np.any(arrays["x"] != original_x, axis=1) | (arrays["y"] != original_y)
            self.same(len(ids), max(1, int(len(original_y) * row["ratio"])), "Selected training records")
            self.same(len(ids), row["selected_count"], "Recorded selection count")
            self.same(int(modified.sum()), row["modified_count"], "Recorded actual modified-record count")
            self.check(int(modified.sum()) <= len(ids), "Modified records cannot exceed the selected records")
            self.same(len(ids) / len(original_y), row["actual_ratio"], "Recorded actual selection share")
            self.same(details["actual_ratio"], row["actual_ratio"], "Artifact actual selection share")
            _, _, base_metadata = self.model(base)
            self.arrays_equal(base_metadata["extra"]["poison_ids"], ids, "Pinned model metadata selection positions")
            self.evidence.setdefault("training_selections", {})[base] = {
                "selected_count": len(ids),
                "modified_count": int(modified.sum()),
                "actual_selected_share": len(ids) / len(original_y),
                "selected_source_row_ids": self.data["id_train"][ids].tolist(),
                "modified_source_row_ids": self.data["id_train"][modified].tolist(),
            }
            outside = np.ones(len(original_y), dtype=bool)
            outside[ids] = False
            self.arrays_equal(arrays["x"][outside], original_x[outside], "Untouched training measurements")
            self.arrays_equal(arrays["y"][outside], original_y[outside], "Untouched training labels")
            self.constraints(arrays["x"], original_x, self.cfg["epsilon"] if row["kind"] != "label-flip" else None, np.inf, "Changed training measurements")
            if row["kind"] == "collision":
                self.arrays_equal(arrays["y"], original_y, "Preserved collision labels")
            else:
                self.check((arrays["y"][ids] == 0).all(), "Changed spam labels")
            if row["kind"] == "label-flip":
                self.arrays_equal(arrays["x"], original_x, "Label changes must not change measurements")
            model, threshold, _ = self.model(name)
            probabilities = score(model, self.data["x_test"])
            measured = self.metrics(self.data["y_test"], probabilities, threshold)
            self.compare_metrics(row, measured, name)
            self.recomputed_scores[name] = probabilities
            self.same(measured["fnr"] - baseline["fnr"], row["fnr_delta"], name + ": missed-spam shift")
            self.same(measured["brier"] - baseline["brier"], row["brier_delta"], name + ": calibration shift")
            self.same(float(((probabilities >= threshold) != (baseline_p >= baseline_t)).mean()), row["decision_disagreement"], name + ": changed decisions")
            self.same(float(np.mean(np.abs(probabilities - baseline_p))), row["mean_score_shift"], name + ": changed scores")
            for key, mask in subgroups.items():
                value = float((probabilities[mask] < threshold).mean()) if mask.any() else None
                self.same(value, row["subgroup_fnr"][key], name + ": subgroup " + key)
            if row["sanitized"]:
                quarantine, _ = self.artifact(base + "-quarantine")
                keep = quarantine["keep"]
                self.check(keep.dtype == bool and len(keep) == len(original_y), "Cleanup keep mask")
                self.arrays_equal(quarantine["quarantined_ids"], np.flatnonzero(~keep), "Cleanup removed positions")
                self.arrays_equal(quarantine["raw_row_ids"], self.data["id_train"][~keep], "Cleanup original source row ids")
                truth = np.zeros(len(keep), dtype=bool)
                truth[ids] = True
                self.same(int((~keep).sum()), row["filter"]["quarantined"], "Cleanup removed count")
                self.same(int((~keep & truth).sum()), row["filter"]["poison_detected"], "Cleanup changed-record count")
                self.same(int((~keep & ~truth).sum()), row["filter"]["clean_quarantined"], "Cleanup unchanged-record count")
            if row["kind"] == "backdoor":
                inputs = self.trigger_inputs()
                self.same(float((score(model, inputs) < threshold).mean()), row["trigger_asr"], name + ": pattern escape rate")
                clean, clean_t, _ = self.model("clean")
                self.same(float((score(clean, inputs) < clean_t).mean()), row["clean_trigger_asr"], name + ": original pattern escape rate")
        self.evidence["checked_rows"]["changed_training_metrics"] = len(self.report["poisoning"])

    @staticmethod
    def gradients(model, values):
        model.eval()
        inputs = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(model(inputs)[:, 1].sum(), inputs)[0]
        return torch.linalg.vector_norm(gradients, dim=1).detach().numpy()

    def check_detector(self):
        model, threshold, _ = self.model("clean")
        covariance = LedoitWolf().fit(self.data["x_train"])
        distance_cutoff = np.quantile(covariance.mahalanobis(self.data["x_train"]), 0.99)
        gradient_cutoff = np.quantile(self.gradients(model, self.data["x_train"]), 0.99)
        flags = (covariance.mahalanobis(self.data["x_test"]) > distance_cutoff) | (self.gradients(model, self.data["x_test"]) > gradient_cutoff)
        decisions = (self.recomputed_scores["clean"] >= threshold) | flags
        y = self.data["y_test"]
        measured = {"flagged": int(flags.sum()), "flag_rate": float(flags.mean()), "fpr": float(decisions[y == 0].mean()), "fnr": float((~decisions[y == 1]).mean())}
        for key, value in measured.items():
            self.same(value, self.report["detector"][key], "Extra input checks." + key)
        self.evidence["checked_rows"]["detector_metrics"] = 1

    def check_saved_scores(self):
        arrays, details = self.artifact("evaluation-scores")
        self.check(details["versions"] == self.manifest["checkpoints"], "Saved score checkpoint versions")
        self.check(set(arrays) == set(self.recomputed_scores), "Saved score keys differ from independently checked rows")
        for name, calculated in self.recomputed_scores.items():
            self.check(arrays[name].shape == calculated.shape, name + ": saved score shape")
            self.check(np.allclose(arrays[name], calculated, atol=1e-7, rtol=1e-6), name + ": saved scores differ from real inference")
        self.evidence["checked_rows"]["saved_score_arrays"] = len(arrays)

    def run(self):
        self.check(self.evidence["report_sha256"] == self.manifest["report_sha256"], "Report file changed after evaluation")
        for path in self.manifest["files"]:
            self.bound_file(path)
        self.check_source_records()
        self.check_checkpoints()
        self.check_clean()
        self.check_transfers()
        self.check_sweeps()
        self.check_training_changes()
        self.check_detector()
        self.check_saved_scores()
        self.evidence["outcome"] = "passed"
        self.evidence["verified_evidence_files"] = len(self.manifest["files"])
        self.evidence["checked_fields"] = ["threshold", "n", "accuracy", "f1", "roc_auc", "pr_auc (average precision)", "brier", "fpr", "fnr", "adversarial_recall", "asr", "eligible", "attack_n", "raw and final validity", "change distances", "documented input types and separate observed statistics", "actual selected and modified record counts and shares", "cleanup counts", "subgroup missed-spam rates", "score/decision shifts", "pattern escape rates", "detector outcomes", "saved prediction scores"]
        return self.evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check measured results against pinned real data, attacks and saved model weights")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report", help="Report JSON; otherwise inferred from the run configuration")
    parser.add_argument("--output", default="reports/independent_results_audit.json")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--require-current-source", action="store_true", help="Fail if the present package source differs from any checkpoint's recorded source snapshot")
    args = parser.parse_args(argv)
    os.chdir(args.workspace)
    auditor = None
    try:
        cfg = config(args.config)
        directory = Path(cfg.get("report_dir", "reports" if cfg["root"] == "runs/default" else str(Path(cfg["root"]) / "reports")))
        report_path = Path(args.report) if args.report else directory / "benchmark_results.json"
        auditor = Auditor(cfg, report_path, args.require_current_source)
        evidence = auditor.run()
        write_json(args.output, evidence)
        print(f"Passed {evidence['checks']} independent checks. Evidence: {Path(args.output).resolve()}")
        print("Historical timing values were not claimed to be remeasured.")
        return 0
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        evidence = auditor.evidence if auditor else {"checked_at_utc": datetime.now(timezone.utc).isoformat()}
        evidence["outcome"] = "failed"
        evidence["error"] = str(exc)
        write_json(args.output, evidence)
        print(f"Audit failed: {exc}", file=sys.stderr)
        print(f"Evidence: {Path(args.output).resolve()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
