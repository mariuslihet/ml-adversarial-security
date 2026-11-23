from pathlib import Path

import numpy as np
import pytest
import torch

from ml_adv_sec.common import sha256, read_json
from ml_adv_sec.data import prepare, verify, load
from ml_adv_sec.data.pipeline import SHA256
from ml_adv_sec.models import score, calibrate, train_mlp, save, restore
from ml_adv_sec.attacks import generate, poison, trigger, bpda
from ml_adv_sec.defenses import squeeze, sanitize, Detector
from ml_adv_sec.eval.metrics import metrics
from ml_adv_sec.eval.workflow import train, evasion, poisoning, defend, evaluate, Calibrated, load_arrays


def test_authentic_data_and_split_isolation(experiment):
    cfg, data, domain, _ = experiment
    assert sha256(cfg["data_path"]) == SHA256
    assert verify(cfg)["raw_rows"] == 4601
    feature_rows = []
    for split in ["train", "val", "test"]:
        x, y = data["x_" + split], data["y_" + split]
        assert x.shape[1] == 57 and set(y) == {0, 1}
        assert np.isfinite(x).all()
        feature_rows.append({tuple(row) for row in x})
    assert not feature_rows[0] & feature_rows[1]
    assert not feature_rows[0] & feature_rows[2]
    assert not feature_rows[1] & feature_rows[2]
    assert domain.integer[55:].all()


def test_zero_only_training_frequencies_remain_continuous(experiment, tmp_path):
    from ml_adv_sec.data import Domain
    cfg = dict(experiment[0], root=str(tmp_path), max_rows=128, seed=19, epochs=2)
    prepare(cfg)
    data = load(cfg)
    domain = Domain(cfg)
    constant_frequencies = np.flatnonzero(np.ptp(data["x_train"][:, :55], axis=0) == 0)
    assert len(constant_frequencies) > 0
    assert not domain.integer[:55].any()
    assert domain.integer[55:].all()
    assert domain.valid(data["x_test"], data["x_test"], cfg["epsilon"]).all()
    model = train_mlp(cfg, data["x_train"], data["y_train"])
    x = data["x_test"][data["y_test"] == 1]
    adv, _ = generate(model, x, np.ones(len(x), dtype=np.int64), domain, cfg, "cleverhans-fgsm")
    raw = adv * domain.scale + domain.center
    np.testing.assert_allclose(raw[:, domain.integer], np.rint(raw[:, domain.integer]), atol=1e-3)


def test_training_only_scaler_and_reproducible_splits(experiment):
    cfg, data, domain, _ = experiment
    original = np.loadtxt(cfg["data_path"], delimiter=",")
    train_rows = original[data["id_train"], :-1]
    np.testing.assert_allclose(np.median(train_rows, axis=0), domain.center)
    prepare(cfg)
    again = load(cfg)
    for key in data:
        np.testing.assert_array_equal(data[key], again[key])


def test_manifest_detects_real_file_tampering(experiment):
    cfg, _, _, _ = experiment
    path = Path(cfg["root"]) / "prepared/schema.json"
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b" ")
        with pytest.raises(ValueError, match="Integrity"):
            verify(cfg)
    finally:
        path.write_bytes(saved)


def test_seeded_training_and_fpr_calibration(experiment):
    cfg, data, _, model = experiment
    again = train_mlp(cfg, data["x_train"], data["y_train"])
    p = score(model, data["x_val"])
    np.testing.assert_array_equal(p, score(again, data["x_val"]))
    t = calibrate(data["y_val"], p, cfg["fpr"])
    assert (p[data["y_val"] == 0] >= t).mean() <= cfg["fpr"]
    assert metrics(model, data["x_test"], data["y_test"], t)["p95_ms"] > 0
    first = metrics(model, data["x_test"], data["y_test"], t)
    second = metrics(again, data["x_test"], data["y_test"], t)
    for key in ["accuracy", "roc_auc", "pr_auc", "f1", "fpr", "fnr", "brier"]:
        assert first[key] == second[key]


def test_attack_decisions_match_thresholds_on_actual_records(experiment):
    cfg, data, _, model = experiment
    probabilities = score(model, data["x_test"])
    thresholds = [0, 0.5, 1, 1.000001, calibrate(data["y_val"], score(model, data["x_val"]), cfg["fpr"])]
    for threshold in thresholds:
        wrapped = Calibrated(model, threshold).eval()
        with torch.no_grad():
            predicted = wrapped(torch.from_numpy(data["x_test"])).argmax(1).numpy()
        np.testing.assert_array_equal(predicted, probabilities >= threshold)


def test_checkpoint_round_trip_and_integrity(experiment):
    cfg, data, _, model = experiment
    p = score(model, data["x_val"])
    t = calibrate(data["y_val"], p)
    meta = save(cfg, "test-roundtrip", model, t, metrics(model, data["x_val"], data["y_val"], t))
    restored, rt, _ = restore(cfg, "test-roundtrip")
    assert rt == t
    np.testing.assert_array_equal(p, score(restored, data["x_val"]))
    path = Path(cfg["root"]) / "models/test-roundtrip" / meta["weights"]
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b"corrupt")
        with pytest.raises(ValueError, match="hash"):
            restore(cfg, "test-roundtrip")
    finally:
        path.write_bytes(saved)


@pytest.mark.parametrize("method,norm", [("fgsm", np.inf), ("cleverhans-fgsm", np.inf), ("pgd", np.inf), ("pgd", 2), ("cw", 2), ("hsj", np.inf)])
def test_real_attacks_and_constraints(experiment, method, norm):
    cfg, data, domain, model = experiment
    x = data["x_test"][data["y_test"] == 1][:4]
    y = np.ones(len(x), dtype=np.int64)
    adv, details = generate(model, x, y, domain, cfg, method, norm=norm)
    eps = None if method == "cw" else cfg["epsilon"]
    assert adv.shape == x.shape and np.isfinite(adv).all()
    assert domain.valid(adv, x, eps, norm).all()
    np.testing.assert_array_equal(adv[:, domain.immutable], x[:, domain.immutable])
    raw = adv * domain.scale + domain.center
    np.testing.assert_allclose(raw[:, domain.integer], np.rint(raw[:, domain.integer]), atol=1e-3)
    if eps is not None:
        assert np.max(np.linalg.norm(adv - x, ord=norm, axis=1)) <= eps + 2e-5
    if method == "hsj":
        assert details["decision_queries"] > 0
    if method in ["fgsm", "pgd", "cleverhans-fgsm"]:
        assert np.any(adv != x)


@pytest.mark.parametrize("norm", [np.inf, 2])
def test_mutable_integer_projection_and_outlier_anchor(experiment, norm):
    from copy import deepcopy
    _, data, domain, _ = experiment
    domain = deepcopy(domain)
    domain.immutable[55:] = False
    original = data["x_test"][:8]
    candidate = original.copy()
    candidate[:, 55:] += 0.37
    projected = domain.project(candidate, original, 0.5, norm)
    raw = projected * domain.scale + domain.center
    np.testing.assert_allclose(raw[:, 55:], np.rint(raw[:, 55:]), atol=1e-3)
    assert domain.valid(projected, original, 0.5, norm).all()
    assert np.max(np.linalg.norm(projected - original, ord=norm, axis=1)) <= 0.5 + 2e-5
    assert domain.valid(original, original, 0.5, norm).all()


@pytest.mark.parametrize("kind", ["label-flip", "collision", "backdoor"])
def test_actual_poisoning(experiment, kind):
    cfg, data, domain, model = experiment
    x, y, ids = poison(model, data["x_train"], data["y_train"], domain, cfg, kind, 0.03)
    assert len(ids) == int(len(x) * 0.03)
    assert domain.valid(x, data["x_train"]).all()
    if kind == "collision":
        np.testing.assert_array_equal(y, data["y_train"])
        assert np.any(x[ids] != data["x_train"][ids])
    else:
        assert (y[ids] == 0).all() and (data["y_train"][ids] == 1).all()
    if kind == "backdoor":
        assert np.any(x[ids] != data["x_train"][ids])
        assert domain.valid(x, data["x_train"], cfg["epsilon"]).all()


@pytest.mark.parametrize("mode", ["pgd", "trades"])
def test_robust_training_executes(experiment, mode):
    cfg, data, domain, baseline = experiment
    hardened = train_mlp(cfg, data["x_train"], data["y_train"], mode, domain)
    assert np.isfinite(score(hardened, data["x_test"])).all()
    assert any(not torch.equal(a, b) for a, b in zip(baseline.parameters(), hardened.parameters()))


def test_sanitization_detection_and_adaptive_attack(experiment):
    cfg, data, domain, model = experiment
    x, y, ids = poison(model, data["x_train"], data["y_train"], domain, cfg, "backdoor", 0.03)
    mask, report = sanitize(model, x, y)
    assert mask.dtype == bool and mask.shape == y.shape
    assert mask.any() and (~mask).any()
    detector = Detector().fit(model, data["x_train"])
    flags = detector.flags(model, data["x_test"])
    fallback = detector.conservative(model, data["x_test"], score(model, data["x_test"]))
    assert (fallback[flags] == 1).all()
    malicious = data["x_test"][data["y_test"] == 1][:4]
    adv = bpda(model, malicious, np.ones(len(malicious), dtype=np.int64), domain, cfg, cfg["epsilon"])
    assert domain.valid(adv, malicious, cfg["epsilon"]).all()
    squeezed = squeeze(adv, domain, cfg["quantum"])
    assert np.isfinite(score(model, squeezed)).all()


def test_complete_real_report(completed_experiment):
    cfg, _, _, _ = completed_experiment
    path = Path(cfg["report_dir"]) / "benchmark_results.md"
    result = read_json(path.with_suffix(".json"))
    assert len(result["transferability"]) == 12
    assert len(result["pareto"]) == 20
    assert len(result["poisoning"]) == 24
    assert {r["model"] for r in result["clean"]} == {"clean", "tree", "pgd", "trades", "squeezed"}
    assert all(r["projected_valid_fraction"] == 1 for r in result["transferability"])
    for row in result["transferability"]:
        target, threshold, _ = restore(cfg, row["target"])
        arrays, _ = load_arrays(cfg, row["attack"])
        assert row["eligible"] == int((score(target, arrays["original"]) >= threshold).sum())
    assert "Evasion and transfer" in path.read_text(encoding="utf-8")
    assert all("dataset_files" in read_json(p) for p in (Path(cfg["root"]) / "models").glob("*/v*.json"))
