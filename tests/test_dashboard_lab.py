"""Independent real-model checks for ensemble, calibration, and input-loss tools."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from ml_adv_sec.dashboard_lab import ModelLabViews
from ml_adv_sec.data import Domain
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.models import calibrate, restore, score
from test_dashboard_views import view_run, view_http, json_get, csv_get


def predictions(run, model, values):
    target = "clean" if model == "squeezed" else model
    classifier, _, _ = restore(run["cfg"], target, run["payload"]["evidence"]["checkpoints"][target])
    if model == "squeezed":
        values = squeeze(values, Domain(run["cfg"]), run["cfg"]["quantum"])
    return score(classifier, values)


def splits(run):
    with np.load(Path(run["cfg"]["root"])/"prepared/splits.npz") as archive:
        return {k: archive[k] for k in archive.files}


def fit_positions(result, data):
    fit = set(result["split"]["fit_source_rows"])
    threshold = set(result["split"]["threshold_source_rows"])
    assert not fit & threshold
    assert fit | threshold == set((data["id_val"]+1).tolist())
    assert not (fit | threshold) & set((data["id_test"]+1).tolist())
    assert not (fit | threshold) & set((data["id_train"]+1).tolist())
    return np.flatnonzero(np.isin(data["id_val"]+1, list(fit))), np.flatnonzero(np.isin(data["id_val"]+1, list(threshold)))


@pytest.mark.parametrize("mode", ["fit", "equal"])
def test_ensemble_uses_real_member_scores_and_separate_validation_roles(view_run, view_http, mode):
    members = ["clean", "tree", "pgd"]
    query = {"run": view_run["id"], "tool": "ensemble", "members": ",".join(members), "mode": mode}
    result = json_get(view_http, "/api/model-lab", **query)
    data = splits(view_run)
    fit, threshold = fit_positions(result, data)
    val = np.column_stack([predictions(view_run, m, data["x_val"]) for m in members]).astype(float)
    test = np.column_stack([predictions(view_run, m, data["x_test"]) for m in members]).astype(float)
    weights = np.array(result["parameters"]["weights"])
    assert np.isfinite(weights).all() and (weights >= 0).all() and (weights <= 1).all()
    assert abs(weights.sum()-1) < 1e-12
    if mode == "equal":
        np.testing.assert_array_equal(weights, np.full(3, 1/3))
    else:
        gradient = 2*val[fit].T@(val[fit]@weights-data["y_val"][fit])/len(fit)
        assert gradient.min() >= gradient@weights-1e-6
        assert result["fitting"]["fitted_brier"] <= result["fitting"]["uniform_brier"]+1e-10
    expected = test@weights
    np.testing.assert_array_equal([r["score"] for r in result["records"]], expected)
    assert result["threshold"] == calibrate(data["y_val"][threshold], (val@weights)[threshold], 0.01)
    assert result["metrics"]["brier"] == float(np.mean((expected-data["y_test"])**2))
    assert {r["source_row"] for r in result["records"]} == set((data["id_test"]+1).tolist())
    exported = csv_get(view_http, "/api/model-lab.csv", **query)
    assert len(exported) == len(expected)
    np.testing.assert_array_equal([float(r["score"]) for r in exported], expected)
    summary = csv_get(view_http, "/api/model-lab.csv", **query, scope="summary")
    assert {r["model"] for r in summary} == {"ensemble", *members}
    recipe = json_get(view_http, "/api/model-lab-recipe", **query)
    assert recipe["parameters"] == result["parameters"] and recipe["split"] == result["split"]


def test_ensemble_saved_attack_runs_real_member_models(view_run, view_http):
    members = ["clean", "tree", "squeezed"]
    attack = view_run["payload"]["report"]["transferability"][0]["attack"]
    query = {"run": view_run["id"], "tool": "ensemble", "members": ",".join(members), "attack": attack}
    result = json_get(view_http, "/api/model-lab", **query)
    with np.load(Path(view_run["cfg"]["root"])/"artifacts"/(attack+".npz")) as archive:
        before = np.column_stack([predictions(view_run, m, archive["original"]) for m in members])@result["parameters"]["weights"]
        after = np.column_stack([predictions(view_run, m, archive["adversarial"]) for m in members])@result["parameters"]["weights"]
    np.testing.assert_array_equal([r["before"] for r in result["attack_records"]], before)
    np.testing.assert_array_equal([r["after"] for r in result["attack_records"]], after)
    eligible = before >= result["threshold"]
    escaped = eligible & (after < result["threshold"])
    assert result["attack_metrics"]["escape_rate"] == (float(escaped.sum()/eligible.sum()) if eligible.any() else None)
    assert len(csv_get(view_http, "/api/model-lab.csv", **query, scope="attacks")) == len(before)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("model", ["tree", "squeezed"])
def test_calibration_scores_thresholds_and_bands_match_real_fitted_correction(view_run, view_http, method, model):
    query = {"run": view_run["id"], "tool": "calibration", "method": method, "model": model}
    result = json_get(view_http, "/api/model-lab", **query)
    data = splits(view_run);fit, threshold = fit_positions(result, data)
    test = predictions(view_run, model, data["x_test"])
    val = predictions(view_run, model, data["x_val"])
    parameters = result["parameters"]
    if method == "sigmoid":
        def transform(p):
            return expit(parameters["slope"]*logit(np.clip(p.astype(float), 1e-6, 1-1e-6))+parameters["intercept"])
        assert parameters["slope"] > 0
    else:
        regression = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(val[fit], data["y_val"][fit])
        np.testing.assert_array_equal(parameters["score_knots"], regression.X_thresholds_)
        np.testing.assert_array_equal(parameters["probability_knots"], regression.y_thresholds_)
        transform = regression.predict
    corrected = transform(test)
    bounded = np.clip(corrected.astype(float), 1e-12, 1-1e-12)
    expected_loss = float(-np.mean(data["y_test"]*np.log(bounded)+(1-data["y_test"])*np.log1p(-bounded)))
    assert np.isfinite(result["after"]["log_loss"]) and result["after"]["log_loss"] == expected_loss
    np.testing.assert_array_equal([r["before"] for r in result["records"]], test)
    np.testing.assert_array_equal([r["after"] for r in result["records"]], corrected)
    assert result["before_threshold"] == calibrate(data["y_val"][threshold], val[threshold], 0.01)
    assert result["threshold"] == calibrate(data["y_val"][threshold], transform(val)[threshold], 0.01)
    for side, scores in [("before", test), ("after", corrected)]:
        error = 0
        for band in result[side+"_reliability"]["bins"]:
            mask = np.minimum((scores.astype(float)*10).astype(int), 9) == band["bin"]
            assert band["count"] == int(mask.sum())
            assert band["mean_score"] == (float(scores[mask].astype(float).mean()) if mask.any() else None)
            assert band["spam_share"] == (float(data["y_test"][mask].mean()) if mask.any() else None)
            error += float(mask.mean())*abs(band["mean_score"]-band["spam_share"]) if mask.any() else 0
        assert result[side+"_reliability"]["expected_error"] == error
    assert len(csv_get(view_http, "/api/model-lab.csv", **query, scope="bands")) == 20
    assert len(csv_get(view_http, "/api/model-lab.csv", **query)) == len(test)
    recipe = json_get(view_http, "/api/model-lab-recipe", **query)
    assert recipe["model"] == model and recipe["parameters"] == parameters


@pytest.mark.parametrize("replacement", ["median", "zero"])
@pytest.mark.parametrize("model", ["clean", "tree", "squeezed"])
def test_measurement_loss_trials_replay_exact_training_derived_replacements(view_run, view_http, replacement, model):
    columns = [0, 20, 55]
    query = {"run": view_run["id"], "tool": "stress", "columns": ",".join(map(str, columns)), "replacement": replacement, "model": model}
    result = json_get(view_http, "/api/model-lab", **query)
    data = splits(view_run)
    baseline = predictions(view_run, model, data["x_test"])
    assert len(result["trials"]) == 6 and len(result["replacements"]) == 3
    schema = json.loads((Path(view_run["cfg"]["root"])/"prepared/schema.json").read_text(encoding="utf-8"))
    raw = np.median(data["x_train"].astype(float), axis=0)*schema["scale"]+schema["center"] if replacement == "median" else np.zeros(57)
    raw[np.asarray(schema["integer"], dtype=bool)] = np.rint(raw[np.asarray(schema["integer"], dtype=bool)])
    values = ((raw-schema["center"])/schema["scale"]).astype(np.float32)
    for trial in result["trials"]:
        changed = data["x_test"].copy();changed[:, trial["columns"]] = values[trial["columns"]]
        expected = predictions(view_run, model, changed)
        np.testing.assert_array_equal(trial["scores"], expected)
        threshold = result["threshold"]
        old = (baseline >= threshold) == data["y_test"]
        new = (expected >= threshold) == data["y_test"]
        assert trial["new_mistakes"] == int((old & ~new).sum())
        assert trial["fixed_mistakes"] == int((~old & new).sum())
        assert trial["decisions_changed"] == trial["new_mistakes"]+trial["fixed_mistakes"]
    for item in result["replacements"]:
        assert item["raw_value"] == float(raw[item["column"]])
        assert item["scaled_value"] == float(values[item["column"]])
    exported = csv_get(view_http, "/api/model-lab.csv", **query, trial="single-55")
    np.testing.assert_array_equal([float(r["after"]) for r in exported], result["trials"][2]["scores"])
    assert len(csv_get(view_http, "/api/model-lab.csv", **query, scope="summary")) == 6
    recipe = json_get(view_http, "/api/model-lab-recipe", **query, trial="single-55")
    assert recipe["selected_trial"] == "single-55" and recipe["selected_columns"] == [55]


@pytest.mark.parametrize("params", [
    {"tool": "missing"}, {"tool": "ensemble", "members": "clean"}, {"tool": "ensemble", "members": "clean,clean"},
    {"tool": "ensemble", "members": "clean,missing"}, {"tool": "calibration", "method": "missing"},
    {"tool": "calibration", "bins": 3}, {"tool": "calibration", "fpr": "nan"}, {"tool": "calibration", "fpr": 0.3},
    {"tool": "stress", "columns": "0,0"}, {"tool": "stress", "columns": "57"}, {"tool": "stress", "columns": "x"},
    {"tool": "stress", "columns": "0", "replacement": "missing"}, {"tool": "calibration", "report_sha256": "stale"},
])
def test_model_tools_reject_invalid_or_stale_requests(view_run, view_http, params):
    with pytest.raises(HTTPError) as error:
        json_get(view_http, "/api/model-lab", run=view_run["id"], **params)
    assert error.value.code == 400


def test_metadata_and_export_validation(view_run, view_http):
    options = json_get(view_http, "/api/model-lab-options", run=view_run["id"])
    assert len(options["measurements"]) == 57
    assert [r["column"] for r in options["measurements"] if r["locked_for_attacks"]] == view_run["cfg"]["immutable"]
    for params in [{"tool": "stress", "columns": "0", "scope": "bands"}, {"tool": "calibration", "scope": "attacks"}, {"tool": "stress", "columns": "0", "trial": "missing"}]:
        with pytest.raises(HTTPError) as error:
            csv_get(view_http, "/api/model-lab.csv", run=view_run["id"], **params)
        assert error.value.code == 400
    with urlopen(view_http+"/lab.js") as response:
        assert response.status == 200 and b"Fit and test ensemble" in response.read()
