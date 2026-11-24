"""New analysis views exercised with real source records and fitted models."""

import csv
import io
import json
import shutil
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pytest

from ml_adv_sec.common import sha256
from ml_adv_sec.dashboard import Jobs, Reports, handler_for
from ml_adv_sec.dashboard_views import ResultViews
from ml_adv_sec.data import Domain, load, prepare
from ml_adv_sec.data.pipeline import SHA256
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.eval.workflow import defend, evaluate, evasion, poison_name, poisoning, train
from ml_adv_sec.models import restore, score


@pytest.fixture(scope="module")
def view_run(completed_experiment):
    cfg, data, _, _ = completed_experiment
    reports = Reports(cfg["root"], cfg)
    report_path = (Path(cfg["report_dir"]) / "benchmark_results.json").resolve()
    run_id = next(key for key, value in reports.available().items() if value == report_path)
    payload = reports.read(run_id)
    return {"cfg": cfg, "data": data, "reports": reports, "views": ResultViews(reports), "id": run_id, "payload": payload}


@pytest.fixture(scope="module")
def different_cohort(view_run):
    """A second fitted run, with different real records and a renamed real cache."""
    workspace = view_run["reports"].workspace
    source = workspace / "renamed-source" / "actual-mail-records.data"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(view_run["cfg"]["data_path"], source)
    assert sha256(source) == SHA256
    cfg = dict(view_run["cfg"])
    root = workspace / "runs" / "different-cohort"
    cfg.update(root=str(root), report_dir=str(root / "reports"), data_path=str(source), seed=73, max_rows=128, epochs=1, steps=1, cw_steps=1, hsj_steps=1, collision_steps=1, attack_samples=4, poison_ratios=[0.01], epsilons=[0.01, 0.1], dashboard_name="=Different real cohort")
    prepare(cfg)
    train(cfg)
    evasion(cfg)
    poisoning(cfg)
    defend(cfg)
    evaluate(cfg)
    report_path = (root / "reports/benchmark_results.json").resolve()
    run_id = next(key for key, value in view_run["reports"].available().items() if value == report_path)
    return {"cfg": cfg, "id": run_id, "data": load(cfg), "payload": view_run["reports"].read(run_id)}


@pytest.fixture(scope="module")
def same_records_fewer_attacks(view_run):
    """Fit the same real split again, while attacking a smaller actual subset."""
    cfg = dict(view_run["cfg"])
    root = view_run["reports"].workspace / "runs" / "same-records-fewer-attacks"
    cfg.update(root=str(root), report_dir=str(root / "reports"), attack_samples=max(1, cfg["attack_samples"] // 2), dashboard_name="Same test records, fewer attacks")
    prepare(cfg)
    train(cfg)
    evasion(cfg)
    poisoning(cfg)
    defend(cfg)
    evaluate(cfg)
    report_path = (root / "reports/benchmark_results.json").resolve()
    run_id = next(key for key, value in view_run["reports"].available().items() if value == report_path)
    return {"cfg": cfg, "id": run_id, "data": load(cfg), "payload": view_run["reports"].read(run_id)}


@pytest.fixture(scope="module")
def view_http(view_run):
    jobs = Jobs(view_run["reports"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(view_run["reports"], jobs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    jobs.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def json_get(base, route, **query):
    with urlopen(base + route + "?" + urlencode(query), timeout=30) as response:
        assert response.status == 200
        return json.load(response)


def csv_get(base, route, **query):
    with urlopen(base + route + "?" + urlencode(query), timeout=30) as response:
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/csv")
        assert "attachment" in response.headers["Content-Disposition"]
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))


def direct_predictions(context, model):
    report, evidence = context["payload"]["report"], context["payload"]["evidence"]
    row = next(r for r in report["clean"] + report["poisoning"] if r["model"] == model)
    target = "clean" if model == "squeezed" else model
    classifier, _, _ = restore(context["cfg"], target, evidence["checkpoints"][target])
    values = context["data"]["x_test"]
    if model == "squeezed":
        values = squeeze(values, Domain(context["cfg"]), context["cfg"]["quantum"])
    return score(classifier, values), row["threshold"]


def confusion_counts(y, probabilities, threshold):
    flagged = probabilities >= threshold
    return {"caught": int(((y == 1) & flagged).sum()), "missed": int(((y == 1) & ~flagged).sum()), "false-alarm": int(((y == 0) & flagged).sum()), "clear": int(((y == 0) & ~flagged).sum())}


@pytest.mark.parametrize("model", ["clean", "tree", "pgd", "trades", "squeezed"])
def test_record_recheck_uses_each_real_clean_model(view_run, view_http, model):
    probabilities, threshold = direct_predictions(view_run, model)
    data = view_run["data"]
    result = json_get(view_http, "/api/records", run=view_run["id"], model=model, recheck=1, size=100, report_sha256=view_run["payload"]["evidence"]["report_sha256"])
    assert result["scores_recalculated"] is True
    assert result["threshold"] == threshold
    assert result["tested"] == len(data["y_test"])
    assert result["counts"] == confusion_counts(data["y_test"], probabilities, threshold)
    assert sum(result["counts"].values()) == len(data["y_test"])
    with np.load(Path(view_run["cfg"]["root"]) / "artifacts/evaluation-scores.npz") as saved:
        assert result["maximum_score_difference"] == float(np.max(np.abs(probabilities - saved[model])))
    source_to_position = {int(raw_id) + 1: i for i, raw_id in enumerate(data["id_test"])}
    for row in result["rows"]:
        i = source_to_position[row["source_row"]]
        assert row["score"] == float(probabilities[i])
        assert row["actual"] == ("spam" if data["y_test"][i] else "regular")
        assert row["decision"] == ("flagged" if probabilities[i] >= threshold else "passed")


def test_record_recheck_accepts_a_real_changed_training_model(view_run, view_http):
    model = next(row["model"] for row in view_run["payload"]["report"]["poisoning"] if row["kind"] == "backdoor" and row["sanitized"])
    probabilities, threshold = direct_predictions(view_run, model)
    result = json_get(view_http, "/api/records", run=view_run["id"], model=model, recheck=1)
    assert result["scores_recalculated"]
    assert result["counts"] == confusion_counts(view_run["data"]["y_test"], probabilities, threshold)
    assert result["model"] == model


@pytest.mark.parametrize("label", ["all", "spam", "regular"])
@pytest.mark.parametrize("outcome", ["all", "caught", "missed", "false-alarm", "clear"])
def test_record_label_and_outcome_filters_match_real_labels(view_run, label, outcome):
    probabilities, threshold = direct_predictions(view_run, "tree")
    y = view_run["data"]["y_test"]
    ids = view_run["data"]["id_test"]
    result = view_run["views"].records(view_run["id"], model="tree", label=label, outcome=outcome, size=100)
    expected = []
    for i, value in enumerate(y):
        flagged = probabilities[i] >= threshold
        actual_outcome = "caught" if value == 1 and flagged else "missed" if value == 1 else "false-alarm" if flagged else "clear"
        if (label == "all" or value == (1 if label == "spam" else 0)) and (outcome == "all" or outcome == actual_outcome):
            expected.append(i)
    expected.sort(key=lambda i: (-float(probabilities[i]), int(ids[i])))
    assert result["total"] == len(expected)
    assert [row["source_row"] for row in result["rows"]] == [int(ids[i]) + 1 for i in expected[:100]]
    assert result["counts"] == confusion_counts(y, probabilities, threshold)


@pytest.mark.parametrize("order", ["score-desc", "score-asc", "source"])
def test_record_pagination_and_stable_sorting_cover_the_real_test_set(view_run, order):
    probabilities, _ = direct_predictions(view_run, "tree")
    ids = view_run["data"]["id_test"]
    if order == "source":
        expected = sorted(range(len(ids)), key=lambda i: int(ids[i]))
    else:
        direction = -1 if order == "score-desc" else 1
        expected = sorted(range(len(ids)), key=lambda i: (direction * float(probabilities[i]), int(ids[i])))
    records = []
    first = view_run["views"].records(view_run["id"], model="tree", order=order, size=10)
    for page in range(1, first["pages"] + 1):
        result = view_run["views"].records(view_run["id"], model="tree", order=order, page=page, size=10)
        assert result["page"] == page
        records.extend(row["source_row"] for row in result["rows"])
    assert records == [int(ids[i]) + 1 for i in expected]
    assert len(set(records)) == len(ids)
    last = view_run["views"].records(view_run["id"], model="tree", order=order, page=100000, size=10)
    assert last["page"] == first["pages"]
    assert [row["source_row"] for row in last["rows"]] == records[(last["page"] - 1) * 10:]


def test_record_source_search_and_csv_export_use_original_UCI_values(view_run, view_http):
    raw = np.loadtxt(view_run["cfg"]["data_path"], delimiter=",")
    assert sha256(view_run["cfg"]["data_path"]) == SHA256
    schema = json.loads((Path(view_run["cfg"]["root"]) / "prepared/schema.json").read_text())
    needle = str(int(view_run["data"]["id_test"][0]) + 1)
    searched = view_run["views"].records(view_run["id"], search=" " + needle + " ", order="source", size=100)
    expected = sorted(int(i) + 1 for i in view_run["data"]["id_test"] if needle in str(int(i) + 1))
    assert [r["source_row"] for r in searched["rows"]] == expected
    rows = csv_get(view_http, "/api/records.csv", run=view_run["id"], model="tree", features=1, sort="source", page=2, size=10)
    assert len(rows) == len(view_run["data"]["y_test"]), "Export must include the whole filtered selection, not only the visible page"
    probabilities, threshold = direct_predictions(view_run, "tree")
    positions = {int(raw_id) + 1: i for i, raw_id in enumerate(view_run["data"]["id_test"])}
    for row in rows:
        source = int(row["source_row"]) - 1
        np.testing.assert_array_equal([float(row[name]) for name in schema["names"]], raw[source, :-1])
        assert row["actual_label"] == ("spam" if raw[source, -1] == 1 else "regular")
        assert float(row["spam_score"]) == float(probabilities[positions[source + 1]])
        assert float(row["decision_threshold"]) == threshold
        assert row["report_sha256"] == view_run["payload"]["evidence"]["report_sha256"]
    filtered = csv_get(view_http, "/api/records.csv", run=view_run["id"], model="tree", label="spam", outcome="missed", sort="source")
    assert len(filtered) == confusion_counts(view_run["data"]["y_test"], probabilities, threshold)["missed"]
    assert all(r["actual_label"] == "spam" and r["outcome"] == "missed" for r in filtered)


def test_same_run_comparison_has_measured_zero_deltas(view_run, view_http):
    fingerprint = view_run["payload"]["evidence"]["report_sha256"]
    result = json_get(view_http, "/api/compare", left=view_run["id"], right=view_run["id"], left_sha=fingerprint, right_sha=fingerprint)
    assert result["same_run"] and result["same_test_records"]
    assert result["same_attacked_records"] is True
    assert len(result["models"]) == len(view_run["payload"]["report"]["clean"])
    for model in result["models"]:
        assert model["left"] == model["right"]
        assert all(value == 0 for value in model["change"].values())
    for attack in result["attacks"]:
        assert attack["same_attacked_records"] is True
        assert attack["left"] == attack["right"]
        assert attack["escape_change"] is None if attack["left"]["asr"] is None else attack["escape_change"] == 0


def test_different_real_cohort_comparison_and_csv(view_run, different_cohort, view_http):
    views = view_run["views"]
    result = views.compare(view_run["id"], different_cohort["id"])
    assert not result["same_run"] and not result["same_test_records"]
    original_attacked = view_run["data"]["id_test"][view_run["data"]["y_test"] == 1][:view_run["cfg"]["attack_samples"]]
    other_attacked = different_cohort["data"]["id_test"][different_cohort["data"]["y_test"] == 1][:different_cohort["cfg"]["attack_samples"]]
    assert set(map(int, original_attacked)) != set(map(int, other_attacked))
    assert result["same_attacked_records"] is False
    assert all(r["same_attacked_records"] is False for r in result["attacks"])
    assert result["left"]["tested"] == len(view_run["data"]["y_test"])
    assert result["right"]["tested"] == len(different_cohort["data"]["y_test"])
    assert set(view_run["data"]["id_test"].tolist()) != set(different_cohort["data"]["id_test"].tolist())
    original = {r["model"]: r for r in view_run["payload"]["report"]["clean"]}
    changed = {r["model"]: r for r in different_cohort["payload"]["report"]["clean"]}
    for row in result["models"]:
        a, b = original[row["model"]], changed[row["model"]]
        assert row["change"]["spam_caught"] == a["fnr"] - b["fnr"]
        assert row["change"]["false_alarms"] == b["fpr"] - a["fpr"]
        assert row["change"]["f1"] == b["f1"] - a["f1"]
        assert row["change"]["p95_ms"] == b["p95_ms"] - a["p95_ms"]
    exported = csv_get(view_http, "/api/compare.csv", left=view_run["id"], right=different_cohort["id"])
    assert len(exported) == len(result["models"])
    for row in exported:
        assert row["right_run"] == "'=Different real cohort", "Editable names must not become spreadsheet formulas"
        assert row["same_test_records"] == "False"
        assert row["left_report_sha256"] == result["left"]["report_sha256"]
        assert row["right_report_sha256"] == result["right"]["report_sha256"]


def test_same_test_cohort_does_not_imply_the_same_attacked_records(view_run, same_records_fewer_attacks, view_http):
    other = same_records_fewer_attacks
    np.testing.assert_array_equal(view_run["data"]["id_test"], other["data"]["id_test"])
    np.testing.assert_array_equal(view_run["data"]["x_test"], other["data"]["x_test"])
    np.testing.assert_array_equal(view_run["data"]["y_test"], other["data"]["y_test"])
    result = json_get(view_http, "/api/compare", left=view_run["id"], right=other["id"])
    assert result["same_run"] is False
    assert result["same_test_records"] is True
    assert result["same_attacked_records"] is False
    assert result["left"]["tested"] == result["right"]["tested"]
    assert result["left"]["attacked"] > result["right"]["attacked"] > 0
    for row in result["attacks"]:
        name = row["attack"]
        with np.load(Path(view_run["cfg"]["root"]) / "artifacts" / (name + ".npz")) as saved:
            original_ids = saved["record_ids"]
        with np.load(Path(other["cfg"]["root"]) / "artifacts" / (name + ".npz")) as saved:
            smaller_ids = saved["record_ids"]
        assert len(original_ids) > len(smaller_ids) > 0
        np.testing.assert_array_equal(smaller_ids, original_ids[:len(smaller_ids)])
        assert row["same_attacked_records"] is False
        assert row["left"]["attack_n"] == len(original_ids)
        assert row["right"]["attack_n"] == len(smaller_ids)


@pytest.mark.parametrize("kind", ["label-flip", "collision", "backdoor"])
@pytest.mark.parametrize("sanitized", [False, True])
def test_training_record_selection_cleanup_and_four_real_scores(view_run, view_http, kind, sanitized):
    report = view_run["payload"]["report"]
    row = next(r for r in report["poisoning"] if r["kind"] == kind and r["sanitized"] == sanitized and r["ratio"] == 0.03)
    name = poison_name(kind, row["ratio"])
    root = Path(view_run["cfg"]["root"])
    with np.load(root / "artifacts" / (name + ".npz")) as altered:
        px, py, selected_ids = altered["x"], altered["y"], altered["poison_ids"]
    with np.load(root / "artifacts" / (name + "-quarantine.npz")) as quarantine:
        keep = quarantine["keep"]
    data = view_run["data"]
    modified = np.any(px != data["x_train"], axis=1) | (py != data["y_train"])
    expected_counts = {"selected": len(selected_ids), "modified": int(modified.sum()), "removed": int((~keep).sum()), "all": len(keep)}
    expected_sets = {"selected": set(map(int, selected_ids)), "modified": set(map(int, np.flatnonzero(modified))), "removed": set(map(int, np.flatnonzero(~keep))), "all": set(range(len(keep)))}
    for mode in ["selected", "modified", "removed", "all"]:
        page = view_run["views"].training(view_run["id"], row["model"], mode=mode, size=100)
        assert page["cleanup_model"] == name + "-sanitized"
        assert page["scores_after_cleanup"] is sanitized
        assert page["counts"] == expected_counts
        assert page["total"] == len(expected_sets[mode])
        expected_order = sorted(expected_sets[mode], key=lambda i: int(data["id_train"][i]))
        assert [r["position"] for r in page["rows"]] == expected_order[:100]
    selection = json_get(view_http, "/api/training-records", run=view_run["id"], model=row["model"], mode="selected", size=10)
    assert selection["cleanup_model"] == name + "-sanitized"
    assert selection["scores_after_cleanup"] is sanitized
    assert selection["counts"]["selected"] == row["selected_count"]
    position = selection["rows"][0]["position"]
    detail = json_get(view_http, "/api/training-record", run=view_run["id"], model=row["model"], position=position, report_sha256=view_run["payload"]["evidence"]["report_sha256"])
    assert detail["cleanup_model"] == name + "-sanitized"
    assert detail["scores_after_cleanup"] is sanitized
    assert detail["model"] == row["model"]
    assert detail["source_row"] == int(data["id_train"][position]) + 1
    assert detail["selected"] and detail["modified"] == bool(modified[position])
    assert detail["removed"] == bool(~keep[position])
    assert detail["before_label"] == int(data["y_train"][position])
    assert detail["after_label"] == int(py[position])
    inputs = np.stack([data["x_train"][position], px[position]])
    versions = view_run["payload"]["evidence"]["checkpoints"]
    clean, clean_threshold, _ = restore(view_run["cfg"], "clean", versions["clean"])
    target, target_threshold, _ = restore(view_run["cfg"], row["model"], versions[row["model"]])
    np.testing.assert_array_equal(detail["clean_scores"], score(clean, inputs))
    np.testing.assert_array_equal(detail["changed_scores"], score(target, inputs))
    assert detail["clean_threshold"] == clean_threshold
    assert detail["changed_threshold"] == target_threshold
    domain = Domain(view_run["cfg"])
    np.testing.assert_allclose([f["before"] for f in detail["features"]], inputs[0] * domain.scale + domain.center)
    np.testing.assert_allclose([f["after"] for f in detail["features"]], inputs[1] * domain.scale + domain.center)
    np.testing.assert_allclose([f["scaled_change"] for f in detail["features"]], inputs[1] - inputs[0])
    assert all(not f["changed"] for f in detail["features"] if f["locked"])


@pytest.mark.parametrize("method,kwargs", [
    ("records", {"model": "missing-filter"}),
    ("records", {"label": "other"}),
    ("records", {"outcome": "other"}),
    ("records", {"search": "not-a-row"}),
    ("records", {"order": "unknown"}),
    ("records", {"page": 0}),
    ("records", {"size": 7}),
    ("records", {"expected": "outdated-fingerprint"}),
    ("training", {"model": "clean"}),
])
def test_analysis_views_reject_unlisted_inputs_without_answers(view_run, method, kwargs):
    with pytest.raises(ValueError):
        getattr(view_run["views"], method)(view_run["id"], **kwargs)


def test_http_analysis_views_reject_stale_fingerprints_and_invalid_positions(view_run, view_http):
    model = view_run["payload"]["report"]["poisoning"][0]["model"]
    requests = [
        ("/api/records", {"run": view_run["id"], "report_sha256": "outdated-fingerprint"}),
        ("/api/compare", {"left": view_run["id"], "right": view_run["id"], "left_sha": "outdated-fingerprint"}),
        ("/api/training-record", {"run": view_run["id"], "model": model, "position": len(view_run["data"]["y_train"])}),
        ("/api/training-record", {"run": view_run["id"], "model": model, "position": -1}),
        ("/api/training-records", {"run": view_run["id"], "model": model, "mode": "other"}),
    ]
    for route, query in requests:
        with pytest.raises(HTTPError) as error:
            json_get(view_http, route, **query)
        assert error.value.code >= 400
        assert "error" in json.load(error.value)


@pytest.mark.parametrize("model", ["clean", "tree", "pgd", "trades", "squeezed"])
@pytest.mark.parametrize("mode", ["saved", "manual", "target"])
def test_decision_lab_measures_actual_validation_and_test_predictions(view_run, view_http, model, mode):
    from sklearn.metrics import confusion_matrix
    result = json_get(view_http, "/api/decision-lab", run=view_run["id"], model=model, mode=mode, threshold=0.37, target_fpr=0.03)
    probabilities, original_threshold = direct_predictions(view_run, model)
    classifier, _, _ = restore(view_run["cfg"], "clean" if model == "squeezed" else model, view_run["payload"]["evidence"]["checkpoints"]["clean" if model == "squeezed" else model])
    values = view_run["data"]["x_val"]
    if model == "squeezed":
        values = squeeze(values, Domain(view_run["cfg"]), view_run["cfg"]["quantum"])
    validation_scores = score(classifier, values)
    t = result["threshold"]
    assert t == (original_threshold if mode == "saved" else 0.37) if mode != "target" else 0 <= t <= 1.000001
    for partition, key, p in [("validation", "val", validation_scores), ("test", "test", probabilities)]:
        y = view_run["data"]["y_"+key]
        tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
        actual = result["metrics"][partition]
        assert [actual["clear"], actual["false_alarms"], actual["missed"], actual["caught"]] == [tn, fp, fn, tp]
        assert actual["tested"] == len(y)
    if mode == "target":
        negatives = validation_scores[view_run["data"]["y_val"] == 0]
        assert np.mean(negatives >= t) <= 0.03
        candidates = np.unique(np.r_[0.0, negatives, np.nextafter(negatives, np.float32(np.inf)), 1.000001])
        assert all(np.mean(negatives >= previous) > 0.03 for previous in candidates[candidates < t])
    assert result["changed_count"] == int(((probabilities >= t) != (probabilities >= original_threshold)).sum())
    assert result["brier"] == float(np.mean((probabilities-view_run["data"]["y_test"])**2))
    assert sum(r["count"] for r in result["histogram"]) == len(probabilities)
    for r in result["histogram"]:
        assert r["count"] == r["spam"] + r["regular"]
        if r["count"] == 0:
            assert r["mean_score"] is None and r["observed_spam"] is None
    curve = result["validation_curve"]
    assert len(curve) <= 101
    assert all(a["threshold"] <= b["threshold"] and a["caught"] >= b["caught"] and a["false_alarms"] >= b["false_alarms"] for a, b in zip(curve, curve[1:]))
    for point in curve:
        flagged = validation_scores >= point["threshold"]
        assert point["caught"] == int((flagged & (view_run["data"]["y_val"] == 1)).sum())
        assert point["false_alarms"] == int((flagged & (view_run["data"]["y_val"] == 0)).sum())


def test_decision_export_uses_all_real_partitions_and_reject_all(view_run, view_http):
    result = json_get(view_http, "/api/decision-lab", run=view_run["id"], mode="manual", threshold=1.000001)
    assert result["metrics"]["test"]["caught"] == result["metrics"]["test"]["false_alarms"] == 0
    assert result["metrics"]["test"]["precision"] is None
    rows = csv_get(view_http, "/api/decision-lab.csv", run=view_run["id"], model="clean", mode="manual", threshold=1.000001)
    data = view_run["data"]
    assert len(rows) == len(data["y_test"]) + len(data["y_val"])
    assert {int(r["source_row"])-1 for r in rows} == set(data["id_test"].tolist()) | set(data["id_val"].tolist())
    assert all(r["preview_flagged"] == "False" for r in rows)
    assert all(r["report_sha256"] == view_run["payload"]["evidence"]["report_sha256"] for r in rows)


@pytest.mark.parametrize("category", ["disagree", "left-only", "right-only", "both-wrong", "all"])
def test_matched_filter_decisions_are_computed_from_actual_vectors(view_run, view_http, category):
    pa, ta = direct_predictions(view_run, "clean")
    pb, tb = direct_predictions(view_run, "tree")
    y = view_run["data"]["y_test"]
    ca, cb = (pa >= ta) == y, (pb >= tb) == y
    masks = {"disagree": ca != cb, "left-only": ca & ~cb, "right-only": ~ca & cb, "both-wrong": ~ca & ~cb, "all": np.ones(len(y), dtype=bool)}
    result = json_get(view_http, "/api/disagreements", left=view_run["id"], category=category, size=100)
    assert result["matched"] == len(y) and result["left_unmatched"] == result["right_unmatched"] == 0
    assert all(result["counts"][key] == int(mask.sum()) for key, mask in masks.items())
    assert result["counts"]["both_correct"] == int((ca & cb).sum())
    lookup = {int(raw_id)+1: i for i, raw_id in enumerate(view_run["data"]["id_test"])}
    assert [r["source_row"] for r in result["rows"]] == sorted(int(view_run["data"]["id_test"][i])+1 for i in np.flatnonzero(masks[category]))[:100]
    for r in result["rows"]:
        i = lookup[r["source_row"]]
        assert r["left_score"] == float(pa[i]) and r["right_score"] == float(pb[i])
        assert r["left_correct"] == bool(ca[i]) and r["right_correct"] == bool(cb[i])
    rows = csv_get(view_http, "/api/disagreements.csv", left=view_run["id"], category=category, size=10)
    assert len(rows) == result["total"]
    assert all(float(r["left_threshold"]) == ta and float(r["right_threshold"]) == tb for r in rows)


def test_cross_run_disagreements_match_only_shared_source_ids(view_run, different_cohort, view_http):
    result = json_get(view_http, "/api/disagreements", left=view_run["id"], right=different_cohort["id"], category="all", size=100)
    ids = set(map(int, view_run["data"]["id_test"])) & set(map(int, different_cohort["data"]["id_test"]))
    assert result["matched"] == len(ids)
    assert result["left_unmatched"] == len(view_run["data"]["y_test"])-len(ids)
    assert result["right_unmatched"] == len(different_cohort["data"]["y_test"])-len(ids)
    assert {r["source_row"]-1 for r in result["rows"]} == ids
    same = json_get(view_http, "/api/disagreements", left=view_run["id"], left_model="clean", right_model="clean")
    assert same["total"] == 0 and same["counts"]["left-only"] == same["counts"]["right-only"] == 0


@pytest.mark.parametrize("changed", [False, True])
def test_source_record_inspection_recalculates_pinned_models_and_exact_raw_values(view_run, view_http, changed):
    source = int(view_run["data"]["id_test"][0])+1
    model = view_run["payload"]["report"]["poisoning"][0]["model"] if changed else "clean"
    result = json_get(view_http, "/api/record-detail", run=view_run["id"], source_row=source, model=model)
    raw = np.loadtxt(view_run["cfg"]["data_path"], delimiter=",")[source-1]
    np.testing.assert_array_equal([r["value"] for r in result["features"]], raw[:-1])
    np.testing.assert_array_equal([r["model_input"] for r in result["features"]], view_run["data"]["x_test"][0])
    assert result["actual"] == int(raw[-1])
    assert len(result["models"]) == (6 if changed else 5)
    for row in result["models"]:
        p, threshold = direct_predictions(view_run, row["model"])
        assert row["score"] == float(p[0]) and row["threshold"] == threshold
        assert row["correct"] == bool((p[0] >= threshold) == raw[-1])


@pytest.mark.parametrize("route,query", [
    ("/api/decision-lab", {"threshold": "nan"}),
    ("/api/decision-lab", {"threshold": -1}),
    ("/api/decision-lab", {"target_fpr": 0.5}),
    ("/api/decision-lab", {"mode": "unknown"}),
    ("/api/decision-lab", {"model": "missing"}),
    ("/api/decision-lab", {"report_sha256": "outdated"}),
    ("/api/disagreements", {"right_sha": "outdated"}),
    ("/api/disagreements", {"category": "unknown"}),
    ("/api/record-detail", {"source_row": -1}),
])
def test_new_analysis_routes_reject_invalid_or_outdated_requests(view_run, view_http, route, query):
    with pytest.raises(HTTPError) as error:
        json_get(view_http, route, run=view_run["id"], left=view_run["id"], **query)
    assert "error" in json.load(error.value)


def test_analysis_script_is_served_by_the_actual_dashboard(view_http):
    with urlopen(view_http+"/analysis.js") as response:
        assert response.status == 200 and b"function decisionLabPage" in response.read()


@pytest.mark.parametrize("sanitized", [False, True])
def test_decision_lab_runs_changed_and_cleaned_models(view_run, view_http, sanitized):
    row = next(r for r in view_run["payload"]["report"]["poisoning"] if r["sanitized"] is sanitized)
    result = json_get(view_http, "/api/decision-lab", run=view_run["id"], model=row["model"], mode="target", target_fpr=0)
    assert result["metrics"]["validation"]["false_alarms"] == 0
    probabilities, _ = direct_predictions(view_run, row["model"])
    assert result["metrics"]["test"]["caught"] == int(((probabilities >= result["threshold"]) & (view_run["data"]["y_test"] == 1)).sum())


@pytest.mark.parametrize("model", ["clean", "tree", "pgd", "trades", "squeezed"])
def test_importance_trials_equal_independent_real_model_runs(view_run, view_http, model):
    result = json_get(view_http, "/api/importance", run=view_run["id"], model=model, repeats=2, seed=19)
    p, threshold = direct_predictions(view_run, model)
    checkpoint = "clean" if model == "squeezed" else model
    classifier, _, _ = restore(view_run["cfg"], checkpoint, view_run["payload"]["evidence"]["checkpoints"][checkpoint])
    x, y = view_run["data"]["x_test"], view_run["data"]["y_test"]
    assert result["records"] == len(y) and result["feature_checks"] == 114
    assert result["baseline"]["brier"] == float(np.mean((p-y)**2))
    assert len(result["rows"]) == 57
    for row in result["rows"]:
        assert row["distinct_values"] == len(np.unique(x[:, row["column"]]))
        for repeat, trial in enumerate(row["trials"]):
            shuffled = x.copy()
            order = np.random.default_rng(19+row["column"]*1009+repeat).permutation(len(x))
            shuffled[:, row["column"]] = x[order, row["column"]]
            if model == "squeezed":
                shuffled = squeeze(shuffled, Domain(view_run["cfg"]), view_run["cfg"]["quantum"])
            fresh = score(classifier, shuffled)
            assert trial["brier_increase"] == float(np.mean((fresh-y)**2))-float(np.mean((p-y)**2))
            assert trial["changed_decisions"] == int(((fresh >= threshold) != (p >= threshold)).sum())
            assert trial["mean_score_change"] == float(np.mean(np.abs(fresh-p)))
        for key, value in row["mean"].items():
            assert value == float(np.mean([trial[key] for trial in row["trials"]]))
            assert row["spread"][key] == float(np.std([trial[key] for trial in row["trials"]], ddof=1))


@pytest.mark.parametrize("sanitized", [False, True])
def test_importance_validation_partition_and_changed_models(view_run, view_http, sanitized):
    model = next(row["model"] for row in view_run["payload"]["report"]["poisoning"] if row["sanitized"] is sanitized)
    result = json_get(view_http, "/api/importance", run=view_run["id"], model=model, partition="val", repeats=1, seed=0)
    assert result["records"] == len(view_run["data"]["y_val"])
    assert all(all(v == 0 for v in row["spread"].values()) for row in result["rows"])
    classifier, threshold, _ = restore(view_run["cfg"], model, view_run["payload"]["evidence"]["checkpoints"][model])
    p = score(classifier, view_run["data"]["x_val"])
    y = view_run["data"]["y_val"]
    assert result["baseline"]["brier"] == float(np.mean((p-y)**2))
    assert result["baseline"]["caught"] == int(((p >= threshold) & (y == 1)).sum())


def test_importance_repeats_and_csv_are_reproducible_real_results(view_run, view_http):
    args = {"run": view_run["id"], "model": "tree", "partition": "test", "repeats": 2, "seed": 2147483647}
    first, second = json_get(view_http, "/api/importance", **args), json_get(view_http, "/api/importance", **args)
    assert first == second
    rows = csv_get(view_http, "/api/importance.csv", **args)
    assert len(rows) == 114
    expected = {(r["feature"], t["repeat"]): t for r in first["rows"] for t in r["trials"]}
    for row in rows:
        trial = expected[(row["feature"], int(row["repeat"]))]
        assert float(row["brier_increase"]) == trial["brier_increase"]
        assert int(row["changed_decisions"]) == trial["changed_decisions"]
        assert row["report_sha256"] == view_run["payload"]["evidence"]["report_sha256"]
        assert float(row["threshold"]) == first["threshold"]


@pytest.mark.parametrize("target", ["clean", "tree"])
@pytest.mark.parametrize("category", ["all", "escaped", "resisted", "one"])
def test_attack_overlap_recalculates_shared_record_outcomes(view_run, view_http, target, category):
    report = view_run["payload"]["report"]
    root = Path(view_run["cfg"]["root"])
    names = sorted({r["attack"] for r in report["transferability"] if r["target"] == target})
    classifier, threshold, _ = restore(view_run["cfg"], target, view_run["payload"]["evidence"]["checkpoints"][target])
    artifacts = []
    for name in names:
        with np.load(root / "artifacts" / (name+".npz")) as saved:
            artifacts.append({"ids": saved["record_ids"], "before": score(classifier, saved["original"]), "after": score(classifier, saved["adversarial"])})
    common = sorted(set.intersection(*(set(map(int, a["ids"])) for a in artifacts)))
    positions = [{int(raw_id): i for i, raw_id in enumerate(a["ids"])} for a in artifacts]
    eligible = np.array([artifacts[0]["before"][positions[0][i]] >= threshold for i in common])
    matrix = np.array([[a["after"][positions[j][raw_id]] for j, a in enumerate(artifacts)] for raw_id in common])
    escapes = (matrix < threshold) & eligible[:, None]
    number = escapes.sum(axis=1)
    masks = {"all": np.ones(len(common), dtype=bool), "escaped": number > 0, "resisted": eligible & (number == 0), "one": number == 1}
    result = json_get(view_http, "/api/attack-overlap", run=view_run["id"], target=target, category=category, size=100)
    assert result["matched"] == len(common)
    assert result["counts"]["eligible"] == int(eligible.sum())
    assert result["counts"]["already_missed"] == int((~eligible).sum())
    assert all(result["counts"][key] == int(mask.sum()) for key, mask in masks.items())
    assert [r["source_row"]-1 for r in result["rows"]] == np.array(common)[masks[category]].tolist()
    for row in result["rows"]:
        i = common.index(row["source_row"]-1)
        assert row["escaped_count"] == int(number[i]) and row["eligible"] == bool(eligible[i])
        for j, attack in enumerate(row["attacks"]):
            assert attack["score"] == float(matrix[i, j]) and attack["escaped"] == bool(escapes[i, j])
            assert int(artifacts[j]["ids"][attack["sample"]]) == row["source_row"]-1
    for pair in result["overlaps"]:
        a, b = names.index(pair["left"]), names.index(pair["right"])
        both, either = int((escapes[:, a] & escapes[:, b]).sum()), int((escapes[:, a] | escapes[:, b]).sum())
        assert pair["both"] == both and pair["either"] == either
        assert pair["jaccard"] == (both/either if either else None)
    exported = csv_get(view_http, "/api/attack-overlap.csv", run=view_run["id"], target=target, category=category, size=10)
    assert len(exported) == result["total"]*len(names)
    assert all(r["report_sha256"] == view_run["payload"]["evidence"]["report_sha256"] for r in exported)
    if category == "all":
        for attack in result["rows"][0]["attacks"]:
            inspected = json_get(view_http, "/api/inspect", run=view_run["id"], target=target, attack=attack["attack"], sample=attack["sample"])
            assert inspected["source_row"]+1 == result["rows"][0]["source_row"]
            assert inspected["after"] == attack["score"]


@pytest.mark.parametrize("route,query", [
    ("/api/importance", {"repeats": 0}),
    ("/api/importance", {"repeats": 11}),
    ("/api/importance", {"seed": -1}),
    ("/api/importance", {"partition": "train"}),
    ("/api/importance", {"model": "missing"}),
    ("/api/importance", {"report_sha256": "outdated"}),
    ("/api/attack-overlap", {"target": "missing"}),
    ("/api/attack-overlap", {"category": "missing"}),
    ("/api/attack-overlap", {"page": 0}),
    ("/api/attack-overlap", {"size": 7}),
    ("/api/attack-overlap", {"report_sha256": "outdated"}),
])
def test_research_routes_reject_invalid_or_stale_evidence(view_run, view_http, route, query):
    with pytest.raises(HTTPError) as error:
        json_get(view_http, route, run=view_run["id"], **query)
    assert "error" in json.load(error.value)


def test_research_script_is_served(view_http):
    with urlopen(view_http+"/research.js") as response:
        assert response.status == 200 and b"function importancePage" in response.read()
