"""Real HTTP attackers, actual trained targets, and independent evidence checks."""
import json
import hashlib
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from ml_adv_sec.attacks.local_oracle import LocalOracle
from ml_adv_sec.attacks.query_search import EditRules, query_attack
from ml_adv_sec.common import sha256
from ml_adv_sec.dashboard_adaptive import AdaptiveTests
from ml_adv_sec.dashboard_lab import ModelLabViews
from ml_adv_sec.data import Domain
from ml_adv_sec.defenses import Detector, squeeze
from ml_adv_sec.models import restore, score
from test_dashboard_views import view_run, view_http, json_get, csv_get
from test_dashboard_investigations import post


def settings(run_state, **changes):
    return {"run": run_state["id"], "report_sha256": run_state["payload"]["evidence"]["report_sha256"], "records": 2, "seeds": [42], "epsilons": [.1], "queries": 8, **changes}


def independent_target(run, result):
    cfg, recipe = run["cfg"], result["target"]
    with np.load(Path(cfg["root"])/"prepared/splits.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    cache = {}
    def predict(model, x):
        name = "clean" if model == "squeezed" else model
        if name not in cache:
            cache[name] = restore(cfg, name, run["payload"]["evidence"]["checkpoints"][name])[0]
        return score(cache[name], squeeze(x, Domain(cfg), cfg["quantum"]) if model == "squeezed" else x)
    p = recipe["parameters"]
    if recipe["kind"] == "ensemble":
        return lambda x: np.column_stack([predict(m, x) for m in p["members"]])@p["weights"]
    if recipe["kind"] == "calibrated":
        base = lambda x: predict(p["model"], x)
        if p["method"] == "sigmoid":
            return lambda x: expit(p["slope"]*logit(np.clip(base(x).astype(float), p["probability_clip"], 1-p["probability_clip"]))+p["intercept"])
        fit, _ = ModelLabViews.lab_split(data, cfg["seed"])
        regression = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(base(data["x_val"])[fit], data["y_val"][fit])
        return lambda x: regression.predict(base(x))
    if recipe["kind"] == "detector":
        classifier = restore(cfg, "clean", run["payload"]["evidence"]["checkpoints"]["clean"])[0]
        detector = Detector().fit(classifier, data["x_train"])
        return lambda x: detector.conservative(classifier, x, score(classifier, x))
    return lambda x: predict(p["model"], x)


@pytest.mark.parametrize("target,model,method", [("filter", "tree", "sigmoid"), ("filter", "squeezed", "sigmoid"), ("ensemble", "tree", "sigmoid"), ("calibrated", "tree", "sigmoid"), ("calibrated", "tree", "isotonic"), ("detector", "clean", "sigmoid")])
def test_real_target_queries_and_final_inputs_reproduce_exactly(view_run, target, model, method):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    result = manager.run(settings(view_run, target=target, model=model, method=method))
    predictor = independent_target(view_run, result)
    raw = np.loadtxt(view_run["cfg"]["data_path"], delimiter=",")
    domain = Domain(view_run["cfg"])
    with np.load(Path(view_run["cfg"]["root"])/"prepared/splits.npz") as archive:
        ids, test = archive["id_test"], archive["x_test"]
    saved = manager.read(result["id"], traces=True)
    assert saved["sampling_reference"] == "clean"
    assert saved["paired_trials"] == len(saved["rows"]) == 8
    assert saved["attacker_queries"] == sum(r["queries"] for r in saved["rows"])
    assert saved["defender_audit_queries"] == 2*len(saved["rows"])
    assert saved["attacker_queries"] <= 8*8
    for r in saved["rows"]:
        source = r["source_row"]-1
        position = int(np.flatnonzero(ids == source)[0])
        original = np.array(saved["examples"]["original"][r["index"]])
        changed = np.array(saved["examples"]["changed"][r["index"]])
        np.testing.assert_array_equal(original, raw[source, :-1])
        rules = EditRules(domain, test[position], original, .1, np.inf, 4, "both", "paired")
        assert rules.valid(changed)
        assert abs(original[:48].sum()-changed[:48].sum()) < 1e-8
        assert abs(original[48:54].sum()-changed[48:54].sum()) < 1e-8
        np.testing.assert_array_equal(changed[domain.immutable], original[domain.immutable])
        assert r["before"] == float(predictor(test[position][None])[0])
        assert r["after"] == float(predictor(rules.scaled(changed)[None])[0])
        assert r["queries"] == len(r["trace"]) <= 8
        assert [t["query"] for t in r["trace"]] == list(range(1, r["queries"]+1))
        assert all(t["request_ms"] >= 0 for t in r["trace"])
        for observation in r["trace"]:
            requested = original.copy()
            for column, value in observation["edits"]:
                requested[column] = value
            assert hashlib.sha256(requested.astype("<f8").tobytes()).hexdigest() == observation["input_sha256"]
            assert rules.valid(requested)
            probability = float(predictor(rules.scaled(requested)[None])[0])
            feedback = int(probability >= saved["target"]["threshold"]) if r["algorithm"] == "decision" else probability
            assert observation["feedback"] == feedback
        if r["algorithm"] == "decision":
            assert all(type(t["feedback"]) is int and t["feedback"] in {0, 1} for t in r["trace"])
    assert AdaptiveTests(ModelLabViews(view_run["reports"])).read(result["id"])["evidence_sha256"] == result["evidence_sha256"]


@pytest.mark.parametrize("access", ["score", "decision"])
def test_prediction_api_enforces_quota_and_feedback_privacy(view_run, access):
    views = ModelLabViews(view_run["reports"])
    payload, root, data, _ = views.lab_data(view_run["id"], None)
    row, predict, _, _ = views.lab_model(payload, root, data, "tree")
    source = int(data["id_test"][0]);raw = np.loadtxt(view_run["cfg"]["data_path"], delimiter=",")[source, :-1]
    rules = EditRules(Domain(view_run["cfg"]), data["x_test"][0], raw, .1, np.inf, 4, "both", "paired")
    oracle = LocalOracle(predict, row["threshold"])
    try:
        with oracle.session(rules, access, 8) as client:
            assert not hasattr(client, "state") and not hasattr(client, "host")
            invalid = raw.copy();invalid[55] += 1
            with pytest.raises(ValueError, match="declared measurement rules"):
                client.ask(invalid)
            assert client.queries == 0
            for i in range(8):
                answer = client.ask(raw)
                assert answer["query"] == i+1
                assert ("score" in answer) == (access == "score")
            with pytest.raises(ValueError, match="budget is exhausted"):
                client.ask(raw)
            assert client.queries == 8
            client._queries = 0
            with pytest.raises(ValueError, match="budget is exhausted"):
                client.ask(raw)
            assert oracle.query_count(client.token) == 8
    finally:
        oracle.close()


def test_repeated_seeds_levels_and_both_goals_use_paired_records(view_run, view_http):
    result = post(view_http, "/api/adaptive-tests", settings(view_run, records=4, seeds=[7, 19], epsilons=[.05, .1]))
    assert len(result["rows"]) == 64
    expected = {(r["source_row"], r["actual"]) for r in result["cohort"]}
    for epsilon in [.05, .1]:
        for algorithm in ["score", "decision"]:
            for seed in [7, 19]:
                rows = [r for r in result["rows"] if r["epsilon"] == epsilon and r["algorithm"] == algorithm and r["seed"] == seed]
                assert {(r["source_row"], r["actual"]) for r in rows} == expected
    for row in result["summary"]:
        trials = [r for r in result["rows"] if (r["actual"], r["epsilon"], r["algorithm"]) == (row["actual"], row["epsilon"], row["algorithm"])]
        eligible = [r for r in trials if r["eligible"]]
        assert row["eligible_trials"] == len(eligible)
        rates = [sum(r["escaped"] for r in eligible if r["seed"] == seed)/sum(r["seed"] == seed for r in eligible) for seed in [7, 19]] if eligible else []
        assert row["mean_escape_rate"] == (float(np.mean(rates)) if rates else None)
        assert row["escape_rate_spread"] == (float(np.std(rates, ddof=1)) if rates else None)
    for curve in result["query_curves"]:
        eligible = [r for r in result["rows"] if r["eligible"] and (r["actual"], r["epsilon"], r["algorithm"]) == (curve["actual"], curve["epsilon"], curve["algorithm"])]
        found = sum(r["first_error_query"] is not None and r["first_error_query"] <= curve["query_limit"] for r in eligible)
        assert curve["errors_found"] == found
    for scope, count in [("summary", 8), ("records", 64), ("queries", len(result["query_curves"])), ("changes", 57)]:
        exported = csv_get(view_http, "/api/adaptive-test.csv", id=result["id"], scope=scope, index=0)
        assert len(exported) == count
        assert all(r["evidence_sha256"] == result["evidence_sha256"] for r in exported)
    detail = json_get(view_http, "/api/adaptive-record", id=result["id"], index=0)
    assert len(detail["features"]) == 57 and detail["row"]["queries"] == len(detail["row"]["trace"])
    queried = json_get(view_http, "/api/adaptive-record", id=result["id"], index=0, query=detail["row"]["queries"])
    assert queried["input_scope"] == f"API query {detail['row']['queries']}"
    changes = csv_get(view_http, "/api/adaptive-test.csv", id=result["id"], scope="changes", index=0, query=1)
    assert all(int(r["source_row"]) == result["rows"][0]["source_row"] and r["input_scope"] == "API query 1" for r in changes)
    assert json_get(view_http, "/api/adaptive-test.json", id=result["id"])["examples"]


@pytest.mark.parametrize("policy,norm,space", [("independent", "l2", "words"), ("paired", "l2", "characters"), ("paired", "linf", "both")])
def test_independent_and_paired_edits_obey_declared_geometry(view_run, policy, norm, space):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    result = manager.run(settings(view_run, records=1, goal="spam", algorithms=["score"], policy=policy, norm=norm, space=space, queries=16))
    full = manager.read(result["id"], traces=True)
    for row in full["rows"]:
        original = np.array(full["examples"]["original"][row["index"]]);changed = np.array(full["examples"]["changed"][row["index"]])
        assert len(np.flatnonzero(np.abs(changed-original)>1e-9)) <= 4
        assert row["l2" if norm == "l2" else "linf"] <= .1+1e-7
        if space == "words":np.testing.assert_array_equal(original[48:], changed[48:])
        if space == "characters":np.testing.assert_array_equal(original[:48], changed[:48])


def test_saved_evidence_and_request_changes_are_rejected(view_run):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    result = manager.run(settings(view_run, records=1, algorithms=["decision"], goal="regular"))
    directory = manager.directory/result["id"]
    request = json.loads((directory/"request.json").read_text(encoding="utf-8"));request["queries"] = 512
    (directory/"request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        manager.read(result["id"])
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"));manifest["files"]["request.json"] = sha256(directory/"request.json")
    (directory/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="settings disagree"):
        manager.read(result["id"])


@pytest.mark.parametrize("change", [{"queries": True}, {"queries": 1000}, {"records": 0}, {"epsilons": [float("nan")]}, {"algorithms": [{}]}, {"seeds": [1, 1]}, {"policy": "paired", "max_fields": 1}, {"members": [1]}, {"target": "external"}, {"run": "../outside"}, {"report_sha256": "stale"}, {"records": 32, "queries": 512, "seeds": [1, 2, 3, 4, 5]}, {"target": "ensemble", "members": ["clean"]}])
def test_invalid_or_unbounded_studies_rejected(view_run, view_http, change):
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/adaptive-tests", settings(view_run, **change))
    assert error.value.code in {400, 503}


def test_adaptive_write_requires_local_authorization(view_http):
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/adaptive-tests", {}, token="invalid")
    assert error.value.code == 403


def test_fixed_target_and_attack_seed_reproduce_inputs_and_feedback(view_run):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    a = manager.run(settings(view_run, records=1, queries=16))
    b = manager.run(settings(view_run, records=1, queries=16))
    a, b = [manager.read(r["id"], traces=True) for r in [a, b]]
    np.testing.assert_array_equal(a["examples"]["changed"], b["examples"]["changed"])
    assert a["summary"] == b["summary"] and a["query_curves"] == b["query_curves"]
    assert a["implementation"] == b["implementation"]
    for ra, rb in zip(a["rows"], b["rows"]):
        assert [{k: v for k, v in t.items() if k != "request_ms"} for t in ra["trace"]] == [{k: v for k, v in t.items() if k != "request_ms"} for t in rb["trace"]]


def test_changed_source_report_does_not_rebind_saved_study(view_run):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    result = manager.run(settings(view_run, records=1, algorithms=["score"], goal="regular"))
    report = Path(view_run["cfg"]["report_dir"])/"benchmark_results.json"
    original = report.read_bytes()
    try:
        report.write_bytes(original+b"\n")
        with pytest.raises(ValueError, match="result file changed"):
            manager.read(result["id"])
    finally:
        report.write_bytes(original)
    assert manager.read(result["id"])["evidence_sha256"] == result["evidence_sha256"]


def test_concurrent_real_requests_allow_only_one_expensive_study(view_run, view_http):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    barrier = threading.Barrier(2)
    def start():
        barrier.wait()
        try:
            return post(view_http, "/api/adaptive-tests", settings(view_run, queries=32))
        except HTTPError as error:
            assert error.code == 400
            return json.loads(error.read())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start(), range(2)))
    assert sum("id" in r for r in results) == 1
    assert sum("Another adaptive test is running" in r.get("error", "") for r in results) == 1


def test_original_score_cohort_is_fixed_across_different_defenders(view_run):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    first = manager.run(settings(view_run, records=2, target="filter", model="tree"))
    second = manager.run(settings(view_run, records=2, target="ensemble", seeds=[73]))
    assert first["cohort"] == second["cohort"]
