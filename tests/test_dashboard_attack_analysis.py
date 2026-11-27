"""Fresh real-model transfer, exhaustive edit subsets, and deduplicated coverage."""
import hashlib
import itertools
import json
from urllib.error import HTTPError

import numpy as np
import pytest

from ml_adv_sec.dashboard_adaptive import AdaptiveTests
from ml_adv_sec.dashboard_attack_analysis import AttackAnalysis
from ml_adv_sec.dashboard_lab import ModelLabViews
from ml_adv_sec.data import Domain
from ml_adv_sec.models import restore, score
from test_dashboard_adaptive import settings, independent_target
from test_dashboard_views import view_run, view_http
from test_dashboard_investigations import post


@pytest.fixture(scope="module")
def studies(view_run):
    manager = AdaptiveTests(ModelLabViews(view_run["reports"]))
    first = manager.run(settings(view_run, queries=24, max_fields=6, epsilons=[.5]))
    second = manager.run(settings(view_run, queries=16, seeds=[73], max_fields=4, epsilons=[.2], policy="independent"))
    return manager, [first["id"], second["id"]]


@pytest.mark.parametrize("tool", ["transfer", "coverage"])
def test_actual_model_feedback_and_unique_coverage(view_run, view_http, studies, tool):
    manager, ids = studies
    selected = ids[:1] if tool == "transfer" else ids
    result = post(view_http, "/api/attack-analysis", {"tool": tool, "studies": selected, "members": ["clean", "tree", "squeezed"]})
    payload = dict(result);digest = payload.pop("result_sha256")
    assert digest == hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert result["defender_predictions"] == 2*len(result["rows"])
    data = view_run["data"]
    source_data = np.loadtxt(view_run["cfg"]["data_path"], delimiter=",")
    from ml_adv_sec.defenses import squeeze
    for model in ["clean", "tree", "squeezed"]:
        fitted = restore(view_run["cfg"], "clean" if model == "squeezed" else model, view_run["payload"]["evidence"]["checkpoints"]["clean" if model == "squeezed" else model])[0]
        rows = [r for r in result["rows"] if r["model"] == model]
        for row in rows:
            pos = int(np.flatnonzero(data["id_test"]+1 == row["source_row"])[0])
            original = source_data[row["source_row"]-1, :-1]
            edited = original.copy()
            x = data["x_test"][pos].copy()
            domain = Domain(view_run["cfg"])
            for col, value in row["edits"]:
                edited[col] = value
                x[col] = (value-domain.center[col])/domain.scale[col]
            assert hashlib.sha256(edited.astype("<f8").tobytes()).hexdigest() == row["input_sha256"]
            prepared = squeeze(x[None], domain, view_run["cfg"]["quantum"]) if model == "squeezed" else x[None]
            assert row["after"] == float(score(fitted, prepared)[0])
            assert row["caused_error"] == (row["eligible"] and (row["after"] >= row["threshold"]) != bool(row["actual"]))
        summary = next(s for s in result["summary"] if s["model"] == model)
        failed = {r["source_row"] for r in rows if r["caused_error"]}
        assert summary["sources_failing_at_least_once"] == len(failed)
        single = [{r["source_row"] for r in rows if r["study"] == i and r["caused_error"]} for i in selected]
        assert summary["added_beyond_best_single_study"] == len(failed)-max(map(len, single))
        assert summary["sources_failed_in_every_study"] == len(set.intersection(*single))


def test_reduction_exhausts_exact_valid_subsets(view_run, view_http, studies):
    manager, ids = studies
    saved = manager.read(ids[0], traces=True)
    index = max(range(len(saved["rows"])), key=lambda i: saved["rows"][i]["changed_fields"])
    result = post(view_http, "/api/attack-analysis", {"tool": "reduction", "studies": ids[:1], "index": index})
    from ml_adv_sec.attacks.query_search import EditRules
    r = saved["rows"][index]
    pos = int(np.flatnonzero(view_run["data"]["id_test"]+1 == r["source_row"])[0])
    original, changed = np.array(saved["examples"]["original"][index]), np.array(saved["examples"]["changed"][index])
    rules = EditRules(Domain(view_run["cfg"]), view_run["data"]["x_test"][pos], original, .5, np.inf, 6, "both", "paired")
    valid = {}
    fields = rules.changed(changed).tolist()
    for n in range(len(fields)+1):
        for subset in itertools.combinations(fields, n):
            raw = original.copy();raw[list(subset)] = changed[list(subset)]
            if rules.valid(raw):valid[subset] = raw
    assert len(result["rows"])+result["invalid_subsets"] == 2**len(fields)
    assert len(valid) == len(result["rows"])
    predictor = independent_target(view_run, saved)
    for row in result["rows"]:
        raw = valid[tuple(row["retained_columns"])]
        assert row["after"] == float(predictor(rules.scaled(raw)[None])[0])
    assert result["summary"][0]["minimum_retained_fields"] == min((r["changed_fields"] for r in result["rows"] if r["caused_error"]), default=None)


@pytest.mark.parametrize("change", [{"tool":"unknown"}, {"tool":[]}, {"studies":[]}, {"members":[]}, {"members":["clean","clean"]}, {"members":["missing"]}, {"tool":"reduction","members":None,"index":0}, {"tool":"reduction","index":-1}, {"tool":"reduction","index":True}, {"extra":1}])
def test_invalid_analysis_is_rejected(view_http, studies, change):
    _, ids = studies
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/attack-analysis", {"tool":"transfer", "studies":ids[:1], "members":["clean"], **change})
    assert error.value.code == 400


def test_analysis_rejects_damaged_evidence_and_requires_local_token(view_http, studies):
    manager, ids = studies
    p={"tool":"transfer","studies":ids[:1],"members":["clean"]}
    with pytest.raises(HTTPError) as error:post(view_http,"/api/attack-analysis",p,token="invalid")
    assert error.value.code == 403
    path=manager.directory/ids[0]/"examples.npz"
    original=path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        with pytest.raises(HTTPError) as error:post(view_http,"/api/attack-analysis",p)
        assert error.value.code == 400
    finally:path.write_bytes(original)


def test_analysis_concurrency_guard(studies):
    manager, ids=studies
    analysis=AttackAnalysis(manager)
    analysis.lock.acquire()
    try:
        with pytest.raises(ValueError, match="Another attack analysis"):
            analysis.run({"tool":"transfer","studies":ids[:1],"members":["clean"]})
    finally:analysis.lock.release()


@pytest.mark.parametrize("target", [{"target":"ensemble"}, {"target":"calibrated","method":"sigmoid"}, {"target":"calibrated","method":"isotonic"}, {"target":"detector"}, {"target":"filter","model":"squeezed"}])
def test_reduction_preserves_actual_target_recipe(view_run, view_http, studies, target):
    manager, _ = studies
    saved = manager.run(settings(view_run, records=1, **target))
    result = post(view_http, "/api/attack-analysis", {"tool":"reduction", "studies":[saved["id"]], "index":0})
    evidence = manager.read(saved["id"], traces=True)
    fresh = independent_target(view_run, evidence)
    original = np.array(evidence["examples"]["original"][0])
    source_row = evidence["rows"][0]["source_row"]
    pos = int(np.flatnonzero(view_run["data"]["id_test"]+1 == source_row)[0])
    domain = Domain(view_run["cfg"])
    for row in result["rows"]:
        x = view_run["data"]["x_test"][pos].copy()
        for column, value in row["edits"]:x[column] = (value-domain.center[column])/domain.scale[column]
        assert row["before"] == float(fresh(view_run["data"]["x_test"][pos][None])[0])
        assert row["after"] == float(fresh(x[None])[0])
        assert row["threshold"] == evidence["target"]["threshold"]
