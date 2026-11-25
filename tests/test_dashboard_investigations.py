"""Real-data budget experiments and actual transactional notebook workflows."""
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from ml_adv_sec.dashboard_investigations import InvestigationViews, CaseNotebook
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.data import Domain
from ml_adv_sec.eval.workflow import evaluate
from ml_adv_sec.models import restore, score
from test_dashboard_views import view_run, view_http, different_cohort, json_get, csv_get


def token_for(base):
    with urlopen(base) as response:
        return re.search(r'name="dashboard-token" content="([^"]+)"', response.read().decode()).group(1)


def post(base, route, values, token=None, origin=None):
    headers = {"Content-Type": "application/json", "X-Dashboard-Token": token_for(base) if token is None else token}
    if origin is not None:
        headers["Origin"] = origin
    with urlopen(Request(base+route, data=json.dumps(values).encode(), headers=headers, method="POST"), timeout=60) as response:
        return json.load(response)


def case_values(view_run, **changes):
    return {"run": view_run["id"], "report_sha256": view_run["payload"]["evidence"]["report_sha256"], "source_row": int(view_run["data"]["id_test"][0])+1, "model": "clean", "title": "Investigation "+uuid.uuid4().hex, "notes": "", "status": "open", "tags": [], **changes}


@pytest.mark.parametrize("model", ["clean", "pgd", "trades", "squeezed-bpda"])
@pytest.mark.parametrize("category", ["all", "escaped", "resisted", "recovered", "already-missed"])
def test_budget_outcomes_equal_independent_runs_of_saved_examples(view_run, view_http, model, category):
    report = view_run["payload"]["report"]
    cfg = view_run["cfg"]
    checkpoint = "clean" if model == "squeezed-bpda" else model
    classifier, _, _ = restore(cfg, checkpoint, view_run["payload"]["evidence"]["checkpoints"][checkpoint])
    threshold = next(r["threshold"] for r in report["clean"] if r["model"] == ("squeezed" if model == "squeezed-bpda" else model))
    measured = sorted([r for r in report["pareto"] if r["model"] == model], key=lambda r: r["epsilon"])
    def predict(x):
        return score(classifier, squeeze(x, Domain(cfg), cfg["quantum"]) if model == "squeezed-bpda" else x)
    after, originals, candidates = [], [], []
    for row in measured:
        with np.load(Path(cfg["root"])/"artifacts"/f"sweep-{model}-{row['epsilon']!r}.npz") as archive:
            original, candidate = archive["original"], archive["adversarial"]
        originals.append(original); candidates.append(candidate); after.append(predict(candidate))
    before = predict(originals[0])
    eligible = before >= threshold
    escapes = np.stack(after, axis=1) < threshold
    escapes &= eligible[:, None]
    recovered = (escapes[:, :-1] & ~escapes[:, 1:]).any(axis=1)
    masks = {"all": np.ones(len(before), dtype=bool), "escaped": escapes.any(axis=1), "resisted": eligible & ~escapes.any(axis=1), "recovered": recovered, "already-missed": ~eligible}
    ids = view_run["data"]["id_test"][view_run["data"]["y_test"] == 1][:len(before)]
    result = json_get(view_http, "/api/budgets", run=view_run["id"], model=model, category=category, size=100)
    assert result["threshold"] == threshold
    assert all(result["counts"][key] == int(mask.sum()) for key, mask in masks.items())
    assert [r["source_row"] for r in result["rows"]] == sorted((ids[masks[category]]+1).tolist())
    for j, row in enumerate(result["levels"]):
        assert row["epsilon"] == measured[j]["epsilon"]
        assert row["escaped"] == int(escapes[:, j].sum())
        assert row["caught_again"] == (int((escapes[:, j-1] & ~escapes[:, j]).sum()) if j else 0)
        assert row["newly_escaped"] == int((escapes[:, j] & ~escapes[:, :j].any(axis=1)).sum())
    for record in result["rows"]:
        i = ids.tolist().index(record["source_row"]-1)
        first = next((row["epsilon"] for j, row in enumerate(measured) if escapes[i, j]), None)
        assert record["first_observed_escape"] == first
        assert record["recovered"] == bool(recovered[i])
        assert [r["score"] for r in record["levels"]] == [float(p[i]) for p in after]
    exported = csv_get(view_http, "/api/budgets.csv", run=view_run["id"], model=model, category=category, scope="records", size=10)
    assert len(exported) == result["total"]*len(measured)
    assert all(r["report_sha256"] == view_run["payload"]["evidence"]["report_sha256"] for r in exported)
    if category == "all":
        schema = json.loads((Path(cfg["root"])/"prepared/schema.json").read_text())
        first_record = result["rows"][0]
        i = ids.tolist().index(first_record["source_row"]-1)
        for j in range(len(measured)):
            detail = json_get(view_http, "/api/budget-record", run=view_run["id"], model=model, level=j, source_row=first_record["source_row"])
            assert detail["after_score"] == float(after[j][i])
            np.testing.assert_allclose([r["before"] for r in detail["features"]], originals[j][i]*schema["scale"]+schema["center"])
            np.testing.assert_allclose([r["after"] for r in detail["features"]], candidates[j][i]*schema["scale"]+schema["center"])
            assert detail["linf"] == float(np.abs(candidates[j][i]-originals[j][i]).max())
            assert detail["linf"] <= detail["epsilon"]+1e-5
            assert all(r["scaled_change"] == 0 for r in detail["features"] if r["locked"])
        summary = csv_get(view_http, "/api/budgets.csv", run=view_run["id"], model=model, scope="summary")
        assert len(summary) == len(measured)


def test_case_create_reopens_real_evidence_and_persists_across_instances(view_run, view_http):
    values = case_values(view_run, notes="First observation\nSecond observation", tags=["review", "review", "actual-record"])
    saved = post(view_http, "/api/cases/create", values)
    assert saved["revision"] == 1 and saved["tags"] == ["review", "actual-record"]
    reopened = json_get(view_http, "/api/case", id=saved["id"])
    assert reopened["case"] == saved and len(reopened["history"]) == 1
    assert reopened["evidence_error"] is None and len(reopened["evidence"]["record"]["features"]) == 57
    assert reopened["evidence"]["record"]["source_row"] == values["source_row"]
    assert reopened["evidence"]["checkpoints"] == view_run["payload"]["evidence"]["checkpoints"]
    persisted = CaseNotebook(InvestigationViews(view_run["reports"])).get(saved["id"])
    assert persisted["case"] == saved and persisted["evidence_error"] is None
    exported = json_get(view_http, "/api/case-export", id=saved["id"])
    assert exported["case"]["notes"] == values["notes"] and exported["evidence"]["record"]["models"] == reopened["evidence"]["record"]["models"]


def test_case_revisions_preserve_bindings_and_reject_lost_updates(view_run, view_http):
    created = post(view_http, "/api/cases/create", case_values(view_run))
    edit = {"id": created["id"], "revision": 1, "title": created["title"], "notes": "Reviewed actual scores", "status": "reviewed", "tags": ["checked"]}
    updated = post(view_http, "/api/cases/update", edit)
    assert updated["revision"] == 2 and updated["report_sha256"] == created["report_sha256"]
    with pytest.raises(HTTPError) as conflict:
        post(view_http, "/api/cases/update", {**edit, "notes": "Outdated edit"})
    assert conflict.value.code == 400
    with pytest.raises(HTTPError):
        post(view_http, "/api/cases/update", {**edit, "revision": 2, "run": "replacement"})
    final = post(view_http, "/api/cases/update", {**edit, "revision": 2, "status": "archived", "notes": "Preserve this evidence reference"})
    reopened = json_get(view_http, "/api/case", id=created["id"])
    assert final["revision"] == 3 and len(reopened["history"]) == 3
    assert [r["revision"] for r in reopened["history"]] == [3, 2, 1]
    assert reopened["history"][1]["notes"] == edit["notes"]
    assert all(r["report_sha256"] == created["report_sha256"] and r["source_row"] == created["source_row"] for r in reopened["history"])


def test_two_real_concurrent_case_edits_allow_only_one_revision(view_run, view_http):
    created = post(view_http, "/api/cases/create", case_values(view_run))
    token = token_for(view_http)
    def edit(number):
        try:
            result = post(view_http, "/api/cases/update", {"id": created["id"], "revision": 1, "title": created["title"], "notes": f"Concurrent observation {number}", "status": "reviewed", "tags": []}, token=token)
            return result["notes"]
        except HTTPError as exc:
            assert exc.code == 400
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, [1, 2]))
    assert sum(r is not None for r in results) == 1
    case = json_get(view_http, "/api/case", id=created["id"])
    assert case["case"]["revision"] == 2 and len(case["history"]) == 2
    assert case["case"]["notes"] == next(r for r in results if r is not None)


def test_case_filters_and_csv_keep_user_text_as_text(view_run, view_http):
    marker = uuid.uuid4().hex[:20]
    values = case_values(view_run, title="=1+1 "+marker, notes="@SUM(1,1)\n<script>user text</script>", tags=["+local", marker], status="resolved")
    created = post(view_http, "/api/cases/create", values)
    matched = json_get(view_http, "/api/cases", run=view_run["id"], status="resolved", search=marker, size=10)
    assert matched["total"] == 1 and matched["rows"][0]["id"] == created["id"]
    assert json_get(view_http, "/api/cases", status="open", search=marker)["total"] == 0
    assert json_get(view_http, "/api/cases", search="' OR 1=1 --"+marker)["total"] == 0
    exported = csv_get(view_http, "/api/cases.csv", search=marker)
    assert len(exported) == 1 and exported[0]["title"] == "'"+values["title"]
    assert exported[0]["notes"] == "'"+values["notes"] and exported[0]["tags"].startswith("'+local")
    assert json_get(view_http, "/api/case", id=created["id"])["case"]["notes"] == values["notes"]


def test_case_keeps_notes_but_no_replacement_scores_after_real_report_update(view_run, different_cohort, view_http):
    other = different_cohort
    values = case_values(view_run, run=other["id"], report_sha256=other["payload"]["evidence"]["report_sha256"], source_row=int(other["data"]["id_test"][0])+1)
    created = post(view_http, "/api/cases/create", values)
    evaluate(other["cfg"])
    reopened = json_get(view_http, "/api/case", id=created["id"])
    assert reopened["evidence"] is None and reopened["evidence_error"]
    assert reopened["case"]["report_sha256"] == values["report_sha256"]
    updated = post(view_http, "/api/cases/update", {"id": created["id"], "revision": 1, "title": created["title"], "notes": "The report was recalculated; keep original reference", "status": "reviewed", "tags": []})
    assert updated["revision"] == 2 and updated["report_sha256"] == values["report_sha256"]
    exported = json_get(view_http, "/api/case-export", id=created["id"])
    assert exported["evidence"] is None and exported["case"]["notes"] == updated["notes"]


@pytest.mark.parametrize("changes", [
    {"source_row": -1}, {"source_row": True}, {"source_row": 1.5}, {"run": []}, {"model": []}, {"model": "missing"},
    {"report_sha256": "0"*64}, {"report_sha256": ""}, {"title": " "}, {"title": []}, {"notes": "x"*4001},
    {"status": "unknown"}, {"tags": "not-a-list"}, {"tags": ["x"]*9}, {"tags": ["x"*25]}, {"extra_field": True},
])
def test_case_creation_rejects_unverified_bindings_and_invalid_notes(view_run, view_http, changes):
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/cases/create", case_values(view_run, **changes))
    assert error.value.code == 400
    assert "error" in json.load(error.value)


def test_case_writes_require_token_and_local_origin_and_cannot_use_get(view_run, view_http):
    values = case_values(view_run)
    for headers in [{"token": "invalid"}, {"origin": "https://outside.example"}]:
        with pytest.raises(HTTPError) as error:
            post(view_http, "/api/cases/create", values, **headers)
        assert error.value.code == 403
    for route in ["/api/cases/create", "/api/cases/update"]:
        with pytest.raises(HTTPError) as error:
            json_get(view_http, route)
        assert error.value.code == 404
    train_id = int(view_run["data"]["id_train"][0])+1
    with pytest.raises(HTTPError):
        post(view_http, "/api/cases/create", {**values, "source_row": train_id})


@pytest.mark.parametrize("route,query", [
    ("/api/budgets", {"model": "tree"}), ("/api/budgets", {"category": "unknown"}), ("/api/budgets", {"page": 0}),
    ("/api/budgets", {"report_sha256": "outdated"}), ("/api/budgets.csv", {"scope": "unknown"}),
    ("/api/budget-record", {"level": -1}), ("/api/budget-record", {"level": 999}), ("/api/budget-record", {"source_row": -1}),
    ("/api/cases", {"status": "unknown"}), ("/api/cases", {"size": 7}), ("/api/case", {"id": "missing"}),
])
def test_investigation_routes_reject_invalid_inputs(view_run, view_http, route, query):
    with pytest.raises(HTTPError) as error:
        json_get(view_http, route, run=view_run["id"], **query)
    assert "error" in json.load(error.value)


def test_investigation_script_is_served(view_http):
    with urlopen(view_http+"/investigations.js") as response:
        assert response.status == 200 and b"function notebookPage" in response.read()
