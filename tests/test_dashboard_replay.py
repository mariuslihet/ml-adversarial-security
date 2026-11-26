"""Replay complete real attack batches through independently restored models."""
import json
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest

from ml_adv_sec.dashboard_replay import ReplayViews
from ml_adv_sec.data import Domain
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.models import restore, score
from test_dashboard_views import view_run, view_http, same_records_fewer_attacks, different_cohort, json_get, csv_get


def attacks(run):
    return sorted({r["attack"] for r in run["payload"]["report"]["transferability"]})


def independent(run, attack, model):
    cfg = run["cfg"]
    with np.load(Path(cfg["root"])/"artifacts"/(attack+".npz")) as archive:
        ids, x, adv = [archive[k] for k in ["record_ids", "original", "adversarial"]]
    target = "clean" if model == "squeezed" else model
    predictor, _, _ = restore(cfg, target, run["payload"]["evidence"]["checkpoints"][target])
    if model == "squeezed":
        x, adv = [squeeze(values, Domain(cfg), cfg["quantum"]) for values in [x, adv]]
    threshold = next(r["threshold"] for r in run["payload"]["report"]["clean"]+run["payload"]["report"]["poisoning"] if r["model"] == model)
    return ids, score(predictor, x), score(predictor, adv), threshold


@pytest.mark.parametrize("attack_index", range(6))
def test_all_filters_run_actual_saved_attacks_with_correct_eligibility(view_run, attack_index):
    attack = attacks(view_run)[attack_index]
    result = ReplayViews(view_run["reports"]).attack_replay(view_run["id"], attack, "tree")
    assert len(result["models"]) == len(view_run["payload"]["report"]["clean"]+view_run["payload"]["report"]["poisoning"])
    for actual in result["models"]:
        ids, before, after, threshold = independent(view_run, attack, actual["model"])
        eligible, escaped = before >= threshold, (before >= threshold) & (after < threshold)
        assert actual["initially_caught"] == int(eligible.sum())
        assert actual["escaped"] == int(escaped.sum())
        assert actual["escape_rate"] == (float(escaped.sum()/eligible.sum()) if eligible.any() else None)
        assert actual["caught_after"] == int((after >= threshold).sum())
        assert actual["newly_caught"] == int(((before < threshold) & (after >= threshold)).sum())
        assert actual["mean_score_change"] == float(np.mean(after-before))
    assert result["counts"]["changed"] == 0
    assert result["counts"]["all"] == len(ids)


@pytest.mark.parametrize("category", ["all", "first-only", "second-only", "both-caught", "both-missed", "changed"])
def test_attack_pairing_filters_and_complete_exports_match_independent_decisions(view_run, view_http, category):
    first, second = attacks(view_run)[0], attacks(view_run)[-1]
    query = {"run": view_run["id"], "attack": first, "other_attack": second, "model": "tree", "category": category, "size": 10}
    result = json_get(view_http, "/api/attack-replay", **query)
    ids_a, before_a, after_a, t_a = independent(view_run, first, "tree")
    ids_b, before_b, after_b, t_b = independent(view_run, second, "tree")
    expected = {}
    for raw_id in np.intersect1d(ids_a, ids_b):
        i, j = np.flatnonzero(ids_a == raw_id)[0], np.flatnonzero(ids_b == raw_id)[0]
        a, b = after_a[i] >= t_a, after_b[j] >= t_b
        outcome = "both-caught" if a and b else "first-only" if a else "second-only" if b else "both-missed"
        if category == "all" or outcome == category or (category == "changed" and a != b):
            expected[int(raw_id)+1] = (before_a[i], after_a[i], before_b[j], after_b[j], outcome)
    exported = csv_get(view_http, "/api/attack-replay.csv", **query)
    assert len(exported) == result["total"] == len(expected)
    assert len({r["source_row"] for r in exported}) == len(exported)
    for row in exported:
        actual = expected[int(row["source_row"])]
        assert tuple(float(row[k]) for k in ["first_before", "first_after", "second_before", "second_after"]) == tuple(float(v) for v in actual[:4])
        assert row["outcome"] == actual[4]
    pages = []
    for page in range(1, result["pages"]+1):
        pages.extend(json_get(view_http, "/api/attack-replay", **query, page=page)["rows"])
    assert {r["source_row"] for r in pages} == set(expected)
    assert json_get(view_http, "/api/attack-replay", **query, page=999)["page"] == result["pages"]


def test_compare_runs_pairs_only_shared_attacked_source_records(view_run, same_records_fewer_attacks, view_http):
    attack = attacks(view_run)[0]
    result = json_get(view_http, "/api/attack-replay", run=view_run["id"], attack=attack, other=same_records_fewer_attacks["id"], model="squeezed")
    a = independent(view_run, attack, "squeezed")
    b = independent(same_records_fewer_attacks, attack, "squeezed")
    assert result["counts"]["all"] == len(np.intersect1d(a[0], b[0]))
    assert result["first_unmatched"] == len(a[0])-result["counts"]["all"] > 0
    assert result["second_unmatched"] == 0
    for row in result["rows"]:
        index = np.flatnonzero(b[0] == row["source_row"]-1)[0]
        assert row["second_after"] == float(b[2][index])


def test_different_source_selection_is_never_paired_by_position(view_run, different_cohort):
    attack = attacks(view_run)[0]
    result = ReplayViews(view_run["reports"]).attack_replay(view_run["id"], attack, other=different_cohort["id"])
    a = independent(view_run, attack, "clean")[0]
    b = independent(different_cohort, attack, "clean")[0]
    assert {r["source_row"] for r in result["rows"]} == set(int(i)+1 for i in np.intersect1d(a, b))
    assert result["first_unmatched"] == len(a)-len(np.intersect1d(a, b))


def test_record_measurements_and_exports_reconstruct_both_real_attacks(view_run, view_http):
    first, second = attacks(view_run)[0], attacks(view_run)[-1]
    query = {"run": view_run["id"], "attack": first, "other_attack": second, "model": "tree"}
    result = json_get(view_http, "/api/attack-replay", **query)
    row = result["rows"][0]
    detail = json_get(view_http, "/api/attack-replay-record", **query, source_row=row["source_row"])
    assert detail["record"] == row
    assert len(detail["changes"]) == 57
    schema = json.loads((Path(view_run["cfg"]["root"])/"prepared/schema.json").read_text())
    for side, attack in [("first", first), ("second", second)]:
        with np.load(Path(view_run["cfg"]["root"])/"artifacts"/(attack+".npz")) as archive:
            i = np.flatnonzero(archive["record_ids"] == row["source_row"]-1)[0]
            for r in detail["changes"]:
                column = r["column"]
                assert r[side+"_original"] == float(archive["original"][i, column])*schema["scale"][column]+schema["center"][column]
                assert r[side+"_changed"] == float(archive["adversarial"][i, column])*schema["scale"][column]+schema["center"][column]
                assert r[side+"_locked"] == (column in schema["immutable"])
    exported = csv_get(view_http, "/api/attack-replay.csv", **query, scope="changes", source_row=row["source_row"])
    assert len(exported) == 57 and len({r["name"] for r in exported}) == 57
    models = csv_get(view_http, "/api/attack-replay.csv", **query, scope="models")
    assert len({r["model"] for r in models}) == len(result["models"])


@pytest.mark.parametrize("params", [
    {"attack": "../outside"}, {"model": "missing"}, {"category": "unknown"}, {"page": 0}, {"size": 1},
    {"report_sha256": "outdated"}, {"other_sha256": "outdated"}, {"source_row": 999999}, {"other_attack": "missing"},
])
def test_invalid_replay_requests_rejected(view_run, view_http, params):
    query = {"run": view_run["id"], "attack": attacks(view_run)[0], **params}
    with pytest.raises(HTTPError) as error:
        json_get(view_http, "/api/attack-replay", **query)
    assert error.value.code == 400


@pytest.mark.parametrize("params", [{"scope": "unknown"}, {"scope": "changes"}, {"scope": "models", "source_row": 1}, {"scope": "records", "source_row": 1}])
def test_invalid_export_shapes_return_useful_errors(view_run, view_http, params):
    with pytest.raises(HTTPError) as error:
        csv_get(view_http, "/api/attack-replay.csv", run=view_run["id"], attack=attacks(view_run)[0], **params)
    assert error.value.code == 400
