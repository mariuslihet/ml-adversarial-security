"""Actual preset revisions, fresh pipeline reruns, and corrected retry recovery."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from ml_adv_sec.dashboard_investigations import InvestigationViews
from ml_adv_sec.dashboard_presets import ExperimentPresets
from test_dashboard_views import view_run, view_http, json_get
from test_dashboard_campaigns import SMALL, owned_manager, wait_job
from test_dashboard_investigations import post


def library(tmp_path, view_run):
    jobs, manager = owned_manager(tmp_path, view_run)
    return jobs, manager, ExperimentPresets(jobs, manager, InvestigationViews(jobs.reports))


def test_versioned_presets_persist_and_keep_old_configurations(tmp_path, view_run):
    jobs, manager, presets = library(tmp_path, view_run)
    try:
        first = presets.save({"name": "Real baseline", "notes": "One training round", "settings": SMALL})
        second = presets.save({"id": first["id"], "revision": 1, "name": "Real baseline", "notes": "Two training rounds", "settings": {**SMALL, "epochs": 2}})
        assert second["revision"] == 2 and second["source"]["kind"] == "preset-edit"
        reopened = ExperimentPresets(jobs, manager, InvestigationViews(jobs.reports)).get(first["id"], 1)
        assert reopened["preset"]["settings"]["epochs"] == 1
        assert [r["revision"] for r in reopened["history"]] == [2, 1]
        assert presets.get(first["id"])["preset"]["settings"]["epochs"] == 2
        with pytest.raises(ValueError):
            presets.save({"id": first["id"], "revision": 1, "name": "Outdated edit", "settings": SMALL})
        assert presets.get(first["id"])["preset"] == second
    finally:
        manager.close(); jobs.close()


def test_archiving_does_not_rebind_incompatible_presets_to_new_fixed_settings(tmp_path, view_run):
    jobs, manager, presets = library(tmp_path, view_run)
    try:
        first = presets.save({"name": "Keep fixed context", "settings": SMALL})
        jobs.reports.cfg = {**jobs.reports.cfg, "threads": 2}
        assert not presets.get(first["id"])["compatible"]
        archived = presets.save({"id": first["id"], "revision": 1, "name": first["name"], "settings": first["settings"], "archived": True})
        assert archived["fixed_context"] == first["fixed_context"] and archived["config_sha256"] == first["config_sha256"]
        assert presets.listing() == [] and len(presets.listing(True)) == 1
        with pytest.raises(ValueError):
            presets.run({"id": first["id"], "revision": 1})
        assert jobs.active is None
    finally:
        manager.close(); jobs.close()


def test_preset_and_configuration_replay_execute_two_fresh_real_pipelines(tmp_path, view_run):
    jobs, manager, presets = library(tmp_path, view_run)
    try:
        saved = presets.save({"name": "Measured repeatable configuration", "settings": SMALL})
        first = presets.run({"id": saved["id"], "revision": 1})
        wait_job(jobs)
        first_job = json.loads((jobs.directory/first["id"]/"job.json").read_text())
        assert first_job["status"] == "complete" and first_job["exit_code"] == 0
        assert first_job["origin"]["kind"] == "preset" and first_job["origin"]["revision"] == 1
        first_root = Path(first_job["settings"]["root"])
        original_report = (first_root/"reports/benchmark_results.json").read_bytes()
        snapshot = presets.replay_settings(job_id=first["id"])
        with pytest.raises(ValueError):
            presets.replay({"job_id": first["id"], "config_sha256": "outdated"})
        second = presets.replay({"job_id": first["id"], "config_sha256": snapshot["config_sha256"], "name": "Fresh rerun of actual process"})
        wait_job(jobs)
        second_job = json.loads((jobs.directory/second["id"]/"job.json").read_text())
        assert second_job["status"] == "complete" and second_job["exit_code"] == 0
        assert second["id"] != first["id"] and second_job["settings"]["root"] != first_job["settings"]["root"]
        assert second_job["origin"]["job"] == first["id"]
        assert (first_root/"reports/benchmark_results.json").read_bytes() == original_report
        second_root = Path(second_job["settings"]["root"])
        with np.load(first_root/"prepared/splits.npz") as a, np.load(second_root/"prepared/splits.npz") as b:
            np.testing.assert_array_equal(a["id_test"], b["id_test"])
        with np.load(first_root/"artifacts/evaluation-scores.npz") as a, np.load(second_root/"artifacts/evaluation-scores.npz") as b:
            for model in ["clean", "tree", "pgd", "trades", "squeezed"]:
                np.testing.assert_array_equal(a[model], b[model])
        assert "Running train" in (jobs.directory/second["id"]/"output.log").read_text()
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_corrected_retry_forks_only_failed_started_entries_and_preserves_original(tmp_path, view_run):
    jobs, manager, _ = library(tmp_path, view_run)
    try:
        original = manager.create({"name": "Actual failed data constraint", "base": {**SMALL, "poison_ratios": [0.49]}, "seeds": [42, 73]})
        manager.tick(); wait_job(jobs); manager.tick()
        before = manager.listing()[0]
        assert [e["status"] for e in before["entries"]] == ["failed", "queued"]
        retry = manager.retry({"id": original["id"], "overrides": {"poison_ratios": [0.01]}})
        assert len(retry["entries"]) == 1 and retry["entries"][0]["seed"] == 42
        assert retry["entries"][0]["origin"]["entry"] == 0 and retry["retry_of"] == original["id"]
        manager.tick(); wait_job(jobs); manager.tick()
        completed = next(c for c in manager.listing() if c["id"] == retry["id"])
        unchanged = next(c for c in manager.listing() if c["id"] == original["id"])
        assert completed["status"] == "complete" and manager.results(retry["id"])["verified_runs"] == 1
        assert unchanged["entries"] == before["entries"]
        new_job = manager.entry(retry["id"], 0)["job"]
        assert new_job["origin"]["job"] == before["entries"][0]["job_id"]
        assert new_job["settings"]["poison_ratios"] == [0.01]
        assert manager.entry(original["id"], 0)["job"]["settings"]["poison_ratios"] == [0.49]
        with pytest.raises(ValueError):
            manager.retry({"id": original["id"], "overrides": {"seed": 99}})
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_retry_does_not_repeat_a_successful_entry_when_another_process_was_cancelled(tmp_path, view_run):
    jobs, manager, _ = library(tmp_path, view_run)
    try:
        original = manager.create({"name": "Keep successful work", "base": SMALL, "seeds": [42, 73]})
        manager.tick(); manager.action(original["id"], "pause"); wait_job(jobs); manager.tick()
        first = manager.entry(original["id"], 0)
        report_path = Path(first["job"]["settings"]["report_dir"])/"benchmark_results.json"
        preserved = report_path.read_bytes()
        manager.action(original["id"], "resume"); manager.tick()
        assert jobs.active
        manager.action(original["id"], "cancel"); wait_job(jobs); manager.tick()
        original_entries = next(c for c in manager.listing() if c["id"] == original["id"])["entries"]
        assert [e["status"] for e in original_entries] == ["complete", "cancelled"]
        fork = manager.retry({"id": original["id"]})
        assert len(fork["entries"]) == 1 and fork["entries"][0]["seed"] == 73
        assert fork["entries"][0]["origin"]["entry"] == 1
        manager.tick(); wait_job(jobs); manager.tick()
        assert manager.results(fork["id"])["verified_runs"] == 1
        assert report_path.read_bytes() == preserved
        assert next(c for c in manager.listing() if c["id"] == original["id"])["entries"] == original_entries
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_preset_api_captures_verified_report_settings_and_keeps_user_text(view_run, view_http):
    saved = post(view_http, "/api/presets/save", {"name": "Verified configuration <script>", "notes": "User notes\nNo generated model answer", "run": view_run["id"], "report_sha256": view_run["payload"]["evidence"]["report_sha256"]})
    assert saved["source"]["report_sha256"] == view_run["payload"]["evidence"]["report_sha256"]
    for key in ["seed", "epochs", "epsilon", "poison_ratios", "epsilons", "hidden"]:
        assert saved["settings"][key] == view_run["cfg"][key]
    exported = json_get(view_http, "/api/preset-export", id=saved["id"])
    assert exported["preset"] == saved and exported["compatible"]
    snapshot = json_get(view_http, "/api/replay-settings", run=view_run["id"], report_sha256=view_run["payload"]["evidence"]["report_sha256"])
    assert snapshot["settings"] == saved["settings"]
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/replay", {"run": view_run["id"], "report_sha256": view_run["payload"]["evidence"]["report_sha256"], "config_sha256": "wrong"})
    assert error.value.code == 400


@pytest.mark.parametrize("values", [
    {"name": "", "settings": SMALL}, {"name": "Bad", "settings": {"data_path": "outside"}},
    {"name": "Bad", "settings": {"seed": True}}, {"name": "Bad", "settings": SMALL, "notes": "x"*3001},
    {"name": "Bad", "settings": SMALL, "fixed_context": {}}, {"name": "Bad", "settings": SMALL, "archived": "yes"},
])
def test_preset_api_rejects_unvalidated_settings(view_http, values):
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/presets/save", values)
    assert error.value.code == 400


def test_reuse_and_retry_api_require_protected_local_writes(view_http):
    for route in ["/api/presets/save", "/api/presets/run", "/api/replay", "/api/campaigns/retry"]:
        with pytest.raises(HTTPError) as error:
            post(view_http, route, {}, token="invalid")
        assert error.value.code == 403
        with pytest.raises(HTTPError) as get_error:
            json_get(view_http, route)
        assert get_error.value.code == 404
    with urlopen(view_http+"/presets.js") as response:
        assert response.status == 200 and b"Repeat configuration" in response.read()
