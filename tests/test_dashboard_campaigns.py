"""Real subprocess campaign execution, recovery, ownership, and measured statistics."""
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from ml_adv_sec.dashboard import Jobs, Reports
from ml_adv_sec.dashboard_campaigns import Campaigns
from ml_adv_sec.eval.workflow import evaluate
from test_dashboard_views import view_run, view_http, json_get, csv_get
from test_dashboard_investigations import post


SMALL = {"epochs": 1, "max_rows": 128, "hidden": 16, "batch_size": 16, "attack_samples": 2, "steps": 1, "cw_steps": 1, "hsj_steps": 1, "collision_steps": 1, "restarts": 0, "epsilons": [0.01, 0.1], "poison_ratios": [0.01]}


def owned_manager(tmp_path, view_run):
    cfg = {**view_run["cfg"], "root": str(tmp_path/"baseline"), "report_dir": str(tmp_path/"baseline/reports")}
    jobs = Jobs(Reports(tmp_path, cfg))
    return jobs, Campaigns(jobs, start_worker=False)


def wait_job(jobs, timeout=60):
    end = time.monotonic()+timeout
    while jobs.active and time.monotonic() < end:
        time.sleep(0.05)
    assert not jobs.active, "Real process did not finish within the bounded test timeout"


def test_campaign_api_runs_two_real_seed_experiments_and_verifies_aggregates(view_run, view_http):
    created = post(view_http, "/api/campaigns/create", {"name": "Actual seed comparison", "base": SMALL, "seeds": [11, 29], "axis": "none"})
    end = time.monotonic()+90
    while time.monotonic() < end:
        current = next(c for c in json_get(view_http, "/api/campaigns") if c["id"] == created["id"])
        if current["status"] not in {"running", "pausing"}:
            break
        time.sleep(0.1)
    assert current["status"] == "complete", current
    assert len({entry["job_id"] for entry in current["entries"]}) == 2
    measured = json_get(view_http, "/api/campaign-results", id=created["id"])
    assert measured["verified_runs"] == 2 and len(measured["rows"]) == 10 and not measured["evidence_errors"]
    reports = []
    for entry in current["entries"]:
        detail = json_get(view_http, "/api/campaign-entry", id=created["id"], index=entry["index"])
        assert detail["job"]["exit_code"] == 0
        assert detail["job"]["campaign"] == {"id": created["id"], "index": entry["index"]}
        assert detail["job"]["settings"]["seed"] == entry["seed"]
        assert "Running train" in detail["log"] and "Running attack-evasion" in detail["log"] and "Verification completed" in detail["log"]
        root = Path(detail["job"]["settings"]["root"])
        assert list((root/"models/clean").glob("v*.pt")) and list((root/"artifacts").glob("evasion-*.npz"))
        report = json.loads((root/"reports/benchmark_results.json").read_text())
        reports.append((report, detail["job"]["settings"]))
    for aggregate in measured["aggregates"]:
        assert aggregate["runs"] == 2
        values = [next(r for r in report["clean"] if r["model"] == aggregate["model"]) for report, _ in reports]
        for key, source in [("spam_caught", "fnr"), ("false_alarms", "fpr"), ("f1", "f1"), ("brier", "brier")]:
            numbers = [1-r[source] if key == "spam_caught" else r[source] for r in values]
            result = aggregate["metrics"][key]
            assert result["mean"] == float(np.mean(numbers))
            assert result["std"] == float(np.std(numbers, ddof=1))
            assert result["minimum"] == min(numbers) and result["maximum"] == max(numbers)
    assert len(csv_get(view_http, "/api/campaign-results.csv", id=created["id"])) == 10
    assert len(measured["attack_rows"]) == 24 and len(measured["attack_aggregates"]) == 12
    for group in measured["attack_aggregates"]:
        values = [next(row for row in report["transferability"] if row["attack"] == group["attack"] and row["target"] == group["target"]) for report, _ in reports]
        rates = [row["asr"] for row in values if row["asr"] is not None]
        assert group["defined_runs"] == len(rates)
        assert group["mean_escape_rate"] == (float(np.mean(rates)) if rates else None)
        assert group["std_escape_rate"] == (float(np.std(rates, ddof=1)) if len(rates)>1 else None)
        assert group["mean_spam_caught"] == float(np.mean([row["adversarial_recall"] for row in values]))
    assert len(csv_get(view_http, "/api/campaign-results.csv", id=created["id"], scope="attacks")) == 24
    # A real subsequent evaluation changes measured processing times and the
    # report fingerprint; the campaign must not silently mix report generations.
    evaluate(reports[0][1])
    changed = json_get(view_http, "/api/campaign-results", id=created["id"])
    assert changed["verified_runs"] == 1 and len(changed["evidence_errors"]) == 1
    assert all(a["runs"] == 1 and a["metrics"]["f1"]["std"] is None for a in changed["aggregates"])


def test_campaign_pause_restart_and_resume_do_not_repeat_completed_processes(tmp_path, view_run):
    jobs, manager = owned_manager(tmp_path, view_run)
    try:
        created = manager.create({"name": "Pause and recover", "base": SMALL, "seeds": [19, 23]})
        manager.tick()
        first_job = jobs.active
        assert first_job
        assert manager.action(created["id"], "pause")["status"] == "pausing"
        wait_job(jobs); manager.tick()
        saved = manager.listing()[0]
        assert saved["status"] == "paused" and [e["status"] for e in saved["entries"]] == ["complete", "queued"]
        manager.close()
        manager = Campaigns(jobs, start_worker=False)
        assert manager.listing()[0]["status"] == "paused" and jobs.active is None
        manager.action(created["id"], "resume"); manager.tick()
        assert jobs.active and jobs.active != first_job
        wait_job(jobs); manager.tick()
        complete = manager.listing()[0]
        assert complete["status"] == "complete"
        assert complete["entries"][0]["job_id"] == first_job
        assert len(list(jobs.directory.glob("*/job.json"))) == 2
        assert manager.results(created["id"])["verified_runs"] == 2
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_campaign_queue_waits_for_manual_run_and_cancel_does_not_kill_it(tmp_path, view_run):
    jobs, manager = owned_manager(tmp_path, view_run)
    try:
        manual = jobs.launch({**SMALL, "epochs": 200, "max_rows": 640, "name": "Actual manual process"})
        campaign = manager.create({"name": "Wait for manual process", "base": SMALL, "seeds": [42]})
        manager.tick()
        assert manager.listing()[0]["entries"][0]["status"] == "queued"
        manager.action(campaign["id"], "cancel"); manager.tick()
        assert jobs.active == manual["id"] and jobs.process.poll() is None
        assert manager.listing()[0]["entries"][0]["status"] == "cancelled"
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_campaign_cancel_stops_its_actual_owned_process_and_all_queued_entries(tmp_path, view_run):
    jobs, manager = owned_manager(tmp_path, view_run)
    try:
        campaign = manager.create({"name": "Cancel owned process", "base": {**SMALL, "epochs": 200, "max_rows": 640}, "seeds": [42, 73]})
        manager.tick(); assert jobs.active
        manager.action(campaign["id"], "cancel")
        wait_job(jobs); manager.tick()
        result = manager.listing()[0]
        assert result["status"] == "cancelled" and all(e["status"] == "cancelled" for e in result["entries"])
        assert manager.results(campaign["id"])["verified_runs"] == 0
        assert len(list(jobs.directory.glob("*/job.json"))) == 1
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_campaign_stops_after_actual_dataset_constraint_failure(tmp_path, view_run):
    jobs, manager = owned_manager(tmp_path, view_run)
    try:
        campaign = manager.create({"name": "Real invalid change share", "base": {**SMALL, "poison_ratios": [0.49]}, "seeds": [42, 73]})
        manager.tick(); wait_job(jobs); manager.tick()
        result = manager.listing()[0]
        assert result["status"] == "failed" and result["entries"][0]["status"] == "failed" and result["entries"][1]["status"] == "queued"
        assert "Error:" in manager.entry(campaign["id"], 0)["log"]
        assert manager.results(campaign["id"])["rows"] == []
    finally:
        manager.close(); jobs.close(); wait_job(jobs)


def test_campaign_grid_freezes_validated_settings_before_execution(tmp_path, view_run):
    jobs, manager = owned_manager(tmp_path, view_run)
    try:
        created = manager.create({"name": "Validated grid", "base": SMALL, "axis": "steps", "values": [1, 2], "seeds": [11, 29]})
        assert [(e["variant"], e["seed"]) for e in created["entries"]] == [(1, 11), (1, 29), (2, 11), (2, 29)]
        assert all(e["values"]["steps"] == e["variant"] for e in created["entries"])
        manager.action(created["id"], "pause"); manager.tick()
        assert not list(jobs.directory.glob("*/job.json"))
        jobs.reports.cfg = {**jobs.reports.cfg, "threads": 2}
        manager.action(created["id"], "resume"); manager.tick()
        assert manager.listing()[0]["status"] == "failed" and jobs.active is None
    finally:
        manager.close(); jobs.close()


@pytest.mark.parametrize("changes", [
    {"seeds": []}, {"seeds": [True]}, {"seeds": [1, 1]}, {"seeds": list(range(13))},
    {"axis": "unknown"}, {"axis": "steps", "values": [0]}, {"axis": "steps", "values": [1.5]},
    {"axis": "epsilon", "values": [0.1, 0.1]}, {"axis": "epsilon", "values": [float("nan")]},
    {"axis": "steps", "values": [1, 2], "seeds": list(range(7))}, {"base": {"max_rows": 64}},
    {"base": {"data_path": "outside"}}, {"fail_fast": "yes"}, {"name": ""},
])
def test_campaign_rejects_invalid_plans_before_spawning(tmp_path, view_run, changes):
    jobs, manager = owned_manager(tmp_path, view_run)
    try:
        with pytest.raises(ValueError):
            manager.create({"name": "Invalid plan", "base": SMALL, "seeds": [42], **changes})
        assert jobs.active is None and not manager.listing()
    finally:
        manager.close(); jobs.close()


def test_campaign_write_api_requires_local_token(view_http):
    with pytest.raises(HTTPError) as error:
        post(view_http, "/api/campaigns/create", {"name": "No authorization", "base": SMALL, "seeds": [42]}, token="wrong")
    assert error.value.code == 403
    with urlopen(view_http+"/campaigns.js") as response:
        assert b"Build campaign from these settings" in response.read()
