import json
import re
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from ml_adv_sec.dashboard import Reports, Jobs, handler_for


@pytest.fixture
def dashboard(completed_experiment):
    cfg, _, _, _ = completed_experiment
    reports = Reports(cfg["root"], cfg)
    jobs = Jobs(reports)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(reports, jobs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    with urlopen(base) as response:
        html = response.read().decode()
    token = re.search(r'name="dashboard-token" content="([^"]+)"', html).group(1)
    yield base, token, reports, jobs
    jobs.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def get_json(url):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def post(url, token, values, **headers):
    request = Request(url, data=json.dumps(values).encode(), headers={"Content-Type": "application/json", "X-Dashboard-Token": token, **headers}, method="POST")
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def test_dashboard_reads_actual_reports_and_saved_email_changes(dashboard, completed_experiment):
    base, _, _, _ = dashboard
    cfg, _, _, _ = completed_experiment
    runs = get_json(base + "/api/reports")
    run = runs[0]["id"]
    report = get_json(base + "/api/report/" + run)["report"]
    assert report == json.loads((Path(cfg["report_dir"]) / "benchmark_results.json").read_text())
    details = get_json(base + f"/api/inspect?run={run}&attack=evasion-pgd-linf&sample=0&target=clean")
    assert len(details["features"]) == 57
    assert any(f["changed"] for f in details["features"])
    assert all(not f["changed"] for f in details["features"] if f["locked"])
    with np.load(Path(cfg["root"]) / "artifacts/evasion-pgd-linf.npz") as arrays:
        actual = arrays["adversarial"][0] - arrays["original"][0]
    np.testing.assert_allclose([f["scaled_change"] for f in details["features"]], actual)
    assert details["caught_before"] == (details["before"] >= details["threshold"])
    assert details["caught_after"] == (details["after"] >= details["threshold"])
    for asset in ["/app.js", "/style.css"]:
        with urlopen(base + asset) as response:
            assert response.status == 200
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_dashboard_rejects_outside_requests_and_unlisted_settings(dashboard):
    base, token, _, _ = dashboard
    for headers in [{"X-Dashboard-Token": "wrong"}, {"Origin": "https://outside.example"}]:
        with pytest.raises(HTTPError) as error:
            post(base + "/api/jobs", headers.pop("X-Dashboard-Token", token), {}, **headers)
        assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        post(base + "/api/jobs", token, {"root": "outside"})
    assert error.value.code == 400
    for route in ["/../../pyproject.toml", "/api/report/missing"]:
        with pytest.raises(HTTPError) as error:
            urlopen(base + route)
        assert error.value.code == 404
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/api/reports", headers={"Host": "outside.example"}))
    assert error.value.code == 403


def wait_for_job(jobs, job_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = next(j for j in jobs.list() if j["id"] == job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("The real local experiment did not finish within 120 seconds")


def test_start_button_endpoint_executes_a_real_full_pipeline(dashboard):
    base, token, reports, jobs = dashboard
    job = post(base + "/api/jobs", token, {"name": "HTTP integration run", "max_rows": 128, "epochs": 1, "steps": 1, "cw_steps": 1, "hsj_steps": 1, "collision_steps": 1, "attack_samples": 4, "poison_ratios": [0.01], "epsilons": [0.03, 0.1]})
    finished = wait_for_job(jobs, job["id"])
    assert finished["status"] == "complete", finished["log"]
    assert finished["exit_code"] == 0
    assert "Verification completed on checksum-verified real data." in finished["log"]
    report = json.loads((Path(job["settings"]["report_dir"]) / "benchmark_results.json").read_text())
    assert report["dataset"]["raw_rows"] == 4601
    assert len(report["clean"]) == 5
    assert len(report["poisoning"]) == 6
    assert len(report["pareto"]) == 8
    assert any(item["name"] == "HTTP integration run" for item in reports.list())
    with pytest.raises(HTTPError) as error:
        post(base + "/api/jobs/cancel", token, {"id": job["id"]})
    assert error.value.code == 400
    assert next(j for j in jobs.list() if j["id"] == job["id"])["status"] == "complete"


def test_stop_button_terminates_only_the_active_process(dashboard):
    base, token, _, jobs = dashboard
    job = post(base + "/api/jobs", token, {"name": "Cancellation check", "epochs": 200})
    with pytest.raises(HTTPError) as error:
        post(base + "/api/jobs", token, {"name": "Concurrent run"})
    assert error.value.code == 400
    post(base + "/api/jobs/cancel", token, {"id": job["id"]})
    stopped = wait_for_job(jobs, job["id"])
    assert stopped["status"] == "cancelled"
    assert stopped["exit_code"] != 0
    assert not (Path(job["settings"]["report_dir"]) / "benchmark_results.json").exists()


def test_missing_dataset_produces_a_real_failed_job(experiment, tmp_path):
    cfg, _, _, _ = experiment
    settings = dict(cfg, data_path=str(tmp_path / "missing.data"))
    jobs = Jobs(Reports(tmp_path, settings))
    try:
        job = jobs.launch({"name": "Missing data check"})
        finished = wait_for_job(jobs, job["id"])
        assert finished["status"] == "failed"
        assert finished["exit_code"] == 2
        assert "Supply" in finished["log"]
        assert not (Path(job["settings"]["report_dir"]) / "benchmark_results.json").exists()
        assert next(j for j in Jobs(jobs.reports).list() if j["id"] == job["id"])["status"] == "failed"
    finally:
        jobs.close()


@pytest.mark.parametrize("evidence", ["report", "weights", "attack", "schema"])
def test_changed_evidence_never_returns_results(dashboard, completed_experiment, evidence):
    base, _, reports, _ = dashboard
    cfg, _, _, _ = completed_experiment
    run_id = next(r["id"] for r in reports.list() if not r.get("error") and r["name"] == "640-record check")
    report_path = Path(cfg["report_dir"]) / "benchmark_results.json"
    manifest = json.loads(report_path.with_name("benchmark_results.manifest.json").read_text())
    paths = {"report": report_path, "weights": Path(cfg["root"]) / "models/clean" / (manifest["checkpoints"]["clean"] + ".pt"), "attack": Path(cfg["root"]) / "artifacts/evasion-pgd-linf.npz", "schema": Path(cfg["root"]) / "prepared/schema.json"}
    path = paths[evidence]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(HTTPError) as error:
            get_json(base + "/api/report/" + run_id)
        assert error.value.code == 503
        body = json.loads(error.value.read())
        assert "report" not in body and body["error"]
    finally:
        path.write_bytes(original)
    assert get_json(base + "/api/report/" + run_id)["evidence"]["report_sha256"] == manifest["report_sha256"]


def test_inspection_uses_the_evaluated_version_after_new_training(dashboard, completed_experiment):
    from ml_adv_sec.models import train_mlp, save, calibrate, score
    from ml_adv_sec.eval.metrics import metrics
    base, _, reports, _ = dashboard
    cfg, data, domain, _ = completed_experiment
    run = next(r["id"] for r in reports.list() if not r.get("error") and r["name"] == "640-record check")
    url = base + f"/api/inspect?run={run}&attack=evasion-pgd-linf&sample=0&target=clean"
    before = get_json(url)
    latest_path = Path(cfg["root"]) / "models/clean/latest.json"
    pointer = latest_path.read_bytes()
    try:
        model = train_mlp(cfg, data["x_train"], data["y_train"], "pgd", domain)
        t = calibrate(data["y_val"], score(model, data["x_val"]), cfg["fpr"])
        save(cfg, "clean", model, t, metrics(model, data["x_val"], data["y_val"], t), {"training_mode": "pgd"})
        after = get_json(url)
        assert before == after
    finally:
        latest_path.write_bytes(pointer)


def test_close_ratios_and_large_requested_attack_set_run_without_aliasing(dashboard):
    from ml_adv_sec.eval.workflow import poison_name
    base, token, _, jobs = dashboard
    ratios = [0.0101, 0.0104]
    assert poison_name("label-flip", ratios[0]) != poison_name("label-flip", ratios[1])
    job = post(base + "/api/jobs", token, {"name": "Close shares audit", "max_rows": 128, "epochs": 1, "steps": 1, "cw_steps": 1, "hsj_steps": 1, "collision_steps": 1, "attack_samples": 512, "poison_ratios": ratios, "epsilons": [0.0001, 0.0002]})
    result = wait_for_job(jobs, job["id"])
    assert result["status"] == "complete", result["log"]
    report = get_json(base + "/api/report/" + result["report_id"])["report"]
    assert 0 < report["counts"]["attack_samples"] < 512
    assert {r["attack_n"] for r in report["transferability"]} == {report["counts"]["attack_samples"]}
    assert report["counts"]["selected_records"] == 128
    assert len({r["model"] for r in report["poisoning"]}) == 12
    for ratio in ratios:
        path = Path(job["settings"]["root"]) / "artifacts" / (poison_name("label-flip", ratio) + ".json")
        assert json.loads(path.read_text())["details"]["requested_ratio"] == ratio


def test_impossible_training_share_fails_before_model_training(dashboard):
    base, token, _, jobs = dashboard
    job = post(base + "/api/jobs", token, {"name": "Too many selected records", "max_rows": 128, "poison_ratios": [0.49]})
    finished = wait_for_job(jobs, job["id"])
    assert finished["status"] == "failed"
    assert "Choose a smaller share" in finished["log"]
    assert "Running train" not in finished["log"]
    assert "report_id" not in finished


def test_record_inspection_rejects_a_report_replaced_after_page_load(dashboard, completed_experiment):
    from ml_adv_sec.eval.workflow import evaluate
    base, _, reports, _ = dashboard
    cfg, _, _, _ = completed_experiment
    run = next(r["id"] for r in reports.list() if r["name"] == "640-record check" and not r.get("error"))
    previous = get_json(base + "/api/report/" + run)["evidence"]["report_sha256"]
    evaluate(cfg)
    current = get_json(base + "/api/report/" + run)["evidence"]["report_sha256"]
    assert current != previous
    with pytest.raises(HTTPError) as error:
        get_json(base + f"/api/inspect?run={run}&attack=evasion-pgd-linf&sample=0&target=clean&report_sha256={previous}")
    assert error.value.code == 503
    body = json.loads(error.value.read())
    assert "updated after you opened it" in body["error"]
    assert "features" not in body
    assert get_json(base + f"/api/inspect?run={run}&attack=evasion-pgd-linf&sample=0&target=clean&report_sha256={current}")["index"] == 0
