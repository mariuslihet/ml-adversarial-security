"""Live real-process campaign recovery and durable source-bound case checks."""
import csv
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

BASE = "http://127.0.0.1:8873"
STATE = Path("reports/dashboard_capability_checks.json")


def get(route, **query):
    with urlopen(BASE+route+("?"+urlencode(query) if query else ""), timeout=120) as response:
        body = response.read()
        return json.loads(body) if "json" in response.headers["Content-Type"] else body


def post(route, values):
    token = re.search(r'name="dashboard-token" content="([^"]+)"', get("/").decode()).group(1)
    request = Request(BASE+route, data=json.dumps(values).encode(), headers={"Content-Type": "application/json", "X-Dashboard-Token": token, "Origin": BASE}, method="POST")
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def wait_campaign(campaign_id, predicate, timeout=180):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        current = next(c for c in get("/api/campaigns") if c["id"] == campaign_id)
        if predicate(current):
            return current
        if current["status"] in {"failed", "finished-with-errors", "cancelled"}:
            raise AssertionError(current)
        time.sleep(0.25)
    raise AssertionError("Actual campaign did not reach the required state")


def save(evidence):
    STATE.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def begin():
    settings = {"epochs": 1, "max_rows": 128, "hidden": 16, "batch_size": 16, "attack_samples": 2, "steps": 1, "cw_steps": 1, "hsj_steps": 1, "collision_steps": 1, "restarts": 0, "epsilons": [0.01, 0.1], "poison_ratios": [0.01]}
    campaign = post("/api/campaigns/create", {"name": "Verified real seed and attack-setting grid", "base": settings, "axis": "epsilon", "values": [0.03, 0.1], "seeds": [53, 89]})
    wait_campaign(campaign["id"], lambda c: any(e["status"] == "running" for e in c["entries"]))
    post("/api/campaigns/action", {"id": campaign["id"], "action": "pause"})
    paused = wait_campaign(campaign["id"], lambda c: c["status"] == "paused")
    assert [e["status"] for e in paused["entries"]] == ["complete", "queued", "queued", "queued"]
    evidence = {"started": datetime.now(timezone.utc).isoformat(), "campaign_id": campaign["id"], "paused_first_job": paused["entries"][0]["job_id"], "settings": settings, "paused_after_real_completion": True, "status": "paused; ready for server-restart check"}
    save(evidence)
    print(json.dumps(evidence), flush=True)


def finish():
    evidence = json.loads(STATE.read_text())
    campaign_id = evidence["campaign_id"]
    recovered = next(c for c in get("/api/campaigns") if c["id"] == campaign_id)
    assert recovered["status"] == "paused"
    assert recovered["entries"][0]["job_id"] == evidence["paused_first_job"]
    assert all(e["status"] == "queued" for e in recovered["entries"][1:])
    post("/api/campaigns/action", {"id": campaign_id, "action": "resume"})
    completed = wait_campaign(campaign_id, lambda c: c["status"] == "complete")
    assert len({e["job_id"] for e in completed["entries"]}) == 4
    assert completed["entries"][0]["job_id"] == evidence["paused_first_job"]
    results = get("/api/campaign-results", id=campaign_id)
    assert results["verified_runs"] == 4 and len(results["rows"]) == 20 and len(results["attack_rows"]) == 48
    assert not results["evidence_errors"]
    reports = {}
    for entry in completed["entries"]:
        detail = get("/api/campaign-entry", id=campaign_id, index=entry["index"])
        assert detail["job"]["status"] == "complete" and detail["job"]["exit_code"] == 0
        assert "Running prepare" in detail["log"] and "Running evaluate" in detail["log"] and "Verification completed" in detail["log"]
        cfg = detail["job"]["settings"]
        assert cfg["seed"] == entry["seed"] and cfg["epsilon"] == entry["variant"]
        reports[entry["index"]] = json.loads((Path(cfg["report_dir"])/"benchmark_results.json").read_text())
    for group in results["aggregates"]:
        rows = [row for row in results["rows"] if row["variant"] == group["variant"] and row["model"] == group["model"]]
        assert len(rows) == group["runs"] == 2
        for key in ["spam_caught", "false_alarms", "f1", "brier"]:
            values = [r[key] for r in rows]
            assert group["metrics"][key]["mean"] == float(np.mean(values))
            assert group["metrics"][key]["std"] == float(np.std(values, ddof=1))
    for group in results["attack_aggregates"]:
        rows = [row for row in results["attack_rows"] if (row["variant"], row["target"], row["attack"]) == (group["variant"], group["target"], group["attack"])]
        values = [row["asr"] for row in rows if row["asr"] is not None]
        assert group["defined_runs"] == len(values)
        assert group["mean_escape_rate"] == (float(np.mean(values)) if values else None)
    for scope, count in [("utility", 20), ("attacks", 48)]:
        exported = list(csv.DictReader(io.StringIO(get("/api/campaign-results.csv", id=campaign_id, scope=scope).decode("utf-8-sig"))))
        assert len(exported) == count
        assert {r["report_sha256"] for r in exported} == {e["report_sha256"] for e in completed["entries"]}
    # Save one actual measured record as a notebook case, then preserve a real
    # review history without manufacturing any model answer or example row.
    entry = completed["entries"][0]
    actual = get("/api/records", run=entry["report_id"], model="clean", report_sha256=entry["report_sha256"])
    source = actual["rows"][0]
    title = f"Measured source {source['source_row']}: {source['outcome']}"
    notes = f"Actual model score {source['score']!r}; threshold {actual['threshold']!r}; source label {source['actual']}. Bound report {entry['report_sha256']}."
    case = post("/api/cases/create", {"run": entry["report_id"], "report_sha256": entry["report_sha256"], "source_row": source["source_row"], "model": "clean", "title": title, "notes": notes, "tags": ["measured-campaign"], "status": "open"})
    verified = get("/api/case", id=case["id"])
    assert verified["evidence_error"] is None and len(verified["evidence"]["record"]["features"]) == 57
    assert verified["evidence"]["record"]["models"][0]["score"] == source["score"]
    update = {"id": case["id"], "revision": 1, "title": title, "notes": notes, "status": "reviewed", "tags": ["measured-campaign"]}
    reviewed = post("/api/cases/update", update)
    try:
        post("/api/cases/update", update)
        raise AssertionError("An outdated notebook edit was accepted")
    except HTTPError as error:
        assert error.code == 400
    archived = post("/api/cases/update", {**update, "revision": reviewed["revision"], "status": "archived"})
    exported_case = get("/api/case-export", id=case["id"])
    assert archived["revision"] == 3 and len(exported_case["history"]) == 3
    assert exported_case["evidence"]["record"]["models"][0]["score"] == source["score"]
    evidence.update(status="passed", completed=datetime.now(timezone.utc).isoformat(), server_restart_recovered_without_duplicate=True, process_ids=[e["job_id"] for e in completed["entries"]], verified_runs=results["verified_runs"], utility_rows=len(results["rows"]), attack_rows=len(results["attack_rows"]), utility_groups=len(results["aggregates"]), attack_groups=len(results["attack_aggregates"]), archived_measured_case=case["id"], case_revisions=len(exported_case["history"]), browser="Not yet checked through browser input")
    save(evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    (begin if sys.argv[1:] == ["begin"] else finish)()
