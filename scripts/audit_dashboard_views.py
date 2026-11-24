"""Exercise the running dashboard's analysis endpoints against saved evidence."""
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen


def main():
    base = "http://127.0.0.1:8873"
    checks = []

    def request(route, **query):
        with urlopen(base + route + ("?" + urlencode(query) if query else ""), timeout=120) as response:
            raw = response.read()
            return json.loads(raw) if "json" in response.headers["Content-Type"] else raw

    runs = request("/api/reports")
    assert runs
    for run in runs:
        payload = request("/api/report/" + run["id"])
        report, fingerprint = payload["report"], payload["evidence"]["report_sha256"]
        tested = report["dataset"]["split_sizes"]["test"]
        for model in report["clean"]:
            result = request("/api/records", run=run["id"], model=model["model"], recheck=1, report_sha256=fingerprint)
            assert sum(result["counts"].values()) == tested
            assert result["tested"] == tested and result["scores_recalculated"]
            assert result["maximum_score_difference"] <= 1e-5
            checks.append({"run": run["id"], "model": model["model"], "tested": tested, "maximum_score_difference": result["maximum_score_difference"]})
        exported = list(csv.DictReader(io.StringIO(request("/api/records.csv", run=run["id"], model="clean", features=1).decode("utf-8-sig"))))
        assert len(exported) == tested
        assert all(row["report_sha256"] == fingerprint for row in exported)
        for row in report["poisoning"]:
            selected = request("/api/training-records", run=run["id"], model=row["model"], report_sha256=fingerprint)
            assert selected["counts"]["selected"] == row["selected_count"]
            assert selected["counts"]["modified"] == row["modified_count"]
            assert selected["scores_after_cleanup"] == row["sanitized"]
            if selected["rows"]:
                detail = request("/api/training-record", run=run["id"], model=row["model"], position=selected["rows"][0]["position"], report_sha256=fingerprint)
                assert len(detail["features"]) == 57
                assert len(detail["clean_scores"]) == len(detail["changed_scores"]) == 2
        comparison = request("/api/compare", left=run["id"], right=run["id"])
        assert comparison["same_test_records"] and comparison["same_attacked_records"]
        assert all(all(value == 0 for value in row["change"].values()) for row in comparison["models"])
        try:
            request("/api/records", run=run["id"], report_sha256="outdated")
            raise AssertionError("Outdated evidence was accepted")
        except HTTPError as error:
            assert error.code >= 400
        print(f"Checked {run['id']}: {tested} records, {len(report['poisoning'])} training models", flush=True)
    if len(runs) > 1:
        comparison = request("/api/compare", left=runs[0]["id"], right=runs[-1]["id"])
        assert comparison["left"]["id"] != comparison["right"]["id"]
        assert len(comparison["attacks"]) == 12
        assert request("/api/compare.csv", left=runs[0]["id"], right=runs[-1]["id"])
    assert b"Record explorer" in request("/")
    assert b"function fxLoadRecords" in request("/features.js")
    evidence = {"checked_at": datetime.now(timezone.utc).isoformat(), "base": base, "runs": len(runs), "score_recalculations": checks, "csv_exports": "all test rows plus original measurements", "training_inspection": "every changed and cleaned model, four actual scores per inspected record", "browser_clicks": "not performed: Windows browser tool startup failed"}
    output = Path("reports/dashboard_feature_checks.json")
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
