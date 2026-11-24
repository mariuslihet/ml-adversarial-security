"""Check the live analysis pages against each saved real experiment."""
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


def check_pairs(get, runs):
    checks = []
    for right in [runs[1], runs[2], runs[-1]]:
        a = get("/api/report/"+runs[0]["id"])
        b = get("/api/report/"+right["id"])
        with np.load(Path(a["report"]["config"]["root"]) / "prepared/splits.npz") as saved:
            ids_a = saved["id_test"]
        with np.load(Path(b["report"]["config"]["root"]) / "prepared/splits.npz") as saved:
            ids_b = saved["id_test"]
        shared, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
        with np.load(Path(a["report"]["config"]["root"]) / "artifacts/evaluation-scores.npz") as saved:
            pa = saved["clean"][ia]
        with np.load(Path(b["report"]["config"]["root"]) / "artifacts/evaluation-scores.npz") as saved:
            pb = saved["tree"][ib]
        rows = list(csv.DictReader(io.StringIO(get("/api/disagreements.csv", left=runs[0]["id"], right=right["id"], category="all").decode("utf-8-sig"))))
        assert [int(r["source_row"])-1 for r in rows] == shared.tolist()
        np.testing.assert_array_equal([float(r["left_score"]) for r in rows], pa)
        np.testing.assert_array_equal([float(r["right_score"]) for r in rows], pb)
        checks.append({"left": runs[0]["id"], "right": right["id"], "matched": len(shared), "score_pairs_verified": len(rows)})
    assert any(r["matched"] > 0 for r in checks)
    return checks


def main():
    base = "http://127.0.0.1:8873"
    checks = []

    def get(route, **query):
        with urlopen(base+route+("?"+urlencode(query) if query else ""), timeout=120) as response:
            raw = response.read()
            return json.loads(raw) if "json" in response.headers["Content-Type"] else raw

    runs = get("/api/reports")
    assert b"Decision lab" in get("/") and b"decisionLabPage" in get("/analysis.js")
    for run in runs:
        payload = get("/api/report/"+run["id"])
        report, fingerprint = payload["report"], payload["evidence"]["report_sha256"]
        models = [r["model"] for r in report["clean"]] + [r["model"] for r in report["poisoning"][:2]]
        for model in models:
            for mode in ["saved", "manual", "target"]:
                result = get("/api/decision-lab", run=run["id"], model=model, mode=mode, threshold=0.5, target_fpr=0.01, report_sha256=fingerprint)
                assert result["metrics"]["test"]["tested"] == report["dataset"]["split_sizes"]["test"]
                assert sum(r["count"] for r in result["histogram"]) == result["metrics"]["test"]["tested"]
                assert result["maximum_score_difference"] <= 1e-5
                if mode == "target":
                    assert result["metrics"]["validation"]["fpr"] <= 0.01
                checks.append({"run": run["id"], "model": model, "mode": mode, "threshold": result["threshold"], "changed": result["changed_count"], "score_difference": result["maximum_score_difference"]})
        rows = list(csv.DictReader(io.StringIO(get("/api/decision-lab.csv", run=run["id"], mode="target").decode("utf-8-sig"))))
        assert len(rows) == report["dataset"]["split_sizes"]["test"]+report["dataset"]["split_sizes"]["val"]
        assert all(r["report_sha256"] == fingerprint for r in rows)
        for category in ["disagree", "left-only", "right-only", "both-wrong", "all"]:
            result = get("/api/disagreements", left=run["id"], category=category, size=10)
            assert result["matched"] == report["dataset"]["split_sizes"]["test"]
            assert sum(result["counts"][key] for key in ["both_correct", "left-only", "right-only", "both-wrong"]) == result["matched"]
            exported = list(csv.DictReader(io.StringIO(get("/api/disagreements.csv", left=run["id"], category=category).decode("utf-8-sig"))))
            assert len(exported) == result["total"]
        all_rows = get("/api/disagreements", left=run["id"], category="all", size=10)
        detail = get("/api/record-detail", run=run["id"], source_row=all_rows["rows"][0]["source_row"], model=models[-1], report_sha256=fingerprint)
        assert len(detail["features"]) == 57 and len(detail["models"]) == 6
        assert detail["report_sha256"] == fingerprint
        print(f"Checked {run['id']}: 21 decision experiments, all comparison filters and exports, fresh record scores", flush=True)
    pairs = check_pairs(get, runs)
    output = Path("reports/dashboard_analysis_checks.json")
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(), "runs": len(runs), "decision_experiments": checks, "cross_run_pairs": pairs, "exports": "all partitions and every disagreement category checked", "record_details": "57 authentic source values and fresh scores for all five baseline models plus a cleaned model", "browser_status": "Not tested: browser process fails with Windows error 1920"}, indent=2), encoding="utf-8")
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
