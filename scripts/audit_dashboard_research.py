"""Exercise live importance and overlap workflows with saved real model runs."""
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


def main():
    base = "http://127.0.0.1:8873"

    def get(route, **query):
        with urlopen(base+route+("?"+urlencode(query) if query else ""), timeout=120) as response:
            body = response.read()
            return json.loads(body) if "json" in response.headers["Content-Type"] else body

    def csv_get(route, **query):
        return list(csv.DictReader(io.StringIO(get(route, **query).decode("utf-8-sig"))))

    runs = get("/api/reports")
    importance_checks, overlap_checks = [], []
    assert b"Input importance" in get("/") and b"importancePage" in get("/research.js")
    for run in runs:
        payload = get("/api/report/"+run["id"])
        report, fingerprint = payload["report"], payload["evidence"]["report_sha256"]
        models = [r["model"] for r in report["clean"]]+[r["model"] for r in report["poisoning"][:2]]
        for model in models:
            result = get("/api/importance", run=run["id"], model=model, repeats=2, seed=73, report_sha256=fingerprint)
            assert len(result["rows"]) == 57 and result["feature_checks"] == 114
            assert result["records"] == report["dataset"]["split_sizes"]["test"]
            assert result["maximum_score_difference"] <= 1e-5
            assert all(r["mean"]["mean_score_change"] >= 0 for r in result["rows"])
            for row in result["rows"]:
                assert row["mean"]["brier_increase"] == np.mean([t["brier_increase"] for t in row["trials"]])
                if row["distinct_values"] == 1:
                    assert all(t["mean_score_change"] == 0 for t in row["trials"])
            importance_checks.append({"run": run["id"], "model": model, "records": result["records"], "checks": result["feature_checks"], "score_difference": result["maximum_score_difference"]})
            if model == "tree":
                exported = csv_get("/api/importance.csv", run=run["id"], model=model, repeats=2, seed=73, report_sha256=fingerprint)
                expected = {(r["feature"], t["repeat"]): t for r in result["rows"] for t in r["trials"]}
                assert len(exported) == len(expected)
                for row in exported:
                    assert float(row["brier_increase"]) == expected[(row["feature"], int(row["repeat"]))]["brier_increase"]
        for target in ["clean", "tree"]:
            all_rows = []
            page = 1
            while True:
                result = get("/api/attack-overlap", run=run["id"], target=target, size=10, page=page, report_sha256=fingerprint)
                all_rows.extend(result["rows"])
                if page >= result["pages"]:
                    break
                page += 1
            assert len(all_rows) == result["matched"]
            assert len({r["source_row"] for r in all_rows}) == result["matched"]
            assert result["counts"]["eligible"]+result["counts"]["already_missed"] == result["matched"]
            assert result["counts"]["escaped"]+result["counts"]["resisted"] == result["counts"]["eligible"]
            for row in all_rows:
                assert row["escaped_count"] == sum(a["escaped"] for a in row["attacks"])
                if not row["eligible"]:
                    assert row["escaped_count"] == 0
            for category in ["all", "escaped", "resisted", "one"]:
                filtered = get("/api/attack-overlap", run=run["id"], target=target, category=category, report_sha256=fingerprint)
                exported = csv_get("/api/attack-overlap.csv", run=run["id"], target=target, category=category, report_sha256=fingerprint)
                assert len(exported) == filtered["total"]*len(filtered["attacks"])
                assert all(r["report_sha256"] == fingerprint for r in exported)
            root = Path(report["config"]["root"])
            for attack in all_rows[0]["attacks"]:
                with np.load(root / "artifacts" / (attack["attack"]+".npz")) as archive:
                    assert int(archive["record_ids"][attack["sample"]])+1 == all_rows[0]["source_row"]
                detail = get("/api/inspect", run=run["id"], target=target, attack=attack["attack"], sample=attack["sample"], report_sha256=fingerprint)
                assert detail["source_row"]+1 == all_rows[0]["source_row"] and detail["after"] == attack["score"]
            overlap_checks.append({"run": run["id"], "target": target, "records": result["matched"], "pages_checked": page, "attack_pairs": len(result["overlaps"]), "originally_caught": result["counts"]["eligible"], "already_missed": result["counts"]["already_missed"], "escaped_any": result["counts"]["escaped"]})
        print(f"Checked {run['id']}: seven importance models, both attack targets, every filter/export/page and six attack inspector links per target", flush=True)
    evidence = {"checked_at": datetime.now(timezone.utc).isoformat(), "runs": len(runs), "importance": importance_checks, "attack_overlap": overlap_checks, "browser": "Not run: Windows browser process startup fails with error 1920"}
    output = Path("reports/dashboard_research_checks.json")
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
