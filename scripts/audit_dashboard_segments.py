"""Exercise error groups and source-distribution views on all real saved runs."""
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
    assert b"Error breakdowns" in get("/") and b"function shiftPage" in get("/segments.js")
    slices, shifts = [], []
    pages_checked = 0
    for run in runs:
        payload = get("/api/report/"+run["id"])
        report, fingerprint = payload["report"], payload["evidence"]["report_sha256"]
        cfg, root = report["config"], Path(report["config"]["root"])
        source = np.loadtxt(cfg["data_path"], delimiter=",")
        with np.load(root / "prepared/splits.npz") as archive:
            data = {key: archive[key] for key in archive.files}
        models = [r["model"] for r in report["clean"]]+[r["model"] for r in report["poisoning"][:2]]
        for model in models:
            for feature, bands in [(0, 4), (54, 5), (56, 10)]:
                result = get("/api/error-slices", run=run["id"], model=model, feature=feature, bands=bands, report_sha256=fingerprint)
                assert result["maximum_score_difference"] <= 1e-5
                assert sum(r["tested"] for r in result["bands"]) == len(data["y_test"])
                assert sum(r["training_records"] for r in result["bands"]) == len(data["y_train"])
                for field in ["caught", "missed", "false_alarms", "clear"]:
                    assert sum(r[field] for r in result["bands"]) == result["overall"][field]
                slices.append({"run": run["id"], "model": model, "feature": feature, "bands": len(result["bands"]), "score_difference": result["maximum_score_difference"]})
        with np.load(root / "artifacts/evaluation-scores.npz") as archive:
            scores = archive["clean"]
        threshold = next(r["threshold"] for r in report["clean"] if r["model"] == "clean")
        y, ids = data["y_test"], data["id_test"]
        for category in ["all", "missed", "false-alarm", "correct"]:
            mask = np.ones(len(y), dtype=bool)
            if category == "missed":
                mask = (y == 1) & (scores < threshold)
            elif category == "false-alarm":
                mask = (y == 0) & (scores >= threshold)
            elif category == "correct":
                mask = (scores >= threshold) == y
            result = get("/api/error-slices", run=run["id"], feature=56, outcome=category, size=100, report_sha256=fingerprint)
            exported = csv_get("/api/error-slices.csv", run=run["id"], feature=56, outcome=category, scope="records", report_sha256=fingerprint)
            assert [int(r["source_row"])-1 for r in exported] == sorted(ids[mask].tolist())
            assert result["total"] == len(exported)
            if category == "all":
                observed = []
                for page in range(1, result["pages"]+1):
                    batch = get("/api/error-slices", run=run["id"], feature=56, size=100, page=page, report_sha256=fingerprint)
                    observed.extend(r["source_row"]-1 for r in batch["rows"])
                    pages_checked += 1
                assert observed == sorted(ids.tolist())
                row = result["rows"][0]
                detail = get("/api/record-detail", run=run["id"], source_row=row["source_row"], report_sha256=fingerprint)
                assert detail["models"][0]["score"] == row["score"]
        summary = csv_get("/api/error-slices.csv", run=run["id"], feature=56, scope="summary", report_sha256=fingerprint)
        assert sum(int(r["tested"]) for r in summary) == len(y)
        for partition in ["test", "val"]:
            for label in ["all", "spam", "regular"]:
                a_ids, b_ids = data["id_train"], data["id_"+partition]
                if label != "all":
                    a_ids = a_ids[source[a_ids, -1] == int(label == "spam")]
                    b_ids = b_ids[source[b_ids, -1] == int(label == "spam")]
                result = get("/api/data-shift", run=run["id"], partition=partition, label=label, report_sha256=fingerprint)
                assert result["training_records"] == len(a_ids) and result["comparison_records"] == len(b_ids)
                for row in result["rows"]:
                    a, b = source[a_ids, row["column"]], source[b_ids, row["column"]]
                    support = np.union1d(a, b)
                    cdf_a = np.searchsorted(np.sort(a), support, side="right")/len(a)
                    cdf_b = np.searchsorted(np.sort(b), support, side="right")/len(b)
                    assert abs(row["cdf_distance"]-float(np.max(np.abs(cdf_a-cdf_b)))) < 1e-12
                    assert row["outside_count"] == int(((b < a.min()) | (b > a.max())).sum())
                    assert row["comparison"]["median"] == float(np.median(b))
                feature = max(result["rows"], key=lambda r: r["outside_count"])["column"]
                outside = (source[b_ids, feature] < source[a_ids, feature].min()) | (source[b_ids, feature] > source[a_ids, feature].max())
                filtered = get("/api/data-shift", run=run["id"], partition=partition, label=label, feature=feature, outside=1, report_sha256=fingerprint)
                assert filtered["total"] == int(outside.sum())
                exported = csv_get("/api/data-shift.csv", run=run["id"], partition=partition, label=label, feature=feature, outside=1, scope="records", report_sha256=fingerprint)
                assert [int(r["source_row"])-1 for r in exported] == sorted(b_ids[outside].tolist())
                for row in exported:
                    source_id = int(row["source_row"])-1
                    np.testing.assert_array_equal([float(row[name]) for name in result["features"]], source[source_id, :-1])
                    assert row["report_sha256"] == fingerprint
                histogram = filtered["histogram"]
                assert sum(r["reference_count"] for r in histogram) == len(a_ids)
                assert sum(r["comparison_count"] for r in histogram) == len(b_ids)
                summary = csv_get("/api/data-shift.csv", run=run["id"], partition=partition, label=label, scope="summary", report_sha256=fingerprint)
                assert len(summary) == 57
                shifts.append({"run": run["id"], "partition": partition, "label": label, "features_verified": len(result["rows"]), "training_records": len(a_ids), "comparison_records": len(b_ids), "outside_records_exported": len(exported)})
        print(f"Checked {run['id']}: 21 model error breakdowns, all record filters/pages, six distribution comparisons, exact source CSVs", flush=True)
    evidence = {"checked_at": datetime.now(timezone.utc).isoformat(), "runs": len(runs), "slices": slices, "shift": shifts, "all_record_pages_checked": pages_checked, "browser": "Not run: browser control fails before page access with Windows error 1920"}
    output = Path("reports/dashboard_segment_checks.json")
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
