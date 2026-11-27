"""Independently replay actual browser-exported attack analysis results."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from ml_adv_sec.common import config, seed
from ml_adv_sec.dashboard import Reports
from ml_adv_sec.dashboard_adaptive import AdaptiveTests
from ml_adv_sec.dashboard_lab import ModelLabViews
from ml_adv_sec.data import Domain
from ml_adv_sec.models import restore, score
from ml_adv_sec.defenses import squeeze
from audit_adaptive_studies import predictor


def main():
    workspace = Path.cwd()
    reports = Reports(workspace, config("configs/default.yaml"))
    views = ModelLabViews(reports)
    adaptive = AdaptiveTests(views)
    downloads = Path.home()/"Downloads"
    output = Path("reports/attack-analysis-browser")
    output.mkdir(exist_ok=True)
    checks = {"analyses": [], "model_predictions_replayed": 0, "raw_measurements_checked": 0, "browser_files_verified": 0}
    for tool in ["transfer", "reduction", "coverage"]:
        path = downloads/f"attack-{tool}-json.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        digest = result.pop("result_sha256")
        assert digest == hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        result["result_sha256"] = digest
        payload, _, data, _ = views.lab_data(result["run"], result["report_sha256"])
        cfg = payload["report"]["config"]
        seed(cfg["seed"], cfg["threads"])
        domain, cache = Domain(cfg), {}
        studies = {s["id"]: adaptive.read(s["id"], traces=True) for s in result["studies"]}
        for s in result["studies"]:
            assert s["evidence_sha256"] == studies[s["id"]]["evidence_sha256"]
        thresholds = {r["model"]: r["threshold"] for r in payload["report"]["clean"]+payload["report"]["poisoning"]}
        for row in result["rows"]:
            study = studies[row["study"]]
            original = np.array(study["examples"]["original"][row["trial"]])
            raw = original.copy()
            pos = int(np.flatnonzero(data["id_test"]+1 == row["source_row"])[0])
            x = data["x_test"][pos].copy()
            for column, value in row["edits"]:
                raw[column] = value
                x[column] = (value-domain.center[column])/domain.scale[column]
            assert hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest() == row["input_sha256"]
            if result["tool"] == "reduction":
                predict = predictor(payload, study["target"], data, domain)
                assert row["threshold"] == study["target"]["threshold"]
            else:
                name = "clean" if row["model"] == "squeezed" else row["model"]
                if name not in cache:cache[name] = restore(cfg, name, payload["evidence"]["checkpoints"][name])[0]
                def predict(values):
                    return score(cache[name], squeeze(values, domain, cfg["quantum"]) if row["model"] == "squeezed" else values)
                assert row["threshold"] == thresholds[row["model"]]
            assert row["before"] == float(predict(data["x_test"][pos][None])[0])
            assert row["after"] == float(predict(x[None])[0])
            assert row["eligible"] == ((row["before"] >= row["threshold"]) == bool(row["actual"]))
            assert row["caused_error"] == (row["eligible"] and ((row["after"] >= row["threshold"]) != bool(row["actual"])))
            checks["model_predictions_replayed"] += 2
            checks["raw_measurements_checked"] += 57
        assert result["defender_predictions"] == 2*len(result["rows"])
        for summary in result["summary"]:
            if result["tool"] == "reduction":
                assert summary["valid_subsets"] == len(result["rows"])
                assert summary["minimum_retained_fields"] == min((r["changed_fields"] for r in result["rows"] if r["caused_error"]), default=None)
            else:
                rows = [r for r in result["rows"] if r["model"] == summary["model"]]
                failed = {r["source_row"] for r in rows if r["caused_error"]}
                eligible = {r["source_row"] for r in rows if r["eligible"]}
                assert summary["sources_failing_at_least_once"] == len(failed)
                assert summary["observed_failure_rate"] == (len(failed)/len(eligible) if eligible else None)
                best = max(len({r["source_row"] for r in rows if r["study"] == s and r["caused_error"]}) for s in studies)
                assert summary["added_beyond_best_single_study"] == len(failed)-best
        for scope, values in [("summary", result["summary"]), ("rows", result["rows"])]:
            csv_path = downloads/f"attack-{tool}-{scope}.csv"
            contents = csv_path.read_bytes()
            rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))))
            assert len(rows) == len(values)
            for row, expected in zip(rows, values):
                for key, value in expected.items():
                    if value is None:assert row[key] == ""
                    elif isinstance(value, bool):assert row[key] == str(value).lower()
                    elif isinstance(value, (int, float)):assert float(row[key]) == value
                    elif isinstance(value, list):assert json.loads(row[key]) == value
                    else:assert row[key] == value
            (output/csv_path.name).write_bytes(contents)
            checks["browser_files_verified"] += 1
        (output/path.name).write_bytes(path.read_bytes())
        checks["browser_files_verified"] += 1
        checks["analyses"].append({"tool": result["tool"], "predictions": result["defender_predictions"], "rows": len(result["rows"]), "result_sha256": digest, "summary": result["summary"]})
    Path("reports/dashboard_attack_analysis_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":main()
