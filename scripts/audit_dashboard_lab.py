"""Read-only live checks for all three model tools against independent scoring."""
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from ml_adv_sec.common import seed
from ml_adv_sec.dashboard import Reports
from ml_adv_sec.data import Domain
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.models import calibrate, restore, score


def get(route, **query):
    with urlopen("http://127.0.0.1:8873"+route+("?"+urlencode(query) if query else ""), timeout=120) as response:
        body = response.read()
        return json.loads(body) if "json" in response.headers["Content-Type"] else body


def main():
    reports = Reports(Path.cwd(), get("/api/settings"))
    selected = [r for r in get("/api/reports") if r["name"] in {"Full dataset", "Browser preset full pipeline"}]
    assert len(selected) == 2
    evidence = {"checked": datetime.now(timezone.utc).isoformat(), "checks": [], "predictions_checked": 0, "csv_rows_checked": 0}
    for run in selected:
        payload = reports.read(run["id"]);cfg = payload["report"]["config"]
        seed(cfg["seed"], cfg["threads"])
        with np.load(Path(cfg["root"])/"prepared/splits.npz") as archive:
            data = {k: archive[k] for k in archive.files}
        schema = json.loads((Path(cfg["root"])/"prepared/schema.json").read_text(encoding="utf-8"))
        models = {name: restore(cfg, name, payload["evidence"]["checkpoints"][name])[0] for name in ["clean", "tree", "pgd"]}
        def predict(name, values):
            return score(models["clean" if name == "squeezed" else name], squeeze(values, Domain(cfg), cfg["quantum"]) if name == "squeezed" else values)
        attack = sorted({r["attack"] for r in payload["report"]["transferability"]})[0]
        plans = [{"tool": "ensemble", "members": "clean,tree,pgd", "mode": mode, "attack": attack} for mode in ["fit", "equal"]]
        plans += [{"tool": "calibration", "model": model, "method": method, "bins": 20} for model in ["tree", "squeezed"] for method in ["sigmoid", "isotonic"]]
        plans += [{"tool": "stress", "model": model, "columns": "0,20,55", "replacement": rule} for model in ["clean", "tree", "squeezed"] for rule in ["median", "zero"]]
        for plan in plans:
            query = {"run": run["id"], "report_sha256": payload["evidence"]["report_sha256"], **plan}
            result = get("/api/model-lab", **query)
            tool = plan["tool"]
            if tool != "stress":
                fitting = set(result["split"]["fit_source_rows"]);threshold_rows = set(result["split"]["threshold_source_rows"])
                assert not fitting & threshold_rows and fitting | threshold_rows == set((data["id_val"]+1).tolist())
                assert not (fitting | threshold_rows) & set((data["id_test"]+1).tolist())
                fit = np.flatnonzero(np.isin(data["id_val"]+1, list(fitting)))
                threshold = np.flatnonzero(np.isin(data["id_val"]+1, list(threshold_rows)))
            if tool == "ensemble":
                names = plan["members"].split(",");weights = np.array(result["parameters"]["weights"])
                test = np.column_stack([predict(m, data["x_test"]) for m in names]).astype(float)
                val = np.column_stack([predict(m, data["x_val"]) for m in names]).astype(float)
                expected = test@weights
                np.testing.assert_array_equal([r["score"] for r in result["records"]], expected)
                assert result["threshold"] == calibrate(data["y_val"][threshold], (val@weights)[threshold], 0.01)
                if plan["mode"] == "fit":
                    gradient = 2*val[fit].T@(val[fit]@weights-data["y_val"][fit])/len(fit)
                    assert gradient.min() >= gradient@weights-1e-6
                else:
                    np.testing.assert_array_equal(weights, np.full(len(names), 1/len(names)))
                with np.load(Path(cfg["root"])/"artifacts"/(attack+".npz")) as archive:
                    expected_attack = np.column_stack([predict(m, archive["adversarial"]) for m in names])@weights
                np.testing.assert_array_equal([r["after"] for r in result["attack_records"]], expected_attack)
                evidence["predictions_checked"] += len(expected)+len(expected_attack)
            elif tool == "calibration":
                model = plan["model"];test = predict(model, data["x_test"]);val = predict(model, data["x_val"])
                p = result["parameters"]
                if plan["method"] == "sigmoid":
                    transform = lambda values: expit(p["slope"]*logit(np.clip(values.astype(float), 1e-6, 1-1e-6))+p["intercept"])
                else:
                    regression = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(val[fit], data["y_val"][fit])
                    transform = regression.predict
                    np.testing.assert_array_equal(p["score_knots"], regression.X_thresholds_)
                    np.testing.assert_array_equal(p["probability_knots"], regression.y_thresholds_)
                expected = transform(test)
                np.testing.assert_array_equal([r["after"] for r in result["records"]], expected)
                assert result["threshold"] == calibrate(data["y_val"][threshold], transform(val)[threshold], 0.01)
                assert np.isfinite(result["after"]["log_loss"])
                assert sum(r["count"] for r in result["after_reliability"]["bins"]) == len(test)
                evidence["predictions_checked"] += len(expected)
            else:
                raw = np.median(data["x_train"].astype(float), axis=0)*schema["scale"]+schema["center"] if plan["replacement"] == "median" else np.zeros(57)
                integer = np.array(schema["integer"], dtype=bool);raw[integer] = np.rint(raw[integer])
                fill = ((raw-schema["center"])/schema["scale"]).astype(np.float32)
                for trial in result["trials"]:
                    inputs = data["x_test"].copy();inputs[:, trial["columns"]] = fill[trial["columns"]]
                    expected = predict(plan["model"], inputs)
                    np.testing.assert_array_equal(trial["scores"], expected)
                    evidence["predictions_checked"] += len(expected)
                query["trial"] = "single-55"
            scopes = ["records", "summary"]+(["attacks"] if tool == "ensemble" else ["bands"] if tool == "calibration" else [])
            for scope in scopes:
                content = get("/api/model-lab.csv", **query, scope=scope)
                reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
                assert len(reader.fieldnames) == len(set(reader.fieldnames))
                rows = list(reader)
                assert rows and all(r["report_sha256"] == result["report_sha256"] for r in rows)
                if scope == "records":
                    assert len(rows) == len(data["y_test"])
                evidence["csv_rows_checked"] += len(rows)
            recipe = get("/api/model-lab-recipe", **query)
            assert recipe["report_sha256"] == result["report_sha256"] and recipe["parameters"] == result["parameters"]
            if tool == "stress":
                assert recipe["selected_trial"] == "single-55" and recipe["selected_columns"] == [55]
            evidence["checks"].append({"run": run["id"], "plan": plan, "test_records": result["test_records"], "report_sha256": result["report_sha256"]})
    Path("reports/dashboard_model_lab_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k != "checks"}, indent=2))
    print("Tool configurations checked:", len(evidence["checks"]))


if __name__ == "__main__":
    main()
