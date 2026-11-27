"""Reconstruct every saved API request and independently rerun its real target."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from ml_adv_sec.attacks.query_search import EditRules
from ml_adv_sec.common import config, seed
from ml_adv_sec.dashboard import Reports
from ml_adv_sec.dashboard_adaptive import AdaptiveTests
from ml_adv_sec.dashboard_lab import ModelLabViews
from ml_adv_sec.data import Domain
from ml_adv_sec.defenses import Detector, squeeze
from ml_adv_sec.models import restore, score


def predictor(payload, recipe, data, domain):
    cfg = payload["report"]["config"]
    cache = {}
    def base(name, x):
        target = "clean" if name == "squeezed" else name
        if target not in cache:
            cache[target] = restore(cfg, target, payload["evidence"]["checkpoints"][target])[0]
        return score(cache[target], squeeze(x, domain, cfg["quantum"]) if name == "squeezed" else x)
    parameters = recipe["parameters"]
    if recipe["kind"] == "ensemble":
        weights = np.array(parameters["weights"])
        assert (weights >= 0).all() and abs(weights.sum()-1) < 1e-12
        return lambda x: np.column_stack([base(name, x) for name in parameters["members"]])@weights
    if recipe["kind"] == "calibrated":
        if parameters["method"] == "sigmoid":
            return lambda x: expit(parameters["slope"]*logit(np.clip(base(parameters["model"], x).astype(float), 1e-6, 1-1e-6))+parameters["intercept"])
        positions = np.flatnonzero(np.isin(data["id_val"]+1, parameters["fit_source_rows"]))
        validation = base(parameters["model"], data["x_val"])
        fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(validation[positions], data["y_val"][positions])
        np.testing.assert_array_equal(fitted.X_thresholds_, parameters["score_knots"])
        np.testing.assert_array_equal(fitted.y_thresholds_, parameters["probability_knots"])
        return lambda x: fitted.predict(base(parameters["model"], x))
    if recipe["kind"] == "detector":
        model = restore(cfg, "clean", payload["evidence"]["checkpoints"]["clean"])[0]
        detector = Detector().fit(model, data["x_train"])
        assert detector.gradient_cutoff == parameters["gradient_cutoff"] and detector.distance_cutoff == parameters["distance_cutoff"]
        return lambda x: detector.conservative(model, x, score(model, x))
    return lambda x: base(parameters["model"], x)


def main():
    reports = Reports(Path.cwd(), config("configs/default.yaml"))
    manager = AdaptiveTests(ModelLabViews(reports))
    evidence = {"studies": [], "offline_query_predictions": 0, "final_prediction_checks": 0, "measurement_values_checked": 0}
    cohorts = {}
    for manifest in manager.directory.glob("*/manifest.json"):
        result = manager.read(manifest.parent.name, traces=True)
        if result.get("sampling_reference") != "clean" or not all("edits" in t for r in result["rows"] for t in r["trace"]):
            continue
        p = result["settings"]
        payload = reports.read(p["run"]);cfg = payload["report"]["config"]
        seed(cfg["seed"], cfg["threads"])
        domain = Domain(cfg)
        with np.load(Path(cfg["root"])/"prepared/splits.npz", allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        predict = predictor(payload, result["target"], data, domain)
        raw_source = np.loadtxt(manager.views.source_path(payload["report"]), delimiter=",")
        threshold = result["target"]["threshold"]
        for role in ["fit_source_rows", "threshold_source_rows"]:
            chosen = set(result["target"]["parameters"].get(role, []))
            assert chosen <= set((data["id_val"]+1).tolist())
            assert not chosen & set((data["id_test"]+1).tolist())
        roles = result["target"]["parameters"]
        assert not set(roles.get("fit_source_rows", [])) & set(roles.get("threshold_source_rows", []))
        key = (p["run"], p["goal"], p["records"])
        cohort = {(r["source_row"], r["actual"]) for r in result["cohort"]}
        if key in cohorts:
            assert cohorts[key] == cohort
        cohorts[key] = cohort
        for row in result["rows"]:
            original = np.array(result["examples"]["original"][row["index"]])
            changed = np.array(result["examples"]["changed"][row["index"]])
            source = row["source_row"]-1
            np.testing.assert_array_equal(original, raw_source[source, :-1])
            assert row["actual"] == int(raw_source[source, -1])
            index = int(np.flatnonzero(data["id_test"] == source)[0])
            rules = EditRules(domain, data["x_test"][index], original, row["epsilon"], np.inf if p["norm"] == "linf" else 2, p["max_fields"], p["space"], p["policy"])
            assert row["queries"] == len(row["trace"]) <= p["queries"]
            observed_hashes, first_error = set(), None
            for observation in row["trace"]:
                requested = original.copy()
                for column, value in observation["edits"]:
                    requested[column] = value
                digest = hashlib.sha256(requested.astype("<f8").tobytes()).hexdigest()
                assert digest == observation["input_sha256"] and rules.valid(requested)
                observed_hashes.add(digest)
                actual_score = float(predict(rules.scaled(requested)[None])[0])
                decision = int(actual_score >= threshold)
                assert observation["feedback"] == (decision if row["algorithm"] == "decision" else actual_score)
                assert observation["error_found"] == (decision != row["actual"])
                if row["eligible"] and decision != row["actual"] and first_error is None:
                    first_error = observation["query"]
                evidence["offline_query_predictions"] += 1
                evidence["measurement_values_checked"] += 57
            assert first_error == row["first_error_query"]
            assert hashlib.sha256(changed.astype("<f8").tobytes()).hexdigest() in observed_hashes
            assert row["before"] == float(predict(data["x_test"][index][None])[0])
            assert row["after"] == float(predict(rules.scaled(changed)[None])[0])
            assert row["escaped"] == (row["eligible"] and int(row["after"] >= threshold) != row["actual"])
            evidence["final_prediction_checks"] += 2
        assert result["attacker_queries"] == sum(r["queries"] for r in result["rows"])
        assert result["defender_audit_queries"] == 2*len(result["rows"])
        for summary in result["summary"]:
            group = [r for r in result["rows"] if (r["actual"], r["epsilon"], r["algorithm"]) == (summary["actual"], summary["epsilon"], summary["algorithm"])]
            eligible = [r for r in group if r["eligible"]]
            rates = []
            for attack_seed in p["seeds"]:
                selected = [r for r in eligible if r["seed"] == attack_seed]
                if selected:rates.append(sum(r["escaped"] for r in selected)/len(selected))
            assert summary["escaped_trials"] == sum(r["escaped"] for r in eligible)
            assert summary["mean_escape_rate"] == (float(np.mean(rates)) if rates else None)
            assert summary["escape_rate_spread"] == (float(np.std(rates, ddof=1)) if len(rates)>1 else None)
        for scope in ["records", "summary", "queries", "changes"]:
            assert manager.export(result["id"], scope, 0)
        evidence["studies"].append({"id": result["id"], "name": result["name"], "target": result["target"]["kind"], "trials": len(result["rows"]), "http_queries": result["attacker_queries"], "errors_caused": sum(r["escaped"] for r in result["rows"]), "evidence_sha256": result["evidence_sha256"]})
        print("Verified", result["name"], "with", result["attacker_queries"], "actual HTTP query inputs", flush=True)
    assert evidence["studies"], "Run a current adaptive study before auditing"
    Path("reports/dashboard_adaptive_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k != "studies"}, indent=2))


if __name__ == "__main__":
    main()
