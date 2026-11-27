"""Persisted adaptive tests with paired cohorts, query budgets, and real targets."""
import json
import re
import threading
import uuid
import hashlib
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from .attacks.query_search import EditRules, query_attack
from .attacks.local_oracle import LocalOracle
from .attacks import query_search, local_oracle
from .common import sha256, write_json
from .data import Domain

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class AdaptiveTests:
    def __init__(self, views):
        self.views = views
        self.directory = views.reports.workspace/"runs/adaptive-tests"
        self.lock = threading.Lock()

    @staticmethod
    def validate(values):
        allowed = {"run", "report_sha256", "name", "target", "model", "members", "mode", "method", "fpr", "goal", "records", "seeds", "epsilons", "algorithms", "queries", "max_fields", "space", "policy", "norm"}
        if not isinstance(values, dict) or set(values)-allowed:
            raise ValueError("Use the adaptive-test settings shown in the form")
        p = {"name": "Adaptive attack trial", "target": "filter", "model": "tree", "members": ["clean", "tree", "pgd"], "mode": "fit", "method": "sigmoid", "fpr": .01,
             "goal": "both", "records": 4, "seeds": [42, 73], "epsilons": [.05, .1], "algorithms": ["score", "decision"], "queries": 96, "max_fields": 4, "space": "both", "policy": "paired", "norm": "linf", **values}
        if not isinstance(p.get("run"), str) or not isinstance(p.get("report_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", p["report_sha256"]):
            raise ValueError("Choose a verified source run and its report fingerprint")
        if not isinstance(p["name"], str) or not 1 <= len(p["name"].strip()) <= 80:
            raise ValueError("Use a test name with 1 to 80 characters")
        for key, lo, hi in [("records", 1, 512), ("queries", 8, 512), ("max_fields", 1, 12)]:
            if type(p[key]) is not int or not lo <= p[key] <= hi:
                raise ValueError(f"{key} must be a whole number from {lo} to {hi}")
        for key, choices in [("target", {"filter", "ensemble", "calibrated", "detector"}), ("goal", {"spam", "regular", "both"}), ("policy", {"paired", "independent"}), ("space", {"words", "characters", "both"}), ("norm", {"linf", "l2"}), ("mode", {"fit", "equal"}), ("method", {"sigmoid", "isotonic"})]:
            if not isinstance(p[key], str) or p[key] not in choices:
                raise ValueError(f"Choose a supported {key}")
        for key in ["seeds", "epsilons", "algorithms", "members"]:
            if not isinstance(p[key], list):
                raise ValueError(f"Choose a list of {key}")
        if not 1 <= len(p["seeds"]) <= 5 or any(type(s) is not int or not 0 <= s <= 2147483647 for s in p["seeds"]) or len(set(p["seeds"])) != len(p["seeds"]):
            raise ValueError("Choose one to five different integer attacker seeds")
        if not 1 <= len(p["epsilons"]) <= 5 or any(type(e) not in {int, float} or not np.isfinite(e) or not .001 <= e <= 2 for e in p["epsilons"]) or len(set(p["epsilons"])) != len(p["epsilons"]):
            raise ValueError("Choose one to five different change limits from 0.001 to 2")
        if not p["algorithms"] or len(p["algorithms"]) > 2 or any(not isinstance(a, str) or a not in {"score", "decision"} for a in p["algorithms"]) or len(set(p["algorithms"])) != len(p["algorithms"]):
            raise ValueError("Choose score-guided search, decision-only search, or both")
        if any(not isinstance(m, str) for m in p["members"]) or not isinstance(p["model"], str) or type(p["fpr"]) not in {int, float} or not np.isfinite(p["fpr"]) or not 0 <= p["fpr"] <= .25:
            raise ValueError("Choose supported filters and a false-alarm target from 0 to 25 percent")
        if p["policy"] == "paired" and p["max_fields"] < 2:
            raise ValueError("Paired frequency changes need at least two editable measurements")
        bound = p["records"]*(2 if p["goal"] == "both" else 1)*len(p["seeds"])*len(p["epsilons"])*len(p["algorithms"])*p["queries"]
        if bound > 32768:
            raise ValueError("This plan exceeds 32,768 attacker queries; reduce records, seeds, levels, or the per-record budget")
        p["name"] = p["name"].strip()
        return p

    def target(self, p):
        payload, root, data, schema = self.views.lab_data(p["run"], p["report_sha256"])
        kind = p["target"]
        if kind == "ensemble":
            fitted = self.views.lab_ensemble(payload, root, data, p["members"], p["mode"], p["fpr"])
            predictors = [self.views.predictor(payload, m) for m in p["members"]]
            weights = np.array(fitted["parameters"]["weights"])
            predict = lambda x: np.column_stack([f(x) for f in predictors])@weights
            threshold = fitted["threshold"]
            recipe = {**fitted["parameters"], "mode": p["mode"], "fit_source_rows": (data["id_val"][fitted["fit"]]+1).tolist(), "threshold_source_rows": (data["id_val"][fitted["threshold_rows"]]+1).tolist()}
            component_cost = len(predictors)
        elif kind == "calibrated":
            fitted = self.views.lab_calibration(payload, root, data, p["model"], p["method"], p["fpr"], 10)
            base = self.views.predictor(payload, p["model"])
            parameters = fitted["parameters"]
            if p["method"] == "sigmoid":
                predict = lambda x: expit(parameters["slope"]*logit(np.clip(base(x).astype(float), 1e-6, 1-1e-6))+parameters["intercept"])
            else:
                correction = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(base(data["x_val"])[fitted["fit"]], data["y_val"][fitted["fit"]])
                if not np.array_equal(correction.X_thresholds_, parameters["score_knots"]) or not np.array_equal(correction.y_thresholds_, parameters["probability_knots"]):
                    raise ValueError("The calibration could not be reproduced")
                predict = lambda x: correction.predict(base(x))
            threshold, recipe, component_cost = fitted["threshold"], {"model": p["model"], **parameters}, 1
            recipe.update(fit_source_rows=(data["id_val"][fitted["fit"]]+1).tolist(), threshold_source_rows=(data["id_val"][fitted["threshold_rows"]]+1).tolist())
        else:
            model = "clean" if kind == "detector" else p["model"]
            row, predict, _, _ = self.views.lab_model(payload, root, data, model)
            threshold, recipe, component_cost = row["threshold"], {"model": model}, 1
            if kind == "detector":
                from .defenses import Detector
                from .models import restore, score
                classifier, _, _ = restore(payload["report"]["config"], "clean", payload["evidence"]["checkpoints"]["clean"])
                detector = Detector().fit(classifier, data["x_train"])
                predict = lambda x: detector.conservative(classifier, x, score(classifier, x))
                recipe.update(fallback="flag records that trigger either detector", distance_cutoff=detector.distance_cutoff, gradient_cutoff=detector.gradient_cutoff)
                component_cost = 2
        test_scores = predict(data["x_test"])
        return payload, data, schema, predict, threshold, {"kind": kind, "threshold": threshold, "parameters": recipe, "component_forward_passes_per_query": component_cost}, test_scores

    @staticmethod
    def cohort(data, scores, p, seed):
        selected = []
        labels = [1, 0] if p["goal"] == "both" else [1] if p["goal"] == "spam" else [0]
        for label in labels:
            available = np.flatnonzero(data["y_test"] == label)
            available = available[np.argsort(scores[available], kind="stable")]
            groups = [g for g in np.array_split(available, 4) if len(g)]
            rng = np.random.default_rng(seed+label)
            groups = [rng.permutation(g).tolist() for g in groups]
            order = rng.permutation(len(groups)).tolist()
            count = min(p["records"], len(available));chosen = 0
            while chosen < count:
                for band in order:
                    group = groups[band]
                    if group and chosen < count:
                        index = int(group.pop())
                        selected.append({"position": index, "source_row": int(data["id_test"][index])+1, "actual": label, "score_band": band+1})
                        chosen += 1
        if not selected:
            raise ValueError("No held-out records match the requested goal")
        return selected

    def run(self, values):
        p = self.validate(values)
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another adaptive test is running; let it finish before starting the next")
        oracle = None
        try:
            payload, data, schema, predict, threshold, target, scores = self.target(p)
            reference_scores = self.views.predictor(payload, "clean")(data["x_test"])
            _, reference_root = self.views.context(p["run"], p["report_sha256"])
            with np.load(reference_root/"artifacts/evaluation-scores.npz", allow_pickle=False) as saved:
                if not np.allclose(reference_scores, saved["clean"], atol=1e-6, rtol=1e-5):
                    raise ValueError("The source-cohort reference model differs from this report")
            cohort = self.cohort(data, reference_scores, p, payload["report"]["config"]["seed"])
            domain = Domain(payload["report"]["config"])
            raw_source = np.loadtxt(self.views.source_path(payload["report"]), delimiter=",")
            oracle = LocalOracle(predict, threshold)
            rows, originals, modified = [], [], []
            for epsilon in sorted(p["epsilons"]):
                for algorithm in p["algorithms"]:
                    for seed in p["seeds"]:
                        for record in cohort:
                            position, source = record["position"], record["source_row"]-1
                            if int(raw_source[source, -1]) != record["actual"]:
                                raise ValueError("The source label differs from the held-out record")
                            rules = EditRules(domain, data["x_test"][position], raw_source[source, :-1], epsilon, np.inf if p["norm"] == "linf" else 2, p["max_fields"], p["space"], p["policy"])
                            with oracle.session(rules, algorithm, p["queries"]) as client:
                                trial = query_attack(client.ask, rules, record["actual"], algorithm, p["queries"], seed+record["source_row"]*1009)
                                if trial["queries"] != oracle.query_count(client.token) or trial["queries"] != client.queries or client.queries > p["queries"]:
                                    raise ValueError("The attacker and prediction API query ledgers disagree")
                            # Defender audits are real, separate evaluations, not free attacker feedback.
                            before = float(predict(rules.original[None])[0])
                            after = float(predict(rules.scaled(trial["raw"])[None])[0])
                            eligible = int(before >= threshold) == record["actual"]
                            escaped = eligible and int(after >= threshold) != record["actual"]
                            if escaped != (trial["first_error_query"] is not None) or not rules.valid(trial["raw"]):
                                raise ValueError("The final audit did not match the recorded attacker outcome")
                            if algorithm == "score" and before != trial["trace"][0]["feedback"]:
                                raise ValueError("The target changed during the attack")
                            index = len(rows)
                            rows.append({"index": index, "source_row": record["source_row"], "actual": record["actual"], "score_band": record["score_band"], "epsilon": epsilon, "algorithm": algorithm, "seed": seed,
                                         "before": before, "after": after, "eligible": bool(eligible), "escaped": bool(escaped), "queries": trial["queries"], "first_error_query": trial["first_error_query"],
                                         "reason": trial["reason"], "invalid_proposals": trial["invalid_proposals"], "duplicate_proposals": trial["duplicate_proposals"], "proposals": trial["proposals"],
                                         "changed_columns": rules.changed(trial["raw"]).tolist(), **rules.distance(trial["raw"]), "trace": trial["trace"]})
                            originals.append(rules.raw);modified.append(trial["raw"])
            # Reject a changed source run rather than publishing mixed model versions.
            self.views.context(p["run"], p["report_sha256"])
            identifier = uuid.uuid4().hex
            result = {"id": identifier, "name": p["name"], "created": datetime.now(timezone.utc).isoformat(), "settings": p, "target": target, "cohort": cohort,
                      "source_sha256": self.views.source_hash(payload["report"]), "checkpoints": payload["evidence"]["checkpoints"], "rows": rows, "baseline": self.views.lab_metrics(data["y_test"], scores, threshold),
                      "attacker_queries": sum(r["queries"] for r in rows), "defender_audit_queries": 2*len(rows), "paired_trials": len(rows),
                      "component_attack_forward_passes": sum(r["queries"] for r in rows)*target["component_forward_passes_per_query"],
                      "implementation": {"runner_sha256": SOURCE_SHA256, "search_sha256": query_search.SOURCE_SHA256, "oracle_sha256": local_oracle.SOURCE_SHA256},
                      "runtime_packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "torch", "scikit-learn", "xgboost"]},
                      "attacker_seed_rule": "seed + source_row * 1009; initial random streams are shared between feedback modes",
                      "precision": {"word_frequency_decimals": 2, "character_frequency_decimals": 3}, "transport": "Private loopback HTTP prediction API; server-enforced quotas; decision-only responses contain no scores", "sampling_reference": "clean",
                      "sampling": "Fixed source cohort across targets, budgets, algorithms, and attacker seeds; sampled across four original-filter score bands per class"}
            result["summary"], result["query_curves"] = self.aggregate(result)
            directory = self.directory/identifier
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory/"request.json", p)
            np.savez_compressed(directory/"examples.npz", original=np.array(originals), changed=np.array(modified))
            write_json(directory/"result.json", result)
            write_json(directory/"manifest.pending", {"id": identifier, "files": {name: sha256(directory/name) for name in ["request.json", "examples.npz", "result.json"]}})
            (directory/"manifest.pending").replace(directory/"manifest.json")
            return self.read(identifier)
        finally:
            if oracle is not None:
                oracle.close()
            self.lock.release()

    @staticmethod
    def aggregate(result):
        rows, summary, curves = result["rows"], [], []
        keys = sorted({(r["actual"], r["epsilon"], r["algorithm"]) for r in rows})
        for actual, epsilon, algorithm in keys:
            group = [r for r in rows if (r["actual"], r["epsilon"], r["algorithm"]) == (actual, epsilon, algorithm)]
            eligible = [r for r in group if r["eligible"]]
            per_seed = []
            for seed in result["settings"]["seeds"]:
                subset = [r for r in eligible if r["seed"] == seed]
                per_seed.append(sum(r["escaped"] for r in subset)/len(subset) if subset else None)
            defined = [r for r in per_seed if r is not None]
            worst = len({r["source_row"] for r in eligible if r["escaped"]})
            sources = len({r["source_row"] for r in eligible})
            base = {"actual": actual, "epsilon": epsilon, "algorithm": algorithm}
            summary.append({**base, "trials": len(group), "eligible_trials": len(eligible), "already_wrong_trials": len(group)-len(eligible), "escaped_trials": sum(r["escaped"] for r in eligible),
                            "mean_escape_rate": float(np.mean(defined)) if defined else None, "escape_rate_spread": float(np.std(defined, ddof=1)) if len(defined)>1 else None,
                            "worst_observed_records": worst, "eligible_records": sources, "worst_observed_rate": worst/sources if sources else None,
                            "queries": sum(r["queries"] for r in group), "proposal_limited_trials": sum(r["reason"] == "proposal-limit" for r in group)})
            milestones = sorted({1, result["settings"]["queries"], *[q for q in [8, 16, 32, 64, 128, 256, 512] if q <= result["settings"]["queries"]]})
            for queries in milestones:
                found = sum(r["first_error_query"] is not None and r["first_error_query"] <= queries for r in eligible)
                curves.append({**base, "query_limit": queries, "eligible_trials": len(eligible), "errors_found": found, "escape_rate": found/len(eligible) if eligible else None,
                               "exhausted_searches": sum(r["queries"] < queries and r["reason"] == "proposal-limit" for r in eligible)})
        return summary, curves

    def read(self, identifier, traces=False):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Choose a saved adaptive test")
        directory = self.directory/identifier
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("id") != identifier or set(manifest.get("files", {})) != {"request.json", "examples.npz", "result.json"}:
            raise ValueError("The saved test manifest is invalid")
        for name, expected in manifest["files"].items():
            if sha256(directory/name) != expected:
                raise ValueError("Saved attack evidence changed; this test cannot be presented as verified")
        result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
        if json.loads((directory/"request.json").read_text(encoding="utf-8")) != result.get("settings"):
            raise ValueError("The recorded request and result settings disagree")
        self.views.context(result["settings"]["run"], result["settings"]["report_sha256"])
        if result["id"] != identifier:
            raise ValueError("The saved test identity is invalid")
        if not traces:
            for row in result["rows"]:
                del row["trace"]
        else:
            with np.load(directory/"examples.npz", allow_pickle=False) as archive:
                result["examples"] = {key: archive[key].tolist() for key in ["original", "changed"]}
        result["evidence_sha256"] = manifest["files"]["result.json"]
        return result

    def listing(self):
        records = []
        for path in self.directory.glob("*/manifest.json"):
            try:
                result = self.read(path.parent.name)
                records.append({key: result[key] for key in ["id", "name", "created", "settings", "attacker_queries", "paired_trials", "evidence_sha256"]})
            except (ValueError, KeyError, OSError) as exc:
                records.append({"id": path.parent.name, "name": "Adaptive test needs recheck", "created": "", "error": str(exc)})
        return sorted(records, key=lambda r: r["created"], reverse=True)

    def record(self, identifier, index, query=None):
        result = self.read(identifier, traces=True)
        if type(index) is not int or not 0 <= index < len(result["rows"]):
            raise ValueError("Choose a recorded attack trial")
        row = result["rows"][index]
        payload, root = self.views.context(result["settings"]["run"], result["settings"]["report_sha256"])
        schema = json.loads((root/"prepared/schema.json").read_text(encoding="utf-8"))
        with np.load(self.directory/identifier/"examples.npz", allow_pickle=False) as archive:
            before, after = archive["original"][index], archive["changed"][index]
        scope = "Final retained input"
        if query is not None:
            if type(query) is not int or not 1 <= query <= len(row["trace"]) or "edits" not in row["trace"][query-1]:
                raise ValueError("Choose a query with recorded input edits")
            after = before.copy()
            observation = row["trace"][query-1]
            for column, value in observation["edits"]:
                after[column] = value
            if hashlib.sha256(after.astype("<f8").tobytes()).hexdigest() != observation["input_sha256"]:
                raise ValueError("The recorded API request digest does not match its input")
            scope = f"API query {query}"
        features = [{"column": i, "name": name, "before": float(before[i]), "after": float(after[i]), "change": float(after[i]-before[i]), "locked": i in schema["immutable"]} for i, name in enumerate(schema["names"])]
        return {"row": row, "features": features, "input_scope": scope, "query_number": query, "settings": result["settings"], "threshold": result["target"]["threshold"], "evidence_sha256": result["evidence_sha256"]}

    def export(self, identifier, scope, index=None, query=None):
        result = self.read(identifier)
        if scope == "changes":
            detail = self.record(identifier, index, query)
            r = detail["row"]
            rows = [{**feature, "trial_index": index, "source_row": r["source_row"], "algorithm": r["algorithm"], "seed": r["seed"], "epsilon": r["epsilon"], "input_scope": detail["input_scope"]} for feature in detail["features"]]
        elif scope in {"summary", "records", "queries"}:
            rows = result["summary" if scope == "summary" else "query_curves" if scope == "queries" else "rows"]
        else:
            raise ValueError("Choose a supported measured export")
        columns = list(rows[0])
        return self.views.csv_bytes(columns+["test_id", "evidence_sha256", "report_sha256"], [[",".join(map(str, r[k])) if isinstance(r[k], list) else r[k] for k in columns]+[identifier, result["evidence_sha256"], result["settings"]["report_sha256"]] for r in rows])
