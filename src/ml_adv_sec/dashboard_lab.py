"""Three real-model tools: fitted ensembles, score calibration, and input loss."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from .dashboard_replay import ReplayViews


class ModelLabViews(ReplayViews):
    def lab_options(self, run, expected):
        payload, _, data, schema = self.lab_data(run, expected)
        return {"measurements": [{"column": i, "name": name, "integer": bool(schema["integer"][i]), "locked_for_attacks": i in schema["immutable"]} for i, name in enumerate(schema["names"])],
                "counts": {part: len(data["y_"+part]) for part in ["train", "val", "test"]}, "report_sha256": payload["evidence"]["report_sha256"]}

    def lab_data(self, run, expected):
        payload, root = self.context(run, expected)
        with np.load(root/"prepared/splits.npz", allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        sets = [set(data["id_"+part].tolist()) for part in ["train", "val", "test"]]
        if any(sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError("Training, validation, and test source rows must be separate")
        schema = json.loads((root/"prepared/schema.json").read_text(encoding="utf-8"))
        return payload, root, data, schema

    def lab_model(self, payload, root, data, model):
        row = next((r for r in self.model_rows(payload["report"]) if r["model"] == model), None)
        if row is None:
            raise ValueError("Choose a trained filter from this run")
        predict = self.predictor(payload, model)
        test = predict(data["x_test"])
        with np.load(root/"artifacts/evaluation-scores.npz", allow_pickle=False) as archive:
            if not np.allclose(test, archive[model], atol=1e-6, rtol=1e-5):
                raise ValueError("Fresh model scores differ from the verified run")
        return row, predict, test, predict(data["x_val"])

    @staticmethod
    def lab_split(data, seed):
        fit, threshold = [], []
        for label in [0, 1]:
            positions = np.flatnonzero(data["y_val"] == label)
            if len(positions) < 2:
                raise ValueError("Each validation class needs at least two records for separate fitting and threshold selection")
            positions = positions[np.argsort(data["id_val"][positions])]
            positions = np.random.default_rng(seed+label).permutation(positions)
            count = (len(positions)+1)//2
            fit.extend(positions[:count]);threshold.extend(positions[count:])
        return np.array(sorted(fit), dtype=int), np.array(sorted(threshold), dtype=int)

    def lab_metrics(self, y, p, threshold):
        p = np.asarray(p, dtype=float)
        bounded = np.clip(p, 1e-12, 1-1e-12)
        return {**self.decisions(y, p, threshold), "brier": float(np.mean((p-y)**2)),
                "log_loss": float(-np.mean(y*np.log(bounded)+(1-y)*np.log1p(-bounded))),
                "accuracy": float(np.mean((p >= threshold) == y))}

    @staticmethod
    def reliability(y, p, bins):
        p = np.asarray(p, dtype=float)
        groups = np.minimum((p*bins).astype(int), bins-1)
        rows, error = [], 0.0
        for i in range(bins):
            mask = groups == i
            mean = float(np.mean(p[mask])) if mask.any() else None
            actual = float(np.mean(y[mask])) if mask.any() else None
            gap = abs(mean-actual) if mask.any() else None
            error += int(mask.sum())/len(y)*gap if mask.any() else 0
            rows.append({"bin": i, "low": i/bins, "high": (i+1)/bins, "count": int(mask.sum()), "mean_score": mean, "spam_share": actual, "gap": gap})
        return {"bins": rows, "expected_error": float(error)}

    def lab_ensemble(self, payload, root, data, members, mode, fpr):
        from .models import calibrate
        valid = {r["model"] for r in self.model_rows(payload["report"])}
        if not 2 <= len(members) <= 8 or len(set(members)) != len(members) or not set(members) <= valid or mode not in {"fit", "equal"}:
            raise ValueError("Choose two to eight different trained filters and a supported weight method")
        fit, threshold = self.lab_split(data, payload["report"]["config"]["seed"])
        fitted = [self.lab_model(payload, root, data, model) for model in members]
        test = np.column_stack([r[2] for r in fitted]).astype(float)
        val = np.column_stack([r[3] for r in fitted]).astype(float)
        x, y = val[fit], data["y_val"][fit]
        uniform = np.full(len(members), 1/len(members))
        objective = lambda weights: float(np.mean((x@weights-y)**2))
        weights, iterations = uniform, 0
        if mode == "fit":
            solution = minimize(objective, uniform, jac=lambda w: 2*x.T@(x@w-y)/len(y), method="SLSQP",
                                bounds=[(0, 1)]*len(members), constraints=[{"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones_like(w)}],
                                options={"maxiter": 200, "ftol": 1e-12})
            if not solution.success or not np.isfinite(solution.x).all():
                raise ValueError("Weight fitting did not converge; choose another set of filters")
            weights = np.clip(solution.x, 0, 1);weights /= weights.sum()
            iterations = int(solution.nit)
        probabilities, validation = test@weights, val@weights
        chosen = calibrate(data["y_val"][threshold], validation[threshold], fpr)
        records = [{"source_row": int(raw_id)+1, "actual": int(label), "score": float(p), "flagged": bool(p >= chosen),
                    "member_low": float(test[i].min()), "member_high": float(test[i].max())}
                   for i, (raw_id, label, p) in enumerate(zip(data["id_test"], data["y_test"], probabilities))]
        comparisons = []
        for i, (row, _, _, _) in enumerate(fitted):
            member_threshold = calibrate(data["y_val"][threshold], val[threshold, i], fpr)
            comparisons.append({"model": members[i], "weight": float(weights[i]), "threshold": member_threshold,
                                "saved_threshold": row["threshold"], "metrics": self.lab_metrics(data["y_test"], test[:, i], member_threshold)})
        return {"threshold": chosen, "metrics": self.lab_metrics(data["y_test"], probabilities, chosen), "members": comparisons, "records": records,
                "attack_metrics": None, "attack_records": [],
                "fitting": {"method": mode, "iterations": iterations, "uniform_brier": objective(uniform), "fitted_brier": objective(weights)},
                "parameters": {"members": members, "weights": weights.tolist(), "threshold": chosen}, "fit": fit, "threshold_rows": threshold}

    def lab_calibration(self, payload, root, data, model, method, fpr, bins):
        from .models import calibrate
        if method not in {"sigmoid", "isotonic"} or bins not in {5, 10, 20}:
            raise ValueError("Choose a supported calibration method and confidence-band count")
        row, _, test, validation = self.lab_model(payload, root, data, model)
        fit, threshold = self.lab_split(data, payload["report"]["config"]["seed"])
        x, y = validation[fit], data["y_val"][fit]
        if method == "sigmoid":
            features = logit(np.clip(x.astype(float), 1e-6, 1-1e-6))
            def objective(ab):
                z = ab[0]*features+ab[1]
                return float(np.mean(np.logaddexp(0, z)-y*z)+1e-4*((ab[0]-1)**2+ab[1]**2))
            def gradient(ab):
                error = expit(ab[0]*features+ab[1])-y
                return np.array([np.mean(error*features)+2e-4*(ab[0]-1), np.mean(error)+2e-4*ab[1]])
            solution = minimize(objective, [1.0, 0.0], jac=gradient, method="L-BFGS-B", bounds=[(0.001, 100), (-30, 30)], options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
            if not solution.success:
                raise ValueError("Score calibration did not converge; try the other method")
            slope, intercept = solution.x
            transform = lambda p: expit(slope*logit(np.clip(p.astype(float), 1e-6, 1-1e-6))+intercept)
            parameters = {"method": method, "slope": float(slope), "intercept": float(intercept), "probability_clip": 1e-6, "penalty": 1e-4}
        else:
            regression = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
            transform = regression.predict
            parameters = {"method": method, "score_knots": regression.X_thresholds_.tolist(), "probability_knots": regression.y_thresholds_.tolist()}
        corrected, val_corrected = transform(test), transform(validation)
        before_threshold = calibrate(data["y_val"][threshold], validation[threshold], fpr)
        after_threshold = calibrate(data["y_val"][threshold], val_corrected[threshold], fpr)
        records = [{"source_row": int(raw_id)+1, "actual": int(label), "before": float(a), "after": float(b), "before_flagged": bool(a >= before_threshold), "after_flagged": bool(b >= after_threshold)}
                   for raw_id, label, a, b in zip(data["id_test"], data["y_test"], test, corrected)]
        return {"model": model, "before_threshold": before_threshold, "threshold": after_threshold, "saved_threshold": row["threshold"],
                "before": self.lab_metrics(data["y_test"], test, before_threshold), "after": self.lab_metrics(data["y_test"], corrected, after_threshold),
                "before_reliability": self.reliability(data["y_test"], test, bins), "after_reliability": self.reliability(data["y_test"], corrected, bins),
                "records": records, "parameters": parameters, "fit": fit, "threshold_rows": threshold}

    def lab_stress(self, payload, root, data, schema, model, columns, replacement):
        if not 1 <= len(columns) <= 12 or any(type(i) is not int or not 0 <= i < len(schema["names"]) for i in columns) or len(set(columns)) != len(columns) or replacement not in {"median", "zero"}:
            raise ValueError("Choose one to twelve different measurement columns and a supported replacement")
        row, predict, baseline, _ = self.lab_model(payload, root, data, model)
        center, scale = np.asarray(schema["center"]), np.asarray(schema["scale"])
        raw = np.median(data["x_train"].astype(float), axis=0)*scale+center if replacement == "median" else np.zeros(len(center))
        raw[np.asarray(schema["integer"], dtype=bool)] = np.rint(raw[np.asarray(schema["integer"], dtype=bool)])
        fill = ((raw-center)/scale).astype(np.float32)
        baseline_correct = (baseline >= row["threshold"]) == data["y_test"]
        trials = []
        plans = [("single-"+str(column), [column]) for column in columns]+[("combined-"+str(i+1), columns[:i+1]) for i in range(len(columns))]
        for trial_id, selected in plans:
            values = data["x_test"].copy();values[:, selected] = fill[selected]
            probabilities = predict(values)
            correct = (probabilities >= row["threshold"]) == data["y_test"]
            trials.append({"id": trial_id, "columns": selected, "metrics": self.lab_metrics(data["y_test"], probabilities, row["threshold"]),
                           "decisions_changed": int(((probabilities >= row["threshold"]) != (baseline >= row["threshold"])).sum()),
                           "new_mistakes": int((baseline_correct & ~correct).sum()), "fixed_mistakes": int((~baseline_correct & correct).sum()),
                           "mean_absolute_score_change": float(np.mean(np.abs(probabilities-baseline))), "scores": probabilities.tolist()})
        return {"model": model, "threshold": row["threshold"], "baseline": self.lab_metrics(data["y_test"], baseline, row["threshold"]),
                "records": [{"source_row": int(raw_id)+1, "actual": int(label), "before": float(p)} for raw_id, label, p in zip(data["id_test"], data["y_test"], baseline)],
                "trials": trials, "replacements": [{"column": i, "name": schema["names"][i], "raw_value": float(raw[i]), "scaled_value": float(fill[i]), "locked_for_attacks": i in schema["immutable"]} for i in columns],
                "parameters": {"columns": columns, "replacement": replacement, "values": {str(i): float(fill[i]) for i in columns}}}

    def model_lab(self, run, tool, expected=None, model="clean", members=None, mode="fit", method="sigmoid", fpr=0.01, bins=10, columns=None, replacement="median", attack="", export=None, trial=""):
        if tool not in {"ensemble", "calibration", "stress"} or not np.isfinite(fpr) or not 0 <= fpr <= 0.25 or export not in {None, "records", "summary", "attacks", "recipe", "bands"}:
            raise ValueError("Choose a listed model tool, false-alarm target, and export")
        if export == "bands" and tool != "calibration":
            raise ValueError("Confidence-band export requires the score calibration tool")
        if export == "attacks" and (tool != "ensemble" or not attack):
            raise ValueError("Choose an ensemble and a saved attack before exporting attack records")
        payload, root, data, schema = self.lab_data(run, expected)
        # Per-request state stays local: the HTTP server shares this view object.
        if tool == "ensemble":
            result = self.lab_ensemble(payload, root, data, members or [], mode, fpr)
            if attack:
                evidence = self.replay_evidence(run, attack, expected)
                predictors = [self.predictor(payload, name) for name in members]
                weights = np.array(result["parameters"]["weights"])
                before = np.column_stack([p(evidence["original"]) for p in predictors])@weights
                after = np.column_stack([p(evidence["changed"]) for p in predictors])@weights
                threshold = result["threshold"];eligible = before >= threshold;escaped = eligible & (after < threshold)
                result["attack_metrics"] = {"attack": attack, "records": len(before), "initially_caught": int(eligible.sum()), "escaped": int(escaped.sum()), "escape_rate": float(escaped.sum()/eligible.sum()) if eligible.any() else None, "caught_after": int((after >= threshold).sum())}
                result["attack_records"] = [{"source_row": int(raw_id)+1, "before": float(a), "after": float(b), "initially_caught": bool(a >= threshold), "caught_after": bool(b >= threshold)} for raw_id, a, b in zip(evidence["ids"], before, after)]
        elif tool == "calibration":
            result = self.lab_calibration(payload, root, data, model, method, fpr, bins)
        else:
            result = self.lab_stress(payload, root, data, schema, model, columns or [], replacement)
        split = {"fit_source_rows": (data["id_val"][result.pop("fit")]+1).tolist(), "threshold_source_rows": (data["id_val"][result.pop("threshold_rows")]+1).tolist()} if tool != "stress" else {"replacement_source": "training partition" if replacement == "median" else "raw zero", "training_records": len(data["id_train"])}
        result.update(tool=tool, report_sha256=payload["evidence"]["report_sha256"], run=run, split=split, test_records=len(data["y_test"]))
        result["recipe"] = {"tool": tool, "run": run, "model": model if tool != "ensemble" else None, "threshold": result["threshold"], "before_threshold": result.get("before_threshold"), "report_sha256": result["report_sha256"], "checkpoints": payload["evidence"]["checkpoints"], "parameters": result["parameters"], "split": split, "target_fpr": fpr if tool != "stress" else None}
        result["recipe"]["evaluation"] = {"confidence_bands": bins if tool == "calibration" else None, "saved_attack": attack if tool == "ensemble" else None}
        if tool == "stress":
            chosen_trial = next((r for r in result["trials"] if r["id"] == (trial or result["trials"][-1]["id"])), None)
            if chosen_trial is None:
                raise ValueError("Choose a measured input-loss trial")
            result["recipe"].update(selected_trial=chosen_trial["id"], selected_columns=chosen_trial["columns"])
        if export == "recipe":
            return result["recipe"]
        if export:
            if export == "attacks":
                rows = result["attack_records"]
            elif export == "bands":
                rows = [{"version": name, **band} for name in ["before", "after"] for band in result[name+"_reliability"]["bins"]]
            elif export == "summary":
                rows = ([{"model": "ensemble", "weight": 1.0, "threshold": result["threshold"], **result["metrics"]}]+[{"model": r["model"], "weight": r["weight"], "threshold": r["threshold"], **r["metrics"]} for r in result["members"]]) if tool == "ensemble" else [{"version": name, **result[name]} for name in ["before", "after"]] if tool == "calibration" else [{"trial": r["id"], "columns": ",".join(map(str, r["columns"])), "new_mistakes": r["new_mistakes"], "fixed_mistakes": r["fixed_mistakes"], **r["metrics"]} for r in result["trials"]]
            else:
                rows = result["records"]
                if tool == "stress":
                    chosen = next((r for r in result["trials"] if r["id"] == (trial or result["trials"][-1]["id"])), None)
                    if chosen is None:
                        raise ValueError("Choose a measured input-loss trial")
                    rows = [{**r, "after": float(p), "trial": chosen["id"], "before_flagged": r["before"] >= result["threshold"], "after_flagged": p >= result["threshold"]} for r, p in zip(rows, chosen["scores"])]
            if not rows:
                raise ValueError("There are no measured rows for this export")
            columns_out = list(rows[0])
            return self.csv_bytes(columns_out+["run", "tool", "selected_filter", "active_threshold", "report_sha256"], [[r[k] for k in columns_out]+[run, tool, "ensemble" if tool == "ensemble" else model, result["threshold"], result["report_sha256"]] for r in rows])
        return result
