"""Measured decision experiments and matched-record analysis."""
import json

import numpy as np

from .dashboard_views import ResultViews


class AnalysisViews(ResultViews):
    @staticmethod
    def decisions(labels, scores, threshold):
        flagged = scores >= threshold
        tp = int(((labels == 1) & flagged).sum())
        fn = int(((labels == 1) & ~flagged).sum())
        fp = int(((labels == 0) & flagged).sum())
        tn = int(((labels == 0) & ~flagged).sum())
        return {"caught": tp, "missed": fn, "false_alarms": fp, "clear": tn,
                "tested": len(labels), "recall": tp / (tp + fn) if tp + fn else None,
                "fpr": fp / (fp + tn) if fp + tn else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}

    def inputs(self, payload, root, model):
        report = payload["report"]
        row = next((r for r in self.model_rows(report) if r["model"] == model), None)
        if row is None:
            raise ValueError("Choose a filter from this run")
        with np.load(root / "prepared/splits.npz", allow_pickle=False) as saved:
            data = {name: saved[name] for name in ("x_test", "y_test", "id_test", "x_val", "y_val", "id_val")}
        with np.load(root / "artifacts/evaluation-scores.npz", allow_pickle=False) as saved:
            scores = saved[model]
        if scores.shape != data["y_test"].shape or not np.isfinite(scores).all():
            raise ValueError("The saved scores do not match the test records")
        return row, data, scores

    @staticmethod
    def predict(payload, model, values):
        from .data import Domain
        from .defenses import squeeze
        from .models import restore, score
        target = "clean" if model == "squeezed" else model
        classifier, _, _ = restore(payload["report"]["config"], target, payload["evidence"]["checkpoints"][target])
        if model == "squeezed":
            cfg = payload["report"]["config"]
            values = squeeze(values, Domain(cfg), cfg["quantum"])
        return score(classifier, values)

    def decision_lab(self, run_id, model="clean", mode="saved", threshold=0.5, target_fpr=0.01, expected=None, export=False):
        from .models import calibrate
        if mode not in {"saved", "manual", "target"}:
            raise ValueError("Choose saved, manual, or validation-target mode")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1.000001:
            raise ValueError("Choose a threshold between zero and 100%, or just above 100% to flag no records")
        if not np.isfinite(target_fpr) or not 0 <= target_fpr <= 0.25:
            raise ValueError("Choose a false-alarm target between zero and 25%")
        payload, root = self.context(run_id, expected)
        row, data, saved = self.inputs(payload, root, model)
        test_scores = self.predict(payload, model, data["x_test"])
        if not np.allclose(test_scores, saved, atol=1e-6, rtol=1e-5):
            raise ValueError("Fresh model scores differ from the report; rerun evaluation")
        validation_scores = self.predict(payload, model, data["x_val"])
        chosen = row["threshold"] if mode == "saved" else threshold if mode == "manual" else calibrate(data["y_val"], validation_scores, target_fpr)
        fingerprint = payload["evidence"]["report_sha256"]
        if export:
            columns = ["partition", "source_row", "actual_label", "spam_score", "saved_threshold", "preview_threshold", "saved_flagged", "preview_flagged", "model", "report_sha256"]
            values = [[partition, int(raw_id)+1, int(label), float(p), row["threshold"], chosen, bool(p >= row["threshold"]), bool(p >= chosen), model, fingerprint]
                      for partition, scores in [("validation", validation_scores), ("test", test_scores)]
                      for raw_id, label, p in zip(data["id_val" if partition == "validation" else "id_test"], data["y_val" if partition == "validation" else "y_test"], scores)]
            return self.csv_bytes(columns, values)
        metrics = {partition: self.decisions(data["y_"+key], scores, chosen) for partition, key, scores in [("validation", "val", validation_scores), ("test", "test", test_scores)]}
        baseline = self.decisions(data["y_test"], test_scores, row["threshold"])
        changed = (test_scores >= chosen) != (test_scores >= row["threshold"])
        edges = np.linspace(0, 1, 11)
        histogram = []
        for i in range(10):
            mask = (test_scores >= edges[i]) & ((test_scores <= edges[i+1]) if i == 9 else (test_scores < edges[i+1]))
            histogram.append({"low": float(edges[i]), "high": float(edges[i+1]), "count": int(mask.sum()), "spam": int(data["y_test"][mask].sum()), "regular": int((data["y_test"][mask] == 0).sum()), "mean_score": float(test_scores[mask].mean()) if mask.any() else None, "observed_spam": float(data["y_test"][mask].mean()) if mask.any() else None})
        boundaries = np.unique(np.r_[0.0, validation_scores, np.nextafter(validation_scores, np.float32(np.inf)), 1.000001])
        boundaries = boundaries[np.unique(np.linspace(0, len(boundaries)-1, min(101, len(boundaries))).astype(int))]
        curve = [{"threshold": float(t), **self.decisions(data["y_val"], validation_scores, t)} for t in boundaries]
        return {"model": model, "mode": mode, "threshold": float(chosen), "saved_threshold": row["threshold"], "target_fpr": target_fpr,
                "metrics": metrics, "saved_test": baseline, "changed_count": int(changed.sum()),
                "changed_rows": [{"source_row": int(data["id_test"][i])+1, "actual": int(data["y_test"][i]), "score": float(test_scores[i]), "was_flagged": bool(test_scores[i] >= row["threshold"]), "now_flagged": bool(test_scores[i] >= chosen)} for i in np.flatnonzero(changed)[:50]],
                "histogram": histogram, "validation_curve": curve, "brier": float(np.mean((test_scores-data["y_test"])**2)),
                "maximum_score_difference": float(np.max(np.abs(test_scores-saved))), "report_sha256": fingerprint}

    def disagreements(self, left, right=None, left_model="clean", right_model="tree", category="disagree", page=1, size=25, expected_left=None, expected_right=None, export=False):
        right = right or left
        if category not in {"disagree", "left-only", "right-only", "both-wrong", "all"} or page < 1 or size not in {10, 25, 50, 100}:
            raise ValueError("Choose a listed comparison filter and page size")
        a, root_a = self.context(left, expected_left)
        b, root_b = (a, root_a) if right == left else self.context(right, expected_right)
        if expected_right and expected_right != b["evidence"]["report_sha256"]:
            raise ValueError("The right run changed; reload the comparison")
        if self.source_hash(a["report"]) != self.source_hash(b["report"]):
            raise ValueError("Record matching requires the same source dataset")
        ma, da, pa = self.inputs(a, root_a, left_model)
        mb, db, pb = self.inputs(b, root_b, right_model)
        ids, ia, ib = np.intersect1d(da["id_test"], db["id_test"], return_indices=True)
        y, pa, pb = da["y_test"][ia], pa[ia], pb[ib]
        if not np.array_equal(y, db["y_test"][ib]):
            raise ValueError("Matched source records have different labels")
        fa, fb = pa >= ma["threshold"], pb >= mb["threshold"]
        ca, cb = fa == y, fb == y
        masks = {"disagree": fa != fb, "left-only": ca & ~cb, "right-only": ~ca & cb, "both-wrong": ~ca & ~cb, "all": np.ones(len(y), dtype=bool)}
        counts = {key: int(mask.sum()) for key, mask in masks.items()}
        counts["both_correct"] = int((ca & cb).sum())
        positions = np.flatnonzero(masks[category])
        pages = max(1, (len(positions)+size-1)//size)
        page = min(page, pages)
        shown = positions if export else positions[(page-1)*size:page*size]
        rows = [{"source_row": int(ids[i])+1, "actual": int(y[i]), "left_score": float(pa[i]), "right_score": float(pb[i]), "left_flagged": bool(fa[i]), "right_flagged": bool(fb[i]), "left_correct": bool(ca[i]), "right_correct": bool(cb[i]), "score_change": float(pb[i]-pa[i])} for i in shown]
        result = {"rows": rows, "counts": counts, "matched": len(ids), "left_unmatched": len(da["y_test"])-len(ids), "right_unmatched": len(db["y_test"])-len(ids), "total": len(positions), "page": page, "pages": pages, "size": size, "category": category,
                  "left": {"run": left, "model": left_model, "threshold": ma["threshold"], "report_sha256": a["evidence"]["report_sha256"]}, "right": {"run": right, "model": right_model, "threshold": mb["threshold"], "report_sha256": b["evidence"]["report_sha256"]}}
        if export:
            columns = ["source_row", "actual", "left_score", "right_score", "left_flagged", "right_flagged", "left_correct", "right_correct", "score_change"]
            return self.csv_bytes(columns+["left_run", "right_run", "left_model", "right_model", "left_threshold", "right_threshold", "left_report_sha256", "right_report_sha256"], [[r[c] for c in columns]+[left, right, left_model, right_model, ma["threshold"], mb["threshold"], result["left"]["report_sha256"], result["right"]["report_sha256"]] for r in rows])
        return result

    def record_detail(self, run_id, source_row, model="clean", expected=None):
        payload, root = self.context(run_id, expected)
        _, data, _ = self.inputs(payload, root, model)
        matches = np.flatnonzero(data["id_test"] == source_row-1)
        if len(matches) != 1:
            raise ValueError("Choose a source row from this run's test records")
        position = int(matches[0])
        models = list(dict.fromkeys([r["model"] for r in payload["report"]["clean"]]+[model]))
        results = []
        for name in models:
            row, _, saved = self.inputs(payload, root, name)
            fresh = self.predict(payload, name, data["x_test"])
            if not np.allclose(fresh, saved, atol=1e-6, rtol=1e-5):
                raise ValueError("Fresh model scores differ from this report")
            results.append({"model": name, **{key: row[key] for key in ("kind", "ratio", "sanitized") if key in row}, "score": float(fresh[position]), "threshold": row["threshold"], "flagged": bool(fresh[position] >= row["threshold"]), "correct": bool((fresh[position] >= row["threshold"]) == data["y_test"][position])})
        raw = np.loadtxt(self.source_path(payload["report"]), delimiter=",")[source_row-1]
        if int(raw[-1]) != int(data["y_test"][position]):
            raise ValueError("The source label differs from the saved test record")
        schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
        return {"run": run_id, "run_name": payload["report"]["config"].get("dashboard_name") or run_id, "source_row": source_row, "actual": int(raw[-1]), "models": results, "features": [{"name": name, "value": float(raw[i]), "model_input": float(data["x_test"][position, i]), "locked": i in schema["immutable"]} for i, name in enumerate(schema["names"])], "report_sha256": payload["evidence"]["report_sha256"]}
