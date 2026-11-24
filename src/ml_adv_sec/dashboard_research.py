"""Measured input importance and attack overlap over authenticated run evidence."""
import json

import numpy as np

from .dashboard_analysis import AnalysisViews


class ResearchViews(AnalysisViews):
    def predictor(self, payload, model):
        from .data import Domain
        from .defenses import squeeze
        from .models import restore, score
        cfg = payload["report"]["config"]
        target = "clean" if model == "squeezed" else model
        classifier, _, _ = restore(cfg, target, payload["evidence"]["checkpoints"][target])
        domain = Domain(cfg) if model == "squeezed" else None
        return lambda x: score(classifier, squeeze(x, domain, cfg["quantum"]) if domain else x)

    def importance(self, run_id, model="clean", partition="test", repeats=3, seed=42, expected=None, export=False):
        if partition not in {"test", "val"} or not 1 <= repeats <= 10 or not 0 <= seed <= 2147483647:
            raise ValueError("Choose a listed partition, one to ten repeats, and a valid seed")
        payload, root = self.context(run_id, expected)
        row, data, saved = self.inputs(payload, root, model)
        predict = self.predictor(payload, model)
        fresh = predict(data["x_test"])
        if not np.allclose(fresh, saved, atol=1e-6, rtol=1e-5):
            raise ValueError("Fresh model scores differ from this report")
        x, y = data["x_"+partition], data["y_"+partition]
        baseline = fresh if partition == "test" else predict(x)
        baseline_metrics = self.decisions(y, baseline, row["threshold"])
        brier = float(np.mean((baseline-y)**2))
        accuracy = float(((baseline >= row["threshold"]) == y).mean())
        schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
        rows = []
        for column, name in enumerate(schema["names"]):
            trials = []
            for repeat in range(repeats):
                order = np.random.default_rng(seed+column*1009+repeat).permutation(len(x))
                values = x.copy()
                values[:, column] = x[order, column]
                p = predict(values)
                metrics = self.decisions(y, p, row["threshold"])
                trials.append({"repeat": repeat+1, "brier_increase": float(np.mean((p-y)**2))-brier,
                               "accuracy_drop": accuracy-float(((p >= row["threshold"]) == y).mean()),
                               "recall_drop": baseline_metrics["recall"]-metrics["recall"], "false_alarm_change": metrics["fpr"]-baseline_metrics["fpr"],
                               "mean_score_change": float(np.mean(np.abs(p-baseline))), "changed_decisions": int(((p >= row["threshold"]) != (baseline >= row["threshold"])).sum())})
            rows.append({"feature": name, "column": column, "locked": column in schema["immutable"], "distinct_values": len(np.unique(x[:, column])), "trials": trials,
                         "mean": {key: float(np.mean([t[key] for t in trials])) for key in trials[0] if key != "repeat"},
                         "spread": {key: float(np.std([t[key] for t in trials], ddof=1 if repeats > 1 else 0)) for key in trials[0] if key != "repeat"}})
        rows.sort(key=lambda r: (-r["mean"]["brier_increase"], r["column"]))
        fingerprint = payload["evidence"]["report_sha256"]
        if export:
            metrics = [key for key in rows[0]["trials"][0] if key != "repeat"]
            columns = ["feature", "column", "locked", "distinct_values", "repeat"]+metrics+["model", "partition", "records", "seed", "threshold", "report_sha256"]
            return self.csv_bytes(columns, [[r["feature"], r["column"], r["locked"], r["distinct_values"], trial["repeat"]]+[trial[k] for k in metrics]+[model, partition, len(y), seed, row["threshold"], fingerprint] for r in rows for trial in r["trials"]])
        return {"rows": rows, "model": model, "partition": partition, "records": len(y), "repeats": repeats, "seed": seed, "threshold": row["threshold"], "baseline": {**baseline_metrics, "brier": brier, "accuracy": accuracy}, "feature_checks": len(rows)*repeats, "maximum_score_difference": float(np.max(np.abs(fresh-saved))), "report_sha256": fingerprint}

    def attack_overlap(self, run_id, target="clean", category="all", page=1, size=25, expected=None, export=False):
        if category not in {"all", "escaped", "resisted", "one"} or page < 1 or size not in {10, 25, 50, 100}:
            raise ValueError("Choose a listed attack view and page size")
        payload, root = self.context(run_id, expected)
        report = payload["report"]
        attacks = sorted({r["attack"] for r in report["transferability"] if r["target"] == target})
        if not attacks or target not in {"clean", "tree"}:
            raise ValueError("Choose a target from the saved attack results")
        row, data, _ = self.inputs(payload, root, target)
        predict = self.predictor(payload, target)
        records = []
        lookup = {int(raw_id): i for i, raw_id in enumerate(data["id_test"])}
        with np.load(root / "artifacts/evaluation-scores.npz", allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        for attack in attacks:
            with np.load(root / "artifacts" / (attack+".npz"), allow_pickle=False) as archive:
                ids, original, modified, labels = archive["record_ids"], archive["original"], archive["adversarial"], archive["labels"]
            if len(np.unique(ids)) != len(ids) or any(int(i) not in lookup for i in ids):
                raise ValueError("The attack source IDs do not match this test set")
            positions = np.array([lookup[int(i)] for i in ids])
            if not np.array_equal(original, data["x_test"][positions]) or not np.array_equal(labels, data["y_test"][positions]) or not (labels == 1).all():
                raise ValueError("Saved attacks do not match the original held-out spam records")
            before, after = predict(original), predict(modified)
            for suffix, values in [("before", before), ("after", after)]:
                if not np.allclose(values, saved[attack+"-"+target+"-"+suffix], atol=1e-6, rtol=1e-5):
                    raise ValueError("Fresh attack scores differ from the saved evaluation")
            records.append({"attack": attack, "ids": ids, "before": before, "after": after})
        common = records[0]["ids"]
        for record in records[1:]:
            common = np.intersect1d(common, record["ids"])
        common = np.sort(common)
        scores, samples, eligibility = [], [], []
        for record in records:
            index = {int(raw_id): i for i, raw_id in enumerate(record["ids"])}
            positions = np.array([index[int(i)] for i in common], dtype=int)
            samples.append(positions)
            scores.append(record["after"][positions])
            eligibility.append(record["before"][positions] >= row["threshold"])
        eligible = eligibility[0]
        if any(not np.array_equal(eligible, other) for other in eligibility[1:]):
            raise ValueError("The attacks disagree about their original model decisions")
        score_matrix = np.stack(scores, axis=1)
        escaped = (score_matrix < row["threshold"]) & eligible[:, None]
        escaped_count = escaped.sum(axis=1)
        masks = {"all": np.ones(len(common), dtype=bool), "escaped": escaped_count > 0, "resisted": eligible & (escaped_count == 0), "one": escaped_count == 1}
        counts = {key: int(mask.sum()) for key, mask in masks.items()}
        counts["eligible"] = int(eligible.sum())
        counts["already_missed"] = int((~eligible).sum())
        selected = np.flatnonzero(masks[category])
        pages = max(1, (len(selected)+size-1)//size)
        page = min(page, pages)
        shown = selected if export else selected[(page-1)*size:page*size]
        rows = [{"source_row": int(common[i])+1, "eligible": bool(eligible[i]), "escaped_count": int(escaped_count[i]), "attacks": [{"attack": attack, "sample": int(samples[j][i]), "score": float(score_matrix[i,j]), "escaped": bool(escaped[i,j]), "flagged": bool(score_matrix[i,j] >= row["threshold"])} for j, attack in enumerate(attacks)]} for i in shown]
        fingerprint = payload["evidence"]["report_sha256"]
        if export:
            columns = ["source_row", "originally_caught", "attacks_escaped", "attack", "sample", "after_score", "escaped", "flagged", "target", "threshold", "report_sha256"]
            return self.csv_bytes(columns, [[r["source_row"], r["eligible"], r["escaped_count"], a["attack"], a["sample"], a["score"], a["escaped"], a["flagged"], target, row["threshold"], fingerprint] for r in rows for a in r["attacks"]])
        overlaps = []
        for i, first in enumerate(attacks):
            for j, second in enumerate(attacks):
                both = int((escaped[:,i] & escaped[:,j]).sum())
                either = int((escaped[:,i] | escaped[:,j]).sum())
                overlaps.append({"left": first, "right": second, "both": both, "either": either, "jaccard": both/either if either else None})
        return {"rows": rows, "attacks": [{"attack": attack, "available": len(records[i]["ids"]), "excluded": len(records[i]["ids"])-len(common), "escaped": int(escaped[:,i].sum()), "escape_rate": float(escaped[:,i].sum()/eligible.sum()) if eligible.any() else None} for i, attack in enumerate(attacks)], "overlaps": overlaps, "counts": counts, "matched": len(common), "total": len(selected), "page": page, "pages": pages, "size": size, "target": target, "threshold": row["threshold"], "report_sha256": fingerprint}
