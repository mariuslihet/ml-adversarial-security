"""Fresh saved-attack replay and source-matched attack comparisons."""
import json

import numpy as np

from .dashboard_investigations import InvestigationViews


class ReplayViews(InvestigationViews):
    def replay_evidence(self, run, attack, expected):
        payload, root = self.context(run, expected)
        report = payload["report"]
        if attack not in {r["attack"] for r in report["transferability"]}:
            raise ValueError("Choose an attack saved in this run")
        with np.load(root/"artifacts"/(attack+".npz"), allow_pickle=False) as archive:
            ids, original, changed, labels = [archive[k] for k in ["record_ids", "original", "adversarial", "labels"]]
        with np.load(root/"prepared/splits.npz", allow_pickle=False) as archive:
            test_ids, x, y = [archive[k] for k in ["id_test", "x_test", "y_test"]]
        lookup = {int(raw_id): i for i, raw_id in enumerate(test_ids)}
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids) or len(set(ids.tolist())) != len(ids) or any(int(i) not in lookup for i in ids):
            raise ValueError("Attack records do not match this run's held-out source rows")
        positions = np.array([lookup[int(i)] for i in ids])
        if original.shape != changed.shape or not np.isfinite(changed).all() or not np.array_equal(original, x[positions]) or not np.array_equal(labels, y[positions]) or not (labels == 1).all():
            raise ValueError("Attack measurements or labels do not match their original records")
        schema = json.loads((root/"prepared/schema.json").read_text(encoding="utf-8"))
        return {"payload": payload, "root": root, "ids": ids, "original": original, "changed": changed, "schema": schema, "positions": positions, "test": x}

    def replay_scores(self, evidence, model):
        payload, root = evidence["payload"], evidence["root"]
        row = next((r for r in self.model_rows(payload["report"]) if r["model"] == model), None)
        if row is None:
            raise ValueError("Choose a trained filter available in both runs")
        predict = self.predictor(payload, model)
        full = predict(evidence["test"])
        before, after = predict(evidence["original"]), predict(evidence["changed"])
        with np.load(root/"artifacts/evaluation-scores.npz", allow_pickle=False) as archive:
            if not np.allclose(full, archive[model], atol=1e-6, rtol=1e-5):
                raise ValueError("Reloaded model scores differ from this verified report")
        if before.shape != evidence["ids"].shape or after.shape != before.shape or not np.isfinite(after).all() or not np.isfinite(before).all():
            raise ValueError("The model did not return one finite score per attacked record")
        caught_before, caught_after = before >= row["threshold"], after >= row["threshold"]
        eligible = int(caught_before.sum())
        escaped = int((caught_before & ~caught_after).sum())
        return {"before": before, "after": after, "caught_before": caught_before, "caught_after": caught_after,
                "summary": {"model": model, "threshold": row["threshold"], "tested": len(before), "initially_caught": eligible,
                            "escaped": escaped, "escape_rate": escaped/eligible if eligible else None,
                            "caught_after": int(caught_after.sum()), "newly_caught": int((~caught_before & caught_after).sum()),
                            "mean_score_change": float(np.mean(after-before))}}

    def attack_replay(self, run, attack, model="clean", other="", other_attack="", category="all", page=1, size=25,
                      expected=None, other_expected=None, export=None, source_row=None):
        categories = {"all", "first-only", "second-only", "both-caught", "both-missed", "changed"}
        if category not in categories or size not in {10, 25, 50, 100} or page < 1 or export not in {None, "records", "models", "changes"}:
            raise ValueError("Choose a listed record view, export, and page size")
        if source_row is not None and export in {"records", "models"}:
            raise ValueError("Use the measurement-changes export for an individual source row")
        a = self.replay_evidence(run, attack, expected)
        other, other_attack = other or run, other_attack or attack
        b = a if other == run and other_attack == attack else self.replay_evidence(other, other_attack, other_expected)
        if other == run and other_expected and other_expected != a["payload"]["evidence"]["report_sha256"]:
            raise ValueError("The comparison run changed; reload it")
        if self.source_hash(a["payload"]["report"]) != self.source_hash(b["payload"]["report"]):
            raise ValueError("Compare runs that use the same authenticated source dataset")
        scores = {r["model"]: self.replay_scores(a, r["model"]) for r in self.model_rows(a["payload"]["report"])}
        if model not in scores:
            raise ValueError("Choose a trained filter from this run")
        sa = scores[model]
        sb = sa if b is a else self.replay_scores(b, model)
        common = np.intersect1d(a["ids"], b["ids"])
        maps = [{int(raw_id): i for i, raw_id in enumerate(e["ids"])} for e in [a, b]]
        rows = []
        for raw_id in common:
            i, j = [m[int(raw_id)] for m in maps]
            first, second = bool(sa["caught_after"][i]), bool(sb["caught_after"][j])
            outcome = "both-caught" if first and second else "first-only" if first else "second-only" if second else "both-missed"
            rows.append({"source_row": int(raw_id)+1, "first_sample": i, "second_sample": j, "first_before": float(sa["before"][i]),
                         "first_after": float(sa["after"][i]), "second_before": float(sb["before"][j]), "second_after": float(sb["after"][j]),
                         "first_initially_caught": bool(sa["caught_before"][i]), "second_initially_caught": bool(sb["caught_before"][j]),
                         "first_caught": first, "second_caught": second, "outcome": outcome,
                         "score_difference": float(sb["after"][j]-sa["after"][i])})
        counts = {key: sum(r["outcome"] == key for r in rows) for key in categories-{"all", "changed"}}
        counts.update(all=len(rows), changed=counts["first-only"]+counts["second-only"])
        hashes = {"first_sha256": a["payload"]["evidence"]["report_sha256"], "second_sha256": b["payload"]["evidence"]["report_sha256"]}
        metadata = {"run": run, "attack": attack, "other": other, "other_attack": other_attack, "selected_model": model, **hashes}
        if source_row is not None or export == "changes":
            selected = next((r for r in rows if r["source_row"] == source_row), None)
            if selected is None:
                raise ValueError("Choose a source row present in both attack sets")
            changes = []
            for column, name in enumerate(a["schema"]["names"]):
                measurement = {"column": column, "name": name}
                for side, evidence in [("first", a), ("second", b)]:
                    index = selected[side+"_sample"]
                    original, changed = evidence["original"][index, column], evidence["changed"][index, column]
                    schema = evidence["schema"]
                    before = float(original)*schema["scale"][column]+schema["center"][column]
                    after = float(changed)*schema["scale"][column]+schema["center"][column]
                    measurement.update({side+"_original": before, side+"_changed": after, side+"_change": after-before,
                                        side+"_scaled_change": float(changed-original), side+"_locked": column in schema["immutable"]})
                changes.append(measurement)
            if export == "changes":
                columns = list(changes[0])+["source_row"]+list(metadata)
                return self.csv_bytes(columns, [[r[k] for k in changes[0]]+[source_row]+list(metadata.values()) for r in changes])
            return {"record": selected, "changes": changes, "first_threshold": sa["summary"]["threshold"], "second_threshold": sb["summary"]["threshold"], **metadata}
        selected = [r for r in rows if category == "all" or (category == "changed" and r["first_caught"] != r["second_caught"]) or r["outcome"] == category]
        selected.sort(key=lambda r: (-abs(r["score_difference"]), r["source_row"]))
        summaries = [s["summary"] for s in scores.values()]
        if export:
            exported = summaries if export == "models" else selected
            columns = list(summaries[0]) if export == "models" else ["source_row", "first_sample", "second_sample", "first_before", "first_after", "second_before", "second_after", "first_initially_caught", "second_initially_caught", "first_caught", "second_caught", "outcome", "score_difference"]
            return self.csv_bytes(columns+list(metadata), [[r[k] for k in columns]+list(metadata.values()) for r in exported])
        pages = max(1, (len(selected)+size-1)//size)
        page = min(page, pages)
        return {"models": summaries, "rows": selected[(page-1)*size:page*size], "counts": counts, "total": len(selected), "page": page, "pages": pages,
                "first_records": len(a["ids"]), "second_records": len(b["ids"]), "first_unmatched": len(a["ids"])-len(common), "second_unmatched": len(b["ids"])-len(common),
                "first_summary": sa["summary"], "second_summary": sb["summary"], **metadata}
