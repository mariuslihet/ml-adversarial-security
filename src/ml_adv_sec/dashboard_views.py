"""Read-only analysis views over fingerprint-verified experiment evidence."""

import csv
import io
import json
from pathlib import Path

import numpy as np


class ResultViews:
    def __init__(self, reports):
        self.reports = reports

    def context(self, run_id, expected=None):
        payload = self.reports.read(run_id)
        if expected and expected != payload["evidence"]["report_sha256"]:
            raise ValueError("This run changed. Reload its results before continuing.")
        root = Path(payload["report"]["config"]["root"])
        root = (self.reports.workspace / root).resolve() if not root.is_absolute() else root.resolve()
        if not root.is_relative_to(self.reports.workspace):
            raise ValueError("This run is outside the workspace")
        return payload, root

    @staticmethod
    def csv_bytes(columns, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            # Editable run names must remain text when opened in a spreadsheet.
            writer.writerow(["'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for v in row])
        return stream.getvalue().encode("utf-8-sig")

    @staticmethod
    def model_rows(report):
        return report["clean"] + report["poisoning"]

    def source_path(self, report):
        path = Path(report["config"]["data_path"])
        return (self.reports.workspace / path).resolve() if not path.is_absolute() else path.resolve()

    def source_hash(self, report):
        raw = self.source_path(report)
        for filename, digest in report["dataset"]["files"].items():
            if Path(filename).resolve() == raw:
                return digest
        raise ValueError("The report is missing its source-data fingerprint")

    def records(self, run_id, model="clean", label="all", outcome="all", search="", order="score-desc", page=1, size=25, expected=None, recheck=False, export=False, features=False):
        payload, root = self.context(run_id, expected)
        report = payload["report"]
        row = next((r for r in self.model_rows(report) if r["model"] == model), None)
        if row is None:
            raise ValueError("Choose a filter from this run")
        if label not in {"all", "spam", "regular"} or outcome not in {"all", "caught", "missed", "false-alarm", "clear"}:
            raise ValueError("Choose a listed record filter")
        if order not in {"score-desc", "score-asc", "source"} or size not in {10, 25, 50, 100} or page < 1:
            raise ValueError("Choose a listed sort order and page size")
        search = search.strip()
        if len(search) > 24 or (search and not search.isdecimal()):
            raise ValueError("Search with a source row number")
        with np.load(root / "prepared/splits.npz", allow_pickle=False) as data:
            x, y, ids = data["x_test"], data["y_test"], data["id_test"]
        with np.load(root / "artifacts/evaluation-scores.npz", allow_pickle=False) as saved:
            probabilities = saved[model]
        if probabilities.shape != y.shape or not np.isfinite(probabilities).all():
            raise ValueError("The saved scores do not match the test records")
        difference = None
        if recheck:
            from .models import restore, score
            from .data import Domain
            from .defenses import squeeze
            target = "clean" if model == "squeezed" else model
            classifier, _, _ = restore(report["config"], target, payload["evidence"]["checkpoints"][target])
            inputs = squeeze(x, Domain(report["config"]), report["config"]["quantum"]) if model == "squeezed" else x
            recalculated = score(classifier, inputs)
            difference = float(np.max(np.abs(recalculated - probabilities)))
            if not np.allclose(recalculated, probabilities, atol=1e-6, rtol=1e-5):
                raise ValueError("Recalculated scores differ from this report. Run the evaluation again.")
        flagged = probabilities >= row["threshold"]
        masks = {"caught": (y == 1) & flagged, "missed": (y == 1) & ~flagged, "false-alarm": (y == 0) & flagged, "clear": (y == 0) & ~flagged}
        counts = {key: int(mask.sum()) for key, mask in masks.items()}
        mask = np.ones(len(y), dtype=bool)
        if label != "all":
            mask &= y == (1 if label == "spam" else 0)
        if outcome != "all":
            mask &= masks[outcome]
        if search:
            mask &= np.array([search in str(int(i) + 1) for i in ids])
        selected = np.flatnonzero(mask)
        if order == "source":
            selected = selected[np.argsort(ids[selected], kind="stable")]
        else:
            selected = selected[np.lexsort((ids[selected], -probabilities[selected] if order == "score-desc" else probabilities[selected]))]
        total = len(selected)
        pages = max(1, (total + size - 1) // size)
        page = min(page, pages)
        displayed = selected if export else selected[(page - 1) * size:page * size]
        def result(i):
            return "caught" if y[i] == 1 and flagged[i] else "missed" if y[i] == 1 else "false-alarm" if flagged[i] else "clear"
        rows = [{"source_row": int(ids[i]) + 1, "actual": "spam" if y[i] else "regular", "decision": "flagged" if flagged[i] else "passed", "score": float(probabilities[i]), "outcome": result(i)} for i in displayed]
        if export:
            columns = ["source_row", "actual_label", "filter_decision", "spam_score", "outcome", "decision_threshold", "model", "report_sha256"]
            values = [[r["source_row"], r["actual"], r["decision"], r["score"], r["outcome"], row["threshold"], model, payload["evidence"]["report_sha256"]] for r in rows]
            if features:
                raw = np.loadtxt(self.source_path(report), delimiter=",")
                schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
                if not np.array_equal(raw[ids, -1].astype(int), y):
                    raise ValueError("The original source labels do not match this test set")
                columns += schema["names"]
                for value, index in zip(values, displayed):
                    value.extend(raw[ids[index], :-1].tolist())
            return self.csv_bytes(columns, values)
        return {"rows": rows, "total": total, "tested": len(y), "counts": counts, "page": page, "pages": pages, "size": size, "model": model, "threshold": row["threshold"], "scores_recalculated": bool(recheck), "maximum_score_difference": difference, "report_sha256": payload["evidence"]["report_sha256"]}

    def compare(self, left, right, expected_left=None, expected_right=None, export=False):
        a, root_a = self.context(left, expected_left)
        if left == right:
            if expected_right and expected_right != a["evidence"]["report_sha256"]:
                raise ValueError("This comparison changed. Reload its results.")
            b, root_b = a, root_a
        else:
            b, root_b = self.context(right, expected_right)
        ra, rb = a["report"], b["report"]
        with np.load(root_a / "prepared/splits.npz", allow_pickle=False) as data:
            ids_a = data["id_test"]
        with np.load(root_b / "prepared/splits.npz", allow_pickle=False) as data:
            ids_b = data["id_test"]
        source_a, source_b = self.source_hash(ra), self.source_hash(rb)
        same_records = source_a == source_b and np.array_equal(np.sort(ids_a), np.sort(ids_b))
        rows = []
        right_models = {r["model"]: r for r in rb["clean"]}
        for row_a in ra["clean"]:
            row_b = right_models.get(row_a["model"])
            if row_b:
                rows.append({"model": row_a["model"], "left": row_a, "right": row_b, "change": {"spam_caught": row_a["fnr"] - row_b["fnr"], "false_alarms": row_b["fpr"] - row_a["fpr"], "accuracy": row_b["accuracy"] - row_a["accuracy"], "f1": row_b["f1"] - row_a["f1"], "p95_ms": row_b["p95_ms"] - row_a["p95_ms"]}})
        attacks = []
        right_attacks = {(r["target"], r["attack"]): r for r in rb["transferability"]}
        for row_a in ra["transferability"]:
            row_b = right_attacks.get((row_a["target"], row_a["attack"]))
            if row_b:
                with np.load(root_a / "artifacts" / (row_a["attack"] + ".npz"), allow_pickle=False) as saved:
                    attacked_a = saved["record_ids"]
                with np.load(root_b / "artifacts" / (row_b["attack"] + ".npz"), allow_pickle=False) as saved:
                    attacked_b = saved["record_ids"]
                same_attacked = source_a == source_b and np.array_equal(np.sort(attacked_a), np.sort(attacked_b))
                attacks.append({"target": row_a["target"], "attack": row_a["attack"], "left": row_a, "right": row_b, "same_attacked_records": same_attacked, "same_limit": row_a["epsilon"] == row_b["epsilon"], "escape_change": None if row_a["asr"] is None or row_b["asr"] is None else row_b["asr"] - row_a["asr"]})
        def identity(key, p):
            report = p["report"]
            return {"id": key, "name": report["config"].get("dashboard_name") or Path(report["config"]["root"]).name, "tested": report["dataset"]["split_sizes"]["test"], "attacked": report["counts"]["attack_samples"], "seed": report["config"]["seed"], "report_sha256": p["evidence"]["report_sha256"]}
        result = {"left": identity(left, a), "right": identity(right, b), "same_test_records": same_records, "same_attacked_records": all(r["same_attacked_records"] for r in attacks), "same_run": left == right, "models": rows, "attacks": attacks}
        if export:
            columns = ["left_run", "right_run", "same_test_records", "model", "left_spam_caught", "right_spam_caught", "spam_caught_change", "left_false_alarms", "right_false_alarms", "false_alarm_change", "left_f1", "right_f1", "left_p95_ms", "right_p95_ms", "left_report_sha256", "right_report_sha256"]
            values = [[result["left"]["name"], result["right"]["name"], same_records, r["model"], 1-r["left"]["fnr"], 1-r["right"]["fnr"], r["change"]["spam_caught"], r["left"]["fpr"], r["right"]["fpr"], r["change"]["false_alarms"], r["left"]["f1"], r["right"]["f1"], r["left"]["p95_ms"], r["right"]["p95_ms"], result["left"]["report_sha256"], result["right"]["report_sha256"]] for r in rows]
            return self.csv_bytes(columns, values)
        return result

    def training(self, run_id, model, mode="selected", page=1, size=25, expected=None, position=None):
        from .eval.workflow import poison_name
        payload, root = self.context(run_id, expected)
        report = payload["report"]
        row = next((r for r in report["poisoning"] if r["model"] == model), None)
        if row is None:
            raise ValueError("Choose a training-change result from this run")
        if mode not in {"selected", "modified", "removed", "all"} or size not in {10, 25, 50, 100} or page < 1:
            raise ValueError("Choose a listed training-record view")
        name = poison_name(row["kind"], row["ratio"])
        cleanup_model = name + "-sanitized"
        with np.load(root / "prepared/splits.npz", allow_pickle=False) as data:
            x, y, ids = data["x_train"], data["y_train"], data["id_train"]
        with np.load(root / "artifacts" / (name + ".npz"), allow_pickle=False) as changed:
            px, py, selected_ids = changed["x"], changed["y"], changed["poison_ids"]
        with np.load(root / "artifacts" / (name + "-quarantine.npz"), allow_pickle=False) as filtered:
            keep = filtered["keep"]
        if px.shape != x.shape or py.shape != y.shape or keep.shape != y.shape:
            raise ValueError("The training evidence does not match the original records")
        selected = np.zeros(len(y), dtype=bool)
        selected[selected_ids] = True
        modified = np.any(px != x, axis=1) | (py != y)
        masks = {"selected": selected, "modified": modified, "removed": ~keep, "all": np.ones(len(y), dtype=bool)}
        counts = {key: int(mask.sum()) for key, mask in masks.items()}
        cleanup_context = {"cleanup_model": cleanup_model, "scores_after_cleanup": bool(row["sanitized"])}
        if position is not None:
            if not 0 <= position < len(y):
                raise ValueError("Training record number is out of range")
            from .models import restore, score
            clean, clean_t, _ = restore(report["config"], "clean", payload["evidence"]["checkpoints"]["clean"])
            target, target_t, _ = restore(report["config"], model, payload["evidence"]["checkpoints"][model])
            inputs = np.stack([x[position], px[position]])
            clean_scores, target_scores = score(clean, inputs), score(target, inputs)
            schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
            before, after = inputs * np.asarray(schema["scale"]) + np.asarray(schema["center"])
            return {**cleanup_context, "source_row": int(ids[position])+1, "position": position, "selected": bool(selected[position]), "modified": bool(modified[position]), "removed": bool(~keep[position]), "before_label": int(y[position]), "after_label": int(py[position]), "clean_threshold": clean_t, "changed_threshold": target_t, "clean_scores": clean_scores.tolist(), "changed_scores": target_scores.tolist(), "model": model, "features": [{"name": feature, "before": float(before[i]), "after": float(after[i]), "change": float(after[i]-before[i]), "scaled_change": float(px[position,i]-x[position,i]), "changed": bool(abs(px[position,i]-x[position,i])>1e-6), "locked": i in schema["immutable"]} for i, feature in enumerate(schema["names"])], "report_sha256": payload["evidence"]["report_sha256"]}
        positions = np.flatnonzero(masks[mode])
        positions = positions[np.argsort(ids[positions], kind="stable")]
        pages = max(1, (len(positions)+size-1)//size)
        page = min(page, pages)
        return {**cleanup_context, "rows": [{"position": int(i), "source_row": int(ids[i])+1, "before_label": int(y[i]), "after_label": int(py[i]), "selected": bool(selected[i]), "modified": bool(modified[i]), "removed": bool(~keep[i])} for i in positions[(page-1)*size:page*size]], "counts": counts, "mode": mode, "total": len(positions), "page": page, "pages": pages, "size": size, "model": model, "report_sha256": payload["evidence"]["report_sha256"]}
