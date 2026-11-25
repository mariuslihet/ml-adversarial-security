"""Attack-budget evidence and a durable, revision-checked local case notebook."""
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from .dashboard_segments import SegmentViews


class InvestigationViews(SegmentViews):
    def budget_evidence(self, run_id, model, expected):
        payload, root = self.context(run_id, expected)
        measured = sorted([r for r in payload["report"]["pareto"] if r["model"] == model], key=lambda r: r["epsilon"])
        if model not in {"clean", "pgd", "trades", "squeezed-bpda"} or not measured:
            raise ValueError("Choose a model from the saved budget experiments")
        name = "squeezed" if model == "squeezed-bpda" else model
        row, data, _ = self.inputs(payload, root, name)
        predict = self.predictor(payload, name)
        threshold = row["threshold"]
        checkpoint = payload["evidence"]["checkpoints"]["clean" if model == "squeezed-bpda" else model]
        ids = data["id_test"][data["y_test"] == 1]
        original_test = data["x_test"][data["y_test"] == 1]
        levels = []
        for result in measured:
            artifact = f"sweep-{model}-{result['epsilon']!r}"
            with np.load(root / "artifacts" / (artifact+".npz"), allow_pickle=False) as archive:
                original, changed, labels = archive["original"], archive["adversarial"], archive["labels"]
            metadata = json.loads((root / "artifacts" / (artifact+".json")).read_text(encoding="utf-8"))["details"]
            if metadata["version"] != checkpoint or metadata["threshold"] != threshold or metadata["epsilon"] != result["epsilon"]:
                raise ValueError("The saved budget experiment uses a different model version or threshold")
            if not np.array_equal(original, original_test[:len(original)]) or not (labels == 1).all() or changed.shape != original.shape:
                raise ValueError("The budget examples do not match the held-out source records")
            before, after = predict(original), predict(changed)
            eligible = before >= threshold
            escaped = eligible & (after < threshold)
            asr = float(escaped.sum()/eligible.sum()) if eligible.any() else None
            recall = float((after >= threshold).mean())
            if int(eligible.sum()) != result["eligible"] or not np.isclose(recall, result["adversarial_recall"], atol=1e-7) or (asr is None) != (result["asr"] is None) or (asr is not None and not np.isclose(asr, result["asr"], atol=1e-7)):
                raise ValueError("Fresh budget outcomes differ from the saved report")
            levels.append({"epsilon": result["epsilon"], "artifact": artifact, "ids": ids[:len(original)], "original": original, "changed": changed, "before": before, "after": after, "eligible": eligible, "escaped": escaped, "asr": asr, "recall": recall})
        if any(not np.array_equal(level["ids"], levels[0]["ids"]) or not np.array_equal(level["eligible"], levels[0]["eligible"]) for level in levels[1:]):
            raise ValueError("Budget levels do not share the same source records and original decisions")
        return payload, root, row, levels

    def budgets(self, run_id, model="clean", category="all", page=1, size=25, expected=None, export=None):
        if category not in {"all", "escaped", "resisted", "recovered", "already-missed"} or page < 1 or size not in {10, 25, 50, 100} or export not in {None, "summary", "records"}:
            raise ValueError("Choose a listed budget view and page size")
        payload, _, row, levels = self.budget_evidence(run_id, model, expected)
        eligible = levels[0]["eligible"]
        ids = levels[0]["ids"]
        escaped = np.stack([level["escaped"] for level in levels], axis=1)
        recovered = (escaped[:, :-1] & ~escaped[:, 1:]).any(axis=1)
        masks = {"all": np.ones(len(ids), dtype=bool), "escaped": escaped.any(axis=1), "resisted": eligible & ~escaped.any(axis=1), "recovered": recovered, "already-missed": ~eligible}
        counts = {key: int(mask.sum()) for key, mask in masks.items()}
        counts["eligible"] = int(eligible.sum())
        selected = np.flatnonzero(masks[category])
        selected = selected[np.argsort(ids[selected], kind="stable")]
        pages = max(1, (len(selected)+size-1)//size)
        page = min(page, pages)
        shown = selected if export == "records" else selected[(page-1)*size:page*size]
        rows = [{"source_row": int(ids[i])+1, "before_score": float(levels[0]["before"][i]), "eligible": bool(eligible[i]), "first_observed_escape": next((level["epsilon"] for j, level in enumerate(levels) if escaped[i, j]), None), "recovered": bool(recovered[i]), "levels": [{"index": j, "epsilon": level["epsilon"], "score": float(level["after"][i]), "escaped": bool(escaped[i,j]), "flagged": bool(level["after"][i] >= row["threshold"])} for j, level in enumerate(levels)]} for i in shown]
        summary = [{"index": j, "epsilon": level["epsilon"], "records": len(ids), "eligible": int(eligible.sum()), "escaped": int(level["escaped"].sum()), "escape_rate": level["asr"], "spam_caught": level["recall"], "newly_escaped": int((escaped[:, j] & ~escaped[:, :j].any(axis=1)).sum()), "caught_again": int((escaped[:, j-1] & ~escaped[:, j]).sum()) if j else 0} for j, level in enumerate(levels)]
        fingerprint = payload["evidence"]["report_sha256"]
        if export == "summary":
            keys = list(summary[0])
            return self.csv_bytes(keys+["model", "threshold", "report_sha256"], [[r[k] for k in keys]+[model, row["threshold"], fingerprint] for r in summary])
        if export == "records":
            columns = ["source_row", "originally_caught", "before_score", "first_observed_escape", "recovered", "epsilon", "after_score", "escaped", "flagged", "model", "threshold", "report_sha256"]
            return self.csv_bytes(columns, [[r["source_row"], r["eligible"], r["before_score"], r["first_observed_escape"], r["recovered"], level["epsilon"], level["score"], level["escaped"], level["flagged"], model, row["threshold"], fingerprint] for r in rows for level in r["levels"]])
        return {"model": model, "levels": summary, "counts": counts, "rows": rows, "total": len(selected), "page": page, "pages": pages, "size": size, "threshold": row["threshold"], "report_sha256": fingerprint}

    def budget_record(self, run_id, model, level, source_row, expected=None):
        payload, root, row, levels = self.budget_evidence(run_id, model, expected)
        if not 0 <= level < len(levels):
            raise ValueError("Choose a measured budget level")
        selected = levels[level]
        indices = np.flatnonzero(selected["ids"] == source_row-1)
        if len(indices) != 1:
            raise ValueError("Choose a source record in this budget experiment")
        i = int(indices[0])
        schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
        before = selected["original"][i]*np.asarray(schema["scale"])+np.asarray(schema["center"])
        after = selected["changed"][i]*np.asarray(schema["scale"])+np.asarray(schema["center"])
        delta = selected["changed"][i]-selected["original"][i]
        return {"source_row": source_row, "model": model, "epsilon": selected["epsilon"], "before_score": float(selected["before"][i]), "after_score": float(selected["after"][i]), "threshold": row["threshold"], "eligible": bool(selected["eligible"][i]), "escaped": bool(selected["escaped"][i]), "linf": float(np.max(np.abs(delta))), "l2": float(np.linalg.norm(delta)), "features": [{"name": name, "before": float(before[j]), "after": float(after[j]), "scaled_change": float(delta[j]), "changed": bool(delta[j] != 0), "locked": j in schema["immutable"]} for j, name in enumerate(schema["names"])], "report_sha256": payload["evidence"]["report_sha256"]}


class CaseNotebook:
    statuses = {"open", "reviewed", "resolved", "archived"}

    def __init__(self, views):
        self.views = views
        self.path = views.reports.workspace / "runs/dashboard-notebook.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self):
        try:
            db = sqlite3.connect(self.path, timeout=10)
        except sqlite3.Error as exc:
            raise ValueError("The local notebook could not be opened; its saved data has not been replaced") from exc
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, run TEXT NOT NULL, report_sha256 TEXT NOT NULL, source_row INTEGER NOT NULL, model TEXT NOT NULL, title TEXT NOT NULL, notes TEXT NOT NULL, status TEXT NOT NULL, tags TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS case_history (id TEXT NOT NULL, revision INTEGER NOT NULL, snapshot TEXT NOT NULL, PRIMARY KEY(id, revision))")
                yield db
        except sqlite3.Error as exc:
            raise ValueError("The local notebook could not be read or saved; its data has not been replaced") from exc
        finally:
            db.close()

    @staticmethod
    def decoded(row):
        result = dict(row)
        result["tags"] = json.loads(result["tags"])
        return result

    @staticmethod
    def fields(values):
        title, notes, status, tags = values.get("title"), values.get("notes", ""), values.get("status", "open"), values.get("tags", [])
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120 or not isinstance(notes, str) or len(notes) > 4000 or not isinstance(status, str) or status not in CaseNotebook.statuses:
            raise ValueError("Enter a title, notes up to 4,000 characters, and a listed status")
        if not isinstance(tags, list) or len(tags) > 8 or any(not isinstance(tag, str) or not 1 <= len(tag.strip()) <= 24 for tag in tags):
            raise ValueError("Use up to eight tags, each one to 24 characters")
        return {"title": title.strip(), "notes": notes, "status": status, "tags": list(dict.fromkeys(tag.strip() for tag in tags))}

    def create(self, values):
        allowed = {"run", "report_sha256", "source_row", "model", "title", "notes", "status", "tags"}
        if set(values)-allowed:
            raise ValueError("The case contains unsupported fields")
        fields = self.fields(values)
        if not isinstance(values.get("run"), str) or not 1 <= len(values["run"]) <= 80 or not isinstance(values.get("model", "clean"), str) or len(values.get("model", "clean")) > 128:
            raise ValueError("Choose a saved run and a model from its results")
        if type(values.get("source_row")) is not int or not isinstance(values.get("report_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", values["report_sha256"]):
            raise ValueError("Choose a real source row and the current report fingerprint")
        detail = self.views.record_detail(values.get("run", ""), values["source_row"], values.get("model", "clean"), values["report_sha256"])
        now = datetime.now(timezone.utc).isoformat()
        record = {"id": uuid.uuid4().hex, "revision": 1, "run": values["run"], "report_sha256": detail["report_sha256"], "source_row": detail["source_row"], "model": values.get("model", "clean"), **fields, "created": now, "updated": now}
        stored = {**record, "tags": json.dumps(record["tags"])}
        with self.connection() as db:
            db.execute("INSERT INTO cases VALUES (:id,:revision,:run,:report_sha256,:source_row,:model,:title,:notes,:status,:tags,:created,:updated)", stored)
            db.execute("INSERT INTO case_history VALUES (?,?,?)", (record["id"], 1, json.dumps(record)))
        return record

    def update(self, values):
        if set(values)-{"id", "revision", "title", "notes", "status", "tags"}:
            raise ValueError("A saved case's evidence binding cannot be changed")
        fields = self.fields(values)
        if type(values.get("revision")) is not int:
            raise ValueError("Reload the current case before saving")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            found = db.execute("SELECT * FROM cases WHERE id=?", (values.get("id"),)).fetchone()
            if found is None:
                raise ValueError("Case not found")
            record = self.decoded(found)
            if record["revision"] != values["revision"]:
                raise ValueError("This case changed in another window. Reload it before saving; your draft has not been applied.")
            record.update(fields, revision=record["revision"]+1, updated=datetime.now(timezone.utc).isoformat())
            db.execute("UPDATE cases SET revision=:revision,title=:title,notes=:notes,status=:status,tags=:tags,updated=:updated WHERE id=:id", {**record, "tags": json.dumps(record["tags"])})
            db.execute("INSERT INTO case_history VALUES (?,?,?)", (record["id"], record["revision"], json.dumps(record)))
        return record

    def listing(self, run="", status="all", search="", page=1, size=25, export=False):
        if status not in self.statuses | {"all"} or not isinstance(search, str) or len(search) > 200 or page < 1 or size not in {10, 25, 50, 100}:
            raise ValueError("Choose a listed case filter and page size")
        clauses, args = [], []
        if run:
            clauses.append("run=?"); args.append(run)
        if status != "all":
            clauses.append("status=?"); args.append(status)
        if search.strip():
            clauses.append("instr(lower(title || ' ' || notes || ' ' || tags),lower(?))>0"); args.append(search.strip())
        where = " WHERE "+" AND ".join(clauses) if clauses else ""
        with self.connection() as db:
            db.execute("BEGIN")
            total = db.execute("SELECT COUNT(*) FROM cases"+where, args).fetchone()[0]
            pages = max(1, (total+size-1)//size)
            page = min(page, pages)
            query = "SELECT * FROM cases"+where+" ORDER BY updated DESC,id"
            records = db.execute(query if export else query+" LIMIT ? OFFSET ?", args if export else args+[size, (page-1)*size]).fetchall()
        rows = [self.decoded(row) for row in records]
        if export:
            columns = ["id", "revision", "title", "notes", "status", "run", "source_row", "model", "report_sha256", "created", "updated"]
            return self.views.csv_bytes(columns+["tags"], [[r[k] for k in columns]+[", ".join(r["tags"])] for r in rows])
        return {"rows": rows, "total": total, "page": page, "pages": pages, "size": size}

    def get(self, case_id):
        with self.connection() as db:
            db.execute("BEGIN")
            found = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if found is None:
                raise ValueError("Case not found")
            record = self.decoded(found)
            history = [json.loads(row[0]) for row in db.execute("SELECT snapshot FROM case_history WHERE id=? ORDER BY revision DESC", (case_id,)).fetchall()]
        try:
            payload, _ = self.views.context(record["run"], record["report_sha256"])
            detail = self.views.record_detail(record["run"], record["source_row"], record["model"], record["report_sha256"])
            evidence = {"record": detail, "source_sha256": self.views.source_hash(payload["report"]), "checkpoints": payload["evidence"]["checkpoints"], "recalculated_at": datetime.now(timezone.utc).isoformat()}
            return {"case": record, "history": history, "evidence": evidence, "evidence_error": None}
        except (ValueError, KeyError, OSError) as exc:
            return {"case": record, "history": history, "evidence": None, "evidence_error": str(exc)}
