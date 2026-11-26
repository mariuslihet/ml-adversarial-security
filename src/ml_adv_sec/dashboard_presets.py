"""Versioned local experiment settings, exact configuration reuse, and real reruns."""
import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import yaml


class ExperimentPresets:
    def __init__(self, jobs, campaigns, views):
        self.jobs, self.campaigns, self.views = jobs, campaigns, views
        self.path = jobs.reports.workspace / "runs/dashboard-presets.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self):
        try:
            db = sqlite3.connect(self.path, timeout=10)
        except sqlite3.Error as exc:
            raise ValueError("The preset library could not be opened; saved settings were not replaced") from exc
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS presets (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, name TEXT NOT NULL, notes TEXT NOT NULL, archived INTEGER NOT NULL, snapshot TEXT NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS preset_history (id TEXT NOT NULL, revision INTEGER NOT NULL, snapshot TEXT NOT NULL, PRIMARY KEY(id, revision))")
                yield db
        except sqlite3.Error as exc:
            raise ValueError("The local preset library could not be read or saved") from exc
        finally:
            db.close()

    def freeze(self, values):
        cfg, name = self.jobs.validate(values)
        settings = {k: cfg[k] for k in set(self.jobs.integer_limits) | set(self.jobs.float_limits) | {"epsilons", "poison_ratios"}}
        settings["name"] = name
        return settings

    @staticmethod
    def digest(settings, context):
        return hashlib.sha256(json.dumps({"settings": settings, "fixed_context": context}, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def from_report(self, run, expected):
        if not isinstance(run, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Choose a verified run and its exact report fingerprint")
        payload, _ = self.views.context(run, expected)
        cfg = payload["report"]["config"]
        if self.campaigns.fixed_context(cfg) != self.campaigns.fixed_context():
            raise ValueError("This run uses different fixed data or execution settings; it cannot be repeated silently with current defaults")
        fields = set(self.jobs.integer_limits) | set(self.jobs.float_limits) | {"epsilons", "poison_ratios"}
        settings = self.freeze({**{k: cfg[k] for k in fields}, "name": cfg.get("dashboard_name", "Repeated experiment")})
        return settings, {"kind": "report", "run": run, "report_sha256": expected}

    def replay_settings(self, run="", job_id="", expected=""):
        if bool(run) == bool(job_id):
            raise ValueError("Choose one saved report or process to repeat")
        if run:
            settings, origin = self.from_report(run, expected)
        else:
            if not isinstance(job_id, str) or not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", job_id):
                raise ValueError("Choose a saved process identity")
            directory = self.jobs.directory / job_id
            job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
            cfg = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            if cfg != job["settings"] or self.campaigns.fixed_context(cfg) != self.campaigns.fixed_context():
                raise ValueError("The original process configuration changed or uses different fixed settings")
            fields = set(self.jobs.integer_limits) | set(self.jobs.float_limits) | {"epsilons", "poison_ratios"}
            settings = self.freeze({**{k: cfg[k] for k in fields}, "name": job["name"]})
            origin = {"kind": "job", "job": job_id}
        context = self.campaigns.fixed_context()
        return {"settings": settings, "fixed_context": context, "source": origin, "config_sha256": self.digest(settings, context)}

    def replay(self, values):
        if set(values)-{"run", "job_id", "report_sha256", "config_sha256", "name"}:
            raise ValueError("Unsupported rerun setting")
        snapshot = self.replay_settings(values.get("run", ""), values.get("job_id", ""), values.get("report_sha256", ""))
        if snapshot["config_sha256"] != values.get("config_sha256"):
            raise ValueError("The saved configuration changed; review it again before rerunning")
        name = values.get("name", "Repeat · "+snapshot["settings"]["name"][:70])
        return self.jobs.launch({**snapshot["settings"], "name": name}, origin={**snapshot["source"], "config_sha256": snapshot["config_sha256"]})

    def save(self, values):
        if set(values)-{"id", "revision", "name", "notes", "settings", "run", "report_sha256", "archived"}:
            raise ValueError("Unknown preset field")
        name, notes, archived = values.get("name"), values.get("notes", ""), values.get("archived", False)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or not isinstance(notes, str) or len(notes)>3000 or type(archived) is not bool:
            raise ValueError("Enter a name, notes up to 3,000 characters, and an archive choice")
        if "run" in values:
            if "settings" in values:
                raise ValueError("Choose current settings or a verified report, not both")
            settings, source = self.from_report(values["run"], values.get("report_sha256"))
        else:
            if not isinstance(values.get("settings"), dict):
                raise ValueError("Save actual validated experiment settings")
            settings, source = self.freeze(values["settings"]), {"kind": "settings"}
        context = self.campaigns.fixed_context()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            preset_id = values.get("id")
            old = db.execute("SELECT snapshot FROM presets WHERE id=?", (preset_id,)).fetchone() if preset_id else None
            if preset_id and (old is None or type(values.get("revision")) is not int or json.loads(old[0])["revision"] != values["revision"]):
                raise ValueError("This preset changed or no longer exists. Reload before saving; the draft has not replaced it.")
            if old and "run" not in values:
                previous = json.loads(old[0])
                if previous["settings"] == settings:
                    source, context = previous["source"], previous["fixed_context"]
                else:
                    source = {"kind": "preset-edit", "id": preset_id, "revision": previous["revision"], "previous_config_sha256": previous["config_sha256"]}
            now = datetime.now(timezone.utc).isoformat()
            record = {"id": preset_id or uuid.uuid4().hex, "revision": json.loads(old[0])["revision"]+1 if old else 1, "name": name.strip(), "notes": notes, "archived": archived, "settings": settings, "fixed_context": context, "source": source, "config_sha256": self.digest(settings, context), "created": json.loads(old[0])["created"] if old else now, "updated": now}
            body = json.dumps(record, allow_nan=False)
            db.execute("INSERT INTO presets VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,name=excluded.name,notes=excluded.notes,archived=excluded.archived,snapshot=excluded.snapshot", (record["id"], record["revision"], record["name"], notes, int(archived), body))
            db.execute("INSERT INTO preset_history VALUES (?,?,?)", (record["id"], record["revision"], body))
        return record

    def listing(self, archived=False):
        with self.connection() as db:
            rows = db.execute("SELECT snapshot FROM presets"+("" if archived else " WHERE archived=0")+" ORDER BY name,id").fetchall()
        return [json.loads(row[0]) for row in rows]

    def get(self, preset_id, revision=None):
        with self.connection() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT snapshot FROM presets WHERE id=?", (preset_id,)).fetchone() if revision is None else db.execute("SELECT snapshot FROM preset_history WHERE id=? AND revision=?", (preset_id, revision)).fetchone()
            if row is None:
                raise ValueError("Preset revision not found")
            preset = json.loads(row[0])
            history = [json.loads(r[0]) for r in db.execute("SELECT snapshot FROM preset_history WHERE id=? ORDER BY revision DESC", (preset_id,)).fetchall()]
        if preset["config_sha256"] != self.digest(preset["settings"], preset["fixed_context"]):
            raise ValueError("The saved preset configuration changed outside its revision history")
        return {"preset": preset, "history": history, "compatible": preset["fixed_context"] == self.campaigns.fixed_context()}

    def run(self, values):
        if set(values)-{"id", "revision", "name"} or type(values.get("revision")) is not int:
            raise ValueError("Choose an exact preset revision")
        result = self.get(values.get("id"), values["revision"])
        preset = result["preset"]
        if not result["compatible"]:
            raise ValueError("Fixed data or execution settings changed; review this preset before using it")
        return self.jobs.launch({**preset["settings"], "name": values.get("name", preset["name"])}, origin={"kind": "preset", "id": preset["id"], "revision": preset["revision"], "config_sha256": preset["config_sha256"]})
