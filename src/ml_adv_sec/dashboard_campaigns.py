"""Durable sequential campaigns that execute the real local experiment process."""
import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class Campaigns:
    axes = {"epsilon", "steps", "epochs", "learning_rate", "hidden", "quantum", "none"}

    def __init__(self, jobs, start_worker=True):
        self.jobs, self.reports = jobs, jobs.reports
        self.directory = self.reports.workspace / "runs/dashboard-campaigns"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.campaigns = {}
        for path in self.directory.glob("*/campaign.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not re.fullmatch(r"[0-9a-f]{32}", data["id"]) or path.parent.name != data["id"]:
                raise ValueError("A saved campaign has an invalid identity")
            if data["status"] in {"running", "pausing"}:
                data["status"] = "interrupted"
                data["message"] = "Server tracking restarted. Resume only the remaining queued runs; interrupted runs are not retried."
            self.campaigns[data["id"]] = data
            self.save(data)
        self.tick()
        self.worker = threading.Thread(target=self.loop, daemon=True) if start_worker else None
        if self.worker:
            self.worker.start()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def save(self, campaign):
        directory = self.directory / campaign["id"]
        directory.mkdir(exist_ok=True)
        pending = directory / "campaign.pending"
        pending.write_text(json.dumps(campaign, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(directory / "campaign.json")

    def fixed_context(self):
        editable = set(self.jobs.integer_limits) | set(self.jobs.float_limits) | {"name", "epsilons", "poison_ratios", "root", "report_dir", "dashboard_name"}
        context = {k: v for k, v in self.reports.cfg.items() if k not in editable}
        context["data_path"] = str((self.reports.workspace / self.reports.cfg["data_path"]).resolve())
        return context

    def create(self, values):
        if set(values)-{"name", "base", "axis", "values", "seeds", "fail_fast"}:
            raise ValueError("Unknown campaign setting")
        name, base = values.get("name"), values.get("base", {})
        axis, variants, seeds = values.get("axis", "none"), values.get("values", []), values.get("seeds")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60 or not isinstance(base, dict) or not isinstance(axis, str) or axis not in self.axes:
            raise ValueError("Enter a campaign name, base run settings, and a listed setting to vary")
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 12 or any(type(seed) is not int or not 0 <= seed <= 2147483647 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("Use one to twelve distinct whole-number seeds")
        if axis == "none":
            variants = [None]
        elif not isinstance(variants, list) or not 1 <= len(variants) <= 6 or any(type(v) not in {int, float} or not np.isfinite(v) for v in variants) or len(set(variants)) != len(variants):
            raise ValueError("Use one to six distinct finite values for the selected setting")
        if len(variants)*len(seeds) > 12:
            raise ValueError("A campaign can contain at most twelve real runs")
        fail_fast = values.get("fail_fast", True)
        if type(fail_fast) is not bool:
            raise ValueError("Stop-on-failure must be true or false")
        entries = []
        for variant in variants:
            for seed in seeds:
                run = {**base, "seed": seed, "name": f"{name.strip()[:50]} · {len(entries)+1}/{len(variants)*len(seeds)}"}
                if axis != "none":
                    run[axis] = variant
                resolved, run_name = self.jobs.validate(run)
                frozen = {key: resolved[key] for key in set(self.jobs.integer_limits) | set(self.jobs.float_limits) | {"epsilons", "poison_ratios"}}
                frozen["name"] = run_name
                entries.append({"index": len(entries), "seed": seed, "variant": variant, "values": frozen, "status": "queued", "job_id": None})
        with self.lock:
            if any(c["status"] in {"running", "pausing"} for c in self.campaigns.values()):
                raise ValueError("Pause or finish the active campaign before starting another")
            campaign = {"id": uuid.uuid4().hex, "name": name.strip(), "axis": axis, "variants": variants, "seeds": seeds, "fixed_context": self.fixed_context(), "fail_fast": fail_fast, "status": "running", "created": self.now(), "updated": self.now(), "message": "Runs execute sequentially using the real local pipeline.", "entries": entries}
            self.campaigns[campaign["id"]] = campaign
            self.save(campaign)
            return json.loads(json.dumps(campaign))

    def tick(self):
        with self.lock:
            owned = {}
            # Read persisted status and in-memory ownership under the same lock
            # used by process completion, avoiding false interruption at exit.
            with self.jobs.lock:
                tracked = self.jobs.active
                for path in self.jobs.directory.glob("*/job.json"):
                    job = json.loads(path.read_text(encoding="utf-8"))
                    owner = job.get("campaign")
                    if owner:
                        key = (owner["id"], owner["index"])
                        if key in owned:
                            raise ValueError("Multiple process records claim the same campaign entry")
                        owned[key] = job
            for campaign in self.campaigns.values():
                previous = json.dumps(campaign, sort_keys=True)
                for entry in campaign["entries"]:
                    job = owned.get((campaign["id"], entry["index"]))
                    if not job:
                        if entry["status"] == "starting":
                            entry["status"] = "interrupted"
                        continue
                    entry["job_id"] = job["id"]
                    entry["status"] = job["status"] if job["status"] != "running" or tracked == job["id"] else "interrupted"
                    entry["exit_code"] = job.get("exit_code")
                    entry["finished"] = job.get("finished")
                    if job.get("report_sha256"):
                        path = (Path(job["settings"]["report_dir"])/"benchmark_results.json").resolve()
                        entry["report_id"] = hashlib.sha256(str(path).encode()).hexdigest()[:16]
                        entry["report_sha256"] = job["report_sha256"]
                active = any(e["status"] == "running" for e in campaign["entries"])
                queued = [e for e in campaign["entries"] if e["status"] == "queued"]
                if campaign["status"] == "pausing" and not active:
                    campaign["status"] = "paused"
                if campaign["status"] == "running":
                    failures = [e for e in campaign["entries"] if e["status"] in {"failed", "cancelled", "interrupted"} and not e.get("failure_acknowledged")]
                    if failures and campaign["fail_fast"]:
                        campaign["status"] = "failed"
                        campaign["message"] = "A real run failed or lost tracking. Successful results remain available; remaining runs are queued."
                    elif not queued and not active:
                        campaign["status"] = "complete" if all(e["status"] == "complete" for e in campaign["entries"]) else "finished-with-errors"
                    elif queued and not active and not self.jobs.active and not self.stop.is_set():
                        if campaign["fixed_context"] != self.fixed_context():
                            campaign["status"] = "failed"
                            campaign["message"] = "Fixed data or run settings changed. Create a new campaign rather than changing the queued experiments."
                            self.save(campaign)
                            continue
                        entry = queued[0]
                        entry["status"] = "starting"
                        self.save(campaign)
                        try:
                            job = self.jobs.launch(entry["values"], campaign={"id": campaign["id"], "index": entry["index"]})
                            entry.update(status="running", job_id=job["id"])
                        except RuntimeError:
                            entry["status"] = "queued"
                        except (OSError, ValueError) as exc:
                            entry.update(status="failed", error=str(exc))
                if json.dumps(campaign, sort_keys=True) != previous:
                    campaign["updated"] = self.now()
                    self.save(campaign)

    def action(self, campaign_id, action):
        with self.lock:
            campaign = self.campaigns.get(campaign_id)
            if campaign is None:
                raise ValueError("Campaign not found")
            if action == "pause":
                if campaign["status"] != "running":
                    raise ValueError("Only a running campaign can be paused")
                campaign["status"] = "pausing" if any(e["status"] == "running" for e in campaign["entries"]) else "paused"
                campaign["message"] = "The current run may finish; no queued run will start while paused."
            elif action == "resume":
                if campaign["status"] not in {"paused", "interrupted", "failed"} or not any(e["status"] == "queued" for e in campaign["entries"]):
                    raise ValueError("This campaign has no resumable queued runs")
                if any(c["id"] != campaign_id and c["status"] in {"running", "pausing"} for c in self.campaigns.values()):
                    raise ValueError("Another campaign is active")
                for entry in campaign["entries"]:
                    if entry["status"] in {"failed", "cancelled", "interrupted"}:
                        entry["failure_acknowledged"] = True
                campaign["status"] = "running"
                campaign["message"] = "Continuing queued entries. Previously failed or interrupted runs are not retried."
            elif action == "cancel":
                if campaign["status"] in {"complete", "cancelled", "finished-with-errors"}:
                    raise ValueError("This campaign has already finished")
                campaign["status"] = "cancelled"
                for entry in campaign["entries"]:
                    if entry["status"] == "queued":
                        entry["status"] = "cancelled"
                    if entry["status"] == "running" and entry["job_id"] == self.jobs.active:
                        try:
                            self.jobs.cancel(entry["job_id"])
                        except ValueError:
                            pass
                campaign["message"] = "Queued entries were cancelled; completed evidence is retained."
            else:
                raise ValueError("Choose pause, resume, or cancel")
            campaign["updated"] = self.now(); self.save(campaign)
            return json.loads(json.dumps(campaign))

    def listing(self):
        with self.lock:
            return json.loads(json.dumps(sorted(self.campaigns.values(), key=lambda c: c["created"], reverse=True)))

    def results(self, campaign_id):
        with self.lock:
            if campaign_id not in self.campaigns:
                raise ValueError("Campaign not found")
            campaign = json.loads(json.dumps(self.campaigns[campaign_id]))
        rows, attack_rows, errors = [], [], []
        for entry in campaign["entries"]:
            if entry["status"] != "complete":
                continue
            try:
                payload = self.reports.read(entry["report_id"])
                if payload["evidence"]["report_sha256"] != entry["report_sha256"]:
                    raise ValueError("The result changed after its process completed")
                report = payload["report"]
                for model in report["clean"]:
                    rows.append({"entry": entry["index"], "seed": entry["seed"], "variant": entry["variant"], "model": model["model"], "spam_caught": 1-model["fnr"], "false_alarms": model["fpr"], "f1": model["f1"], "brier": model["brier"], "tested": report["dataset"]["split_sizes"]["test"], "report_id": entry["report_id"], "report_sha256": entry["report_sha256"]})
                for attack in report["transferability"]:
                    attack_rows.append({"entry": entry["index"], "seed": entry["seed"], "variant": entry["variant"], "target": attack["target"], "attack": attack["attack"], "epsilon": attack["epsilon"], "eligible": attack["eligible"], "asr": attack["asr"], "adversarial_recall": attack["adversarial_recall"], "attack_n": attack["attack_n"], "report_id": entry["report_id"], "report_sha256": entry["report_sha256"]})
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"entry": entry["index"], "error": str(exc)})
        aggregates = []
        groups = {(r["variant"], r["model"]) for r in rows}
        for variant, model in sorted(groups, key=lambda key: (str(key[0]), key[1])):
            selected = [r for r in rows if r["variant"] == variant and r["model"] == model]
            aggregates.append({"variant": variant, "model": model, "runs": len(selected), "metrics": {key: {"mean": float(np.mean([r[key] for r in selected])), "std": float(np.std([r[key] for r in selected], ddof=1)) if len(selected)>1 else None, "minimum": min(r[key] for r in selected), "maximum": max(r[key] for r in selected)} for key in ["spam_caught", "false_alarms", "f1", "brier"]}})
        attack_aggregates = []
        for variant, target, attack in sorted({(r["variant"], r["target"], r["attack"]) for r in attack_rows}, key=lambda key: (str(key[0]), key[1], key[2])):
            selected = [r for r in attack_rows if (r["variant"], r["target"], r["attack"]) == (variant, target, attack)]
            rates = [r["asr"] for r in selected if r["asr"] is not None]
            attack_aggregates.append({"variant": variant, "target": target, "attack": attack, "runs": len(selected), "defined_runs": len(rates), "limits": list(dict.fromkeys(r["epsilon"] for r in selected)), "mean_escape_rate": float(np.mean(rates)) if rates else None, "std_escape_rate": float(np.std(rates, ddof=1)) if len(rates)>1 else None, "mean_spam_caught": float(np.mean([r["adversarial_recall"] for r in selected]))})
        return {"campaign": campaign, "rows": rows, "attack_rows": attack_rows, "aggregates": aggregates, "attack_aggregates": attack_aggregates, "evidence_errors": errors, "verified_runs": len({r["entry"] for r in rows})}

    def entry(self, campaign_id, index):
        with self.lock:
            campaign = self.campaigns.get(campaign_id)
            if campaign is None or not 0 <= index < len(campaign["entries"]):
                raise ValueError("Campaign entry not found")
            entry = json.loads(json.dumps(campaign["entries"][index]))
        if not entry["job_id"]:
            return {"entry": entry, "job": None, "log": ""}
        directory = self.jobs.directory / entry["job_id"]
        job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        if job.get("campaign") != {"id": campaign_id, "index": index}:
            raise ValueError("The saved process does not belong to this campaign entry")
        log = directory / "output.log"
        return {"entry": entry, "job": job, "log": log.read_text(encoding="utf-8", errors="replace")[-24000:] if log.exists() else ""}

    def loop(self):
        while not self.stop.wait(0.5):
            try:
                self.tick()
            except (OSError, ValueError, KeyError) as exc:
                with self.lock:
                    for campaign in self.campaigns.values():
                        if campaign["status"] in {"running", "pausing"}:
                            campaign["status"] = "failed"; campaign["message"] = str(exc); self.save(campaign)

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=2)
        with self.lock:
            for campaign in self.campaigns.values():
                if campaign["status"] in {"running", "pausing"}:
                    campaign["status"] = "interrupted"; self.save(campaign)
