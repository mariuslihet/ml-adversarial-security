"""Local experiment controls and fingerprint-verified model results."""

import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs


class Reports:
    def __init__(self, workspace, cfg):
        self.workspace = Path(workspace).resolve()
        self.cfg = cfg

    def available(self):
        owned = {}
        for job_path in (self.workspace / "runs/dashboard-jobs").glob("*/job.json"):
            try:
                job = json.loads(job_path.read_text(encoding="utf-8"))
                owned[str(Path(job["settings"]["root"]).resolve())] = job["status"] == "complete" and job["exit_code"] == 0
            except (OSError, ValueError, KeyError):
                continue
        candidates = [self.workspace / "reports/benchmark_results.json"]
        candidates += sorted((self.workspace / "runs").glob("*/reports/benchmark_results.json"))
        candidates.append(Path(self.cfg["root"]).resolve() / "reports/benchmark_results.json")
        if self.cfg.get("report_dir"):
            candidates.append(Path(self.cfg["report_dir"]).resolve() / "benchmark_results.json")
        found = {}
        for path in candidates:
            path = path.resolve()
            if owned.get(str(path.parent.parent.resolve())) is False:
                continue
            if path.is_file() and path not in found.values():
                found[hashlib.sha256(str(path).encode()).hexdigest()[:16]] = path
        return found

    def read(self, run_id):
        from .common import sha256
        path = self.available().get(run_id)
        if path is None:
            raise FileNotFoundError("Report not found")
        manifest_path = path.with_name("benchmark_results.manifest.json")
        if not manifest_path.exists():
            raise ValueError("This older run needs a fresh evaluation before its results can be verified.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Verify exactly the bytes we parse, including during concurrent evaluation.
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != manifest["report_sha256"]:
            raise ValueError("The result file changed. Run the evaluation again before using it.")
        for filename, expected in manifest["files"].items():
            if not Path(filename).is_file() or sha256(filename) != expected:
                raise ValueError("The saved data, model, or attack evidence changed. Run the evaluation again.")
        report = json.loads(body)
        if not report.get("counts") or not manifest.get("checkpoints"):
            raise ValueError("The run is missing its measured counts or model versions.")
        return {"report": report, "updated": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "evidence": {"report_sha256": manifest["report_sha256"], "checkpoints": manifest["checkpoints"]}}

    def list(self):
        rows = []
        for key in self.available():
            try:
                payload = self.read(key)
                report = payload["report"]
                limit = report["config"].get("max_rows", 0)
                name = report["config"].get("dashboard_name") or ("Full dataset" if not limit else f"{limit:,}-record check")
                rows.append({"id": key, "name": name, "updated": payload["updated"], "records": report["dataset"]["split_sizes"]["test"]})
            except (ValueError, KeyError, OSError) as exc:
                rows.append({"id": key, "name": self.available()[key].parent.parent.name + " · needs recheck", "error": str(exc)})
        return rows

    def inspect(self, run_id, attack, sample, target, expected_report=None):
        import numpy as np
        from .common import sha256
        from .models import restore, score
        payload = self.read(run_id)
        if expected_report and expected_report != payload["evidence"]["report_sha256"]:
            raise ValueError("This run was updated after you opened it. Reload the results before inspecting it.")
        report = payload["report"]
        if not any(r["attack"] == attack and r["target"] == target for r in report["transferability"]):
            raise ValueError("Choose an attack from this run")
        root = Path(report["config"]["root"])
        root = (self.workspace / root).resolve() if not root.is_absolute() else root.resolve()
        if not root.is_relative_to(self.workspace) or Path(attack).name != attack:
            raise ValueError("This run is outside the workspace")
        path = root / "artifacts" / (attack + ".npz")
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256(path) != metadata["sha256"]:
            raise ValueError("The saved attack file changed. Run the attack again.")
        with np.load(path, allow_pickle=False) as arrays:
            originals, attacks = arrays["original"], arrays["adversarial"]
            record_ids = arrays["record_ids"]
        if not 0 <= sample < len(originals):
            raise ValueError("Email number is out of range")
        schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
        clean, clean_threshold, clean_meta = restore(report["config"], "clean", payload["evidence"]["checkpoints"]["clean"])
        if metadata["details"]["source_checkpoint"] != clean_meta["weights_sha256"]:
            raise ValueError("The model changed after these attacks were saved")
        model, threshold = (clean, clean_threshold) if target == "clean" else restore(report["config"], target, payload["evidence"]["checkpoints"][target])[:2]
        # Match the evaluation's batch shapes: tiny rounding differences from a
        # two-record batch can otherwise disagree at a decision boundary.
        fresh_before, fresh_after = score(model, originals), score(model, attacks)
        with np.load(root / "artifacts/evaluation-scores.npz", allow_pickle=False) as recorded:
            expected_before = recorded[attack + "-" + target + "-before"]
            expected_after = recorded[attack + "-" + target + "-after"]
        if not np.allclose(fresh_before, expected_before, atol=1e-6, rtol=1e-5) or not np.allclose(fresh_after, expected_after, atol=1e-6, rtol=1e-5):
            raise ValueError("The current model scores differ from the saved evaluation. Re-evaluate this run.")
        p = [fresh_before[sample], fresh_after[sample]]
        before = originals[sample] * np.asarray(schema["scale"]) + np.asarray(schema["center"])
        after = attacks[sample] * np.asarray(schema["scale"]) + np.asarray(schema["center"])
        features = [{"name": name, "before": float(before[i]), "after": float(after[i]), "change": float(after[i] - before[i]), "scaled_change": float(attacks[sample, i] - originals[sample, i]), "locked": i in schema["immutable"], "changed": bool(abs(attacks[sample, i] - originals[sample, i]) > 1e-6)} for i, name in enumerate(schema["names"])]
        with np.load(root / "prepared/splits.npz", allow_pickle=False) as splits:
            mask = splits["y_test"] == 1
            if not np.array_equal(originals, splits["x_test"][mask][:len(originals)]) or not np.array_equal(record_ids, splits["id_test"][mask][:len(originals)]):
                raise ValueError("The attacked records do not match this run's test data.")
            raw_id = int(record_ids[sample])
        return {"index": sample, "count": len(originals), "source_row": raw_id, "before": float(p[0]), "after": float(p[1]), "threshold": threshold, "caught_before": bool(p[0] >= threshold), "caught_after": bool(p[1] >= threshold), "features": features, "attack": metadata["details"]}


class Jobs:
    """One real child process at a time, with logs and restart-visible history."""

    integer_limits = {"epochs": (1, 200), "steps": (1, 300), "attack_samples": (1, 512), "batch_size": (8, 2048), "hidden": (8, 512), "restarts": (0, 10), "cw_steps": (1, 500), "hsj_steps": (1, 100), "collision_steps": (1, 300), "seed": (0, 2147483647), "max_rows": (0, 4601)}
    float_limits = {"epsilon": (0.0001, 2), "fpr": (0, 0.25), "learning_rate": (0.000001, 0.1), "quantum": (0.0001, 2)}

    def __init__(self, reports):
        self.reports = reports
        self.directory = reports.workspace / "runs/dashboard-jobs"
        self.lock = threading.Lock()
        self.process = None
        self.active = None

    @staticmethod
    def _save(directory, job):
        from .common import write_json
        temporary = directory / "job.pending"
        write_json(temporary, job)
        temporary.replace(directory / "job.json")

    def list(self):
        jobs = []
        for path in sorted(self.directory.glob("*/job.json"), reverse=True):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] == "running" and job["id"] != self.active:
                    job["status"] = "interrupted"
                log = path.with_name("output.log")
                text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                job["log"] = text[-24000:]
                stages = [line.removeprefix("Running ").strip() for line in text.splitlines() if line.startswith("Running ")]
                job["stage"] = stages[-1] if stages else None
                result_path = Path(job["settings"]["report_dir"]) / "benchmark_results.json"
                if job["status"] == "complete" and job["exit_code"] == 0 and result_path.exists():
                    job["report_id"] = hashlib.sha256(str(result_path.resolve()).encode()).hexdigest()[:16]
                    from .common import sha256
                    job["report_updated"] = not job.get("report_sha256") or sha256(result_path) != job["report_sha256"]
                jobs.append(job)
            except (ValueError, OSError, KeyError):
                continue
        return jobs[:30]

    def launch(self, values):
        import math
        import yaml
        if not isinstance(values, dict):
            raise ValueError("Run settings must be an object")
        allowed = set(self.integer_limits) | set(self.float_limits) | {"name", "poison_ratios", "epsilons"}
        if set(values) - allowed:
            raise ValueError("Unknown run setting")
        cfg = dict(self.reports.cfg)
        for key, (low, high) in self.integer_limits.items():
            value = values.get(key, cfg[key])
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{key} must be a whole number between {low} and {high}")
            cfg[key] = value
        if 0 < cfg["max_rows"] < 128:
            raise ValueError("Use at least 128 records, or 0 for the full dataset")
        for key, (low, high) in self.float_limits.items():
            value = values.get(key, cfg[key])
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            cfg[key] = value
        for key, low, high in [("poison_ratios", 0.001, 0.49), ("epsilons", 0.0001, 2)]:
            value = values.get(key, cfg[key])
            if not isinstance(value, list) or not 1 <= len(value) <= 12 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high for v in value):
                raise ValueError(f"{key} needs 1–12 numbers between {low} and {high}")
            cfg[key] = sorted(set(value))
        name = values.get("name", "Local experiment")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Use a run name with 1–80 characters")
        with self.lock:
            if self.active:
                raise RuntimeError("A run is already in progress")
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            directory = self.directory / run_id
            directory.mkdir(parents=True)
            cfg["root"] = str(self.reports.workspace / "runs" / ("experiment-" + run_id))
            cfg["report_dir"] = str(Path(cfg["root"]) / "reports")
            cfg["data_path"] = str((self.reports.workspace / cfg["data_path"]).resolve())
            cfg["dashboard_name"] = name.strip()
            config_path = directory / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            job = {"id": run_id, "name": name.strip(), "status": "running", "started": datetime.now(timezone.utc).isoformat(), "finished": None, "exit_code": None, "settings": cfg}
            self._save(directory, job)
            try:
                with (directory / "output.log").open("wb") as output:
                    self.process = subprocess.Popen([sys.executable, "-u", "-m", "ml_adv_sec", "--config", str(config_path), "verify-all"], cwd=self.reports.workspace, stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError as exc:
                job["status"] = "failed"
                job["finished"] = datetime.now(timezone.utc).isoformat()
                (directory / "output.log").write_text(str(exc), encoding="utf-8")
                self._save(directory, job)
                raise
            self.active = run_id
            threading.Thread(target=self._finish, args=(self.process, directory, job), daemon=True).start()
            return job

    def _finish(self, process, directory, job):
        code = process.wait()
        with self.lock:
            job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
            job["exit_code"] = code
            job["status"] = "complete" if code == 0 else ("cancelled" if job.get("cancel_requested") else "failed")
            job["finished"] = datetime.now(timezone.utc).isoformat()
            result = Path(job["settings"]["report_dir"]) / "benchmark_results.json"
            if code == 0 and result.exists():
                from .common import sha256
                job["report_sha256"] = sha256(result)
            self._save(directory, job)
            self.active = None
            self.process = None

    def cancel(self, run_id):
        with self.lock:
            if run_id != self.active or self.process is None:
                raise ValueError("This run is not active")
            if self.process.poll() is not None:
                raise ValueError("This run has already finished. Reload its status.")
            directory = self.directory / run_id
            job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
            job["cancel_requested"] = True
            self._save(directory, job)
            self.process.terminate()

    def close(self):
        active = self.active
        if active:
            try:
                self.cancel(active)
            except ValueError:
                pass


def handler_for(reports, jobs=None):
    from .dashboard_research import ResearchViews
    assets = Path(__file__).with_name("web")
    views = ResearchViews(reports)
    jobs = jobs or Jobs(reports)
    token = secrets.token_urlsafe(32)
    static = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/features.js": ("features.js", "text/javascript"), "/analysis.js": ("analysis.js", "text/javascript"), "/research.js": ("research.js", "text/javascript"), "/style.css": ("style.css", "text/css")}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                return self.send_error(403, "Use the dashboard's local address")
            route = urlsplit(self.path).path
            query = parse_qs(urlsplit(self.path).query)
            def q(name, default=""):
                return query.get(name, [default])[0]
            try:
                if route in {"/api/importance", "/api/importance.csv"}:
                    exporting = route.endswith(".csv")
                    result = views.importance(q("run"), q("model", "clean"), q("partition", "test"), int(q("repeats", "3")), int(q("seed", "42")), q("report_sha256"), exporting)
                    return self.respond(result if exporting else json.dumps(result, allow_nan=False).encode(), "text/csv" if exporting else "application/json", filename="input-importance.csv" if exporting else None)
                if route in {"/api/attack-overlap", "/api/attack-overlap.csv"}:
                    exporting = route.endswith(".csv")
                    result = views.attack_overlap(q("run"), q("target", "clean"), q("category", "all"), int(q("page", "1")), int(q("size", "25")), q("report_sha256"), exporting)
                    return self.respond(result if exporting else json.dumps(result, allow_nan=False).encode(), "text/csv" if exporting else "application/json", filename="attack-overlap.csv" if exporting else None)
                if route in {"/api/decision-lab", "/api/decision-lab.csv"}:
                    exporting = route.endswith(".csv")
                    result = views.decision_lab(q("run"), q("model", "clean"), q("mode", "saved"), float(q("threshold", "0.5")), float(q("target_fpr", "0.01")), q("report_sha256"), exporting)
                    return self.respond(result if exporting else json.dumps(result, allow_nan=False).encode(), "text/csv" if exporting else "application/json", filename="decision-preview.csv" if exporting else None)
                if route in {"/api/disagreements", "/api/disagreements.csv"}:
                    exporting = route.endswith(".csv")
                    result = views.disagreements(q("left"), q("right"), q("left_model", "clean"), q("right_model", "tree"), q("category", "disagree"), int(q("page", "1")), int(q("size", "25")), q("left_sha"), q("right_sha"), exporting)
                    return self.respond(result if exporting else json.dumps(result, allow_nan=False).encode(), "text/csv" if exporting else "application/json", filename="filter-disagreements.csv" if exporting else None)
                if route == "/api/record-detail":
                    result = views.record_detail(q("run"), int(q("source_row", "0")), q("model", "clean"), q("report_sha256"))
                    return self.respond(json.dumps(result, allow_nan=False).encode(), "application/json")
                if route in {"/api/records", "/api/records.csv"}:
                    exporting = route.endswith(".csv")
                    result = views.records(q("run"), q("model", "clean"), q("label", "all"), q("outcome", "all"), q("search"), q("sort", "score-desc"), int(q("page", "1")), int(q("size", "25")), q("report_sha256"), q("recheck") == "1", exporting, q("features") == "1")
                    return self.respond(result if exporting else json.dumps(result, allow_nan=False).encode(), "text/csv" if exporting else "application/json", filename="filter-records.csv" if exporting else None)
                if route in {"/api/compare", "/api/compare.csv"}:
                    exporting = route.endswith(".csv")
                    result = views.compare(q("left"), q("right"), q("left_sha"), q("right_sha"), exporting)
                    return self.respond(result if exporting else json.dumps(result, allow_nan=False).encode(), "text/csv" if exporting else "application/json", filename="run-comparison.csv" if exporting else None)
                if route in {"/api/training-records", "/api/training-record"}:
                    result = views.training(q("run"), q("model"), q("mode", "selected"), int(q("page", "1")), int(q("size", "25")), q("report_sha256"), int(q("position", "-1")) if route.endswith("/training-record") else None)
                    return self.respond(json.dumps(result, allow_nan=False).encode(), "application/json")
                if route == "/api/reports":
                    return self.respond(json.dumps(reports.list()).encode(), "application/json")
                if route == "/api/settings":
                    return self.respond(json.dumps(reports.cfg).encode(), "application/json")
                if route == "/api/jobs":
                    return self.respond(json.dumps(jobs.list()).encode(), "application/json")
                if route == "/api/inspect":
                    query = parse_qs(urlsplit(self.path).query)
                    payload = reports.inspect(query.get("run", [""])[0], query.get("attack", [""])[0], int(query.get("sample", ["0"])[0]), query.get("target", ["clean"])[0], query.get("report_sha256", [None])[0])
                    return self.respond(json.dumps(payload, allow_nan=False).encode(), "application/json")
                if route.startswith("/api/report/"):
                    payload = reports.read(route.removeprefix("/api/report/"))
                    return self.respond(json.dumps(payload, allow_nan=False).encode(), "application/json")
                if route in static:
                    name, mime = static[route]
                    body = (assets / name).read_bytes()
                    if route == "/":
                        body = body.replace(b"DASHBOARD_TOKEN", token.encode())
                    return self.respond(body, mime)
                self.send_error(404, "Page not found")
            except FileNotFoundError:
                self.respond(json.dumps({"error": "The saved report or its evidence could not be found."}).encode(), "application/json", 404)
            except (ValueError, KeyError, OSError) as exc:
                self.respond(json.dumps({"error": str(exc)}).encode(), "application/json", 503)

        def do_POST(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if host != f"127.0.0.1:{self.server.server_port}" or (origin and origin != "http://" + host) or not secrets.compare_digest(self.headers.get("X-Dashboard-Token", ""), token):
                return self.send_error(403, "Open the dashboard on this computer to start a run")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384 or self.headers.get("Content-Type") != "application/json":
                    return self.send_error(400, "Send run settings as JSON")
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict):
                    raise ValueError("Send run settings as an object")
                route = urlsplit(self.path).path
                if route == "/api/jobs":
                    result = jobs.launch(values)
                elif route == "/api/jobs/cancel":
                    jobs.cancel(values.get("id"))
                    result = {"stopping": True}
                else:
                    return self.send_error(404, "Page not found")
                self.respond(json.dumps(result).encode(), "application/json")
            except (ValueError, KeyError, RuntimeError) as exc:
                self.respond(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
            except OSError:
                self.respond(json.dumps({"error": "The local process could not start. Its saved output contains the error."}).encode(), "application/json", 503)

        def respond(self, body, mime, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    return Handler


def serve(cfg, port=8873):
    if not 0 < port < 65536:
        raise ValueError("Choose a port between 1 and 65535")
    reports = Reports(Path.cwd(), cfg)
    jobs = Jobs(reports)
    # Windows SO_REUSEADDR can silently share a port with another app.
    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False
    server = LocalServer(("127.0.0.1", port), handler_for(reports, jobs))
    print(f"Dashboard: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard stopped.", flush=True)
    finally:
        jobs.close()
        server.server_close()
