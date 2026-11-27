"""Actual local prediction API with server-enforced quotas and feedback isolation."""
import http.client
import json
import secrets
import socket
import threading
import time
import hashlib
from pathlib import Path
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class OracleClient:
    def __init__(self, port, token):
        self.token, self._queries = token, 0
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        self.connection.connect()
        self.connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def ask(self, raw):
        started = time.perf_counter()
        body = json.dumps({"values": np.asarray(raw).tolist()}, allow_nan=False).encode()
        self.connection.request("POST", "/predict", body=body, headers={"Content-Type": "application/json", "X-Test-Token": self.token})
        response = self.connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise ValueError(result.get("error", "The prediction API rejected this request"))
        if result.get("query") != self._queries+1:
            raise ValueError("The prediction API query sequence changed")
        self._queries += 1
        result["request_ms"] = (time.perf_counter()-started)*1000
        return result

    @property
    def queries(self):
        return self._queries

    def close(self):
        self.connection.close()


class LocalOracle:
    def __init__(self, predict, threshold):
        self.sessions, self.lock = {}, threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.settimeout(30)

            def reply(self, code, body):
                content = json.dumps(body, allow_nan=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers();self.wfile.write(content)

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/predict" or not 0 < length <= 8192 or self.headers.get("Content-Type") != "application/json":
                        self.close_connection = True
                        return self.reply(400, {"error": "Send one bounded measurement vector to /predict"})
                    body = json.loads(self.rfile.read(length))
                    if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                        return self.reply(403, {"error": "Use the private local prediction endpoint"})
                    with owner.lock:
                        state = owner.sessions.get(self.headers.get("X-Test-Token", ""))
                    if state is None:
                        return self.reply(403, {"error": "This test session is unavailable"})
                    if not isinstance(body, dict) or set(body) != {"values"} or not isinstance(body["values"], list) or len(body["values"]) != 57 or any(type(v) not in {int, float} for v in body["values"]):
                        return self.reply(400, {"error": "Supply exactly 57 numeric measurements"})
                    raw = np.array(body["values"], dtype=float)
                    with state["lock"]:
                        if state["queries"] >= state["budget"]:
                            return self.reply(429, {"error": "The model query budget is exhausted"})
                        if not state["rules"].valid(raw):
                            return self.reply(400, {"error": "These edits exceed the declared measurement rules"})
                        state["queries"] += 1
                        probability = float(np.asarray(predict(state["rules"].scaled(raw)[None]))[0])
                        if not np.isfinite(probability) or not 0 <= probability <= 1:
                            return self.reply(503, {"error": "The model returned an invalid probability"})
                        result = {"query": state["queries"], "decision": int(probability >= threshold), "input_sha256": hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest()}
                        if state["access"] == "score":
                            result["score"] = probability
                        return self.reply(200, result)
                except (ValueError, TypeError) as exc:
                    return self.reply(400, {"error": str(exc)})

            def log_message(self, *_):
                return

        class Server(ThreadingHTTPServer):
            allow_reuse_address = False
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.worker.start()

    @contextmanager
    def session(self, rules, access, budget):
        if access not in {"score", "decision"} or type(budget) is not int or budget < 1:
            raise ValueError("Use a supported feedback mode and a positive API budget")
        token = secrets.token_hex(24)
        state = {"rules": rules, "access": access, "budget": budget, "queries": 0, "lock": threading.Lock()}
        with self.lock:
            self.sessions[token] = state
        client = OracleClient(self.server.server_port, token)
        try:
            yield client
        finally:
            client.close()
            with self.lock:
                del self.sessions[token]

    def query_count(self, token):
        with self.lock:
            state = self.sessions[token]
        with state["lock"]:
            return state["queries"]

    def close(self):
        self.server.shutdown();self.server.server_close();self.worker.join(timeout=2)
