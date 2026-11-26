"""Independently check live saved-attack replay against real pinned models."""
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from ml_adv_sec.dashboard import Reports
from ml_adv_sec.common import seed
from ml_adv_sec.data import Domain
from ml_adv_sec.defenses import squeeze
from ml_adv_sec.models import restore, score


def get(route, **query):
    with urlopen("http://127.0.0.1:8873"+route+("?"+urlencode(query) if query else ""), timeout=120) as response:
        body = response.read()
        return json.loads(body) if "json" in response.headers["Content-Type"] else body


def main():
    reports = Reports(Path.cwd(), get("/api/settings"))
    available = [r for r in get("/api/reports") if not r.get("error")]
    selected = available[:2]
    evidence = {"checked": datetime.now(timezone.utc).isoformat(), "runs": [], "model_attack_checks": 0, "record_measurements_checked": 0}
    for run in selected:
        payload = reports.read(run["id"])
        cfg, report = payload["report"]["config"], payload["report"]
        seed(cfg["seed"], cfg["threads"])
        root = Path(cfg["root"])
        schema = json.loads((root/"prepared/schema.json").read_text())
        predictors = {}
        for row in report["clean"]+report["poisoning"]:
            name = row["model"]
            target = "clean" if name == "squeezed" else name
            predictors[name] = restore(cfg, target, payload["evidence"]["checkpoints"][target])[0]
        attacks = sorted({r["attack"] for r in report["transferability"]})
        run_checks = []
        for attack in attacks:
            query = {"run": run["id"], "report_sha256": payload["evidence"]["report_sha256"], "attack": attack, "other_attack": attacks[-1], "model": "tree", "size": 10}
            actual = get("/api/attack-replay", **query)
            with np.load(root/"artifacts"/(attack+".npz")) as archive:
                x, adv, ids = [archive[k] for k in ["original", "adversarial", "record_ids"]]
            for summary in actual["models"]:
                model = summary["model"]
                values = [squeeze(v, Domain(cfg), cfg["quantum"]) for v in [x, adv]] if model == "squeezed" else [x, adv]
                before, after = [score(predictors[model], v) for v in values]
                threshold = next(r["threshold"] for r in report["clean"]+report["poisoning"] if r["model"] == model)
                eligible, escaped = before >= threshold, (before >= threshold) & (after < threshold)
                assert summary["initially_caught"] == int(eligible.sum())
                assert summary["escaped"] == int(escaped.sum())
                assert summary["escape_rate"] == (float(escaped.sum()/eligible.sum()) if eligible.any() else None)
                assert summary["caught_after"] == int((after >= threshold).sum())
                assert summary["mean_score_change"] == float(np.mean(after-before))
                evidence["model_attack_checks"] += 1
            rows = []
            for page in range(1, actual["pages"]+1):
                rows.extend(get("/api/attack-replay", **query, page=page)["rows"])
            assert len(rows) == actual["total"] == len({r["source_row"] for r in rows})
            exported_body = get("/api/attack-replay.csv", **query)
            reader = csv.DictReader(io.StringIO(exported_body.decode("utf-8-sig")))
            assert len(reader.fieldnames) == len(set(reader.fieldnames))
            exported = list(reader)
            assert {int(r["source_row"]) for r in exported} == {r["source_row"] for r in rows}
            model_csv = list(csv.DictReader(io.StringIO(get("/api/attack-replay.csv", **query, scope="models").decode("utf-8-sig"))))
            assert {r["model"] for r in model_csv} == set(predictors)
            if rows:
                detail = get("/api/attack-replay-record", **query, source_row=rows[0]["source_row"])
                i = np.flatnonzero(ids == rows[0]["source_row"]-1)[0]
                for measurement in detail["changes"]:
                    column = measurement["column"]
                    assert measurement["first_original"] == float(x[i, column])*schema["scale"][column]+schema["center"][column]
                    assert measurement["first_changed"] == float(adv[i, column])*schema["scale"][column]+schema["center"][column]
                    evidence["record_measurements_checked"] += 1
                changes = list(csv.DictReader(io.StringIO(get("/api/attack-replay.csv", **query, source_row=rows[0]["source_row"], scope="changes").decode("utf-8-sig"))))
                assert len(changes) == 57
            run_checks.append({"attack": attack, "models": len(actual["models"]), "paired": actual["counts"]["all"], "pages": actual["pages"]})
        evidence["runs"].append({"id": run["id"], "report_sha256": payload["evidence"]["report_sha256"], "checks": run_checks})
    if len(selected) == 2:
        attack = sorted({r["attack"] for r in reports.read(selected[0]["id"])["report"]["transferability"]})[0]
        result = get("/api/attack-replay", run=selected[0]["id"], attack=attack, other=selected[1]["id"], model="squeezed")
        ids = []
        for run in selected:
            cfg = reports.read(run["id"])["report"]["config"]
            with np.load(Path(cfg["root"])/"artifacts"/(attack+".npz")) as archive:
                ids.append(archive["record_ids"])
        assert result["counts"]["all"] == len(np.intersect1d(*ids))
        evidence["cross_run_shared_records"] = result["counts"]["all"]
    Path("reports/dashboard_replay_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
