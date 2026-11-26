"""Read-only verification of the actual runs created by browser acceptance checks."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from ml_adv_sec.dashboard import Reports
from ml_adv_sec.dashboard_views import ResultViews


def get(route, **query):
    with urlopen("http://127.0.0.1:8873"+route+("?"+urlencode(query) if query else ""), timeout=120) as response:
        return json.load(response)


def main():
    jobs = get("/api/jobs")
    first = next(j for j in jobs if j["name"] == "Browser preset full pipeline")
    repeated = next(j for j in jobs if j["name"] == "Browser exact repeat")
    reports = Reports(Path.cwd(), first["settings"])
    views = ResultViews(reports)
    checked = []
    for job in [first, repeated]:
        assert job["status"] == "complete" and job["exit_code"] == 0
        assert all("Running "+stage in job["log"] for stage in ["prepare", "train", "attack-evasion", "attack-poison", "defend", "evaluate"])
        payload = reports.read(job["report_id"])
        assert payload["evidence"]["report_sha256"] == job["report_sha256"]
        differences = {}
        for model in ["clean", "tree", "pgd", "trades", "squeezed"]:
            result = views.records(job["report_id"], model=model, recheck=True)
            assert result["maximum_score_difference"] == 0
            differences[model] = result["maximum_score_difference"]
        checked.append({"job": job["id"], "report": job["report_id"], "sha256": job["report_sha256"], "origin": job["origin"], "fresh_model_maximum_differences": differences})
    assert first["settings"]["root"] != repeated["settings"]["root"]
    assert repeated["origin"]["job"] == first["id"]
    aroot, broot = Path(first["settings"]["root"]), Path(repeated["settings"]["root"])
    with np.load(aroot/"prepared/splits.npz") as a, np.load(broot/"prepared/splits.npz") as b:
        np.testing.assert_array_equal(a["id_test"], b["id_test"])
    with np.load(aroot/"artifacts/evaluation-scores.npz") as a, np.load(broot/"artifacts/evaluation-scores.npz") as b:
        for model in ["clean", "tree", "pgd", "trades", "squeezed"]:
            np.testing.assert_array_equal(a[model], b[model])
    preset_id = first["origin"]["id"]
    export = get("/api/preset-export", id=preset_id)
    revisions = [get("/api/preset", id=preset_id, revision=i)["preset"] for i in range(1, 5)]
    assert [p["settings"]["epochs"] for p in revisions] == [1, 2, 2, 2]
    assert [p["archived"] for p in revisions] == [False, False, True, False]
    assert export["preset"]["revision"] == 4
    campaigns = get("/api/campaigns")
    retry = next(c for c in campaigns if c["name"] == "Browser corrected retry")
    original = next(c for c in campaigns if c["id"] == retry["retry_of"])
    assert original["status"] == "failed" and retry["status"] == "complete"
    assert original["entries"][0]["values"]["poison_ratios"] == [0.49]
    assert retry["entries"][0]["values"]["poison_ratios"] == [0.01]
    assert retry["entries"][0]["origin"]["job"] == original["entries"][0]["job_id"]
    comparison = get("/api/campaign-results", id=retry["id"])
    assert comparison["verified_runs"] == 1 and not comparison["evidence_errors"]
    assert len(comparison["rows"]) == 5 and len(comparison["attack_rows"]) == 12
    evidence = {"checked": datetime.now(timezone.utc).isoformat(), "runs": checked, "test_ids_and_all_five_score_vectors_equal": True, "preset": preset_id, "revisions": [p["revision"] for p in revisions], "original_campaign": original["id"], "retry_campaign": retry["id"], "retry_utility_rows": 5, "retry_attack_rows": 12, "browser": {"save_apply_revision_archive_restore": "observed", "preset_run_and_exact_repeat_completion": "observed", "corrected_retry_and_verified_comparison": "observed", "export_click": "observed; export response independently checked; browser download capture unavailable", "console_errors_or_warnings_at_check": [], "narrow_layout": "DOM dimensions checked; screenshot capture unavailable"}}
    Path("reports/dashboard_reuse_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
