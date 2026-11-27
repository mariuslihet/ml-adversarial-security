"""Compare browser downloads byte for byte with the saved study exports."""
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

downloads = Path.home()/"Downloads"
evidence = json.loads((downloads/"adaptive-test-evidence.json").read_text(encoding="utf-8"))
study = evidence["id"]
output = Path("reports/adaptive-browser")
output.mkdir(exist_ok=True)
verified = []
files = [("adaptive-test-evidence.json", f"/api/adaptive-test.json?id={study}")]
files += [(f"adaptive-{scope}.csv", f"/api/adaptive-test.csv?id={study}&scope={scope}") for scope in ["summary", "records", "queries"]]
changes = downloads/"adaptive-changes (1).csv"
rows = list(csv.DictReader(io.StringIO(changes.read_text(encoding="utf-8-sig"))))
assert len(rows) == 57
assert rows[0]["input_scope"] == "API query 2"
files.append((changes.name, f"/api/adaptive-test.csv?id={study}&scope=changes&index=0&query=2"))
for name, route in files:
    contents = (downloads/name).read_bytes()
    with urlopen("http://127.0.0.1:8873"+route) as response:
        assert contents == response.read(), name
    (output/name).write_bytes(contents)
    verified.append({"file": name, "bytes": len(contents), "route": route})
report_path = Path("reports/dashboard_adaptive_checks.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
report["browser_downloads"] = verified
report["browser_checks"] = ["Three target modes completed from the form", "Ensemble defaults survived target switching", "Study history refresh, open, and load settings", "Per-query input selector and 57 exact measurements", "Five browser downloads matched saved exports byte for byte", "Paired one-field edits rejected", "Oversized query plans rejected", "Empty attacker selection rejected", "Pagination and outcome/class filters", "Desktop screenshot inspected", "No browser console errors observed"]
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(verified, indent=2))
