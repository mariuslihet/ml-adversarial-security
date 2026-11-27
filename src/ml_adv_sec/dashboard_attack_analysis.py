"""Fresh defender evaluations of verified adaptive attack evidence."""
import hashlib
import itertools
import json
import threading
from datetime import datetime, timezone

import numpy as np

from .attacks.query_search import EditRules
from .data import Domain


def portable_numbers(value):
    """Whole-valued floats survive a browser JSON round trip as integers."""
    if isinstance(value, dict):
        return {key: portable_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_numbers(item) for item in value]
    return int(value) if isinstance(value, float) and value.is_integer() else value


class AttackAnalysis:
    def __init__(self, adaptive):
        self.adaptive = adaptive
        self.views = adaptive.views
        self.lock = threading.Lock()

    def run(self, p):
        if not isinstance(p, dict) or set(p)-{"tool", "studies", "members", "index"}:
            raise ValueError("Choose the attack analysis settings shown in the form")
        tool, ids, members = p.get("tool"), p.get("studies"), p.get("members", [])
        if not isinstance(tool, str) or tool not in {"transfer", "reduction", "coverage"}:
            raise ValueError("Choose attack transfer, edit reduction, or combined coverage")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 8 or any(not isinstance(s, str) for s in ids) or len(set(ids)) != len(ids):
            raise ValueError("Choose one to eight different saved studies")
        if tool != "coverage" and len(ids) != 1:
            raise ValueError("This tool uses exactly one saved study")
        if not isinstance(members, list) or not (0 if tool == "reduction" else 1) <= len(members) <= 8 or any(not isinstance(m, str) for m in members) or len(set(members)) != len(members):
            raise ValueError("Choose one to eight different target filters")
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another attack analysis is running; let it finish first")
        try:
            return self._run(p, tool, ids, members)
        finally:
            self.lock.release()

    def _run(self, p, tool, ids, members):
        studies = [self.adaptive.read(s, traces=True) for s in ids]
        source = studies[0]["settings"]
        if any((s["settings"]["run"], s["settings"]["report_sha256"]) != (source["run"], source["report_sha256"]) for s in studies):
            raise ValueError("Combine studies from the same verified source run")
        payload, root, data, _ = self.views.lab_data(source["run"], source["report_sha256"])
        domain = Domain(payload["report"]["config"])
        positions = {int(s)+1: i for i, s in enumerate(data["id_test"])}
        records, predictions, candidates = [], 0, []
        for study in studies:
            for row in study["rows"]:
                index = row["index"]
                original = np.array(study["examples"]["original"][index])
                changed = np.array(study["examples"]["changed"][index])
                position = positions[row["source_row"]]
                if int(data["y_test"][position]) != row["actual"]:
                    raise ValueError("The saved source label differs from this report")
                rules = EditRules(domain, data["x_test"][position], original, row["epsilon"], np.inf if study["settings"]["norm"] == "linf" else 2, study["settings"]["max_fields"], study["settings"]["space"], study["settings"]["policy"])
                if not rules.valid(changed):
                    raise ValueError("The saved attack violates its edit rules")
                candidates.append((study, row, rules, changed))
        if len(candidates)*max(1, len(members))*2 > 32768:
            raise ValueError("This analysis exceeds 32,768 defender predictions; choose fewer studies or filters")
        summary, skipped = [], 0
        def measure(study, row, rules, raw, predict, threshold, model, **extra):
            nonlocal predictions
            before = float(predict(rules.original[None])[0])
            after = float(predict(rules.scaled(raw)[None])[0])
            predictions += 2
            eligible = int(before >= threshold) == row["actual"]
            return {"study": study["id"], "trial": row["index"], "source_row": row["source_row"], "actual": row["actual"], "model": model, "threshold": threshold, "before": before, "after": after, "eligible": eligible, "caused_error": bool(eligible and int(after >= threshold) != row["actual"]), "wrong_after": bool(int(after >= threshold) != row["actual"]), "changed_fields": len(rules.changed(raw)), "input_sha256": hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest(), "edits": [[int(i), float(raw[i])] for i in rules.changed(raw)], **extra}
        if tool == "reduction":
            index = p.get("index")
            if type(index) is not int or not 0 <= index < len(candidates):
                raise ValueError("Choose a recorded trial index")
            study, row, rules, changed = candidates[index]
            _, _, _, predict, threshold, recipe, _ = self.adaptive.target(study["settings"])
            if recipe != study["target"]:
                raise ValueError("The frozen target recipe could not be reproduced")
            target_name = recipe["parameters"]["model"] if recipe["kind"] == "filter" else recipe["kind"]
            fields = rules.changed(changed).tolist()
            if len(fields) > 12:
                raise ValueError("Edit reduction supports at most twelve changed measurements")
            for count in range(len(fields)+1):
                for subset in itertools.combinations(fields, count):
                    raw = rules.raw.copy()
                    raw[list(subset)] = changed[list(subset)]
                    if not rules.valid(raw):
                        skipped += 1
                        continue
                    records.append(measure(study, row, rules, raw, predict, threshold, target_name, retained_columns=list(subset)))
            successful = [r for r in records if r["caused_error"]]
            minimum = min((r["changed_fields"] for r in successful), default=None)
            summary = [{"model": target_name, "valid_subsets": len(records), "invalid_subsets": skipped, "successful_subsets": len(successful), "minimum_retained_fields": minimum, "equally_small_successes": sum(r["changed_fields"] == minimum for r in successful)}]
        else:
            for model in members:
                model_row, predict, _, _ = self.views.lab_model(payload, root, data, model)
                measured = [measure(s, r, rules, raw, predict, model_row["threshold"], model) for s, r, rules, raw in candidates]
                records.extend(measured)
                sources = sorted({r["source_row"] for r in measured})
                eligible_sources = {r["source_row"] for r in measured if r["eligible"]}
                failed = {r["source_row"] for r in measured if r["caused_error"]}
                per_study = {s["id"]: {r["source_row"] for r in measured if r["study"] == s["id"] and r["caused_error"]} for s in studies}
                shared = set.intersection(*[{r["source_row"] for r in measured if r["study"] == s["id"]} for s in studies])
                summary.append({"model": model, "trials": len(measured), "eligible_trials": sum(r["eligible"] for r in measured), "caused_errors": sum(r["caused_error"] for r in measured), "unique_sources": len(sources), "eligible_sources": len(eligible_sources), "sources_failing_at_least_once": len(failed), "observed_failure_rate": len(failed)/len(eligible_sources) if eligible_sources else None, "shared_sources": len(shared), "shared_sources_failing": len(shared & failed), "sources_failed_in_every_study": len(set.intersection(*per_study.values())), "added_beyond_best_single_study": len(failed)-max(map(len, per_study.values()))})
        for study in studies:
            if self.adaptive.read(study["id"])["evidence_sha256"] != study["evidence_sha256"]:
                raise ValueError("Source evidence changed during analysis")
        result = {"tool": tool, "created": datetime.now(timezone.utc).isoformat(), "report_sha256": source["report_sha256"], "run": source["run"], "studies": [{"id": s["id"], "name": s["name"], "evidence_sha256": s["evidence_sha256"]} for s in studies], "summary": summary, "rows": records, "defender_predictions": predictions, "invalid_subsets": skipped, "notes": "Fresh defender audits of fixed saved edits, not additional attacker queries. Trial counts include repeats; unique-source rates count each record once. Reduction is exhaustive only over subsets of the selected final edits, not all possible edits. Coverage is observed on selected records, not a population robustness estimate."}
        result = portable_numbers(result)
        result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result
