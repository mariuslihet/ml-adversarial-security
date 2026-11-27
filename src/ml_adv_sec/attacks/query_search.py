"""Query-limited searches over quantized, sparse, explicitly constrained inputs."""
import numpy as np
import hashlib
from pathlib import Path

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class EditRules:
    def __init__(self, domain, original, raw, epsilon, norm, max_fields, space, policy):
        self.domain, self.original = domain, np.asarray(original, dtype=np.float32)
        self.raw = np.asarray(raw, dtype=float)
        self.epsilon, self.norm, self.max_fields, self.policy = epsilon, norm, max_fields, policy
        self.quantum = np.r_[np.full(48, .01), np.full(6, .001), np.ones(3)]
        permitted = np.arange(48) if space == "words" else np.arange(48, 54) if space == "characters" else np.arange(54)
        self.allowed = np.array([int(i) for i in permitted if not domain.immutable[i]], dtype=int)
        self.low = np.maximum(np.minimum(domain.low*domain.scale+domain.center, self.raw), self.raw-epsilon*domain.scale)
        self.high = np.minimum(np.maximum(domain.high*domain.scale+domain.center, self.raw), self.raw+epsilon*domain.scale)

    def scaled(self, raw):
        result = self.original.copy()
        changed = np.abs(raw-self.raw) > 1e-9
        result[changed] = ((raw[changed]-self.domain.center[changed])/self.domain.scale[changed]).astype(np.float32)
        return result

    def changed(self, raw):
        return np.flatnonzero(np.abs(raw-self.raw) > 1e-9)

    def valid(self, raw):
        if raw.shape != (57,) or not np.isfinite(raw).all():
            return False
        changed = self.changed(raw)
        if len(changed) > self.max_fields or not set(changed.tolist()) <= set(self.allowed.tolist()):
            return False
        if not np.allclose(raw[:48], np.round(raw[:48], 2), atol=1e-9, rtol=0) or not np.allclose(raw[48:54], np.round(raw[48:54], 3), atol=1e-9, rtol=0):
            return False
        if np.any(raw[:54] < 0) or raw[:48].sum() > 100+1e-8 or raw[48:54].sum() > 100+1e-8:
            return False
        if self.policy == "paired" and (abs(raw[:48].sum()-self.raw[:48].sum()) > 1e-8 or abs(raw[48:54].sum()-self.raw[48:54].sum()) > 1e-8):
            return False
        delta = self.scaled(raw).astype(float)-self.original
        distance = np.max(np.abs(delta)) if self.norm == np.inf else np.linalg.norm(delta)
        return bool(distance <= self.epsilon+1e-7 and self.domain.valid(self.scaled(raw)[None], self.original[None], self.epsilon, self.norm)[0])

    def distance(self, raw):
        delta = self.scaled(raw).astype(float)-self.original
        return {"changed_fields": len(self.changed(raw)), "linf": float(np.max(np.abs(delta))), "l2": float(np.linalg.norm(delta))}

    def cost(self, raw):
        d = self.distance(raw)
        return d["changed_fields"], d["l2"], d["linf"]

    def mutate(self, current, rng):
        candidate = current.copy()
        if not len(self.allowed):
            return candidate
        active = np.intersect1d(self.changed(current), self.allowed)
        pool = active if len(active) >= (2 if self.policy == "paired" else 1) and rng.random() < .65 else self.allowed
        if self.policy == "paired":
            groups = [pool[pool < 48], pool[(pool >= 48) & (pool < 54)]]
            groups = [g for g in groups if len(g) >= 2]
            if not groups:
                return candidate
            group = groups[int(rng.integers(len(groups)))]
            donors = group[current[group]-self.low[group] >= self.quantum[group]-1e-9]
            if not len(donors):
                return candidate
            donor = int(rng.choice(donors))
            receivers = group[(group != donor) & (self.high[group]-current[group] >= self.quantum[group]-1e-9)]
            if not len(receivers):
                return candidate
            receiver = int(rng.choice(receivers))
            units = int(np.floor((min(current[donor]-self.low[donor], self.high[receiver]-current[receiver])+1e-9)/self.quantum[donor]))
            if units < 1:
                return candidate
            amount = max(1, int(np.ceil(units*rng.uniform(.15, 1))))*self.quantum[donor]
            candidate[donor] -= amount;candidate[receiver] += amount
        else:
            column = int(rng.choice(pool))
            first = int(np.ceil((self.low[column]-1e-9)/self.quantum[column]))
            last = int(np.floor((self.high[column]+1e-9)/self.quantum[column]))
            if first > last:
                return candidate
            candidate[column] = int(rng.integers(first, last+1))*self.quantum[column]
        candidate[:48] = np.round(candidate[:48], 2);candidate[48:54] = np.round(candidate[48:54], 3)
        candidate[np.abs(candidate-self.raw) < 1e-9] = self.raw[np.abs(candidate-self.raw) < 1e-9]
        return candidate

    def shrink(self, current, fraction):
        candidate = self.raw+(current-self.raw)*fraction
        candidate[:48] = np.round(candidate[:48], 2);candidate[48:54] = np.round(candidate[48:54], 3)
        if self.policy == "paired":
            for group in [np.arange(48), np.arange(48, 54)]:
                changed = np.intersect1d(self.changed(current), group)
                if len(changed):
                    residual = self.raw[group].sum()-candidate[group].sum()
                    candidate[changed[-1]] += residual
                    digits = 2 if changed[-1] < 48 else 3
                    candidate[changed[-1]] = np.round(candidate[changed[-1]], digits)
        candidate[np.abs(candidate-self.raw) < 1e-9] = self.raw[np.abs(candidate-self.raw) < 1e-9]
        return candidate


def query_attack(oracle, rules, actual, access, budget, seed):
    """Decision-only selection never sees probabilities; every ask is a real query."""
    if access not in {"score", "decision"} or type(budget) is not int or budget < 1 or actual not in {0, 1}:
        raise ValueError("Use a supported feedback mode, label, and positive query budget")
    rng = np.random.default_rng(seed)
    desired = 1-int(actual)
    trace, seen = [], set()
    invalid = duplicates = proposals = 0
    first_error = None

    def ask(raw):
        response = oracle(raw)
        fingerprint = hashlib.sha256(np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
        if response.get("query") != len(trace)+1 or response.get("decision") not in {0, 1} or response.get("input_sha256") != fingerprint or (access == "decision" and "score" in response):
            raise ValueError("The prediction endpoint violated its feedback contract")
        decision = int(response["decision"])
        feedback = decision if access == "decision" else float(response["score"])
        trace.append({"query": len(trace)+1, "feedback": feedback, "error_found": decision == desired, "request_ms": float(response["request_ms"]), "input_sha256": fingerprint,
                      "edits": [[int(i), float(raw[i])] for i in rules.changed(raw)], **rules.distance(raw)})
        seen.add(rules.scaled(raw).tobytes())
        return feedback, decision

    initial, decision = ask(rules.raw)
    if decision == desired:
        return {"raw": rules.raw.copy(), "trace": trace, "first_error_query": None, "queries": 1, "reason": "already-wrong", "invalid_proposals": 0, "duplicate_proposals": 0, "proposals": 0}
    best = current = rules.raw.copy()
    objective = initial if actual else -initial
    success = None
    while len(trace) < budget and proposals < budget*30:
        proposals += 1
        if success is not None and rng.random() < .75:
            candidate = rules.shrink(success, float(rng.uniform(.15, .95)))
        else:
            reference = rules.raw if proposals % 11 == 0 else best if access == "score" else current
            candidate = rules.mutate(reference, rng)
        key = rules.scaled(candidate).tobytes()
        if key in seen:
            duplicates += 1;continue
        if not rules.valid(candidate):
            invalid += 1;continue
        feedback, decision = ask(candidate)
        if access == "score":
            candidate_objective = feedback if actual else -feedback
            if candidate_objective < objective:
                objective, best = candidate_objective, candidate.copy()
        else:
            current = candidate.copy()
        if decision == desired:
            if first_error is None:
                first_error = len(trace)
            if success is None or rules.cost(candidate) < rules.cost(success):
                success = candidate.copy()
    result = success if success is not None else best if access == "score" else current
    if not rules.valid(result):
        raise ValueError("The search produced an input outside its declared rules")
    return {"raw": result, "trace": trace, "first_error_query": first_error, "queries": len(trace),
            "reason": "query-budget" if len(trace) == budget else "proposal-limit", "invalid_proposals": invalid, "duplicate_proposals": duplicates, "proposals": proposals}
