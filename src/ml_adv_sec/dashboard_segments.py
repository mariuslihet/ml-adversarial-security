"""Error slices and partition comparisons from authenticated source records."""
import json

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from .dashboard_research import ResearchViews


class SegmentViews(ResearchViews):
    def raw_partitions(self, payload, root):
        raw = np.loadtxt(self.source_path(payload["report"]), delimiter=",")
        with np.load(root / "prepared/splits.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for partition in ("train", "val", "test"):
            ids = data["id_"+partition]
            if not np.array_equal(raw[ids, -1].astype(int), data["y_"+partition]):
                raise ValueError("Original source labels do not match the prepared partitions")
        schema = json.loads((root / "prepared/schema.json").read_text(encoding="utf-8"))
        return raw, data, schema

    @staticmethod
    def cuts(values, bands):
        candidates = np.unique(np.quantile(values, np.linspace(0, 1, bands+1)[1:-1]))
        return candidates[(candidates > values.min()) & (candidates < values.max())]

    @staticmethod
    def bounds(cuts, index):
        return {"lower": float(cuts[index-1]) if index else None, "upper": float(cuts[index]) if index < len(cuts) else None}

    def slices(self, run_id, model="clean", feature=0, bands=4, band=-1, outcome="all", page=1, size=25, expected=None, export=None):
        if not 0 <= feature < 57 or bands not in {4, 5, 10} or outcome not in {"all", "missed", "false-alarm", "correct"} or page < 1 or size not in {10, 25, 50, 100} or export not in {None, "summary", "records"}:
            raise ValueError("Choose a listed feature, band count, record filter, and page size")
        payload, root = self.context(run_id, expected)
        row, prepared, saved = self.inputs(payload, root, model)
        raw, data, schema = self.raw_partitions(payload, root)
        scores = self.predictor(payload, model)(prepared["x_test"])
        if not np.allclose(scores, saved, atol=1e-6, rtol=1e-5):
            raise ValueError("Fresh model scores differ from the report")
        train_values = raw[data["id_train"], feature]
        test_values = raw[data["id_test"], feature]
        cuts = self.cuts(train_values, bands)
        if band < -1 or band > len(cuts):
            raise ValueError("Choose a band from the measured training boundaries")
        assignments = np.searchsorted(cuts, test_values, side="right")
        training_assignments = np.searchsorted(cuts, train_values, side="right")
        y = data["y_test"]
        flagged = scores >= row["threshold"]
        metrics = []
        for index in range(len(cuts)+1):
            mask = assignments == index
            counts = self.decisions(y[mask], scores[mask], row["threshold"])
            errors = counts["missed"]+counts["false_alarms"]
            metrics.append({"band": index, **self.bounds(cuts, index), "training_records": int((training_assignments == index).sum()), **counts,
                            "error_rate": errors/int(mask.sum()) if mask.any() else None,
                            "spam_share": float(y[mask].mean()) if mask.any() else None,
                            "mean_score": float(scores[mask].mean()) if mask.any() else None})
        mask = assignments == band if band >= 0 else np.ones(len(y), dtype=bool)
        if outcome == "missed":
            mask &= (y == 1) & ~flagged
        elif outcome == "false-alarm":
            mask &= (y == 0) & flagged
        elif outcome == "correct":
            mask &= flagged == y
        selected = np.flatnonzero(mask)
        selected = selected[np.argsort(data["id_test"][selected], kind="stable")]
        pages = max(1, (len(selected)+size-1)//size)
        page = min(page, pages)
        shown = selected if export == "records" else selected[(page-1)*size:page*size]
        rows = [{"source_row": int(data["id_test"][i])+1, "value": float(test_values[i]), "band": int(assignments[i]), "actual": int(y[i]), "score": float(scores[i]), "flagged": bool(flagged[i]), "correct": bool(flagged[i] == y[i])} for i in shown]
        fingerprint = payload["evidence"]["report_sha256"]
        if export == "summary":
            keys = ["band", "lower", "upper", "training_records", "tested", "caught", "missed", "false_alarms", "clear", "recall", "fpr", "precision", "error_rate", "spam_share", "mean_score"]
            return self.csv_bytes(keys+["feature", "model", "threshold", "report_sha256"], [[r[k] for k in keys]+[schema["names"][feature], model, row["threshold"], fingerprint] for r in metrics])
        if export == "records":
            keys = ["source_row", "value", "band", "actual", "score", "flagged", "correct"]
            return self.csv_bytes(keys+["feature", "model", "threshold", "report_sha256"], [[r[k] for k in keys]+[schema["names"][feature], model, row["threshold"], fingerprint] for r in rows])
        return {"model": model, "feature": schema["names"][feature], "feature_index": feature, "features": schema["names"], "threshold": row["threshold"], "requested_bands": bands, "cuts": cuts.tolist(), "bands": metrics, "overall": self.decisions(y, scores, row["threshold"]), "rows": rows, "total": len(selected), "page": page, "pages": pages, "size": size, "band": band, "outcome": outcome, "maximum_score_difference": float(np.max(np.abs(scores-saved))), "report_sha256": fingerprint}

    @staticmethod
    def describe(values):
        return {"count": len(values), "minimum": float(values.min()), "maximum": float(values.max()), "mean": float(values.mean()), "median": float(np.median(values)), "std": 0.0 if values.min() == values.max() else float(values.std()), "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)), "zero_share": float((values == 0).mean())}

    def shift(self, run_id, partition="test", label="all", feature=0, outside=False, page=1, size=25, expected=None, export=None):
        if partition not in {"test", "val"} or label not in {"all", "spam", "regular"} or not 0 <= feature < 57 or page < 1 or size not in {10, 25, 50, 100} or export not in {None, "summary", "records"}:
            raise ValueError("Choose a listed partition, label, feature, and page size")
        payload, root = self.context(run_id, expected)
        raw, data, schema = self.raw_partitions(payload, root)
        train_ids, target_ids = data["id_train"], data["id_"+partition]
        if label != "all":
            value = int(label == "spam")
            train_ids = train_ids[raw[train_ids, -1] == value]
            target_ids = target_ids[raw[target_ids, -1] == value]
        if not len(train_ids) or not len(target_ids):
            raise ValueError("Both partitions need records with the selected label")
        reference, target = raw[train_ids, :-1], raw[target_ids, :-1]
        rows = []
        for column, name in enumerate(schema["names"]):
            a, b = reference[:, column], target[:, column]
            before, after = self.describe(a), self.describe(b)
            scale = before["q75"]-before["q25"]
            scaling = "training IQR"
            if scale == 0:
                scale, scaling = before["std"], "training standard deviation"
            distance = float(wasserstein_distance(a, b))
            out = (b < before["minimum"]) | (b > before["maximum"])
            rows.append({"feature": name, "column": column, "reference": before, "comparison": after,
                         "cdf_distance": float(ks_2samp(a, b, method="asymp").statistic), "raw_distance": distance,
                         "scaled_distance": distance/scale if scale > 0 else None, "scaling": scaling if scale > 0 else None,
                         "outside_count": int(out.sum()), "outside_share": float(out.mean()), "zero_share_change": after["zero_share"]-before["zero_share"]})
        selected_feature = rows[feature]
        cuts = np.unique(np.r_[reference[:, feature].min(), self.cuts(reference[:, feature], 10), np.nextafter(reference[:, feature].max(), np.inf)])
        a_bands = np.searchsorted(cuts, reference[:, feature], side="right")
        b_bands = np.searchsorted(cuts, target[:, feature], side="right")
        histogram = [{"band": index, **self.bounds(cuts, index), "reference_count": int((a_bands == index).sum()), "comparison_count": int((b_bands == index).sum()), "reference_share": float((a_bands == index).mean()), "comparison_share": float((b_bands == index).mean())} for index in range(len(cuts)+1)]
        lower, upper = selected_feature["reference"]["minimum"], selected_feature["reference"]["maximum"]
        out = (target[:, feature] < lower) | (target[:, feature] > upper)
        selected = np.flatnonzero(out if outside else np.ones(len(target), dtype=bool))
        selected = selected[np.argsort(target_ids[selected], kind="stable")]
        pages = max(1, (len(selected)+size-1)//size)
        page = min(page, pages)
        shown = selected if export == "records" else selected[(page-1)*size:page*size]
        records = [{"source_row": int(target_ids[i])+1, "actual": int(raw[target_ids[i], -1]), "value": float(target[i, feature]), "outside": bool(out[i])} for i in shown]
        fingerprint = payload["evidence"]["report_sha256"]
        if export == "summary":
            columns = ["feature", "cdf_distance", "raw_distance", "scaled_distance", "scaling", "outside_count", "outside_share", "zero_share_change"]
            summary_keys = list(rows[0]["reference"])
            return self.csv_bytes(columns+["training_"+k for k in summary_keys]+["comparison_"+k for k in summary_keys]+["partition", "label", "report_sha256"], [[r[k] for k in columns]+[r["reference"][k] for k in summary_keys]+[r["comparison"][k] for k in summary_keys]+[partition, label, fingerprint] for r in rows])
        if export == "records":
            columns = ["source_row", "actual", "value", "outside"]
            return self.csv_bytes(columns+["selected_feature", "partition", "label_filter", "report_sha256"]+schema["names"], [[r[k] for k in columns]+[schema["names"][feature], partition, label, fingerprint]+target[i].tolist() for r, i in zip(records, shown)])
        return {"rows": rows, "selected": selected_feature, "features": schema["names"], "feature_index": feature, "partition": partition, "label": label, "training_records": len(train_ids), "comparison_records": len(target_ids), "histogram": histogram, "records": records, "outside_only": bool(outside), "total": len(selected), "page": page, "pages": pages, "size": size, "report_sha256": fingerprint}
