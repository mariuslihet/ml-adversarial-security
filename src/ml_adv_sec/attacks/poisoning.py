import numpy as np

from ..models import art_classifier


def trigger(x, domain, epsilon=0.1, indices=None):
    mutable = np.flatnonzero(~domain.immutable & ~domain.integer)
    if len(mutable) < 2:
        raise ValueError("A trigger requires two mutable continuous features")
    candidate = x.copy()
    chosen = mutable[:2] if indices is None else indices
    if epsilon <= 0:
        raise ValueError("Trigger budget must be positive")
    candidate[:, chosen] += epsilon / 2
    return domain.project(candidate, x, epsilon)


def poison(model, x, y, domain, cfg, kind, ratio):
    if not 0 < ratio < 0.5:
        raise ValueError("Poison ratio must be in (0, .5)")
    rng = np.random.default_rng(cfg["seed"])
    count = max(1, int(len(x) * ratio))
    eligible = np.flatnonzero(y == 1)
    if count > len(eligible):
        raise ValueError("Not enough malicious records for requested ratio")
    ids = rng.choice(eligible, count, replace=False)
    px, py = x.copy(), y.copy()
    if kind == "label-flip":
        py[ids] = 0
    elif kind == "backdoor":
        px[ids] = trigger(x[ids], domain, cfg["epsilon"])
        py[ids] = 0
    elif kind == "collision":
        from art.attacks.poisoning import FeatureCollisionAttack
        low = domain.low.astype(np.float32)
        high = domain.high.astype(np.float32)
        # ART requires a strict interval even for constant training columns.
        # Its numerical interval is widened by one float; Domain restores constants.
        high = np.maximum(high, np.nextafter(low, np.float32(np.inf)))
        classifier = art_classifier(model, (low, high))
        benign = x[y == 0]
        # Collision is targeted toward an actual benign representation; labels stay malicious.
        layer = classifier.layer_names[-2]
        attack = FeatureCollisionAttack(classifier, target=benign[:1], feature_layer=layer, max_iter=cfg["collision_steps"], learning_rate=0.01, similarity_coeff=1.0, verbose=False)
        generated, _ = attack.poison(x[ids], np.eye(2, dtype=np.float32)[y[ids]])
        px[ids] = domain.project(generated, x[ids], cfg["epsilon"])
    else:
        raise ValueError(f"Unknown poisoning method: {kind}")
    return px, py, ids
