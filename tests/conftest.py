from pathlib import Path

import pytest

from ml_adv_sec.common import config
from ml_adv_sec.data import prepare, load, Domain
from ml_adv_sec.models import train_mlp

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def experiment(tmp_path_factory):
    cfg = config(REPOSITORY / "configs/quick.yaml")
    root = tmp_path_factory.mktemp("real-spambase")
    cfg["root"] = str(root)
    cfg["report_dir"] = str(root / "reports")
    cfg["data_path"] = str(REPOSITORY / "data/spambase.data")
    prepare(cfg)
    data = load(cfg)
    model = train_mlp(cfg, data["x_train"], data["y_train"])
    return cfg, data, Domain(cfg), model


@pytest.fixture(scope="session")
def completed_experiment(experiment):
    from ml_adv_sec.eval.workflow import train, evasion, poisoning, defend, evaluate
    cfg, _, _, _ = experiment
    train(cfg)
    evasion(cfg)
    poisoning(cfg)
    defend(cfg)
    evaluate(cfg)
    return experiment
