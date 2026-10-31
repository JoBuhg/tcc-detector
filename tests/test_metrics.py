import numpy as np
from tccdet.metrics import basic_metrics, bootstrap_ci


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.8, 0.9])
    m = basic_metrics(y, s)
    assert m["acc"] == m["f1"] == m["auc"] == 1.0 and m["fpr"] == 0.0


def test_confusion_counts():
    y = np.array([0, 0, 1, 1]); s = np.array([0.6, 0.1, 0.9, 0.2])
    m = basic_metrics(y, s)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400); s = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    lo, hi = bootstrap_ci(y, s, "auc", n_boot=200)
    assert lo <= basic_metrics(y, s)["auc"] <= hi
