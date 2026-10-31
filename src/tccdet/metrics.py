import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def basic_metrics(y, s, thr=0.5):
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {
        "acc": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "auc": roc_auc_score(y, s) if both else float("nan"),
        "ap": average_precision_score(y, s) if both else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "n": int(len(y)),
    }


def bootstrap_ci(y, s, key, thr=0.5, n_boot=1000, seed=0):
    """IC 95% por bootstrap sobre as imagens de teste."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(basic_metrics(y[i], s[i], thr)[key])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
