"""Avaliação nos Testes A e B, global e por gerador/família. python -m tccdet.evaluate --split fixed --seed 0"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from . import config
from .dataset import loader
from .metrics import basic_metrics, bootstrap_ci
from .model import build_model

KEYS = ["acc", "precision", "recall", "f1", "fpr", "auc", "ap"]


@torch.no_grad()
def predict(m, csv, cfg, dev):
    dl = loader(csv, cfg["train"]["batch_size"], False, cfg["train"]["num_workers"],
                size=cfg["preprocess"]["size"])
    m.eval(); s = []
    for x, _ in dl:
        s.append(torch.sigmoid(m(x.to(dev)).squeeze(1)).cpu().numpy())
    df = pd.read_csv(csv)
    df["score"] = np.concatenate(s)
    return df


def summarize(df, thr, n_boot, seed):
    r = basic_metrics(df.label, df.score, thr)
    for k in ("f1", "auc", "recall", "acc"):
        r[f"{k}_ci95"] = bootstrap_ci(df.label, df.score, k, thr, n_boot, seed)
    return r


def per_group(df, col, thr, seed):
    """Cada grupo sintético pareado com a mesma quantidade de reais do mesmo conjunto."""
    rng = np.random.default_rng(seed)
    reals = df[df.label == 0]
    rows = []
    for g, syn in df[df.label == 1].groupby(col):
        rr = reals.iloc[rng.permutation(len(reals))[: len(syn)]]
        rows.append({col: g, **basic_metrics(pd.concat([syn, rr]).label, pd.concat([syn, rr]).score, thr)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    thr, nb = cfg["train"]["threshold"], cfg["eval"]["bootstrap"]
    sp = Path(cfg["paths"]["splits"]) / a.split
    out = Path(cfg["paths"]["results"]) / a.split / f"seed{a.seed}"
    m = build_model(False).to(dev)
    m.load_state_dict(torch.load(out / "model.pt", map_location=dev))

    res, per = {}, []
    for name in ("test_a", "test_b"):
        df = predict(m, str(sp / f"{name}.csv"), cfg, dev)
        df.to_csv(out / f"scores_{name}.csv", index=False)
        res[name] = summarize(df, thr, nb, a.seed)
        for col in ("source", "family"):
            for r in per_group(df, col, thr, a.seed):
                per.append({"test": name, "group_by": col, "group": r.pop(col), **r})
    res["delta_test_a_minus_b"] = {  # em pontos percentuais; AUC em pontos de AUC x100
        "f1_pp": 100 * (res["test_a"]["f1"] - res["test_b"]["f1"]),
        "acc_pp": 100 * (res["test_a"]["acc"] - res["test_b"]["acc"]),
        "auc_pp": 100 * (res["test_a"]["auc"] - res["test_b"]["auc"]),
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(per).to_csv(out / "per_group.csv", index=False)
    print(json.dumps(res["delta_test_a_minus_b"], indent=2))
    print(pd.DataFrame(per)[["test", "group_by", "group", "recall", "f1", "auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
