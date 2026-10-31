"""Média e desvio padrão entre sementes. python -m tccdet.aggregate --split fixed"""
import argparse, json
from pathlib import Path
import pandas as pd
from . import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="fixed")
    a = ap.parse_args()
    cfg = config.load(a.config)
    base = Path(cfg["paths"]["results"]) / a.split
    rows = []
    for f in sorted(base.glob("seed*/metrics.json")):
        r = json.loads(f.read_text())
        for t in ("test_a", "test_b"):
            rows.append({"seed": f.parent.name, "test": t, **{k: r[t][k] for k in ("acc", "f1", "auc", "recall", "fpr")}})
        rows.append({"seed": f.parent.name, "test": "delta_pp", **r["delta_test_a_minus_b"]})
    df = pd.DataFrame(rows)
    summ = df.drop(columns="seed").groupby("test").agg(["mean", "std"]).round(4)
    summ.to_csv(base / "summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
