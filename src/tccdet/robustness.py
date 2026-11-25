"""Teste de sensibilidade: AUC/F1 nos Testes A e B sob perturbações (JPEG, desfoque, redução de escala).
Se o desempenho cai muito, o modelo pode depender de artefatos frágeis (incluindo marcas d'água invisíveis).
python -m tccdet.robustness --split fixed --seed 0"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from . import config, perturb
from .dataset import ImageCSV
from .metrics import basic_metrics
from .model import build_model


@torch.no_grad()
def scores(m, df, cfg, dev, pre):
    dl = DataLoader(ImageCSV(df, size=cfg["preprocess"]["size"], pre=pre),
                    batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])
    m.eval(); out = []
    for x, _ in dl:
        out.append(torch.sigmoid(m(x.to(dev)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp = Path(cfg["paths"]["splits"]) / a.split
    out = Path(cfg["paths"]["results"]) / a.split / f"seed{a.seed}"
    m = build_model(False).to(dev)
    m.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    thr = cfg["train"]["threshold"]
    rows = []
    for spec in cfg["robustness"]["perturbations"]:
        pre = perturb.make(spec)
        for name in ("test_a", "test_b"):
            df = pd.read_csv(sp / f"{name}.csv")
            r = basic_metrics(df.label, scores(m, df, cfg, dev, pre), thr)
            rows.append({"perturbation": spec["name"], "test": name, "auc": r["auc"], "f1": r["f1"],
                         "recall": r["recall"], "fpr": r["fpr"]})
    res = pd.DataFrame(rows)
    res.to_csv(out / "robustness.csv", index=False)
    (out / "robustness.json").write_text(json.dumps(rows, indent=2))
    print(res.pivot(index="perturbation", columns="test", values="auc").round(3).to_string())


if __name__ == "__main__":
    main()
