"""Baseline de atalhos: um classificador SEM olhar o conteúdo semântico.
Se acerta muito, o conjunto tem viés (formato/tamanho/ruído). python -m tccdet.shortcuts"""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from . import config

FEATURES = {
    # atributos do arquivo ORIGINAL (mostram o que a padronização precisa remover)
    "raw": ["orig_w", "orig_h", "orig_bytes", "orig_exif", "bytes_per_px", "is_png"],
    # atributos do arquivo JÁ padronizado (deve ficar próximo de AUC 0,5)
    "processed": ["proc_bytes", "lap_var", "mean_r", "mean_g", "mean_b", "std_all"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="fixed")
    a = ap.parse_args()
    cfg = config.load(a.config)
    sp = Path(cfg["paths"]["splits"]) / a.split
    d = {n: pd.read_csv(sp / f"{n}.csv") for n in ("train", "val", "test_a", "test_b")}
    for df in d.values():
        df["bytes_per_px"] = df.orig_bytes / (df.orig_w * df.orig_h)
        df["is_png"] = (df.orig_fmt == "png").astype(int)
    res = {}
    for fs, cols in FEATURES.items():
        clf = GradientBoostingClassifier(random_state=cfg["seed"]).fit(d["train"][cols], d["train"].label)
        res[fs] = {n: round(roc_auc_score(d[n].label, clf.predict_proba(d[n][cols])[:, 1]), 4)
                   for n in ("val", "test_a", "test_b")}
    out = Path(cfg["paths"]["results"]) / "shortcuts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("Interpretação: AUC ≈ 0,5 em 'processed' = sem atalho óbvio; AUC alto = revisar padronização.")


if __name__ == "__main__":
    main()
