"""Divisão sem vazamento. python -m tccdet.split
Visto -> train/val/test_a (70/15/15 por gerador x categoria); Não visto -> test_b.
Reais: subconjuntos disjuntos, mesma quantidade por categoria em cada conjunto."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from . import config
from .config import seen_unseen


def make_split(df, seen, unseen, seed, ratios=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    syn = df[df.label == 1]
    reals = df[df.label == 0]
    parts = {"train": [], "val": [], "test_a": []}
    for (_, _), grp in syn[syn.source.isin(seen)].groupby(["source", "category"]):
        g = grp.iloc[rng.permutation(len(grp))]
        n_tr, n_va = int(round(len(g) * ratios[0])), int(round(len(g) * ratios[1]))
        parts["train"].append(g.iloc[:n_tr])
        parts["val"].append(g.iloc[n_tr:n_tr + n_va])
        parts["test_a"].append(g.iloc[n_tr + n_va:])
    out = {k: pd.concat(v) for k, v in parts.items() if v}
    out["test_b"] = syn[syn.source.isin(unseen)]
    used = set()
    final = {}
    for name, s in out.items():
        chosen = [s]
        for cat, n in s.groupby("category").size().items():
            pool = reals[(reals.category == cat) & ~reals.path.isin(used)]
            if len(pool) < n:
                raise ValueError(f"Reais insuficientes em '{cat}' para '{name}': {len(pool)} < {n}")
            pick = pool.sample(n, random_state=int(rng.integers(1_000_000_000)))
            used.update(pick.path)
            chosen.append(pick)
        d = pd.concat(chosen).reset_index(drop=True)
        d["split"] = name
        final[name] = d
    return final


def write(final, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, d in final.items():
        d.to_csv(out_dir / f"{name}.csv", index=False)
        info[name] = {"n": len(d), "real": int((d.label == 0).sum()), "synthetic": int((d.label == 1).sum()),
                      "by_source": d.groupby("source").size().to_dict()}
    (out_dir / "split_info.json").write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = config.load(a.config)
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "manifest.csv")
    seen, unseen = seen_unseen(cfg)
    base = Path(cfg["paths"]["splits"])
    if cfg["split"]["protocol"] == "fixed":
        write(make_split(df, seen, unseen, cfg["seed"], cfg["split"]["ratios"]), base / "fixed")
    else:  # leave-one-generator-out
        gens = [g for g in cfg["generators"] if g in set(df.source)]
        for held in gens:
            write(make_split(df, [g for g in gens if g != held], [held], cfg["seed"],
                             cfg["split"]["ratios"]), base / f"logo_{held}")
    print("splits gravados em", base)


if __name__ == "__main__":
    main()
