"""Treino em duas fases. python -m tccdet.train --split fixed --seed 0"""
import argparse, copy, json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
from . import config
from .dataset import loader
from .model import build_model, set_phase, set_train_mode


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_epoch(m, dl, crit, dev, phase, opt=None):
    train = opt is not None
    set_train_mode(m, phase) if train else m.eval()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(dl, leave=False):
            x, y = x.to(dev), y.to(dev)
            loss = crit(m(x).squeeze(1), y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="fixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    tc = cfg["train"]
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(a.seed)
    sp = Path(cfg["paths"]["splits"]) / a.split
    out = Path(cfg["paths"]["results"]) / a.split / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["preprocess"]["size"]
    kw = dict(bs=tc["batch_size"], workers=tc["num_workers"], hflip=tc["hflip"], size=size)
    tr = loader(str(sp / "train.csv"), train=True, **kw)
    va = loader(str(sp / "val.csv"), train=False, **kw)

    m = build_model(tc["pretrained"]).to(dev)
    crit = nn.BCEWithLogitsLoss()
    hist = []

    set_phase(m, 1)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=tc["phase1"]["lr"])
    for ep in range(tc["phase1"]["epochs"]):
        l_tr = run_epoch(m, tr, crit, dev, 1, opt); l_va = run_epoch(m, va, crit, dev, 1)
        hist.append({"phase": 1, "epoch": ep, "train_loss": l_tr, "val_loss": l_va}); print(hist[-1])

    set_phase(m, 2)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=tc["phase2"]["lr"])
    best, best_state, bad = float("inf"), copy.deepcopy(m.state_dict()), 0
    for ep in range(tc["phase2"]["max_epochs"]):
        l_tr = run_epoch(m, tr, crit, dev, 2, opt); l_va = run_epoch(m, va, crit, dev, 2)
        hist.append({"phase": 2, "epoch": ep, "train_loss": l_tr, "val_loss": l_va}); print(hist[-1])
        if l_va < best:
            best, best_state, bad = l_va, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["phase2"]["patience"]:
                break
    torch.save(best_state, out / "model.pt")
    (out / "history.json").write_text(json.dumps({"best_val_loss": best, "history": hist}, indent=2))
    print("melhor val_loss:", best)


if __name__ == "__main__":
    main()
