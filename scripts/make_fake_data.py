"""Cria imagens FALSAS (ruído colorido) só para testar o pipeline de ponta a ponta."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tccdet import config

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/smoke.yaml")
p.add_argument("--n", type=int, default=30)
a = p.parse_args()
cfg = config.load(a.config)
rng = np.random.default_rng(0)
raw = Path(cfg["paths"]["raw"])
base = {"cat": (200, 120, 60), "car": (60, 90, 200), "horse": (90, 160, 80)}
noise = {"real": 40, **{g: 25 + 12 * i for i, g in enumerate(cfg["generators"])}}
for src, ns in noise.items():
    for cat in cfg["categories"]:
        d = raw / src / cat
        d.mkdir(parents=True, exist_ok=True)
        n_img = a.n * (len(cfg['generators']) if src == 'real' else 1)
        for i in range(n_img):
            w, h = int(rng.integers(120, 260)), int(rng.integers(120, 260))
            arr = np.clip(np.array(base[cat]) + rng.normal(0, ns, (h, w, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{src}_{cat}_{i:05d}.png")
print("dados falsos em", raw)
