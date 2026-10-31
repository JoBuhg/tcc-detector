"""Padronização: remove metadados, recorte central quadrado, resize, JPEG com qualidade fixa,
deduplicação (dHash) e geração do manifest. python -m tccdet.preprocess"""
import argparse, io
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from . import config

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def center_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s))


def dhash64(im: Image.Image) -> np.uint64:
    g = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def hamming_to_all(hashes: np.ndarray, h: np.uint64) -> np.ndarray:
    x = np.bitwise_xor(hashes, h).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(x, axis=1).sum(1)


def image_stats(arr: np.ndarray) -> dict:
    g = arr.astype(np.float32).mean(2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return {"lap_var": float(lap.var()),
            "mean_r": float(arr[..., 0].mean()), "mean_g": float(arr[..., 1].mean()),
            "mean_b": float(arr[..., 2].mean()), "std_all": float(arr.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = config.load(a.config)
    pp = cfg["preprocess"]
    raw, out_root = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    gens = cfg["generators"]

    sources = ["real"] + [g for g in gens if (raw / g).exists()]
    rows, hashes = [], np.zeros(0, dtype=np.uint64)
    dups = 0
    for src in sources:
        for cat in cfg["categories"]:
            files = sorted(p for p in (raw / src / cat).glob("*") if p.suffix.lower() in EXT)
            for p in tqdm(files, desc=f"{src}/{cat}"):
                with Image.open(p) as im0:
                    fmt, (ow, oh) = (im0.format or p.suffix), im0.size
                    has_exif = bool(im0.info.get("exif")) or bool(getattr(im0, "getexif", lambda: {})())
                    im = center_square(im0.convert("RGB"))
                h = dhash64(im)
                if len(hashes) and hamming_to_all(hashes, h).min() <= pp["dedup_hamming"]:
                    dups += 1
                    continue
                hashes = np.append(hashes, h)
                im = im.resize((pp["size"], pp["size"]), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=pp["jpeg_quality"], subsampling=pp["jpeg_subsampling"])
                data = buf.getvalue()
                dst = out_root / src / cat / (p.stem + ".jpg")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                rows.append({
                    "path": str(dst), "source": src, "category": cat,
                    "label": 0 if src == "real" else 1,
                    "family": "real" if src == "real" else gens[src]["family"],
                    "status": "real" if src == "real" else gens[src]["status"],
                    "orig_w": ow, "orig_h": oh, "orig_bytes": p.stat().st_size,
                    "orig_fmt": str(fmt).lower(), "orig_exif": int(has_exif),
                    "proc_bytes": len(data), **image_stats(arr),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "manifest.csv", index=False)
    print(f"{len(df)} imagens processadas, {dups} duplicatas removidas.")
    print(df.groupby(["source", "category"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
