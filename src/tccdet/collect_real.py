"""Amostra imagens reais de uma base existente para data/raw/real/<categoria>/.
python -m tccdet.collect_real"""
import argparse, random, shutil
from pathlib import Path
from . import config

EXT = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = config.load(a.config)
    r = cfg["real"]
    src_root = Path(r["source_dir"])
    for cat in cfg["categories"]:
        files = []
        for sub in r["sources"][cat]:
            files += [p for p in (src_root / sub).rglob("*") if p.suffix in EXT]
        files.sort()
        random.Random(f"{cfg['seed']}-{cat}").shuffle(files)
        if len(files) < r["n_per_category"]:
            raise ValueError(f"'{cat}': só {len(files)} reais disponíveis, pedido {r['n_per_category']}.")
        out = Path(cfg["paths"]["raw"]) / "real" / cat
        out.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(files[: r["n_per_category"]]):
            shutil.copy(p, out / f"real_{cat}_{i:05d}{p.suffix.lower()}")
        print(f"{cat}: {r['n_per_category']} reais copiadas")


if __name__ == "__main__":
    main()
