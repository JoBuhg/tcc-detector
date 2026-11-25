"""Fluxo de geração MANUAL (ferramentas online gratuitas).

  python -m tccdet.manual prompts   # cria data/manual/prompts.csv (os mesmos prompts para todos os geradores)
  python -m tccdet.manual ingest    # organiza data/inbox/<gerador>/<categoria>/ em data/raw/ + registro
  python -m tccdet.manual status    # conta imagens por gerador/categoria
"""
import argparse, csv, hashlib, shutil
from datetime import date
from pathlib import Path
from PIL import Image
from . import config
from .prompts import build_prompts

EXT = {".png", ".jpg", ".jpeg", ".webp"}
REG_COLS = ["generator", "tool", "category", "idx", "file", "original_name", "sha256", "width", "height",
            "format", "has_exif", "added_on", "prompt", "visible_watermark", "notes"]


def prompts_sheet(cfg, out=None):
    n = cfg["generation"]["n_per_category"]
    out = Path(out or cfg["manual"]["prompts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in cfg["categories"]:
        for i, p in enumerate(build_prompts(cat, n, cfg["generation"]["prompt_style"], cfg["seed"])):
            rows.append({"category": cat, "idx": i, "prompt": p})
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["category", "idx", "prompt"])
        w.writeheader(); w.writerows(rows)
    return rows


def _load_prompts(cfg):
    path = Path(cfg["manual"]["prompts"])
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {(r["category"], int(r["idx"])): r["prompt"] for r in csv.DictReader(f)}


def _load_registry(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["sha256"]: r for r in csv.DictReader(f)}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(cfg, inbox=None):
    """Copia as imagens da caixa de entrada para data/raw/<gerador>/<categoria>/ com nome padronizado.
    A ordem das imagens (e, portanto, o prompt associado) segue a data de modificação e depois o nome."""
    inbox = Path(inbox or cfg["manual"]["inbox"])
    raw = Path(cfg["paths"]["raw"])
    reg_path = Path(cfg["manual"]["registry"])
    reg = _load_registry(reg_path)
    prompts = _load_prompts(cfg)
    added, skipped = 0, 0
    for gen, g in cfg["generators"].items():
        if g.get("kind") != "manual":
            continue
        for cat in cfg["categories"]:
            src = inbox / gen / cat
            if not src.exists():
                continue
            files = sorted((p for p in src.iterdir() if p.suffix.lower() in EXT),
                           key=lambda p: (p.stat().st_mtime, p.name))
            dst = raw / gen / cat
            dst.mkdir(parents=True, exist_ok=True)
            idx = len([p for p in dst.iterdir() if p.suffix.lower() in EXT])
            for p in files:
                sha = _sha(p)
                if sha in reg:
                    skipped += 1
                    continue
                new = dst / f"{gen}_{cat}_{idx:05d}{p.suffix.lower()}"
                shutil.copy2(p, new)
                with Image.open(p) as im:
                    w, h, fmt, exif = im.size[0], im.size[1], im.format, bool(im.getexif())
                reg[sha] = {"generator": gen, "tool": g.get("tool", ""), "category": cat, "idx": idx,
                            "file": str(new), "original_name": p.name, "sha256": sha, "width": w, "height": h,
                            "format": fmt, "has_exif": int(exif), "added_on": date.today().isoformat(),
                            "prompt": prompts.get((cat, idx), ""), "visible_watermark": "", "notes": ""}
                idx += 1
                added += 1
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REG_COLS)
        w.writeheader()
        w.writerows(sorted(reg.values(), key=lambda r: (r["generator"], r["category"], int(r["idx"]))))
    return added, skipped


def status(cfg):
    raw = Path(cfg["paths"]["raw"])
    n = cfg["generation"]["n_per_category"]
    out = []
    for gen in cfg["generators"]:
        for cat in cfg["categories"]:
            d = raw / gen / cat
            c = len([p for p in d.iterdir() if p.suffix.lower() in EXT]) if d.exists() else 0
            out.append((gen, cat, c, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prompts", "ingest", "status"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--inbox")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.command == "prompts":
        rows = prompts_sheet(cfg)
        print(f"{len(rows)} prompts em {cfg['manual']['prompts']}")
    elif a.command == "ingest":
        added, skipped = ingest(cfg, a.inbox)
        print(f"{added} imagens adicionadas, {skipped} já registradas (ignoradas).")
    else:
        print(f"{'gerador':12} {'categoria':10} {'imagens':>8} {'meta':>5}")
        for gen, cat, c, n in status(cfg):
            print(f"{gen:12} {cat:10} {c:8d} {n:5d}{'' if c >= n else '  <- faltam ' + str(n - c)}")


if __name__ == "__main__":
    main()
