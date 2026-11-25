"""Gera imagens sintéticas: python -m tccdet.generate.run --generator sd15 [--n 80]"""
import argparse, hashlib, json
from pathlib import Path
from tqdm import tqdm
from .. import config
from ..prompts import build_prompts
from .generators import build


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--n", type=int, help="imagens por categoria (padrão: config)")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = config.load(a.config)
    g = cfg["generators"][a.generator]
    if g.get("kind") == "manual":
        raise SystemExit(f"'{a.generator}' é um gerador MANUAL ({g.get('tool')}). Gere as imagens na ferramenta online, "
                         "salve em data/inbox/<gerador>/<categoria>/ e rode: python -m tccdet.manual ingest")
    n = a.n or cfg["generation"]["n_per_category"]
    bs = cfg["generation"]["batch_size"]
    cats = a.categories or cfg["categories"]
    if a.device:
        device = a.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = build(a.generator, g, device)
    root = Path(cfg["paths"]["raw"]) / a.generator
    root.mkdir(parents=True, exist_ok=True)
    meta_f = open(root / "meta.jsonl", "a", encoding="utf-8")

    for cat in cats:
        out = root / cat
        out.mkdir(parents=True, exist_ok=True)
        prompts = build_prompts(cat, n, cfg["generation"]["prompt_style"], cfg["seed"])
        todo = [i for i in range(n) if not (out / f"{a.generator}_{cat}_{i:05d}.png").exists()]
        for b in tqdm(range(0, len(todo), bs), desc=f"{a.generator}/{cat}"):
            idxs = todo[b:b + bs]
            seeds = [stable_seed(a.generator, cat, i) for i in idxs]
            imgs = gen.generate(cat, [prompts[i] for i in idxs], seeds, idxs)
            for i, s, im in zip(idxs, seeds, imgs):
                fn = out / f"{a.generator}_{cat}_{i:05d}.png"
                im.convert("RGB").save(fn)
                meta_f.write(json.dumps({"file": fn.name, "category": cat, "prompt": prompts[i],
                                         "seed": s, "size": im.size, "model": g.get("model_id")}) + "\n")
            meta_f.flush()


if __name__ == "__main__":
    main()
