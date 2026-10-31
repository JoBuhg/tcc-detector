"""Carregamento de configuração YAML (com suporte a `extends:`)."""
from pathlib import Path
import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_read(path.parent / parent), cfg)
    return cfg


def load(path: str = "configs/experiment.yaml") -> dict:
    path = Path(path)
    cfg = _read(path)
    gpath = path.parent / cfg.get("generators_file", "generators.yaml")
    with open(gpath, encoding="utf-8") as f:
        gens = yaml.safe_load(f)["generators"]
    # permite sobrescrever geradores no config filho
    cfg["generators"] = _deep_merge(gens, cfg.get("generators_override", {}))
    return cfg


def seen_unseen(cfg: dict):
    g = cfg["generators"]
    seen = [k for k, v in g.items() if v["status"] == "seen"]
    unseen = [k for k, v in g.items() if v["status"] == "unseen"]
    return seen, unseen
