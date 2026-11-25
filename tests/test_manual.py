import os
import numpy as np
from PIL import Image
from tccdet import manual

CFG = {
    "seed": 1, "categories": ["cat", "car"],
    "generation": {"n_per_category": 5, "prompt_style": "a realistic photograph"},
    "manual": {}, "paths": {},
    "generators": {"gen_a": {"kind": "manual", "tool": "Tool A", "status": "seen", "family": "comercial"},
                   "gen_b": {"kind": "manual", "tool": "Tool B", "status": "unseen", "family": "comercial"}},
}


def make_cfg(tmp_path):
    cfg = dict(CFG)
    cfg["manual"] = {"inbox": str(tmp_path / "inbox"), "prompts": str(tmp_path / "prompts.csv"),
                     "registry": str(tmp_path / "registry.csv")}
    cfg["paths"] = {"raw": str(tmp_path / "raw")}
    return cfg


def save_img(path, seed, mtime):
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((20, 20, 3)) * 255).astype("uint8")).save(path)
    os.utime(path, (mtime, mtime))


def test_prompt_sheet_unique_per_category(tmp_path):
    cfg = make_cfg(tmp_path)
    rows = manual.prompts_sheet(cfg)
    assert len(rows) == 10
    for cat in cfg["categories"]:
        ps = [r["prompt"] for r in rows if r["category"] == cat]
        assert len(ps) == len(set(ps)) == 5


def test_ingest_orders_by_time_and_renames(tmp_path):
    cfg = make_cfg(tmp_path)
    manual.prompts_sheet(cfg)
    d = tmp_path / "inbox" / "gen_a" / "cat"
    d.mkdir(parents=True)
    save_img(d / "zzz.png", 1, 1000)   # mais antigo, nome "maior"
    save_img(d / "aaa.png", 2, 2000)
    added, skipped = manual.ingest(cfg)
    assert (added, skipped) == (2, 0)
    raw = tmp_path / "raw" / "gen_a" / "cat"
    names = sorted(p.name for p in raw.iterdir())
    assert names == ["gen_a_cat_00000.png", "gen_a_cat_00001.png"]
    reg = manual._load_registry(cfg["manual"]["registry"])
    first = [r for r in reg.values() if int(r["idx"]) == 0][0]
    assert first["original_name"] == "zzz.png" and first["tool"] == "Tool A" and first["prompt"]


def test_ingest_skips_duplicates_and_continues_index(tmp_path):
    cfg = make_cfg(tmp_path)
    d = tmp_path / "inbox" / "gen_b" / "car"
    d.mkdir(parents=True)
    save_img(d / "x.png", 3, 1000)
    manual.ingest(cfg)
    save_img(d / "y.png", 4, 3000)                 # nova
    added, skipped = manual.ingest(cfg)            # x.png já registrada
    assert (added, skipped) == (1, 1)
    raw = tmp_path / "raw" / "gen_b" / "car"
    assert sorted(p.name for p in raw.iterdir()) == ["gen_b_car_00000.png", "gen_b_car_00001.png"]


def test_status_counts(tmp_path):
    cfg = make_cfg(tmp_path)
    d = tmp_path / "inbox" / "gen_a" / "cat"
    d.mkdir(parents=True)
    save_img(d / "a.png", 5, 1000)
    manual.ingest(cfg)
    st = {(g, c): n for g, c, n, _ in manual.status(cfg)}
    assert st[("gen_a", "cat")] == 1 and st[("gen_b", "car")] == 0
