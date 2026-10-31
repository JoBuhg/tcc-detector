import itertools
import pandas as pd
from tccdet.split import make_split

SEEN, UNSEEN = ["g_a", "g_b"], ["g_c", "g_d"]
CATS = ["x", "y", "z"]


def fake_manifest(n=40):
    rows = []
    for src in ["real"] + SEEN + UNSEEN:
        for c in CATS:
            for i in range(n * (4 if src == "real" else 1)):
                rows.append({"path": f"{src}/{c}/{i}.jpg", "source": src, "category": c,
                             "label": int(src != "real")})
    return pd.DataFrame(rows)


def splits():
    return make_split(fake_manifest(), SEEN, UNSEEN, seed=1)


def test_no_path_overlap_between_sets():
    s = splits()
    for a, b in itertools.combinations(s, 2):
        assert not set(s[a].path) & set(s[b].path), f"vazamento entre {a} e {b}"


def test_unseen_generators_only_in_test_b():
    s = splits()
    for name in ("train", "val", "test_a"):
        assert not set(s[name].source) & set(UNSEEN)
    assert set(s["test_b"][s["test_b"].label == 1].source) == set(UNSEEN)


def test_classes_balanced_per_set_and_category():
    for name, d in splits().items():
        g = d.groupby(["category", "label"]).size().unstack()
        assert (g[0] == g[1]).all(), f"{name} desbalanceado"


def test_proportions_70_15_15():
    s = splits()
    n = [(s[k].label == 1).sum() for k in ("train", "val", "test_a")]
    tot = sum(n)
    assert abs(n[0] / tot - 0.70) < 0.02 and abs(n[1] / tot - 0.15) < 0.02


def test_deterministic():
    a, b = splits(), splits()
    assert a["train"].path.tolist() == b["train"].path.tolist()
