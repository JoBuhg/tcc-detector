import numpy as np
from PIL import Image
from tccdet import perturb


def noisy(size=(64, 48)):
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype("uint8"))


def test_perturbations_keep_size_and_change_pixels():
    im = noisy()
    base = np.asarray(im, dtype=float)
    for spec in ({"kind": "jpeg", "value": 50}, {"kind": "blur", "value": 1.0}, {"kind": "downscale", "value": 0.5}):
        out = perturb.make(spec)(im)
        assert out.size == im.size
        assert np.abs(np.asarray(out, dtype=float) - base).mean() > 1.0


def test_none_is_identity():
    im = noisy()
    out = perturb.make({"kind": "none"})(im)
    assert np.array_equal(np.asarray(out), np.asarray(im))
