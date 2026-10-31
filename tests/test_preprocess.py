import io
import numpy as np
from PIL import Image
from tccdet.preprocess import center_square, dhash64, hamming_to_all


def test_center_square():
    im = Image.new("RGB", (200, 100))
    assert center_square(im).size == (100, 100)


def test_dhash_detects_near_duplicates():
    rng = np.random.default_rng(0)
    a = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-2, 3, a.shape), 0, 255).astype(np.uint8)
    c = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    ha, hb, hc = (dhash64(Image.fromarray(x)) for x in (a, b, c))
    arr = np.array([ha], dtype=np.uint64)
    assert hamming_to_all(arr, hb).min() <= 8
    assert hamming_to_all(arr, hc).min() > 8


def test_saved_jpeg_has_no_exif():
    im = Image.new("RGB", (64, 64), (10, 20, 30))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95, subsampling=0)
    out = Image.open(io.BytesIO(buf.getvalue()))
    assert not out.getexif()
