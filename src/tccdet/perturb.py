"""Perturbações para o teste de sensibilidade (só PIL)."""
import io
from PIL import Image, ImageFilter


def jpeg(im, quality):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(im, radius):
    return im.convert("RGB").filter(ImageFilter.GaussianBlur(float(radius)))


def downscale(im, factor):
    im = im.convert("RGB")
    w, h = im.size
    small = im.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def make(spec: dict):
    kind = spec["kind"]
    if kind == "none":
        return lambda im: im.convert("RGB")
    fn = {"jpeg": jpeg, "blur": blur, "downscale": downscale}[kind]
    return lambda im: fn(im, spec["value"])
