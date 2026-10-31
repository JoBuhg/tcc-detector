"""Adaptadores de geradores. Dependências pesadas são importadas apenas quando usadas.

Interface: gen.generate(category, prompts, seeds, idxs) -> list[PIL.Image]
"""
from pathlib import Path
import sys


class DiffusersGen:
    needs_prompt = True

    def __init__(self, g: dict, device: str):
        import torch
        from diffusers import AutoPipelineForText2Image
        self.g, self.device, self.torch = g, device, torch
        dtype = torch.float16 if device == "cuda" else torch.float32
        kw = {}
        if g.get("disable_safety_checker"):
            kw = {"safety_checker": None, "requires_safety_checker": False}
        self.pipe = AutoPipelineForText2Image.from_pretrained(g["model_id"], torch_dtype=dtype, **kw)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, category, prompts, seeds, idxs):
        gens = [self.torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]
        return self.pipe(
            prompt=list(prompts), num_inference_steps=self.g["steps"],
            guidance_scale=self.g["guidance"], height=self.g["height"], width=self.g["width"],
            generator=gens,
        ).images


class BigGANGen:
    needs_prompt = False

    def __init__(self, g: dict, device: str):
        import torch
        from pytorch_pretrained_biggan import BigGAN
        self.g, self.device, self.torch = g, device, torch
        self.model = BigGAN.from_pretrained(g["model_id"]).to(device).eval()

    def generate(self, category, prompts, seeds, idxs):
        from pytorch_pretrained_biggan import one_hot_from_int, truncated_noise_sample, convert_to_images
        classes = self.g["classes"][category]
        trunc = self.g["truncation"]
        imgs = []
        for i, s in zip(idxs, seeds):
            cls = classes[i % len(classes)]
            noise = self.torch.from_numpy(
                truncated_noise_sample(truncation=trunc, batch_size=1, seed=int(s) % (2**31))
            ).to(self.device)
            vec = self.torch.from_numpy(one_hot_from_int(cls, batch_size=1)).to(self.device)
            with self.torch.no_grad():
                out = self.model(noise, vec, trunc)
            imgs.append(convert_to_images(out.cpu())[0])
        return imgs


class StyleGAN2Gen:
    """Usa o repositório NVlabs/stylegan2-ada-pytorch (git clone em external/)."""
    needs_prompt = False

    def __init__(self, g: dict, device: str):
        import torch
        self.g, self.device, self.torch = g, device, torch
        repo = Path(g["repo_dir"]).resolve()
        if not repo.exists():
            raise FileNotFoundError(
                f"Clone o repositório em {repo}: "
                "git clone https://github.com/NVlabs/stylegan2-ada-pytorch " + str(repo))
        sys.path.insert(0, str(repo))
        self._nets = {}

    def _net(self, category):
        if category in self._nets:
            return self._nets[category]
        import dnnlib, legacy  # do repositório da NVIDIA
        ck = self.g["checkpoints"].get(category)
        if not ck:
            raise ValueError(f"Sem checkpoint StyleGAN2 para '{category}' em generators.yaml. "
                             "Preencha, ou troque a categoria/gerador.")
        with dnnlib.util.open_url(ck) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(self.device)
        self._nets[category] = G
        return G

    def generate(self, category, prompts, seeds, idxs):
        import numpy as np
        from PIL import Image
        G = self._net(category)
        imgs = []
        for s in seeds:
            z = self.torch.from_numpy(np.random.RandomState(int(s) % (2**32)).randn(1, G.z_dim)).to(self.device)
            with self.torch.no_grad():
                x = G(z, None, truncation_psi=self.g["truncation_psi"], noise_mode="const")
            x = (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(self.torch.uint8)[0].cpu().numpy()
            imgs.append(Image.fromarray(x, "RGB"))
        return imgs


KINDS = {"diffusers": DiffusersGen, "biggan": BigGANGen, "stylegan2": StyleGAN2Gen}


def build(name: str, g: dict, device: str):
    return KINDS[g["kind"]](g, device)
