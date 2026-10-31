"""Geração determinística de prompts (mesmo conjunto para todos os geradores de texto)."""
import itertools
import random

SUBJECTS = {
    "cat": ["a tabby cat", "a black cat", "a white cat", "a ginger cat",
            "a siamese cat", "a grey cat", "a calico cat", "a fluffy cat"],
    "car": ["a red car", "a blue sedan", "a white hatchback", "a black sports car",
            "a silver station wagon", "a yellow compact car", "a green convertible", "a grey SUV"],
    "horse": ["a brown horse", "a white horse", "a black horse", "a chestnut horse",
              "a grey horse", "a palomino horse", "a foal", "a spotted horse"],
}
SCENES = ["in a grassy field", "on a city street", "near a wooden fence", "in front of a house",
          "on a gravel road", "in a parking lot", "next to a tree", "on a sunny hill",
          "in a quiet village", "beside a stone wall"]
LIGHT = ["at sunrise", "at midday", "in the late afternoon", "on an overcast day",
         "in soft evening light", "after rain", "under a clear sky"]
CAMERA = ["shot with a 50mm lens", "wide angle shot", "close-up shot", "from a low angle"]


def build_prompts(category: str, n: int, style: str = "a realistic photograph", seed: int = 0):
    combos = list(itertools.product(SUBJECTS[category], SCENES, LIGHT, CAMERA))
    if n > len(combos):
        raise ValueError(f"Só há {len(combos)} prompts únicos para '{category}'; pedido: {n}.")
    random.Random(f"{seed}-{category}").shuffle(combos)
    return [f"{style} of {s} {sc} {li}, {cam}" for s, sc, li, cam in combos[:n]]
