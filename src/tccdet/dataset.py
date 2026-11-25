import pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_tf(train: bool, hflip: bool, size: int):
    tf = [T.Resize((size, size))] if size else []
    if train and hflip:
        tf.append(T.RandomHorizontalFlip())
    return T.Compose(tf + [T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageCSV(Dataset):
    def __init__(self, csv, train=False, hflip=True, size=None, pre=None):
        self.df = pd.read_csv(csv) if isinstance(csv, str) else csv
        self.tf = make_tf(train, hflip, size)
        self.pre = pre

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        if self.pre is not None:
            img = self.pre(img)
        return self.tf(img), torch.tensor(float(r.label))


def loader(csv, bs, train, workers=0, hflip=True, size=None):
    return DataLoader(ImageCSV(csv, train, hflip, size), batch_size=bs, shuffle=train, num_workers=workers)
