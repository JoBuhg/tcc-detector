import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 1)  # logit; classe positiva = sintética
    return m


def set_phase(m, phase: int):
    """Fase 1: só fc. Fase 2: layer4 + fc. BatchNorm dos blocos congelados fica em eval."""
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True
    if phase == 2:
        for p in m.layer4.parameters():
            p.requires_grad = True


def set_train_mode(m, phase: int):
    m.eval()
    m.fc.train()
    if phase == 2:
        m.layer4.train()
