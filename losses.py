import torch
import torch.nn as nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights="IMAGENET1K_V1").features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        return torch.mean((self.vgg(x) - self.vgg(y)) ** 2)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + self.beta * self.perc(pred, target)
