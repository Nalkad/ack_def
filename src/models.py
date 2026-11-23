"""Models used in the experiments.

All architectures are CIFAR-10 variants (3x3 stem, no 7x7 + max-pool), implemented
here from scratch so that the repository has no dependency on a model zoo:

* ``resnet18`` / ``resnet34`` -- He et al. 2016, CIFAR variant (11.2M / 21.3M params)
* ``vgg16``                  -- Simonyan & Zisserman 2015, VGG-16 with BatchNorm,
                                adapted to 32x32 inputs (14.7M params)

``PixelModel`` wraps a network so that the *input space of the attack* is the
[0,1] pixel space: it applies ``(x - mean) / std`` before the network.  Working in
pixel space makes eps = 8/255 an honest "8 grey levels out of 255" budget and makes
``clamp(0, 1)`` a correct projection onto the valid image set.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CIFAR10_MEAN, CIFAR10_STD


# --------------------------------------------------------------------------- #
# pixel-space wrapper
# --------------------------------------------------------------------------- #
class PixelModel(nn.Module):
    """x in [0,1]^3x32x32  ->  logits.  Normalisation is differentiable."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.net((x01 - self.mean) / self.std)


# --------------------------------------------------------------------------- #
# ResNet
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes: int = 10, width: int = 64):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(block, width, layers[0], 1)
        self.layer2 = self._make_layer(block, width * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, width * 4, layers[2], 2)
        self.layer4 = self._make_layer(block, width * 8, layers[3], 2)
        self.fc = nn.Linear(width * 8, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet18(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes)


# --------------------------------------------------------------------------- #
# VGG-16 (BN) for 32x32
# --------------------------------------------------------------------------- #
class VGG16CIFAR(nn.Module):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        layers, in_c = [], 3
        for v in self.cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [
                    nn.Conv2d(in_c, v, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_c = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def vgg16(num_classes: int = 10) -> VGG16CIFAR:
    return VGG16CIFAR(num_classes)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def build_model(name: str, num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name in ("resnet18", "r18"):
        return resnet18(num_classes)
    if name in ("resnet34", "r34"):
        return resnet34(num_classes)
    if name == "vgg16":
        return vgg16(num_classes)
    raise ValueError(f"unknown model {name}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for n in ("resnet18", "resnet34", "vgg16"):
        m = build_model(n)
        print(f"{n:10s} {num_params(m)/1e6:.2f} M params")
