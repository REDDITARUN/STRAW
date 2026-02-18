"""Baseline model builders (e.g. modified ResNet34 for EMNIST)."""

import torch.nn as nn
import torchvision


def build_resnet34(num_classes: int = 47) -> nn.Module:
    """Construct a ResNet34 adapted for single-channel 28x28 inputs.

    Modifications from the standard ImageNet ResNet34:
    - ``conv1``: 1 input channel, 3x3 kernel, stride 1, padding 1
    - ``maxpool``: replaced with ``nn.Identity()``
    - ``fc``: output dimension set to *num_classes*
    """
    model = torchvision.models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
