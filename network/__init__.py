#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-29 17:31:50

@author: JimmyHua
"""

from torch import nn
from torchvision import models


def resnet50(classes, pretrain=True):
    if pretrain:
        net = models.resnet50(pretrained=True)
    else:
        net = models.resnet50()
    net.avgpool = nn.AdaptiveAvgPool2d(1)
    net.fc = nn.Linear(net.fc.in_features, classes)
    return net