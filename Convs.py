#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convs
"""

import torch
import torch.nn as nn
import numpy as np

class Convs(nn.Module):

    def __init__(self, input_channel, output_channel, middle_channel=None):
        super(Convs, self).__init__()
        self.conv_size = 3
        if middle_channel is None:
            middle_channel = output_channel
        print('middle', middle_channel)
        print('output', output_channel)
        self.convs = nn.Sequential(
            nn.Conv2d(input_channel, middle_channel, kernel_size=self.conv_size, padding=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, output_channel, kernel_size=self.conv_size, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)
