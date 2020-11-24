#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode
"""

import torch
import torch.nn as nn
import numpy as np
import Convs

class Encode(nn.Module):

    def __init__(self, input_channel, output_channel, middle_channel=None):
        super(Encode, self).__init__()
        self.pooling_size = 2
        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(2),
            Convs.Convs(input_channel, output_channel, middle_channel)
        )

    def forward(self, x):
        return self.down_sampling(x)
