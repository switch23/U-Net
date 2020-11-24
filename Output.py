#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input
"""

import torch
import torch.nn as nn
import numpy as np

class Output(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(Output, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, 1)

    def forward(self, x):
        return self.conv(x)