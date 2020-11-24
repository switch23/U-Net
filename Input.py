#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input
"""

import torch
import torch.nn as nn
import numpy as np
import Convs

class Input(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(Input, self).__init__()
        self.convs = Convs.Convs(input_channel, output_channel)

    def forward(self, x):
        return self.convs(x)