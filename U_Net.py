#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net
"""

from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import numpy as np

import Input
import Encode
import Decode
import Output

class U_Net(nn.Module):

    def __init__(self, input_channel, num_class):
        super(U_Net, self).__init__()
        self.base_channel = 64
        self.input = Input.Input(input_channel, self.base_channel)
        self.encode1 = Encode.Encode(self.base_channel, 2*self.base_channel)
        self.encode2 = Encode.Encode(2*self.base_channel, 4*self.base_channel)
        self.encode3 = Encode.Encode(4*self.base_channel, 8*self.base_channel)
        self.encode4 = Encode.Encode(8*self.base_channel, 8*self.base_channel, 16*self.base_channel)
        self.decode1 = Decode.Decode(16*self.base_channel, 4*self.base_channel)
        self.decode2 = Decode.Decode(8*self.base_channel, 2*self.base_channel)
        self.decode3 = Decode.Decode(4*self.base_channel, self.base_channel)
        self.decode4 = Decode.Decode(2*self.base_channel, self.base_channel)
        self.output = Output.Output(self.base_channel, num_class)

    def forward(self, x):
        input_x = self.input(x)
        encode1_x = self.encode1(input_x)
        encode2_x = self.encode2(encode1_x)
        encode3_x = self.encode3(encode2_x)
        encode4_x = self.encode4(encode3_x)
        x = self.decode1(encode3_x, encode4_x)
        x = self.decode2(encode2_x, x)
        x = self.decode3(encode1_x, x)
        x = self.decode4(input_x, x)
        x = self.output(x)

        return x
