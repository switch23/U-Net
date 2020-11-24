#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode
"""

import torch
import torch.nn as nn
import numpy as np
import Convs
import torch.nn.functional as F

class Decode(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(Decode, self).__init__()
        self.up_sample_size = 2
        self.up_sample = nn.Upsample(scale_factor=self.up_sample_size, mode='bilinear', align_corners=True)
        self.convs = Convs.Convs(input_channel, output_channel, input_channel//2)

    def forward(self, x_encode, x_decode):
        x_decode = self.up_sample(x_decode)
        row_diff = x_encode.size()[2] - x_decode.size()[2]
        col_diff = x_encode.size()[3] - x_decode.size()[3]
        #row_slice = row_diff // 2
        #col_slice = col_diff // 2
        #x_encode = x_encode[:, :, row_slice:(x_encode.size()[2]-row_slice), col_slice:(x_encode.size()[3]-col_slice)]
        #x_encode = F.pad(x_encode, (row_diff//2, row_diff - row_diff // 2, col_diff // 2, col_diff - col_diff // 2))
        x = torch.cat([x_encode, x_decode], dim=1)
        x = self.convs(x)
        return x
