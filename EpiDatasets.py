#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EpiDatasets
"""

import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvf
import random
from PIL import Image

class EpiDatasets(torch.utils.data.Dataset):
    def __init__(self, transform = None, target_transform = None, train = True):
        self.transform = transform
        self.target_transform = target_transform

        label_files = glob.glob('data/masks/*.png')
        data_files = glob.glob('data/*.tif')

        self.dataset = []
        self.labelset = []

        for label_file in label_files:
            file_name = (os.path.basename(label_file)).split('.')
            file_name = file_name[0].split('_')
            
            self.dataset.append(Image.open('data/'+file_name[0]+'_'+file_name[1]+'.tif').convert('RGB'))
            self.labelset.append(Image.open('data/masks/' + os.path.basename(label_file)))

        if train:
            self.dataset = self.dataset[:30]
            self.labelset = self.labelset[:30]
        else:
            self.dataset = self.dataset[31:]
            self.labelset = self.labelset[31:]

        # Data Augmentation
        self.augmented_dataset = []
        self.augmented_labelset = []
        for num in range(len(self.dataset)):
            data = self.dataset[num]
            label = self.labelset[num]
            for crop_num in range(16):
                # クロップ位置を乱数で決定
                i, j, h, w = transforms.RandomCrop.get_params(data, output_size=(250,250))
                cropped_data = tvf.crop(data, i, j, h, w)
                cropped_label = tvf.crop(label, i, j, h, w)
                for rotation_num in range(4):
                    rotated_data = tvf.rotate(cropped_data, angle=90*rotation_num)
                    rotated_label = tvf.rotate(cropped_label, angle=90*rotation_num)
                    for h_flip_num in range(2):
                        h_flipped_data = transforms.RandomHorizontalFlip(p=h_flip_num)(rotated_data)
                        h_flipped_label = transforms.RandomHorizontalFlip(p=h_flip_num)(rotated_label)
                        self.augmented_dataset.append(h_flipped_data)
                        self.augmented_labelset.append(h_flipped_label)
                        """
                        # 垂直反転にする場合
                        # 回転操作を行う場合, 重複が生じるため水平反転と垂直反転は併用しない
                        for v_flip_num in range(2):
                            v_flipped_data = transforms.RandomVerticalFlip(p=v_flip_num)(h_flipped_data)
                            v_flipped_label = transforms.RandomVerticalFlip(p=v_flip_num)(h_flipped_label)
                            self.augmented_dataset.append(v_flipped_data)
                            self.augmented_labelset.append(v_flipped_label)
                        """

        self.datanum = len(self.augmented_dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.augmented_dataset[idx]
        out_label = self.augmented_labelset[idx]

        if self.transform:
            out_data = self.transform(out_data)

        if self.target_transform:
            out_label = self.target_transform(out_label)

        return out_data, out_label
