#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-01-22

import torch
from torch import nn


class ActivityCNN(nn.Module):
    def __init__(self,
                 seq_len=90,
                 in_ch=6,
                 hidden_ch=20,
                 out_ch=6,
                 dropout_ratio=0.1):
        """
        :param seq_len:    输入序列长度
        :param in_ch:      输入channel
        :param hidden_ch:  隐藏层channel
        :param out_ch:     输出channel，即分类的类别
        """
        super(ActivityCNN, self).__init__()

        # Normalize or standardize parameter
        self.norm = None

        stride = 6
        kernel_size = 6
        self.conv1 = nn.Conv1d(in_ch,
                               hidden_ch,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=0)
        print(f'conv1 weight shape: {self.conv1.weight.shape}')
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.out_len1 = (seq_len - kernel_size) // stride + 1
        # self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.out_len1 = self.out_len1 // 2
        print(f'conv1 output seq len: {self.out_len1}')

        kernel_size = 3
        self.conv2 = nn.Conv1d(hidden_ch,
                               hidden_ch,
                               kernel_size,
                               stride=2,
                               dilation=1)
        print(f'conv2 weight shape: {self.conv2.weight.shape}')
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.out_len2 = (self.out_len1 - kernel_size) // 2 + 1
        print(f'conv2 output seq len: {self.out_len2}')

        self.conv3 = nn.Conv1d(hidden_ch,
                               hidden_ch,
                               kernel_size,
                               stride=2,
                               dilation=1)
        print(f'conv3 weight shape: {self.conv3.weight.shape}')
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.out_len3 = (self.out_len2 - 1 * (kernel_size - 1) - 1) // 2 + 1
        print(f'conv3 output seq len: {self.out_len3}')

        k4 = 3
        self.conv4 = nn.Conv1d(hidden_ch,
                               out_ch,
                               kernel_size=k4,
                               stride=2,
                               dilation=1)
        print(f'conv4 weight shape: {self.conv4.weight.shape}')
        self.out_len = (self.out_len3 - 1 * (k4 - 1) - 1) // 2 + 1
        print(f'conv4 output seq len: {self.out_len}')

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2,
                                 self.conv3, self.relu3, self.dropout3,
                                 self.conv4)

    def forward(self, x):
        return self.net(x)
