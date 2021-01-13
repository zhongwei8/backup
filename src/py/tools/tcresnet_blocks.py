#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:11:29 2020

@author: mi
"""

import keras.layers
from keras import regularizers

parameters = {"kernel_initializer": "he_normal"}


def basic_2d(filters, stage=0, stride=1, kernel_size=9, numerical_name=False):
    """
    A two-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import tcresnet_blocks

        >>> tcresnet_blocks.basic_2d(64)
    """
    if stride > 0 and numerical_name:
        block_char = "b{}".format(stride)
    else:
        block_char = chr(ord('a') + stride)

    stage_char = str(stage + 2)

    def f(x):

        y = keras.layers.Conv2D(filters, (kernel_size, 1),
                                strides=(stride, 1),
                                padding='same',
                                kernel_regularizer=regularizers.l2(0.002),
                                use_bias=False,
                                name="res{}{}_branch2a".format(
                                    stage_char, block_char),
                                **parameters)(x)

        y = keras.layers.BatchNormalization(
            name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = keras.layers.Activation("relu",
                                    name="res{}{}_branch2a_relu".format(
                                        stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters, (kernel_size, 1),
                                strides=(1, 1),
                                padding='same',
                                use_bias=False,
                                kernel_regularizer=regularizers.l2(0.002),
                                name="res{}{}_branch2b".format(
                                    stage_char, block_char),
                                **parameters)(y)

        y = keras.layers.BatchNormalization(
            name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if stride == 2:

            block_char_2 = "b{}".format(stride - 1)

            shortcut = keras.layers.Conv2D(
                filters, (1, 1),
                strides=(stride, 1),
                padding='same',
                use_bias=False,
                kernel_regularizer=regularizers.l2(0.002),
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters)(x)

            shortcut = keras.layers.BatchNormalization(
                name="bn{}{}_branch1".format(stage_char, block_char_2))(
                    shortcut)

            shortcut = keras.layers.Activation("relu",
                                               name="res{}{}_relu".format(
                                                   stage_char,
                                                   block_char_2))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))(
            [y, shortcut])

        y = keras.layers.Activation("relu",
                                    name="res{}{}_relu".format(
                                        stage_char, block_char))(y)

        return y

    return f
