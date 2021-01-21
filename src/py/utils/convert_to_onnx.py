#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-01-20

from pathlib import Path

import click
from keras.models import load_model
import onnxmltools
import torch
from torch import nn


def convert_keras_to_onnx(keras_model, onnx_model_file):
    onnx_model = onnxmltools.convert_keras(keras_model)
    onnxmltools.utils.save_model(onnx_model, onnx_model_file)


def convert_pytorch_to_onnx(pytorch_model, x, onnx_model_file):
    if isinstance(pytorch_model, nn.Module):
        pytorch_model.eval()
    else:
        raise ValueError('Model\'s type must be pytorch')
    torch.onnx.export(
        pytorch_model,
        x,  # model input (or a tuple for multiple inputs)
        onnx_model_file,  # where to save the model
        export_params=True,  # store the trained parameter weights
        do_constant_folding=True,  # Do constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {
                0: 'batch_size'
            },  # variable lenght axes
            'output': {
                0: 'batch_size'
            }
        })


@click.command()
@click.argument('origin-model-file')
@click.argument('origin-model-type')
@click.argument('onnx-model-file')
def main(origin_model_file, origin_model_type, onnx_model_file):
    origin_model_file = Path(origin_model_file)
    onnx_model_file = Path(onnx_model_file)

    if not origin_model_file.exists():
        print(f'Origin model file not exists: {origin_model_file}')
        exit(1)

    if origin_model_type == 'keras':
        keras_model = load_model(origin_model_file, compile=False)
        convert_keras_to_onnx(keras_model, onnx_model_file)
    else:
        supported_types = ['keras']
        raise ValueError(f'Only supported one of [{supported_types}]')


if __name__ == "__main__":
    main()
