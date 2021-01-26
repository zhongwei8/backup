#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-01-22

from keras.models import load_model
import onnxruntime as ort


class GeneralModelPredictor():
    def __init__(self, model_file, model_type='onnx') -> None:
        self._type = model_type
        if model_type == 'keras':
            self._model = load_model(model_file, compile=False)
        elif model_type == 'onnx':
            self._model = ort.InferenceSession(model_file)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    def predict(self, x):
        if self._type == 'keras':
            return self._model.predict(x)
        elif self._type == 'onnx':
            input_name = self._model.get_inputs()[0].name
            input_x = {input_name: x}
            onnx_out = self._model.run(None, input_x)
            return onnx_out[0]
