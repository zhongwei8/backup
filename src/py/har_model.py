# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

import math
from pathlib import Path
import sys

import click
from keras.models import load_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax

cur_dir = Path(__file__).parent.resolve()
project_dir = cur_dir / '../../'
sys.path.append(str(project_dir))
from src.py.utils.model_utils import GeneralModelPredictor

depolyment_dir = cur_dir / '../../../ai-algorithm-depolyment/'
if not depolyment_dir.exists():
    print(f'Warnning: ai-algorithm-depolyment not exits: {depolyment_dir}')
sys.path.append(str(depolyment_dir))
# Import from ai-algorithm-depolyment repo
from utils.base import SensorAlgo

keras_model_file = str(cur_dir / '../../data/model/weights.best-0615.hdf5')
torch_onnx_model_file = str(cur_dir / 'models/model-cnn-20200125.onnx')


class HarModel(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self,
                 model_file=keras_model_file,
                 win_len=8.0,
                 shift=2.0,
                 channel=4,
                 num_classes=6,
                 fs=26,
                 type=0):
        self._version = 100000

        self._model = GeneralModelPredictor(model_file, 'keras')
        self._algo_name = 'HarModel'

        self._input_names = [
            'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'Prob0', 'Prob1', 'Prob2',
            'Prob3', 'Prob4', 'Prob5', 'Predict'
        ]

        self._cur_timestamp = 0

        self._fs = fs
        self._cnt = 0  # count points
        self._buf_len = int(win_len * self._fs)
        self._buffer = np.zeros((1, self._buf_len, channel))  # buffer
        self._channel = channel
        self._shift_len = int(shift * self._fs)
        self._idx = 0

        self._activity = 0
        self._num_classes = num_classes
        self._probs = np.zeros(self._num_classes)
        self._res = {}

    def is_realtime(self):
        return True

    def get_version(self):
        return self._version

    def reset(self):
        self._cnt = 0  # count points
        self._buffer = np.zeros(self._buffer.shape)  # buffer

        self._idx = 0
        self._argmax = 0
        self._probs = np.zeros(self._num_classes)
        self._cur_timestamp = 0

    def feed_data(self, data_point):
        """ main function processes data and count steps"""
        self._cur_timestamp = data_point['EventTimestamp(ns)']
        acc_x = data_point['AccelX']
        acc_y = data_point['AccelY']
        acc_z = data_point['AccelZ']
        activity_type = data_point['Activity']
        acc_amp = math.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)

        self._buffer[0, self._idx, 0] = acc_x
        self._buffer[0, self._idx, 1] = acc_y
        self._buffer[0, self._idx, 2] = acc_z
        self._buffer[0, self._idx, 3] = acc_amp

        self._cnt += 1
        self._idx += 1

        updated = False
        if self._cnt >= self._buf_len and self._idx >= self._buf_len:
            probs = self._model.predict(self._buffer)
            self._probs = probs[0]
            self._argmax = np.argmax(self._probs)

            updated = True
            self._buffer[0,
                         0:self._buf_len - self._shift_len, :] = self._buffer[
                             0, self._shift_len::, :]
            self._idx = self._buf_len - self._shift_len
            self._res = {
                'EventTimestamp(ns)': self._cur_timestamp,
                'Activity': activity_type,
                'Prob0': self._probs[0],
                'Prob1': self._probs[1],
                'Prob2': self._probs[2],
                'Prob3': self._probs[3],
                'Prob4': self._probs[4],
                'Prob5': self._probs[5],
                'Predict': self._argmax
            }
        return updated

    def get_activity(self):
        return self._argmax

    def get_probs(self):
        return self._probs

    def get_result(self):
        return self._res

    def get_model(self):
        return self._model


class HarModelCNN(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self,
                 model_file=torch_onnx_model_file,
                 win_len=8.08,
                 shift=2.0,
                 channel=6,
                 num_classes=6,
                 fs=26,
                 ewma_alpha=0.99,
                 type=0):
        self._version = 100000
        self._init = False
        self._alpha = ewma_alpha
        self._norm = [[
            -0.03710845, -4.35627, 1.2581933, -0.03444214, -4.1099195, 1.267368
        ],
                      [
                          0.1535839, 0.17988862, 0.20019571, 0.17831624,
                          0.30422017, 0.23472616
                      ]]
        self._norm = np.asarray(self._norm).astype(np.float32)

        self._model = GeneralModelPredictor(torch_onnx_model_file, 'onnx')
        self._algo_name = 'HarModelCNN'

        self._input_names = [
            'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'Prob0', 'Prob1', 'Prob2',
            'Prob3', 'Prob4', 'Prob5', 'Predict'
        ]
        self._output_names_type = {
            'EventTimestamp(ns)': int,
            'Prob0': float,
            'Prob1': float,
            'Prob2': float,
            'Prob3': float,
            'Prob4': float,
            'Prob5': float,
            'Predict': int,
            'Activity': int
        }

        self._cur_timestamp = 0

        self._fs = fs
        self._cnt = 0  # count points
        self._buf_len = int(win_len * self._fs)
        self._buffer = []
        self._channel = channel
        self._shift_len = int(shift * self._fs)
        self._idx = 0

        self._activity = 0
        self._num_classes = num_classes
        self._probs = np.zeros(self._num_classes)
        self._res = {}

    def normalize_data(self):
        x = np.asarray(self._buffer).astype(np.float32)
        x = (x - self._norm[0]) * self._norm[1]
        x = x.reshape(1, *x.shape)
        x = np.transpose(x, (0, 2, 1))
        return x

    def is_realtime(self):
        return True

    def get_version(self):
        return self._version

    def reset(self):
        self._init = False
        self._cnt = 0  # count points
        self._buffer = []

        self._idx = 0
        self._argmax = 0
        self._probs = np.zeros(self._num_classes)
        self._cur_timestamp = 0
        self._res = {}

    def feed_data(self, data_point):
        """ main function processes data and count steps"""
        self._cur_timestamp = data_point['EventTimestamp(ns)']
        acc_x = data_point['AccelX']
        acc_y = data_point['AccelY']
        acc_z = data_point['AccelZ']
        activity_type = data_point['Activity']
        if self._init:
            old_x = self._buffer[-1][3]
            old_y = self._buffer[-1][4]
            old_z = self._buffer[-1][5]
            lp_x = old_x * self._alpha + (1 - self._alpha) * acc_x
            lp_y = old_y * self._alpha + (1 - self._alpha) * acc_y
            lp_z = old_z * self._alpha + (1 - self._alpha) * acc_z
        else:
            lp_x = acc_x
            lp_y = acc_y
            lp_z = acc_z
            self._init = True

        self._buffer.append([acc_x, acc_y, acc_z, lp_x, lp_y, lp_z])

        updated = False
        if len(self._buffer) >= self._buf_len:
            x = self.normalize_data()
            probs = self._model.predict(x)
            self._probs = softmax(np.squeeze(probs[0]))
            self._argmax = np.argmax(self._probs)

            updated = True
            del self._buffer[:self._shift_len]

            self._res = {
                'EventTimestamp(ns)': self._cur_timestamp,
                'Activity': activity_type,
                'Prob0': self._probs[0],
                'Prob1': self._probs[1],
                'Prob2': self._probs[2],
                'Prob3': self._probs[3],
                'Prob4': self._probs[4],
                'Prob5': self._probs[5],
                'Predict': self._argmax
            }
        return updated

    def get_activity(self):
        return self._argmax

    def get_probs(self):
        return self._probs

    def get_result(self):
        return self._res

    def get_model(self):
        return self._model

    def process_file(self, file_path):
        df = pd.read_csv(file_path)
        predicts = {}
        for i, row in df.iterrows():
            update = self.feed_data(row)
            if update:
                result = self.get_result()
                for key in result:
                    if key not in predicts:
                        predicts[key] = [result[key]]
                    else:
                        predicts[key].append(result[key])
        result = pd.DataFrame.from_dict(predicts)
        print(result.head(5))
        result = result.drop('EventTimestamp(ns)', axis='columns')
        acc = df[['AccelX', 'AccelY', 'AccelZ']]
        true_pred = result[['Activity', 'Predict']]
        prob = result.drop(['Activity', 'Predict'], axis='columns')
        mpl.use('Qt5Agg')
        plt.figure()
        plt.subplot(311)
        acc.plot(ax=plt.gca())
        plt.legend()
        plt.subplot(312)
        # plt.plot(y_pred, label='y_pred')
        # plt.plot(y_true, label='y_true')
        true_pred.plot(ax=plt.gca())
        plt.subplot(313)
        prob.plot(ax=plt.gca())
        plt.legend()
        plt.show()


@click.command()
@click.argument('file-path')
def main(file_path):
    model = HarModelCNN()
    model.process_file(file_path)


if __name__ == '__main__':
    main()
