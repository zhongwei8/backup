# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

import math
from pathlib import Path
import sys

from keras.models import load_model
import numpy as np

cur_dir = Path(__file__).parent.resolve()
depolyment_dir = cur_dir / '../../../../../ai-algorithm-depolyment/'
if depolyment_dir.exists():
    sys.path.append(depolyment_dir)
else:
    depolyment_dir = cur_dir / '../../../ai-algorithm-depolyment/'
    if depolyment_dir.exists():
        sys.path.append(depolyment_dir)

project_dir = depolyment_dir / 'ai-algorithm-depolyment'
sys.path.append(depolyment_dir)

# Import from ai-algorithm-depolyment repo
# So this file dependent on ai-algorithm-depolyment
from utils.base import SensorAlgo


class HarModel(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self,
                 model_file=cur_dir /
                 '../../data/model/weights.best-0615.hdf5',
                 win_len=8.0,
                 shift=2.0,
                 channel=4,
                 num_classes=6,
                 fs=26,
                 type=0):
        self._version = 100000

        self._model = load_model(model_file, compile=False)
        self._algo_name = 'HarModel'

        self._input_names = [
            'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Prob0', 'Prob1', 'Prob2', 'Prob3', 'Prob4',
            'Prob5', 'Predict'
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
        return updated

    def get_activity(self):
        return self._argmax

    def get_probs(self):
        return self._probs

    def get_result(self):
        return {
            'EventTimestamp(ns)': self._cur_timestamp,
            'Prob0': self._probs[0],
            'Prob1': self._probs[1],
            'Prob2': self._probs[2],
            'Prob3': self._probs[3],
            'Prob4': self._probs[4],
            'Prob5': self._probs[5],
            'Predict': self._argmax
        }

    def get_model(self):
        return self._model
