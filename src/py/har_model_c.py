# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../../")
root_dir = os.path.join(cur_dir, "../../../../../ai-algorithm-depolyment")
sys.path.append(root_dir)
import sports.har.src.c.har_model.har_model as CHarModel

from utils.base import SensorAlgo


class HarModel_C(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self, num_classes=6):
        self._har_model = CHarModel.HarModel()
        self._version = self._har_model.get_version()
        self._algo_name = 'HarModel_C'
        self._input_names = ['EventTimestamp', 'AccelX', 'AccelY', 'AccelZ']
        self._output_names = [
            'EventTimestamp', 'Prob0', 'Prob1', 'Prob2', 'Prob3', 'Prob4',
            'Prob5', 'Predict'
        ]

        self._cur_timestamp = 0
        self._activity = 0
        self._num_classes = num_classes
        self._probs = np.zeros(self._num_classes)

    def is_realtime(self):
        return True

    def get_version(self):
        return self._version

    def reset(self):
        self._cur_timestamp = 0
        self._activity = 0
        self._probs = np.zeros(self._num_classes)
        self._har_model.init_algo()

    def feed_data(self, data_point):
        """ main function processes data and count steps"""
        self._cur_timestamp = data_point['EventTimestamp']
        acc_x = data_point['AccelX']
        acc_y = data_point['AccelY']
        acc_z = data_point['AccelZ']
        updated = self._har_model.process_data(acc_x, acc_y, acc_z)
        if updated > 0:
            self._probs = self._har_model.get_probs()
            self._activity = np.argmax(self._probs)
        return (updated > 0)

    def get_activity(self):
        return self._argmax

    def get_probs(self):
        return self._probs

    def get_result(self):
        return {
            'EventTimestamp': self._cur_timestamp,
            'Prob0': self._probs[0],
            'Prob1': self._probs[1],
            'Prob2': self._probs[2],
            'Prob3': self._probs[3],
            'Prob4': self._probs[4],
            'Prob5': self._probs[5],
            'Predict': self._activity
        }
