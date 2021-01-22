# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../../")
root_dir = os.path.join(cur_dir, "../../../../../ai-algorithm-depolyment")
sys.path.append(root_dir)
import sports.har.src.c.har_detector.har_detector as CHarDetector

from utils.base import SensorAlgo


class HarDetector_C(SensorAlgo):
    def __init__(self,
                 buf_len=30,
                 vote_len=15,
                 num_classes=6,
                 vote_score_thd=0.8):
        self._score_thd = vote_score_thd
        self._vote_len = vote_len
        self._num_classes = num_classes
        self._det = CHarDetector.HarDetector(self._score_thd, vote_len)
        self._version = self._det.get_version()
        self._cnt = 0
        self._activity = 0
        self._cur_timestamp = 0
        self._input_names = [
            'EventTimestamp', 'Prob0', 'Prob1', 'Prob2', 'Prob3', 'Prob4',
            'Prob5'
        ]
        self._output_names = ['EventTimestamp', 'PredictActivity']
        self._algo_name = 'HarDetector_C'

    def is_realtime(self):
        return True

    def reset(self):
        self._cur_timestamp = 0
        self._cnt = 0
        self._activity = 0
        self._det.init_algo(self._score_thd, self._vote_len)

    def feed_data(self, data_point):
        probs = np.zeros(self._num_classes)
        self._cur_timestamp = data_point['EventTimestamp']
        probs[0] = data_point['Prob0']
        probs[1] = data_point['Prob1']
        probs[2] = data_point['Prob2']
        probs[3] = data_point['Prob3']
        probs[4] = data_point['Prob4']
        probs[5] = data_point['Prob5']
        self._det.process_probs(probs, self._num_classes)
        self._activity = self._det.get_activity_type()
        self._cnt += 1
        return True

    def get_result(self):
        return {
            'EventTimestamp': self._cur_timestamp,
            'PredictActivity': self._activity
        }

    def get_version(self):
        return self._version
