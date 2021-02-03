# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

import os
from pathlib import Path
import sys

import numpy as np

current_dir = Path(__file__).parent.resolve()
depolyment_dir = current_dir / '../../../ai-algorithm-depolyment/'
if not depolyment_dir.exists():
    print(f'Warnning: ai-algorithm-depolyment not exits: {depolyment_dir}')
sys.path.append(depolyment_dir)
# Import from ai-algorithm-depolyment repo
from utils.base import SensorAlgo

build_dir = current_dir / '../../build_x86_64/'
if not build_dir.exists():
    print(f'Build dir not exists: {build_dir}')
    print('Please run ./scripts/build_x86_64.sh on project root dir')
    exit(1)
sys.path.append(str(build_dir))
# Import from build directory
import src.c.har_detector.har_detector as CHarDetector


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
            'EventTimestamp(ns)', 'Prob0', 'Prob1', 'Prob2', 'Prob3', 'Prob4',
            'Prob5', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'PredictActivity'
        ]
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
        self._cur_timestamp = data_point['EventTimestamp(ns)']
        probs[0] = data_point['Prob0']
        probs[1] = data_point['Prob1']
        probs[2] = data_point['Prob2']
        probs[3] = data_point['Prob3']
        probs[4] = data_point['Prob4']
        probs[5] = data_point['Prob5']
        self._det.process_probs(probs, self._num_classes)
        self._activity = self._det.get_activity_type()
        self._cnt += 1
        self._res = {
            'EventTimestamp(ns)': self._cur_timestamp,
            'Activity': data_point['Activity'],
            'PredictActivity': self._activity,
        }
        return True

    def get_result(self):
        return self._res

    def get_version(self):
        return self._version
