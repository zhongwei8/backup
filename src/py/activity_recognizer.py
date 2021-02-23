#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-02-19

from enum import Enum, unique
import math
from pathlib import Path
import sys

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = Path(__file__).parent.resolve()
project_dir = current_dir / '../../'
sys.path.append(str(project_dir))
from src.py.utils.model_utils import GeneralModelPredictor

depolyment_dir = current_dir / '../../../ai-algorithm-depolyment/'
if not depolyment_dir.exists():
    print(f'Warnning: ai-algorithm-depolyment not exits: {depolyment_dir}')
sys.path.append(str(depolyment_dir))
# Import from ai-algorithm-depolyment repo
from utils.base import SensorAlgo

keras_model_file = str(current_dir / '../../data/model/har_cnn-20210204.h5')


@unique
class ActivityType(Enum):
    Unknown = 0
    Walk = 1
    Run = 2
    Rowing = 3
    Elliptical = 4
    Bike = 5
    NumType = 6


class ListQueue(object):
    def __init__(self, capacity) -> None:
        self._data = []
        self.capacity = capacity

    def append(self, value):
        if len(self._data) >= self.capacity:
            self._data.pop(0)
        self._data.append(value)

    def shift(self, shift_len):
        if shift_len < self.capacity:
            self._data = self._data[shift_len:]
        else:
            self._data.clear()

    def get(self, idx):
        return self._data[idx]

    def get_data(self):
        return self._data

    def reset(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)

    def size(self):
        return len(self)

    def is_full(self):
        return self.size() >= self.capacity


class ActivityVoter():
    def __init__(self, vote_len, vote_thres, activity_num=6) -> None:
        self._vote_len = vote_len
        self._vote_thres = vote_thres
        self._buf = ListQueue(vote_len)
        self.hist = np.zeros(activity_num)
        self.current_predict = 0
        self.current_activity = 0
        self.current_activity_score = 1.0

    def process(self, activity):
        if self._buf.is_full():
            old = self._buf.get(0)
            self.hist[old] -= 1
        self._buf.append(activity)
        self.hist[activity] += 1

        if self._buf.is_full():
            max_idx = np.argmax(self.hist)
            self.current_activity = max_idx
            self.current_activity_score = self.hist[max_idx] / self._vote_len
            if self.current_activity_score >= self._vote_thres:
                self.current_predict = self.current_activity

            return self.current_predict
        else:
            self._buf.append(activity)
            self.hist[activity] += 1
            return activity

    def reset(self):
        self._buf.reset()
        self.hist.fill(0)
        self.current_predict = 0
        self.current_activity = 0
        self.current_activity_score = 1.0


class ActivityRecognizer(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self,
                 model_file=keras_model_file,
                 win_len=8.0,
                 shift=2.0,
                 num_classes=6,
                 fs=26,
                 vote_len=15,
                 vote_threshold=0.8):
        self._version = 100  # v0.1.0

        self._model = GeneralModelPredictor(model_file, 'keras')
        self._algo_name = 'ActivityRecognizer'

        self._input_names = [
            'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'Prob0', 'Prob1', 'Prob2',
            'Prob3', 'Prob4', 'Prob5', 'Predict', 'PredictActivity'
        ]

        self._cur_timestamp = 0

        self._fs = fs
        self._cnt = 0  # count points
        self._buf_len = int(win_len * self._fs)
        self._data_buffer = ListQueue(self._buf_len)
        self._shift_len = int(shift * self._fs)

        self._model_predict = 0
        self._num_classes = num_classes
        self._probs = np.zeros(self._num_classes)

        self._predcit_activity = 0
        self._voter = ActivityVoter(vote_len, vote_threshold)

        self._res = {}

    def is_realtime(self):
        return True

    def get_version(self):
        return self._version

    def reset(self):
        self._cnt = 0  # count points
        self._data_buffer.reset()
        self._voter.reset()

        self._model_predict = 0
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

        self._data_buffer.append([acc_x, acc_y, acc_z, acc_amp])

        updated = False
        if self._data_buffer.is_full():
            self._cnt += 1
            data = np.asarray(self._data_buffer.get_data())
            data = data.reshape((1, 1, *data.shape))
            probs = self._model.predict(data)
            self._probs = probs[0]
            self._model_predict = np.argmax(self._probs)

            self._predcit_activity = self._voter.process(self._model_predict)

            updated = True
            self._res = {
                'EventTimestamp(ns)': self._cur_timestamp,
                'Activity': activity_type,
                'Prob0': self._probs[0],
                'Prob1': self._probs[1],
                'Prob2': self._probs[2],
                'Prob3': self._probs[3],
                'Prob4': self._probs[4],
                'Prob5': self._probs[5],
                'Predict': self._model_predict,
                'PredictActivity': self._predcit_activity
            }
            self._data_buffer.shift(self._shift_len)
        return updated

    def get_model_predict(self):
        return self._model_predict

    def get_predict_activity(self):
        return self._predcit_activity

    def get_probs(self):
        return self._probs

    def get_result(self):
        return self._res

    def get_model(self):
        return self._model

    def process_file(self, file_path):
        df = pd.read_csv(file_path)
        acc = df[['AccelX', 'AccelY', 'AccelZ']]
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
        return acc, result


def analysis_result(acc, result):
    print(result[340:360])
    result = result.drop('EventTimestamp(ns)', axis='columns')
    result_columns = ['Activity', 'Predict', 'PredictActivity']
    true_pred = result[result_columns]
    prob = result.drop(result_columns, axis='columns')
    mpl.use('Qt5Agg')
    plt.figure()
    plt.subplot(311)
    acc.plot(ax=plt.gca())
    plt.legend()
    plt.subplot(312)
    true_pred.plot(ax=plt.gca(), style='-o')
    plt.subplot(313)
    prob.plot(ax=plt.gca(), style='-o')
    plt.legend()
    plt.show()


@click.command()
@click.argument('file-path')
def main(file_path):
    model = ActivityRecognizer()
    acc, result = model.process_file(file_path)
    analysis_result(acc, result)


if __name__ == '__main__':
    main()
