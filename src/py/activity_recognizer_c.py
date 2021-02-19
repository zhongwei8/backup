#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-02-19

from pathlib import Path
import sys

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = Path(__file__).parent.resolve()
depolyment_dir = current_dir / '../../../ai-algorithm-depolyment/'
if not depolyment_dir.exists():
    print(f'Warnning: ai-algorithm-depolyment not exits: {depolyment_dir}')
sys.path.append(str(depolyment_dir))
# Import from ai-algorithm-depolyment repo
from utils.base import SensorAlgo

build_dir = current_dir / '../../build_linux-x86_64/'
sys.path.append(str(build_dir))
# Import C interface built with pybind11
from mi_har_py import MIActivityRecognizerPy


class ActivityRecognizerC(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self, vote_len=15, vote_threshold=0.8, num_classes=6):
        self._vote_len = vote_len
        self._vote_threshold = vote_threshold
        self._model = MIActivityRecognizerPy(vote_threshold, vote_len)
        self._version = self._model.get_version()
        self._algo_name = 'ActivityRecognizerC'

        self._input_names = [
            'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'Prob0', 'Prob1', 'Prob2',
            'Prob3', 'Prob4', 'Prob5', 'Predict', 'PredictActivity'
        ]

        self._cur_timestamp = 0
        self._cnt = 0  # count points
        self._num_classes = num_classes

        self._model_predict = 0
        self._probs = np.zeros(self._num_classes)
        self._predcit_activity = 0
        self._res = {}

    def is_realtime(self):
        return True

    def get_version(self):
        return self._version

    def reset(self):
        self._cur_timestamp = 0
        self._cnt = 0
        self._model.init_algo(self._vote_threshold, self._vote_len)

        self._model_predict = 0
        self._probs = np.zeros(self._num_classes)
        self._predcit_activity = 0
        self._res = {}

    def feed_data(self, data_point):
        """ main function processes data and count steps"""
        self._cur_timestamp = data_point['EventTimestamp(ns)']
        acc_x = data_point['AccelX']
        acc_y = data_point['AccelY']
        acc_z = data_point['AccelZ']
        activity_type = data_point['Activity']

        updated = self._model.process_data(acc_x, acc_y, acc_z)
        if updated > 0:
            self._probs = self._model.get_probs()
            self._model_predict = np.argmax(self._probs)
            self._predcit_activity = self._model.get_type()
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
        return (updated > 0)

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
    model = ActivityRecognizerC()
    acc, result = model.process_file(file_path)
    analysis_result(acc, result)


if __name__ == '__main__':
    main()
