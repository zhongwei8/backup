# Copyright (c) Xiaomi, 2020. All rights reserved.
from __future__ import absolute_import, division, print_function

from pathlib import Path
import sys

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = Path(__file__).parent
depolyment_dir = current_dir / '../../../ai-algorithm-depolyment/'
if not depolyment_dir.exists():
    print(f'Warnning: ai-algorithm-depolyment not exits: {depolyment_dir}')
sys.path.append(str(depolyment_dir))
# Import from ai-algorithm-depolyment repo
from utils.base import SensorAlgo

build_dir = current_dir / '../../build_x86_64/'
if not build_dir.exists():
    print(f'Build dir not exists: {build_dir}')
    print('Please run ./scripts/build_x86_64.sh on project root dir')
    exit(1)
sys.path.append(str(build_dir))
# Import from build directory
import src.c.har_model.har_model as CHarModel


class HarModel_C(SensorAlgo):
    """ Class for counting steps with accelerate data """
    def __init__(self, num_classes=6):
        self._har_model = CHarModel.HarModel()
        self._version = self._har_model.get_version()
        self._algo_name = 'HarModel_C'

        self._input_names = [
            'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ', 'Activity'
        ]
        self._output_names = [
            'EventTimestamp(ns)', 'Activity', 'Prob0', 'Prob1', 'Prob2',
            'Prob3', 'Prob4', 'Prob5', 'Predict'
        ]

        self._cur_timestamp = 0
        self._activity = 0
        self._num_classes = num_classes
        self._probs = np.zeros(self._num_classes)
        self._res = {}

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
        self._cur_timestamp = data_point['EventTimestamp(ns)']
        acc_x = data_point['AccelX']
        acc_y = data_point['AccelY']
        acc_z = data_point['AccelZ']
        activity_type = data_point['Activity']
        updated = self._har_model.process_data(acc_x, acc_y, acc_z)
        if updated > 0:
            self._probs = self._har_model.get_probs()
            self._activity = np.argmax(self._probs)
            self._res = {
                'EventTimestamp(ns)': self._cur_timestamp,
                'Activity': activity_type,
                'Prob0': self._probs[0],
                'Prob1': self._probs[1],
                'Prob2': self._probs[2],
                'Prob3': self._probs[3],
                'Prob4': self._probs[4],
                'Prob5': self._probs[5],
                'Predict': self._activity
            }
        return (updated > 0)

    def get_activity(self):
        return self._argmax

    def get_probs(self):
        return self._probs

    def get_result(self):
        return self._res

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
    model = HarModel_C()
    model.process_file(file_path)


if __name__ == '__main__':
    main()
