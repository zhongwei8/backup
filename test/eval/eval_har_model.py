# Copyright 2020 Xiaomi
#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(root_dir)
from masl_public.utils.base import Dataset, Evaluator
from masl_public.utils.utils import create_logger, list_csv_files, read_csv
from masl_public.metrics.har.har_metrics import HarModelMetric
from masl_public.metrics.har.har_metrics import LabelDict, Activities, CategoryNames, label_to_category_idx
from src.masl_sports.har.src.py.har_model import HarModel


def slice_data(in_data, frame_len, shift, fs, use_amp=True):
    win_len = int(frame_len * fs)
    stride = int(shift * fs)
    input_shape = np.shape(in_data)
    input_len = input_shape[0]
    n_frames = int((input_len - win_len) / stride) + 1
    if n_frames < 1:
        return None, None
    n_channels = 4 if use_amp else 3
    frames = np.zeros((n_frames, win_len, n_channels))
    labels = np.zeros(n_frames)
    for i in range(n_frames):
        if i * stride + win_len < input_len:
            acc_raw = in_data[i * stride:i * stride + win_len, 2:5]
            frames[i, :, 0:3] = acc_raw
            if use_amp:
                acc_amp = np.sqrt(acc_raw[:, 0]**2 + acc_raw[:, 1]**2 +
                                  acc_raw[:, 2]**2)
                frames[i, :, 3] = acc_amp
            activity = 0
            count = 0
            for n in range(i * stride, i * stride + win_len):
                if in_data[n, -1] > 0:
                    count += 1
                    activity = in_data[n, -1]
            if count >= int(win_len):
                labels[i] = activity
            else:
                labels[i] = 0
    return frames, labels


class HarModelDataset(Dataset):
    def __init__(self, data_dir, duration=8, shift=2, fs=26, logger=None):
        if not os.path.isdir(data_dir):
            raise Exception("invalid input directory: {0}.".format(data_dir))
        self._data_dir = data_dir
        self._csv_data = dict()
        self._duration = duration
        self._shift = shift
        self._fs = fs
        self._processed = False
        self._logger = create_logger(
            name=self._data_dir) if logger is None else logger
        self._logger.info(
            'Create HarModelDataset {0} with duration({1}) shift({2}) fs({3})'.
            format(self._data_dir, duration, shift, fs))

    def load(self):
        data_x = []
        data_y = []
        file_list = list_csv_files(self._data_dir)
        for file_path in file_list:
            header, data = read_csv(file_path)
            frames, labels = slice_data(data,
                                        self._duration,
                                        self._shift,
                                        self._fs,
                                        use_amp=True)
            if frames is None or labels is None:
                continue
            for i in range(len(labels)):
                if labels[i] != 0:
                    data_x.append(frames[i])
                    data_y.append(label_to_category_idx(labels[i]))
        data_x = np.array(data_x, dtype=np.float32)
        data_y = np.array(data_y, dtype=np.float32)
        self._logger.info('test x shape: {0}'.format(data_x.shape))
        self._logger.info('test y shape: {0}'.format(data_y.shape))
        self._csv_data['data_x'] = data_x
        self._csv_data['data_y'] = data_y


class HarModelEvaluator(Evaluator):
    def __init__(self,
                 algo_list,
                 dataset,
                 metric=HarModelMetric(),
                 logger=None):
        self._dataset = dataset
        self._metric = metric
        self._algo_list = algo_list
        self._logger = create_logger(
            name='HarModelEvaluator') if logger is None else logger
        self._metric.update_logger(self._logger)

    def eval(self):
        data_dict = self._dataset.get_data()
        x_test = data_dict['data_x']
        y_test = data_dict['data_y']
        self._logger.info("test x shape: {0}".format(x_test.shape))
        self._logger.info("test y shape: {0}".format(y_test.shape))

        results = dict()
        for algo in self._algo_list:
            model = algo.get_model()
            y_pred_probs = model.predict(x_test)
            algo_name = algo.get_algo_name()
            results[algo_name] = y_pred_probs
        self._metric.calculate(y_test, results)


def evaluate(dataset_dir, algo_names):
    logger = create_logger(name='har model evaluation')
    algo_list = [eval(algo_name + '()') for algo_name in algo_names]
    dataset = HarModelDataset(dataset_dir,
                              duration=8,
                              shift=2,
                              fs=26,
                              logger=logger)
    evaluator = HarModelEvaluator(algo_list, dataset, logger=logger)
    evaluator.eval()


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        required=True,
                        help="Input data files directory")
    parser.add_argument('--algos',
                        type=str,
                        dest='algos',
                        help='Use columns to seperate multiple algorithms.',
                        default='HarModel')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.dir):
        evaluate(args.dir, args.algos.split(','))
    else:
        raise Exception("invalid input directory: {0}.".format(args.dir))
