# Copyright 2020 Xiaomi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import itertools
import argparse
import os
import os.path
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(src_dir)
# from common.py.utils.data_utils import list_csv_files, read_csv_as_dict, write_dict_to_csv
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})


def read_csv_as_dict(file_name, has_header=True, custom_headers=None):
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        if has_header:
            headers = next(reader)
            headers = next(reader)
        if custom_headers is not None:
            headers = custom_headers
        data = np.array(list(reader)).astype(float).transpose()
    d = dict(zip(headers, [data[i] for i in range(data.shape[0])]))
    return headers, d


def all_csv_files(directory):
    """ This function returns all .csv files in the directory, recursively """
    file_list = []
    if not os.path.isdir(directory):
        raise Exception("invalid input directory: {0}.".format(directory))
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if re.search(r'\.csv$', filename):
                file_list.append(os.path.join(root, filename))
    return file_list


def list_csv_files(in_dir, ext='.csv'):
    return [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(in_dir)
        for f in filenames if os.path.splitext(f)[1] == '.csv'
    ]


def check(input_dir, columns):
    file_list = list_csv_files(input_dir)

    for filepath in file_list:
        filename = filepath.split('/')[-1]
        print("process:{0}".format(filename))
        items = filename.split('-')
        scene = 'Unknown'
        for item_str in items:
            vals = item_str.split('_')
            if len(vals) == 2 and vals[0] == 'scene':
                scene = vals[1]
        _, data_dict = read_csv_as_dict(filepath)
        df = pd.DataFrame.from_dict(data_dict)
        print(f'{df.values.shape}')
        ax = df.plot(y=columns, title="scene:{0}".format(scene))
        plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        required=True,
                        dest='dir',
                        help="Input data directory")
    parser.add_argument("--columns",
                        required=True,
                        dest='columns',
                        help="select columns to plot")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.dir):
        check(args.dir, args.columns.split(','))
    else:
        raise Exception("Invalid directory: {0}.".format(args.input_dir))
