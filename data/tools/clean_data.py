# Copyright 2020 Xiaomi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import argparse
import os
import os.path
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(src_dir)
from common.py.utils.data_utils import list_csv_files, read_csv_as_dict, write_dict_to_csv


def clean(input_dir, output_dir):
    file_list = list_csv_files(input_dir)

    for filepath in file_list:
        filename = os.path.split(filepath)[-1]
        abs_path = os.path.abspath(filepath)
        new_filepath = abs_path.replace(os.path.abspath(input_dir),
                                        os.path.abspath(output_dir))
        info, data = read_csv_as_dict(filepath)
        info_list = ['info', 'SampleRate:26']
        new_data_dict = dict()
        if 'Swimming' in filename or 'swimming' in filename:
            new_data_dict[
                'CurrentTimestamp'] = data['CurrentTimeMillis'][::2] // 1000000
        else:
            new_data_dict['CurrentTimestamp'] = data['CurrentTimeMillis'][::2]
        new_data_dict[
            'EventTimestamp'] = data['EventTimestamp(ns)'][::2] // 1000000
        new_data_dict['AccelX'] = data['accel_x'][::2]
        new_data_dict['AccelY'] = data['accel_y'][::2]
        new_data_dict['AccelZ'] = data['accel_z'][::2]
        new_data_dict['GyroX'] = data['gyro_x'][::2]
        new_data_dict['GyroY'] = data['gyro_y'][::2]
        new_data_dict['GyroZ'] = data['gyro_z'][::2]
        new_data_dict['MagX'] = data['mag_x'][::2]
        new_data_dict['MagY'] = data['mag_y'][::2]
        new_data_dict['MagZ'] = data['mag_z'][::2]
        new_data_dict['Activity'] = data['activity'][::2]
        write_dict_to_csv(new_filepath, new_data_dict, info_list)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
                        required=True,
                        dest='input_dir',
                        help="Input data directory")
    parser.add_argument("--output-dir",
                        required=True,
                        dest='output_dir',
                        help="Output data directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if os.path.isdir(args.input_dir):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        clean(args.input_dir, args.output_dir)
    else:
        raise Exception("Invalid directory: {0}.".format(args.input_dir))
