#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-11

from pathlib import Path

import coloredlogs
import click
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd

from activity_data_labeler import LABEL_ITEMS
from activity_data_labeler import LABEL_ITEMS_INDEX_DICT
from activity_data_labeler import LABEL_DAILY
from activity_data_labeler import LABEL_OTHER_SPORTS
from utils.log import Log


ACC_SUFFIX = 'accel-52HZ.csv'
GYRO_SUFFIX = 'gyroscope-52HZ.csv'
MAGNET_SUFFIX = 'magnet-50HZ.csv'
PRESSURE_SUFFIX = 'pressure-25HZ.csv'
LABEL_SUFFIX = 'label-result.csv'

CTM_HEADER_NAME = 'CurrentTimeMillis'
TS_HEADER_NAME = 'EventTimestamp(ns)'
TSS = [CTM_HEADER_NAME, TS_HEADER_NAME]

DATA_SETS_TO_USE = [
    '20201117-20201204',
    '20201205-20201211'
]


exception_record_list = []


def get_activity_type_name_by_file_name(file_name: str):
    metas = file_name.split('-')
    return metas[7].split('_')[1]


def load_label_result(label_file: Path):
    labels = []
    with label_file.open('r') as f:
        for line in f:
            line = line.rstrip('\n')
            labels.append(tuple(map(int, line.split('_'))))
    return labels


def load_sensor_data_and_labels_by_name(record_dir, type_name='accel'):
    print(f'\nProcessing record dir： {record_dir}')
    if type_name == 'accel':
        suffix = ACC_SUFFIX
    elif type_name == 'gyro':
        suffix = GYRO_SUFFIX
    elif type_name == 'magnet':
        suffix = MAGNET_SUFFIX
    else:
        raise ValueError(f'Unsupport type name: {type_name}')

    # Load data and split ts out
    data_file = record_dir / f'{record_dir.name}-{suffix}'
    if not data_file.exists():
        Log.e(f'Record has no {type_name} data, do nothing')
        return None, None, None
    data_all = pd.read_csv(data_file)
    data_all = data_all[:-1]  # Remove the last line, maybe one broken line
    data_ts, data = split_ts_and_data(data_all)

    # Fitler outliner
    print(f'Origing raw data shape: {data.shape}')
    exception_point_process(data.T[0], inplace=True)
    exception_point_process(data.T[1], inplace=True)
    exception_point_process(data.T[2], inplace=True)

    # To get the data labels
    label_file = record_dir / f'{record_dir.name}-{LABEL_SUFFIX}'
    if label_file.exists():
        labels_ts = load_label_result(label_file)
        labels = label_convert_ts2index(labels_ts, data_ts)
    else:
        Log.i('Label file not exists, Using activity type name Infer.')
        activity_type_name = get_activity_type_name_by_file_name(record_dir.name)
        is_daily = activity_type_name in LABEL_DAILY
        is_other_sports = activity_type_name in LABEL_OTHER_SPORTS
        if is_daily or is_other_sports:
            if is_daily:
                activity_type_name_as = 'DailyActivity'
            else:
                activity_type_name_as = 'OtherSports'
            activity_type = LABEL_ITEMS_INDEX_DICT.get(activity_type_name_as)
            print(f'Treat {activity_type_name} as {activity_type_name_as}: {activity_type}')
            labels = [(activity_type, 0, data.shape[0])]
        else:
            return data_ts, data, None

    return data_ts, data, labels


def load_sensor_data_and_labels(record_dir):
    acc_file = record_dir / f'{record_dir.name}-{ACC_SUFFIX}'
    gyro_file = record_dir / f'{record_dir.name}-{GYRO_SUFFIX}'
    magnet_file = record_dir / f'{record_dir.name}-{MAGNET_SUFFIX}'
    label_file = record_dir / f'{record_dir.name}-{LABEL_SUFFIX}'

    acc = pd.read_csv(acc_file)
    acc = acc[:-1]  # Remove the last line, maybe one broken line
    acc_ts, acc_data = split_ts_and_data(acc)
    gyro = pd.read_csv(gyro_file)
    gyro = gyro[:-1]  # Remove the last line, maybe one broken line
    gyro_ts, gyro_data = split_ts_and_data(gyro)
    magnet = pd.read_csv(magnet_file)
    magnet = magnet[:-1]  # Remove the last line, maybe one broken line
    magnet_ts, magnet_data = split_ts_and_data(magnet)

    labels_ts = load_label_result(label_file)
    acc_labels_index = label_convert_ts2index(labels_ts, acc_ts)
    gyro_labels_index = label_convert_ts2index(labels_ts, gyro_ts)
    magnet_labels_index = label_convert_ts2index(labels_ts, magnet_ts)

    ts = {
        'accel': acc_ts,
        'gyro': gyro_ts,
        'magnet': magnet_ts,
    }
    data = {
        'accel': acc_data,
        'gyro': gyro_data,
        'magnet': magnet_data,
    }
    labels = {
        'accel': acc_labels_index,
        'gyro': gyro_labels_index,
        'magnet': magnet_labels_index,
    }
    return ts, data, labels


def load_and_split_data(record_dir: Path, win_size=200, stride=25, downsample=True) -> np.ndarray:
    ts, data, labels = load_sensor_data_and_labels_by_name(record_dir, type_name='accel')
    if ts is None or data is None:
        return None, None
    if labels is None:
        return data, None
    data_segments = []
    segments_label = []
    for (activity_type, start, end) in labels:
        for i in range(start, end - win_size, stride):
            data_segments.append(data[i:i + win_size])
            segments_label.append(activity_type)
    data_segments = np.asarray(data_segments)
    return np.asarray(data_segments), np.asarray(segments_label)


def load_and_split_all_records(dataset_dir: Path, dataset_names):
    global exception_record_list
    all_data = []
    all_labels = []
    for name in dataset_names:
        d = dataset_dir / name
        print(f'\nProcessing dataset: {d}')
        for type_dir in filter(Path.is_dir, d.iterdir()):
            print(f'\nPocessing type dir: {type_dir}')
            for record_dir in filter(Path.is_dir, type_dir.iterdir()):
                data, label = load_and_split_data(record_dir)
                if data is None and label is None:
                    Log.e(f'Record has exception, check it out: {record_dir}')
                    exception_record_list.append(record_dir)
                elif data is not None and label is None:
                    Log.w(f'Record has no label, check it out: {record_dir}')
                else:
                    print(f'Record data segment shape: {data.shape}, label shape: {label.shape}')
                    if data.shape[1] != 3 and data.shape[2] != 3:
                        Log.e(f'Record data shape not correct: {record_dir}')
                    all_data.append(data)
                    all_labels.append(label)
    all_data = np.concatenate(tuple(all_data), axis=0)
    all_labels = np.concatenate(tuple(all_labels), axis=0)
    return all_data, all_labels


def label_convert_ts2index(labels_ts, ts_arr):
    labels_index = []
    max_idx = len(ts_arr) - 1
    Log.i(f'TS arr range: {ts_arr[0]} - {ts_arr[-1]}, Total len: {max_idx + 1}')
    for t, s, e in labels_ts:
        Log.i(f'Finding TS label {t}: {s} - {e}')
        if s > ts_arr[-1] or e < ts_arr[0]:
            Log.e(f'ts has no overlap: {s} > {ts_arr[-1]} or {e} < {ts_arr[0]}')
            continue
        start_index = -1
        end_index = -1
        for i, ts in enumerate(ts_arr):
            if start_index < 0:
                if ts > s:
                    start_index = i
            else:
                if ts > start_index and ts > e:
                    end_index = i
                    break
        if end_index < 0:
            end_index = max_idx
        Log.i(f'Found INDEX label {t}: {start_index} - {end_index}')
        labels_index.append((t, start_index, end_index))
    return labels_index


def split_ts_and_data(df, transpose_data=False):
    ts = df[TS_HEADER_NAME].values
    data = df.drop(columns=TSS).values
    if transpose_data:
        data = data.T
    return ts, data


def exception_point_process(data, inplace=False):
    res = data
    if not inplace:
        res = data.copy()
    max_idx = len(data) - 1
    for i, d in enumerate(data):
        if d > 80 or d < -80:
            if i == 0:
                res[i] = 0
            elif i == max_idx:
                res[i] = data[i - 1]
            else:
                res[i] = (data[i - 1] + data[i + 1]) / 2
    return res


def process_one_record(record_dir: Path):
    """Process one record

    Args:
        record_dir (Path): Path to record directory
    """
    acc_file = record_dir / f'{record_dir.name}-{ACC_SUFFIX}'
    gyro_file = record_dir / f'{record_dir.name}-{GYRO_SUFFIX}'
    magnet_file = record_dir / f'{record_dir.name}-{MAGNET_SUFFIX}'
    pressure_file = record_dir / f'{record_dir.name}-{PRESSURE_SUFFIX}'
    label_file = record_dir / f'{record_dir.name}-{LABEL_SUFFIX}'

    acc = pd.read_csv(acc_file)
    acc = acc[:-1]
    acc_ts, acc_data = split_ts_and_data(acc, transpose_data=True)
    gyro = pd.read_csv(gyro_file)
    gyro_ts, gyro_data = split_ts_and_data(gyro, transpose_data=True)
    magnet = pd.read_csv(magnet_file)
    magnet_ts, magnet_data = split_ts_and_data(magnet, transpose_data=True)
    pressure = pd.read_csv(pressure_file)
    _, pressure_data = split_ts_and_data(pressure, transpose_data=True)

    labels_ts = load_label_result(label_file)
    labels_index_acc = label_convert_ts2index(labels_ts, acc_ts)
    labels_index_gyro = label_convert_ts2index(labels_ts, gyro_ts)
    labels_index_magnet = label_convert_ts2index(labels_ts, magnet_ts)

    data_seg, _ = load_and_split_data(record_dir)
    print(f'data segments shape: {data_seg.shape}')

    # Check the label result
    plt.figure('IMU Sensor data')
    plt.subplot(311)
    ax1 = plt.gca()
    plt.title(f'{record_dir.name}\nAcceleration')
    plt.plot(acc_data[0], label='x')
    plt.plot(acc_data[1], label='y')
    plt.plot(acc_data[2], label='z')
    plt.legend()
    for (t, start, end) in labels_index_acc:
        plt.axvspan(start, end, ymin=0, ymax=0.8, alpha=0.5)
    plt.subplot(312, sharex=ax1)
    plt.title('Gyroscope')
    plt.plot(gyro_data[0])
    plt.plot(gyro_data[1])
    plt.plot(gyro_data[2])
    plt.legend()
    for (t, start, end) in labels_index_gyro:
        plt.axvspan(start, end, ymin=0, ymax=0.8, alpha=0.5)
    plt.subplot(313, sharex=ax1)
    plt.title('Magnetic')
    plt.plot(magnet_data[0])
    plt.plot(magnet_data[1])
    plt.plot(magnet_data[2])
    plt.legend()
    for (t, start, end) in labels_index_magnet:
        plt.axvspan(start, end, ymin=0, ymax=0.8, alpha=0.5)
    plt.tight_layout()
    # plt.show()

    # p1. Process exception points
    acc_re = acc_data.copy()
    exception_point_process(acc_re[0], inplace=True)
    exception_point_process(acc_re[1], inplace=True)
    exception_point_process(acc_re[2], inplace=True)

    # p2. low pass filter, try to get the poseture
    b, a = signal.butter(5, 0.025, 'lowpass', output='ba')
    w, h = signal.freqs(b, a)
    # plt.figure('Frequency response')
    # plt.semilogx(w, 10 * np.log10(abs(h)))
    # plt.title('Butterworth filter frequency response')
    # plt.xlabel('Frequency [radians / second]')
    # plt.ylabel('Amplitude [dB]')
    # plt.margins(0, 0.1)
    # plt.grid(which='both', axis='both')
    # plt.axvline(100, color='green')  # cutoff frequency
    # plt.show()

    acc_lp = signal.filtfilt(b, a, acc_re)
    acc_body = acc_re - acc_lp
    tan = -acc_lp[1] / acc_lp[0]
    tan_angle = np.rad2deg(np.arctan(tan))
    plt.figure('Exception point remove')
    plt.subplot(411)
    ax1 = plt.gca()
    plt.title(f'{record_dir.name}\nAcceleration X')
    plt.plot(acc_data.T[0], '--', label='raw')
    plt.plot(acc_re[0], '-', label='remove exceptions')
    plt.plot(acc_lp[0], '-', label='low pass')
    plt.plot(acc_body[0], '-', label='BA')
    # plt.plot(tan_angle, '-', label='arctan y/x')
    plt.legend()
    plt.subplot(412, sharex=ax1, sharey=ax1)
    plt.title('Acceleration Y')
    plt.plot(acc_data.T[1], '--', label='raw')
    plt.plot(acc_re[1], '-', label='remove exceptions')
    plt.plot(acc_lp[1], '-', label='low pass')
    plt.plot(acc_body[1], '-', label='BA')
    plt.legend()
    plt.subplot(413, sharex=ax1, sharey=ax1)
    plt.title('Acceleration Z')
    plt.plot(acc_data.T[2], '--', label='raw')
    plt.plot(acc_re[2], '-', label='remove exceptions')
    plt.plot(acc_lp[2], '-', label='low pass')
    plt.plot(acc_body[2], '-', label='BA')
    plt.legend()
    plt.subplot(414, sharex=ax1)
    plt.title('arctan -y/x')
    plt.plot(tan_angle, '-', label='arctan -y/x')
    plt.legend()
    plt.tight_layout()
    plt.show()


@click.group(invoke_without_command=True)
@click.argument('data_dir')
@click.pass_context
def main(ctx, data_dir):
    ctx.obj['data_dir'] = Path(data_dir)
    if ctx.invoked_subcommand is None:
        data_dir = Path(data_dir)
        if data_dir.is_file():
            data_dir = data_dir.parent
        process_one_record(data_dir)


@main.command()
@click.option('-f', '--force', is_flag=True, help='Force gennerate labeled activity raw data')
@click.pass_context
def pp(ctx, force):
    global exception_record_list
    dataset_dir = ctx.obj['data_dir']
    data_x_file = Path('../data/activity_data_x_raw.npy')
    data_y_file = Path('../data/activity_data_y_raw.npy')
    if force or not (data_x_file.exists() and data_y_file.exists()):
        data, label = load_and_split_all_records(dataset_dir, DATA_SETS_TO_USE)
        np.save(data_x_file, data)
        np.save(data_y_file, label)
        print(f'Saved data_x to path: {data_x_file}')
        print(f'Saved data_y to path: {data_y_file}')
        print(f'Exception record list: \n{exception_record_list}')
    else:
        print('Activity data exists, loading...')
        data = np.load(data_x_file)
        label = np.load(data_y_file)
    print(f'data shape: {data.shape}')
    print(f'label shape: {label.shape}')


if __name__ == "__main__":
    main(obj={})
