#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-01-04

import multiprocessing as mp
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activity_data_labeler import label_convert_ts2index
from activity_data_labeler import load_label_result

ACC_SUFFIX = 'accel-52HZ.csv'
GYRO_SUFFIX = 'gyroscope-52HZ.csv'
MAGNET_SUFFIX = 'magnet-50HZ.csv'
PRESSURE_SUFFIX = 'pressure-25HZ.csv'
LABEL_SUFFIX = 'label-result.csv'

HEADER_NAMES = [
    'CurrentTimeMillis', 'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ',
    'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Activity'
]

HEADER_TYPES = [
    int, int, float, float, float, float, float, float, float, float, float,
    int
]

HEADER_NAMES_TYPE = dict(zip(HEADER_NAMES, HEADER_TYPES))

SENSOR_TIMESTAMP_NS_ERROR_MAX = 3000_000  # 3ms

DATASET_TO_USE = [
    '20201117-20201204', '20201205-20201211', '20201217-20201231',
    '20210101-20210107'
]


def get_activity_type_name_by_record_name(record_name: str):
    metas = record_name.split('-')
    return metas[7].split('_')[1]


def align_sensor_data(acc, gyro, mag=None, sample_rate=52):
    """Align sensor sensor data
    Columns should be CurrentTimeMillis,EventTimestamp,x,y,z

    If mag sensor data is None, perform 6-axis sensor data align
    Else perform 9-axis sensor data align

    Parameters
    ----------
    acc : np.ndarray
        Accel data with shape (I*5)
    gyro : np.ndarray
        Gyro data with shape (J*5)
    mag : np.ndarray
        Mag data with shape (K*5), Nonable
    sample_rate : int
        Sample rate
    """
    align_to_ref_ts = False
    find_closest = False
    sample_period = 1e9 / sample_rate
    acc_num = len(acc)
    acc_ts = acc[:, 1]
    gyro_num = len(gyro)
    gyro_ts = gyro[:, 1]
    if mag is not None:
        mag_num = len(mag)
        mag_ts = mag[:, 1]
        init_ts = max(gyro_ts[0], mag_ts[0])
    else:
        mag_num = 0
        mag_ts = None
        init_ts = gyro_ts[0]
    i = 0
    j = 0
    k = 0

    # Init
    while i < acc_num and acc_ts[i] < init_ts:
        i += 1

    aligned = []
    ref_ts = acc_ts[i]
    while i < acc_num and j < gyro_num and (mag is None or k < mag_num):
        # Align gyro
        while (j + 1 < gyro_num
               and gyro_ts[j + 1] < ref_ts + SENSOR_TIMESTAMP_NS_ERROR_MAX):
            j += 1
        if find_closest and j + 1 < gyro_num:
            ts_l = gyro_ts[j]
            ts_r = gyro_ts[j + 1]
            if ref_ts - ts_l > ts_r - ref_ts:
                j += 1

        # Align Mag
        if mag is not None:
            while (k + 1 < mag_num
                   and mag_ts[k + 1] < ref_ts + SENSOR_TIMESTAMP_NS_ERROR_MAX):
                k += 1
            if find_closest and k + 1 < mag_num:
                ts_l = mag_ts[k]
                ts_r = mag_ts[k + 1]
                if ref_ts - ts_l > ts_r - ref_ts:
                    k += 1

            # Reconstruct
            aligned.append(np.concatenate((acc[i], gyro[j, 2:5], mag[k, 2:5])))
        else:
            # Reconstruct
            aligned.append(np.concatenate((acc[i], gyro[j, 2:5])))

        # Update reference timestamp
        if align_to_ref_ts:
            ref_ts += sample_period
            if (i + 1 < acc_num):
                if acc_ts[i + 1] < ref_ts + SENSOR_TIMESTAMP_NS_ERROR_MAX:
                    # Avoid Sample period drift, so based on acc timestamp
                    i += 1
                    ref_ts = acc_ts[i]
                else:
                    # Means accelerometer is mssing data
                    pass
            else:
                break
        else:
            # Align to acc timestamp
            i += 1
            if i < acc_num:
                ref_ts = acc_ts[i] + SENSOR_TIMESTAMP_NS_ERROR_MAX

    return np.asarray(aligned)


def plot_aligned_data(x1, x2, name1='x1', name2='x2'):
    res = np.correlate(x1, x2, 'full')
    idx = np.argmax(res) - len(x2) + 1
    print(f'x1 len: {x1.shape[0]}, x2 len: {x2.shape[0]}, idx: {idx}')
    # plt.plot(res)
    plt.plot(range(len(x1)), x1, '-or', label=name1, markersize=3)
    plt.plot(np.arange(len(x2)) + idx, x2, '-og', label=name2, markersize=3)
    plt.legend(loc='lower right')


def align_one_record(record_dir: Path, debug=False):
    acc_file = record_dir / f'{record_dir.name}-{ACC_SUFFIX}'
    gyro_file = record_dir / f'{record_dir.name}-{GYRO_SUFFIX}'
    magnet_file = record_dir / f'{record_dir.name}-{MAGNET_SUFFIX}'
    # print(f'Acc file: {acc_file}')
    # print(f'Gyro file: {gyro_file}')
    # print(f'Magnet file: {magnet_file}')

    if acc_file.exists() and gyro_file.exists() and magnet_file.exists():
        try:
            acc = pd.read_csv(acc_file).values
            gyro = pd.read_csv(gyro_file, ).values
            gyro = gyro[:-1]
            mag = pd.read_csv(magnet_file).values
            mag = mag[:-1]
        except pd.errors.ParserError as e:
            print(f'Error: {e}')
            return None
        acc = pd.read_csv(acc_file).values
        gyro = pd.read_csv(gyro_file).values
        mag = pd.read_csv(magnet_file).values

        aligned = align_sensor_data(acc, gyro, mag)
        if debug:
            plt.figure('Align check: axis x')
            plt.title(record_dir.name)
            ax1 = plt.subplot(311)
            plt.title('acc x')
            plot_aligned_data(acc.T[2], aligned.T[2], 'origin', 'aligned')
            plt.subplot(312, sharex=ax1)
            plt.title('gyro x')
            plot_aligned_data(gyro.T[2], aligned.T[5], 'origin', 'aligned')
            plt.subplot(313, sharex=ax1)
            plt.title('mag x')
            plot_aligned_data(mag.T[2], aligned.T[8], 'origin', 'aligned')
            plt.show()

        return aligned
    else:
        return None


def align_and_relabel_one_record(record_dir: Path,
                                 dst_dir: Path,
                                 sample_rate=26,
                                 debug=False):
    dst_file = None
    if dst_dir is not None:
        activity_type = get_activity_type_name_by_record_name(record_dir.name)
        dst_file = dst_dir / f'{activity_type}-{record_dir.name}.csv'
    # if dst_file is not None and dst_file.exists():
    #     print('Record processed, skipping...')
    #     return

    # print(f'\nProcessing record: {record_dir}')
    label_file = record_dir / f'{record_dir.name}-{LABEL_SUFFIX}'
    if not label_file.exists():
        print('Label file not exists, skipped!')
        return None

    aligned = align_one_record(record_dir, debug)
    if aligned is None:
        return None
    if sample_rate == 52:
        pass
    elif sample_rate == 26:
        aligned = aligned[::2]
    else:
        raise RuntimeError('Unsupported sample rate, must be one of [26, 52]')
    aligned_ts = aligned[:, 1]

    # Load label result
    labels_ts = load_label_result(label_file)
    labels_index = label_convert_ts2index(labels_ts, aligned_ts)

    y = np.zeros([aligned.shape[0], 1])
    for activity_type, start_idx, end_idx in labels_index:
        y[start_idx:end_idx] = activity_type

    labeled = np.hstack((aligned, y))
    df = pd.DataFrame(data=labeled, columns=HEADER_NAMES)
    try:
        df = df.astype(HEADER_NAMES_TYPE)
    except ValueError as e:
        print(f'Convert type error: {e}')
        return

    if dst_file is not None:
        activity_type = get_activity_type_name_by_record_name(record_dir.name)
        dst_file = dst_dir / f'{activity_type}-{record_dir.name}.csv'
        print(f'Saving to file: {dst_file}')
        df.to_csv(dst_file, index=False)


def align_and_relabel_datasets(data_dir: Path,
                               save_dir: Path,
                               dataset_names,
                               sample_rate=26):
    datasets = [data_dir / name for name in dataset_names]
    set_num = len(datasets)
    for i, dataset in enumerate(datasets, 1):
        print(f'\nProcess dataset [{i:02d}/{set_num:02d}]: {dataset.name}')
        scenes = [r for r in dataset.iterdir() if r.is_dir()]
        scene_num = len(scenes)
        dst_dir = save_dir / dataset.name
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)
        for j, scene in enumerate(scenes):
            print(f'\nProcess scene [{j:02d}/{scene_num:02d}]: {scene.name}')
            records = [r for r in scene.iterdir() if r.is_dir()]
            record_num = len(records)
            type_dir = dst_dir / scene.name
            if not type_dir.exists():
                type_dir.mkdir(parents=True)
            for k, record in enumerate(records, 1):
                print(f'\nProcessing record [{k:02d}/{record_num:02d}]: '
                      f'{record.name}')
                align_and_relabel_one_record(record, type_dir, sample_rate)


@click.command()
@click.argument('data-dir')
@click.option('-s', '--save-dir')
def main(data_dir, save_dir):
    if save_dir is not None:
        save_dir = Path(save_dir)
        align_and_relabel_datasets(Path(data_dir), save_dir, DATASET_TO_USE)
    else:
        print('Must set save dir by "-s or --save-dir"')


if __name__ == "__main__":
    main()
