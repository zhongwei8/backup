#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : farmer
# @Date : 2020-11-09

from enum import Enum
from enum import unique
import json
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PySimpleGUI as sg

from utils.data_labeler import DataLabeler
from utils.data_labeler import merge_labels
from utils import log

logger = log.create_logger('ActiDataLabeler')

SENSOR_DATA_NAMES = ['type', 'ts', 'x', 'y', 'z', 'xe', 'ye', 'ze']
SENSOR_DATA_HEADER = [
    'CurrentTimeMillis', 'EventTimestamp(ns)', 'accel_x', 'accel_y', 'accel_z'
]
CTM_HEADER_NAME = 'CurrentTimeMillis'
TS_HEADER_NAME = 'EventTimestamp(ns)'
TSS = [CTM_HEADER_NAME, TS_HEADER_NAME]

# !! In order to be compatible with historical annotations, please add the type at the end instead of inserting
# Refer to list: https://xiaomi.f.mioffice.cn/docs/dock4H2cyIYjwEGAnMmFH5GOcTc#
LABEL_ITEMS = (
    'Undefined',  # 0
    'Static',  # 1
    'DailyActivity',  # 2, Such as Working/Writing/Chatting
    'OtherSports',  # 3, Larger motions, such as Fitness/Playing ball/Washing
    'BriskWalkInDoor',  # 4
    'BriskWalkOutSide',  # 5
    'RuningInDoor',  # 6
    'RuningOutSide',  # 7
    'BikingInDoor',  # 8
    'BikingOutSide',  # 9
    'EBikingOutSide',  # 10
    'MotoBikingOutSide',  # 11
    'SwimmingInDoor',  # 12
    'SwimmingOutSide',  # 13
    'SubwayTaking',  # 14
    'BusTaking',  # 15
    'CarTaking',  # 16
    'CoachTaking',  # 17
    'Sitting',  # 18
    'Standing',  # 19
    'Lying',  # 20
    'EllipticalMachine',  # 21
    'RowingMachine',  # 22
    'Upstairs',  # 23
    'Downstairs',  # 24
    'Driving',  # 25
    'RopeJump',  # 26
    'SlowWalk'  # 27
)

LABEL_DAILY = [
    'DailyActivity', 'SubwayTaking', 'BusTaking', 'CarTaking', 'CoachTaking',
    'Sitting', 'Standing', 'Lying', 'Driving'
]

LABEL_OTHER_SPORTS = ['OtherSports', 'RopeJump']

LABEL_ITEMS_INDEX_DICT = dict(zip(LABEL_ITEMS, list(range(len(LABEL_ITEMS)))))
LABEL_RESULT_FILE_NAME = Path('activity_label_result.csv')

all_files_labels = {}
stat_results = {}


@unique
class MetaType(Enum):
    DATE_TIME = 0
    ACTIVITY_TYPE = 1


def label_convert_ts2index(labels_ts, ts_arr):
    labels_index = []
    max_idx = len(ts_arr) - 1
    # print(f'TS range: {ts_arr[0]} - {ts_arr[-1]}, Total len: {max_idx + 1}')
    for t, s, e in labels_ts:
        # print(f'Finding TS label {t}: {s} - {e}')
        if s > ts_arr[-1] or e < ts_arr[0]:
            # print(f'TS not match: {s} > {ts_arr[-1]} or {e} < {ts_arr[0]}')
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
        # print(f'Found INDEX label {t}: {start_index} - {end_index}')
        labels_index.append((t, start_index, end_index))
    return labels_index


def load_label_result(label_file: Path):
    labels = []
    with label_file.open('r') as f:
        for line in f:
            line = line.rstrip('\n')
            labels.append(tuple(map(int, (map(float, line.split('_'))))))
    return labels


def get_meta_from_file_name(file_name: str,
                            meta_type: MetaType = MetaType.ACTIVITY_TYPE):
    metas = file_name.split('-')
    if MetaType.ACTIVITY_TYPE == meta_type:
        return metas[7].split('_')[1]
    else:
        return ""


def show_relabel_hint(acc_file: Path):
    check_one_file(acc_file)
    res = sg.popup_ok_cancel('已经标注过，是否重新标注？')
    if res == 'Cancel':
        relabel = False
    else:
        relabel = True

    return relabel


def label_one_file(file_path: Path,
                   force=False,
                   result_file: Path = None,
                   label_items=LABEL_ITEMS):
    """
    Label SensorData file that generated by SensorDataCollector app
    Data contents looks like:
    CurrentTimeMillis,EventTimestamp(ns),accel_x,accel_y,accel_z
    1605600680625,5431547485983,0.2299057,4.0800943,10.7349
    ...
    """
    global all_files_labels
    logger.info(f'Processing file: {file_path}')
    file_name = file_path.stem

    # Result label file check
    file_name_prefix = '-'.join(file_name.split('-')[:-2])
    label_file = file_path.parent / f'{file_name_prefix}-label-result.csv'
    logger.info(f'Label result file path: {label_file}')
    labels_ts = []
    if label_file.exists():
        logger.warning('Label result file exists')
        if not force:
            logger.warning('##Skipped')
            return
        labels_ts = load_label_result(label_file)

    # Load data
    df = pd.read_csv(file_path, index_col=False, comment='#')
    ts = df['EventTimestamp(ns)'].values
    utc_ts = df['CurrentTimeMillis'].values
    labels_idx = label_convert_ts2index(labels_ts, ts)
    acc = df.drop(columns=['CurrentTimeMillis', 'EventTimestamp(ns)']).values
    guess_type_name = get_meta_from_file_name(file_name)
    guess_type = LABEL_ITEMS_INDEX_DICT.get(guess_type_name, 0)
    logger.debug(f'Current guess type name: {guess_type_name}:{guess_type}')

    labeler = DataLabeler(label_items)
    labels = labeler.process(utc_ts[0],
                             ts,
                             acc,
                             file_name,
                             selected=guess_type,
                             labels=labels_idx)
    labels = merge_labels(labels)
    if len(labels) > 0:
        all_files_labels[file_name] = labels
        # Write to record fold label result file
        with label_file.open('w+') as f_label:
            for v, s, e in labels:
                f_label.write(f'{v}_{ts[s]}_{ts[e]}\n')
        # Write to global result file
        if result_file is not None:
            with result_file.open('a+') as f_result:
                f_result.write(file_name)
                for v, s, e in labels:
                    f_result.write(f',{v}_{ts[s]}_{ts[e]}')
                f_result.write('\n')


def label_dir(src_dir: Path, force=False, result_file: Path = None):
    if src_dir.is_dir():
        logger.info(f'\nProcessing directory: {src_dir}')
        acc_files = [f for f in src_dir.rglob('*accel-52HZ.csv')]
        total = len(acc_files)
        print(f'Total quantity of files to label: {total}')
        for i, f in enumerate(acc_files, 1):
            print(f'\nProgress [{i}/{total}]')
            label_one_file(f, force, result_file)


def stat_one_file(acc_file: Path):
    global stat_results
    label_file = acc_file.with_name(f'{acc_file.parent.name}-label-result.csv')
    metas = acc_file.stem.split('-')
    activity_type_name = metas[7].split('_')[1]
    is_daily = activity_type_name in LABEL_DAILY
    is_other_sports = activity_type_name in LABEL_OTHER_SPORTS
    if is_daily or is_other_sports:
        if is_daily:
            type_name = 'DailyActivity'
        else:
            type_name = 'OtherSports'
        print(f'stat {activity_type_name} as {type_name}')
        df = pd.read_csv(acc_file)
        ts = df[TS_HEADER_NAME].values
        duration = (ts[-1] - ts[0]) / 1e9 / 60
        if type_name in stat_results:
            stat_results[type_name] += duration
        else:
            stat_results[type_name] = duration
    elif label_file.exists():
        with label_file.open('r') as f:
            for line in f:
                metas = line.rstrip('\n').split('_')
                activity_type = int(metas[0])
                type_name = LABEL_ITEMS[activity_type]
                start_ts = int(metas[1])
                end_ts = int(metas[2])
                duration = (end_ts - start_ts) / 1e9 / 60

                if type_name in stat_results:
                    stat_results[type_name] += duration
                else:
                    stat_results[type_name] = duration
    else:
        print(f'!!Unlabeled record: {acc_file.parent}')


def check_result_file(result_file: Path, force):
    if result_file.exists():
        if force:
            os.remove(result_file)
        else:
            logger.error(
                'Label result file exists. To force override use option "-f".')
            exit(1)


def check_one_file(acc_file: Path, return_fig=False):
    print(f'\nChecking file: {acc_file}')
    label_file = acc_file.with_name(f'{acc_file.parent.name}-label-result.csv')
    df = pd.read_csv(acc_file)
    ts = df[TS_HEADER_NAME].values
    data = df.drop(columns=TSS).values

    labels_idx = None
    has_label = label_file.exists()
    if has_label:
        labels_ts = load_label_result(label_file)
        labels_idx = label_convert_ts2index(labels_ts, ts)

    fig = plt.figure('Check Label Result')
    plt.title(acc_file.parent.name)
    plt.plot(data)
    ax = plt.gca()
    y_value = np.max(data) + 1
    if has_label:
        for style, start_idx, end_idx in labels_idx:
            x_idx = (start_idx + end_idx) // 2
            plt.axvspan(start_idx, end_idx, alpha=0.5)
            plt.text(x_idx,
                     y_value,
                     LABEL_ITEMS[style],
                     horizontalalignment='center',
                     verticalalignment='center')
    else:
        plt.text(0.5,
                 0.9,
                 acc_file.name.split('-')[7].split('_')[1],
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes)
    if return_fig:
        return fig
    else:
        plt.show()


@click.command()
@click.argument('file-path')
@click.option('-f',
              '--force',
              is_flag=True,
              help='Force generate label result file')
@click.option('-t',
              '--statistics',
              is_flag=True,
              help='Statistics the label result')
@click.option('-c', '--check', is_flag=True, help='Check the label result')
def main(file_path, force, statistics, check):
    global stat_results
    file_path = Path(file_path)
    if statistics:
        logger.info('Do statistics...\n')
        record_num = 0
        if file_path.is_file():
            stat_one_file(file_path)
        elif file_path.is_dir():
            for f in file_path.rglob('*accel-52HZ.csv'):
                if f.is_file():
                    record_num += 1
                    stat_one_file(f)
        # Print the result
        print(f'Statistics file num: {record_num}')
        print(json.dumps(stat_results, sort_keys=True, indent=4))
    elif check:
        if file_path.is_file():
            check_one_file(file_path)
        elif file_path.is_dir():
            for f in file_path.rglob('*accel-52HZ.csv'):
                if f.is_file():
                    check_one_file(f)
    else:
        logger.info('Perform labeling...')
        if file_path.is_file():
            label_one_file(file_path, force)
        elif file_path.is_dir():
            label_dir(file_path, force)


if __name__ == '__main__':
    main()
