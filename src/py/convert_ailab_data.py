#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-31

from pathlib import Path

import click
import pandas as pd

from activity_data_labeler import LABEL_ITEMS

ORIGIN_LABEL_NAMES = [
    "None", "WalkOutSide", "WalkInDoor", "RuningOutSide", "InCar", "Lying",
    "Biking", "Sitting", "Upstairs", "Downstairs", "Standing", "Driving",
    "RuningInDoor", "RowingMachine", "EllipticalMachine", "Swimming", "Others",
    "RopeJum", "Falling"
]

LABEL_NAMES_CONVERT_MAP = {
    "None": 'Undefined',
    "WalkOutSide": 'SlowWalk',
    "WalkInDoor": 'SlowWalk',
    "RuningOutSide": 'RuningOutSide',
    "InCar": 'CarTaking',
    "Lying": 'Lying',
    "Biking": 'BikingOutSide',
    "Sitting": 'Sitting',
    "Upstairs": 'Upstairs',
    "Downstairs": 'Downstairs',
    "Standing": 'Standing',
    "Driving": 'Driving',
    "RuningInDoor": 'RuningInDoor',
    "RowingMachine": 'RowingMachine',
    "EllipticalMachine": 'EllipticalMachine',
    "Swimming": 'SwimmingInDoor',
    "Others": 'DailyActivity',  # Not sure
    "RopeJum": 'RopeJump',
    "Falling": 'DailyActivity',  # Negative case for walking
}

LABEL_CONVERT_MAP = {
    i: LABEL_ITEMS.index(LABEL_NAMES_CONVERT_MAP[k])
    for i, k in enumerate(ORIGIN_LABEL_NAMES)
}

OLD_HEADER_NAMES = [
    'CurrentTimestamp', 'EventTimestamp', 'AccelX', 'AccelY', 'AccelZ',
    'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Activity'
]

NEW_HEADER_NAMES = [
    'CurrentTimeMillis', 'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ',
    'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Activity'
]


def process_one_file(src_path: Path, target_dir: Path, down_sample):
    dst_path = target_dir / src_path.name
    print(f'\nProcessing origin file: {src_path}')
    print(f'Saving to file: {dst_path}')
    df = pd.read_csv(src_path, header=1)
    if 'CurrentTimestamp' in df.columns:
        print('Rename CurrentTimestamp to CurrentTimeMillis')
        df.rename(columns={'CurrentTimestamp': 'CurrentTimeMillis'},
                  inplace=True)
        df['CurrentTimeMillis'] = df['CurrentTimeMillis'].astype(int)
    if 'EventTimestamp' in df.columns:
        print('Rename EventTimestamp to EventTimestamp(ns)')
        df.rename(columns={'EventTimestamp': 'EventTimestamp(ns)'},
                  inplace=True)
        df['EventTimestamp(ns)'] = df['EventTimestamp(ns)'].astype(int)
    df['Activity'] = df['Activity'].astype(int)
    # Convert timestamp from ms to ns
    df['EventTimestamp(ns)'] = df['EventTimestamp(ns)'] * 1000_000

    if down_sample:
        print('Applying 1/2 down sample')
        df = df[::2]

    masks = []
    for i in range(len(ORIGIN_LABEL_NAMES)):
        masks.append(df['Activity'] == i)
    for i, mask in enumerate(masks):
        df.loc[mask, 'Activity'] = LABEL_CONVERT_MAP[i]
    # print(df.head(5))

    df.to_csv(dst_path, index=False)


@click.command()
@click.argument('data-dir')
@click.argument('target-dir')
@click.option('-d', '--down-sample', is_flag=True, help='Downsample to 26Hz')
def main(data_dir, target_dir, down_sample):
    data_dir = Path(data_dir)
    target_dir = Path(target_dir)
    if data_dir.exists():
        for f in data_dir.rglob('*.csv'):
            if f.is_file():
                dst_dir = target_dir / f.parent.relative_to(data_dir)
                if not dst_dir.exists():
                    dst_dir.mkdir(parents=True)
                process_one_file(f, dst_dir, down_sample)


if __name__ == "__main__":
    main()
