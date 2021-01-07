#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-18

from pathlib import Path
from random import sample
import subprocess

import click


DATA_SETS_TO_USE = [
    'all'
]


def split_test_set(dataset_dir: Path, dataset_names, test_size=0.2):
    for name in dataset_names:
        src_dir = dataset_dir / name
        dst_dir = dataset_dir / f'{name}-test'
        print(f'\nProcessing dataset: {src_dir}')
        for type_dir in filter(Path.is_dir, src_dir.iterdir()):
            print(f'\nPocessing type dir: {type_dir}')
            target_dir = dst_dir / type_dir.name
            if not target_dir.exists():
                print(f'Creating directory: {target_dir}')
                target_dir.mkdir(parents=True)
            all_records = [record_dir for record_dir in filter(Path.is_dir, type_dir.iterdir())]
            total_num = len(all_records)
            test_num = int(total_num * test_size)
            test_records = sample(all_records, test_num)
            print(f'Sampling: {type_dir}, take {test_num} / {total_num}')
            print(f'Move test record to target: {target_dir}')
            for record_dir in test_records:
                print(f'Moving {record_dir}')
                subprocess.call(f'mv {record_dir} {target_dir}', shell=True)


@click.command()
@click.argument('dataset-dir')
def main(dataset_dir):
    split_test_set(Path(dataset_dir), DATA_SETS_TO_USE)


if __name__ == "__main__":
    main()
