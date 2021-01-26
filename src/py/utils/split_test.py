#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-18

import json
from pathlib import Path
from random import sample, seed
import subprocess

import click

DATA_SETS_TO_USE = ['MIOT-2020Q4-relabeled']


def get_user_id_by_record_name(record_name):
    return record_name.split('-')[2]


def split_test_set(dataset_dir: Path, dataset_names, test_size=0.3):
    total_num = 0
    test_num = 0
    dataset_users_dict = {}
    seed(17)
    for name in dataset_names:
        src_dir = dataset_dir / name
        dst_dir = dataset_dir / f'{name}-test'
        print(f'\nProcessing dataset: {src_dir}')
        dataset_users_dict[name] = {}
        for type_dir in filter(Path.is_dir, src_dir.iterdir()):
            print(f'\nPocessing type dir: {type_dir}')
            target_dir = dst_dir / type_dir.name
            if not target_dir.exists():
                print(f'Creating directory: {target_dir}')
                target_dir.mkdir(parents=True)
            user_records_dict = {}
            for record in type_dir.glob('*.csv'):
                total_num += 1
                user_id = get_user_id_by_record_name((record.stem))
                if user_id in user_records_dict:
                    user_records_dict[user_id].append(record)
                else:
                    user_records_dict[user_id] = [record]
            all_users = user_records_dict.keys()
            dataset_users_dict[name][type_dir.name] = len(all_users)
            total_user_num = len(all_users)
            test_user_num = int(total_user_num * test_size)
            if test_user_num == 0 and total_user_num > 1:
                test_user_num = 1
            test_users = sample(all_users, test_user_num)

            print(f'Sampling: {type_dir}, '
                  f'take user {test_user_num} / {total_user_num}')
            print(f'Move test record to target: {target_dir}')
            for user in test_users:
                for record in user_records_dict[user]:
                    test_num += 1
                    print(f'Moving {record}')
                    subprocess.call(f'mv {record} {target_dir}', shell=True)
    print(json.dumps(dataset_users_dict, indent=2, sort_keys=True))
    print(f'Total records: {total_num}, train: {total_num - test_num}, '
          f'test: {test_num}')


@click.command()
@click.argument('dataset-dir')
def main(dataset_dir):
    split_test_set(Path(dataset_dir), DATA_SETS_TO_USE)


if __name__ == "__main__":
    main()
