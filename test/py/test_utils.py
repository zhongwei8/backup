#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-02-02

import hashlib
from pathlib import Path
import zipfile

import six


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def unzip_file(fz_name, path):
    flag = False
    if zipfile.is_zipfile(fz_name):  # 检查是否为zip文件
        with zipfile.ZipFile(fz_name, 'r') as zipf:
            zipf.extractall(path)
            flag = True
    return flag


def get_dataset(dataset_path, dataset_name, tmp_dir='tmp/', sha256sum=None):
    dataset_dir = dataset_path
    tmp_dir = Path(tmp_dir)
    if dataset_path.startswith('http://') or \
            dataset_path.startswith('https://'):
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)
        dataset_file = tmp_dir / f'{dataset_name}.zip'
        need_download = True
        print('Downloading dataset, please wait ...')
        if dataset_file.exists() and sha256sum is not None:
            print('Dataset exists, checking sha256sum ...')
            sha256 = sha256_checksum(dataset_file)
            if sha256 != sha256sum:
                print(f'Check sum not equal to expected: {sha256}')
                print('Downloading from remote ...')
            else:
                print('Dataset up-to-date, no need to download')
                need_download = False
        if need_download:
            six.moves.urllib.request.urlretrieve(dataset_path, dataset_file)
        print('Dataset downloaded successfully.')
        # Check the sha1sum
        if sha256sum is not None:
            print(f'Checking sha256sum expected: {sha256sum}')
            sha256 = sha256_checksum(dataset_file)
            if sha256 != sha256sum:
                print(f'Check sum not equal to expected: {sha256}')
                raise RuntimeError('Dataset sha256sum check failed')
        dataset_dir = tmp_dir / dataset_name
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True)
        if unzip_file(dataset_file, dataset_dir):
            print(f'unzip dataset to {dataset_dir} successfully')
        else:
            raise Exception(f'unzip dataset to {dataset_dir} failed')
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise Exception(f'Invalid input dataset dir: {dataset_dir}')
    return Path(dataset_dir)
