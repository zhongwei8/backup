#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-02-03

from pathlib import Path
import subprocess


def main():
    current_dir = Path(__file__).parent.resolve()
    test_main_path = current_dir / '../test/py/har_model_test.py'
    subprocess.call(f'python3 {test_main_path}', shell=True)


if __name__ == '__main__':
    main()
