#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-11

from pathlib import Path
import subprocess

import click


def delete(file_path, fds_suffix='fds://new-sensor-bucket'):
    print(f'\nProcessing: {file_path}')
    f = f'{fds_suffix}/{(file_path)}'
    print(f'Deleting: {f}')
    subprocess.run(['fdscli', 'rm', f])


@click.command()
@click.argument('file-path')
@click.argument('fds-suffix')
def main(file_path, fds_suffix):
    file_path = Path(file_path)
    for f in file_path.rglob('*'):
        if f.is_file():
            delete(f.relative_to(file_path), fds_suffix)


if __name__ == "__main__":
    main()
