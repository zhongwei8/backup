#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-01-21

from pathlib import Path
import subprocess

import click

project_parent_dir = (Path(__file__).parent / '../../').resolve()
# Add project parent directory to python path for import issue
# sys.path.append(project_parent_dir)


def run(config_file):
    # Check evaluation environments
    # ai-algorithm-depolyment repo is reuiqred
    # And should be checked out in the same directory as this repo
    depolyment_dir = project_parent_dir / 'ai-algorithm-depolyment'
    if not depolyment_dir.exists():
        print('Evaluation depend on repo: ai-algorithm-depolyment')
        print(
            f'You need to checkout the repo on directory: {project_parent_dir}'
        )
        print(
            f'Or use command:\n\ngit clone git@git.n.xiaomi.com:miot-algorithm/ai-algorithm-depolyment.git {depolyment_dir}'
        )
        exit(1)

    evaluate_py = depolyment_dir / 'run.py'
    subprocess.call(f'python3 {evaluate_py} --config {config_file}',
                    shell=True)


@click.command()
@click.argument('config-file')
def main(config_file):
    run(config_file)


if __name__ == "__main__":
    main()
