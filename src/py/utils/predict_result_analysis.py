#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-01-26

from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import accuracy_score


def process_results(results_dir: Path):
    res = {'FileName': [], 'Accuracy': []}
    for f in results_dir.glob('*.csv'):
        try:
            df = pd.read_csv(f, usecols=['Predict', 'ActivityCategory'])
        except ValueError as e:
            print(e)
            continue

        y_true = df['ActivityCategory'].values
        y_pred = df['Predict'].values
        accuracy = accuracy_score(y_true, y_pred)
        res['FileName'].append(f.name)
        res['Accuracy'].append(accuracy)

    df = pd.DataFrame.from_dict(res)
    df = df.sort_values('Accuracy', ascending=True)
    print(df.head(10))
    df.to_csv('./accuracy_analysis_results.csv')


@click.command()
@click.argument('results-dir')
def main(results_dir):
    process_results(Path(results_dir))


if __name__ == '__main__':
    main()
