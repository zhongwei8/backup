#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2021-02-23

from pathlib import Path

import click
import matplotlib.pyplot as plt
from matplotlib import ticker

import datetime

import numpy as np
import pandas as pd

from indoor_outdoor_recognizer import MIIndoorOutdoorRecognizer


def indoor_outdoor_vote(status_arr):
    status, counts = np.unique(status_arr, return_counts=True)
    print(status)
    print(counts)


def plot_results(res: pd.DataFrame, data_file: Path, is_show: bool,
                 is_save_png: bool):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))

    axes[0].plot(res['update_ts'], res['status'], label='status', marker='o')
    axes[0].set_title(Path(data_file).name)
    axes[0].legend()

    axes[1].plot(res['update_ts'], res['num'], label='num', marker='o')
    axes[1].legend()

    axes[2].plot(res['update_ts'], res['snr_sum'], label='snr_sum', marker='o')
    axes[2].legend()

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    axes[0].set_yticks([-0.5, 0, 1, 2, 2.5])
    axes[0].set_yticklabels(['', 'undefine', 'indoor', 'outdoor', ''])

    def format_func(x_tick, pos=None):
        dt = datetime.datetime.fromtimestamp(x_tick // 1000)
        return dt.strftime("%H:%M:%S")

    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    # axes[2].xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))

    if is_save_png:
        data_file_str = str(data_file)
        sport = data_file_str.split('_')[-1]
        sport = sport.split('-')[0]
        png_dir = Path(f'./src/py/{sport}')
        if not png_dir.exists():
            png_dir.mkdir()
        png_name = data_file.stem
        png_path = png_dir / png_name
        plt.savefig(png_path, dpi=500)

    if is_show:
        plt.show()
    plt.close('all')


def process_file(data_file: Path):
    res = []
    with data_file.open('r') as f:
        nmea_sentences = f.readlines()

        # Remove the header
        nmea_sentences = nmea_sentences[1:]

        recognizer = MIIndoorOutdoorRecognizer()
        current_status = 0
        satel_num = 0
        satel_snr_sum = 0
        for i, nmea_sentence in enumerate(nmea_sentences):
            # print(f'Processing nmea sentence {i}: {nmea_sentence}')
            update = recognizer.process_with_raw_nmea_sentence(nmea_sentence)
            if update:
                update_ts = recognizer.get_update_timestamp()
                current_status = recognizer.get_status()
                satel_num, satel_snr_sum = recognizer.get_satellite_status()
                res.append(
                    [update_ts, current_status, satel_num, satel_snr_sum])
    df = pd.DataFrame(res, columns=['update_ts', 'status', 'num', 'snr_sum'])
    # df['update_ts'] = pd.to_datetime(df['update_ts'], unit = 'ms')
    return df


def summary_nmea_file(data_file: Path,
                      plot: bool,
                      outdoor=False,
                      force=False,
                      is_save_png=False,
                      is_show=False):
    res = process_file(data_file)

    status = res['status'].values

    undefine_cnt = np.sum(status == 0)
    indoor_cnt = np.sum(status == 1)
    outdoor_cnt = np.sum(status == 2)

    cnt = undefine_cnt + indoor_cnt + outdoor_cnt

    if cnt == 0:
        return (data_file.name, 0, 0, 0, 0, np.nan, np.nan, np.nan)

    undefine_rate = undefine_cnt / cnt
    indoor_rate = indoor_cnt / cnt
    outdoor_rate = outdoor_cnt / cnt

    plot_results(
        res,
        data_file,
        is_show=is_show,
        is_save_png=is_save_png,
    )

    return (data_file.name, undefine_cnt, indoor_cnt, outdoor_cnt, cnt,
            undefine_rate, indoor_rate, outdoor_rate)


def summary_nmea_files(dir: Path,
                       plot: bool,
                       outdoor: bool,
                       force=False,
                       is_save_png=False,
                       is_show=False):
    dir_name = str(dir).replace('/', '_')
    columns = [
        'nmea_file', 'undefine_cnt', 'indoor_cnt', 'outdoor_cnt', 'cnt',
        'undefine_rate', 'indoor_rate', 'outdoor_rate'
    ]
    summarys_path = f'~/桌面/{dir_name}_indoor_outdoor_recognizer_summary.csv'
    summarys = []
    nmea_paths = [nmea_path for nmea_path in dir.rglob('*nmea.csv')]
    for nmea_path in nmea_paths:
        print(f'Processing: {nmea_path}')
        summary = summary_nmea_file(nmea_path,
                                    plot=plot,
                                    outdoor=outdoor,
                                    force=force,
                                    is_save_png=is_save_png,
                                    is_show=is_show)
        summarys.append(summary)
    data = pd.DataFrame(summarys, columns=columns)
    data.to_csv(summarys_path, index=False)


def check_nmea_file(nmea_name: str, dir: Path):
    nmea_path = [path for path in dir.rglob(nmea_name)]
    if (nmea_path == []):
        return
    data_file = nmea_path[0]
    print(f'Processing file: {data_file}')
    res = process_file(Path(data_file))
    plot_results(res)
    indoor_outdoor_vote(res['status'].values)


def check_indoor_results(result_path, dir: Path):
    data = pd.read_csv(result_path)
    values = data[['nmea_file', 'indoor_rate']].values
    for nmea_file, indoor_rate in values:
        if (indoor_rate > 0.1):
            check_nmea_file(nmea_file, dir)


@click.command()
@click.argument('data-file')
@click.option('--plot/--no-plot', default=False)
@click.option('--outdoor/--no-outdoor', default=False)
@click.option('--force/--no-force', default=False)
@click.option('-s', is_flag=True)
@click.option('--show/--no-show', default=False)
def main(data_file, plot, outdoor, force, s, show):
    data_file = Path(data_file)
    if data_file.is_file():
        summary_nmea_file(data_file,
                          plot=plot,
                          outdoor=outdoor,
                          force=force,
                          is_save_png=s,
                          is_show=show)
    elif data_file.is_dir():
        summary_nmea_files(data_file,
                           plot=plot,
                           outdoor=outdoor,
                           force=force,
                           is_save_png=s,
                           is_show=show)


if __name__ == '__main__':
    main()
