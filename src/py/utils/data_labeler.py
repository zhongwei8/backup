#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: farmer
# @Date: 2020-04-14

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import widgets
from matplotlib.widgets import Cursor
import numpy as np
import PySimpleGUI as sg
from scipy import signal


FILTER_B, FILTER_A = signal.butter(5, 0.2, 'lowpass', output='ba')


class DataLabel:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    def __str__(self):
        return f'{self.label}:{self.start}-{self.end}'


class DataLabeler:
    def __init__(self, label_items, label_type='index'):
        self.selected = 0
        self.labels = []
        self.label_items = label_items
        self.label_by_index = label_type == 'index'
        self.data_len = 0
        self.label_ax = None
        self.label_vspans = []

    def popup_label_window(self,
                           start,
                           end,
                           location=(400, 300)):
        activity_names = self.label_items
        activity_type_name = activity_names[self.selected]

        name_size = (15, 1)
        layout = [[
            sg.Text('Start Point', size=name_size),
            sg.Text(f'{start}', key='start_point')
        ], [
            sg.Text('End Point', size=name_size),
            sg.Text(f'{end}', key='end_point')
        ], [
            sg.Text('Activity Type', size=name_size),
            sg.Listbox(values=activity_names,
                       default_values=activity_type_name,
                       size=(20, 12),
                       key='selected-type',
                       enable_events=False)
        ], [
            sg.Button('Submit', key='btn_submit'),
            sg.Button('Cancel', key='btn_cancel')
        ]]

        window = sg.Window('Label Window', layout, location=location)

        while True:
            event, values = window.read()
            print(f'event: {event}')
            if event == sg.WIN_CLOSED or event in ('Exit', 'btn_cancel'):
                print('Give up current label')
                activity_type_name = ''
                break
            if event == 'btn_submit':
                activity_type_name = values['selected-type'][0]
                break

        window.close()

        return activity_type_name

    def onselect(self, v_min, v_max):
        # print(f'Processing selected range: {v_min} - {v_max}')
        if self.label_vspans:
            for span in self.label_vspans:
                span.remove()
                plt.draw()
            self.label_vspans = []
        start = int(v_min)
        end = int(v_max)
        if start < 0:
            start = 0
        if end > self.data_len:
            end = self.data_len - 1
        if end <= start:
            print('Start point grater than end point, do nothing')
            return
        label_name = self.popup_label_window(start, end)
        label_index = -1
        if label_name in self.label_items:
            label_index = self.label_items.index(label_name)
            if self.label_by_index:
                self.labels.append([label_index, start, end])
            else:
                self.labels.append([label_name, start, end])
            print(f'Labeling points: {start:^7d} - {end:^7d} '
                  f'as type: {label_index}:{label_name}')
            self.label_ax.axvspan(start, end, alpha=0.5)
        else:
            print(f'Give up label: {start} - {end}')

    def process(self, utc_base, ts, acc, file_name, selected=0, labels=None):
        self.labels = []
        self.selected = selected
        data_len = len(acc)
        self.data_len = data_len
        aligned_utc = ts.copy()
        aligned_utc[0] = utc_base
        for i, t in enumerate(ts[1:], 1):
            aligned_utc[i] = aligned_utc[0] + (t - ts[0]) // 1e6
        utc_date = []
        for t in aligned_utc:
            utc_date.append(
                datetime.fromtimestamp(t / 1000).strftime('%m-%d %H:%M:%S'))
        print(f'UTC date from {utc_date[0]} to {utc_date[-1]}')

        print(f'Acc data shape: {acc.shape}')
        acc_lp = signal.filtfilt(FILTER_B, FILTER_A, acc, axis=0)
        mag = np.linalg.norm(acc, axis=1)
        mag_lp = np.linalg.norm(acc_lp, axis=1)
        print(f'Acc magnitude shape: {mag.shape}')

        # fig, ax = plt.subplots()
        plt.figure(file_name, figsize=(15, 8))
        plt.title(file_name)
        plt.xticks(rotation=0)
        plt.locator_params(nbins=8)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        plt.locator_params(axis='x', nbins=10)
        plt.subplot(311)
        plt.plot(mag_lp, label='Acc low pass magnitude')
        ax = plt.gca()
        self.label_ax = ax
        _ = Cursor(ax, useblit=True, color='red', linewidth=2)

        def format_func(x_tick, pos=None):
            this_index = np.clip(int(x_tick + 0.5), 0, data_len - 1).item()
            return utc_date[this_index]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        _ = widgets.SpanSelector(ax,
                                 self.onselect,
                                 'horizontal',
                                 rectprops=dict(facecolor='blue', alpha=0.5))
        if labels is not None and len(labels) > 0:
            for style, start_idx, end_idx in labels:
                span = ax.axvspan(start_idx, end_idx, alpha=0.5)
                self.label_vspans.append(span)
        plt.grid()
        plt.legend(loc='upper right')
        plt.subplot(312, sharex=ax)
        plt.plot(acc_lp.T[0], label='acc_lp x')
        plt.plot(acc_lp.T[1], label='acc_lp y')
        plt.plot(acc_lp.T[2], label='acc_lp z')
        plt.grid()
        plt.legend(loc='upper right')
        plt.subplot(313, sharex=ax)
        plt.plot(mag, label='Acc raw magnitude')
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()
        print(f'Label done, segments num: {len(self.labels)}: {self.labels}')
        return self.labels


def merge_labels(labels):
    if not labels:
        return []
    res = []
    tmp = labels[:]
    while len(tmp) > 0:
        res.append(tmp.pop(0))
        label = res[-1][0]
        start = res[-1][1]
        end = res[-1][2]
        merged_list = []
        for i, l in enumerate(tmp):
            merged = False
            if l[0] == label:
                if start <= l[1] and l[2] <= end:
                    merged = True
                if l[1] <= start <= l[2]:
                    start = l[1]
                    merged = True
                if l[1] <= end <= l[2]:
                    end = l[2]
                    merged = True
            if merged:
                merged_list.append(l)
        for item in merged_list:
            tmp.remove(item)
        if len(merged_list) > 0:
            print(f'## Expand {res[-1][1]}-{res[-1][2]} to {start}-{end}')
            print(f'#### Merged list: {merged_list}')
        else:
            print(f'## Preserve {res[-1][1]}-{res[-1][2]}')
        res[-1][1] = start
        res[-1][2] = end

    return res


def test_merge():
    test = [[1, 5, 10], [1, 7, 12], [1, 2, 6], [1, 20, 30], [1, 25, 29],
            [1, 1, 14], [1, 15, 16]]
    print(f'List before merge: {test}')
    test = merge_labels(test)
    print(f'List after  merge: {test}')


def main():
    test_merge()


if __name__ == '__main__':
    main()
