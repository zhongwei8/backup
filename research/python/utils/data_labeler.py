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
from tkinter import Button
from tkinter import Entry
from tkinter import Label
from tkinter import StringVar
from tkinter import Tk
from tkinter import W
from tkinter.ttk import Combobox


class DataLabel:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    def __str__(self):
        return f'{self.label}:{self.start}-{self.end}'


class DataLabeler:
    def __init__(self, label_items, label_type='index'):
        self.win = None
        self.selected = 0
        self.labels = []
        self.label_items = label_items
        self.label_by_index = label_type == 'index'
        self.data_len = 0

    def on_cancel(self):
        print('Give up current label')
        self.win.destroy()

    def on_submit(self, com, start, end):
        label_index = com.current()
        label_str = com.get()
        self.win.destroy()
        if self.label_by_index:
            self.labels.append([label_index, start, end])
        else:
            self.labels.append([label_str, start, end])
        print(f'Labeling points: {start:^7d} - {end:^7d} as type: {label_index}:{label_str}')

    def show_label_window(self, start, end):
        self.win = Tk()
        self.win.title(f'Label {start} - {end}')
        Label(self.win, text="Start Point").grid(row=0)
        entry1 = Entry(self.win)
        entry1.grid(row=0, column=1, padx=20, pady=20)
        entry1.insert(0, str(start))

        Label(self.win, text="End Point").grid(row=1)
        entry2 = Entry(self.win)
        entry2.grid(row=1, column=1)
        entry2.insert(0, str(end))

        label_str = StringVar()
        Label(self.win, text="Label").grid(row=2)
        combobox = Combobox(self.win, textvariable=label_str, state='readonly', values=self.label_items)
        combobox.grid(row=2, column=1, padx=20, pady=20)
        combobox.current(self.selected)

        b1 = Button(self.win, text="Submit", command=lambda: self.on_submit(combobox, start, end))
        b1.grid(row=3, column=0, sticky=W, padx=5, pady=5)
        b2 = Button(self.win, text="Cancel", command=lambda: self.on_cancel())
        b2.grid(row=3, column=1, sticky=W, padx=5, pady=5)
        self.win.mainloop()

    def onselect(self, v_min, v_max):
        # print(f'Processing selected range: {v_min} - {v_max}')
        start_point = int(v_min)
        end_point = int(v_max)
        if start_point < 0:
            start_point = 0
        if end_point > self.data_len:
            end_point = self.data_len - 1
        if end_point <= start_point:
            print('Start point grater than end point, do nothing')
            return
        self.show_label_window(start_point, end_point)

    def process(self, utc_base, ts, acc, file_name, selected=0):
        self.labels = []
        self.selected = selected
        self.data_len = len(acc)
        aligned_utc = ts.copy()
        aligned_utc[0] = utc_base
        for i, t in enumerate(ts[1:], 1):
            aligned_utc[i] = aligned_utc[0] + (t - ts[0]) // 1e6
        utc_date = []
        for t in aligned_utc:
            utc_date.append(datetime.fromtimestamp(t / 1000).strftime('%m-%d %H:%M:%S'))
        print(f'UTC date length: {len(utc_date)}, from {utc_date[0]} to {utc_date[-1]}')

        print(f'Acc data shape: {acc.shape}')
        mag = np.linalg.norm(acc, axis=1)
        data_len = len(mag)
        print(f'Acc magnitude shape: {mag.shape}')

        # fig, ax = plt.subplots()
        plt.figure(file_name, figsize=(15, 8))
        plt.title(file_name)
        plt.xticks(rotation=0)
        plt.locator_params(nbins=8)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        plt.locator_params(axis='x', nbins=10)
        plt.subplot(211)
        plt.plot(mag, label='Acc magnitude')
        ax = plt.gca()
        _ = Cursor(ax, useblit=True, color='red', linewidth=2)

        def format_func(x_tick, pos=None):
            this_index = np.clip(int(x_tick + 0.5), 0, data_len - 1).item()
            return utc_date[this_index]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        _ = widgets.SpanSelector(ax, self.onselect, 'horizontal', rectprops=dict(facecolor='blue', alpha=0.5))
        plt.grid()
        plt.legend(loc='upper right')
        plt.subplot(212)
        plt.plot(acc.T[0], label='acc x')
        plt.plot(acc.T[1], label='acc y')
        plt.plot(acc.T[2], label='acc z')
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()
        print(f'Label done, labeled segments num: {len(self.labels)}: {self.labels}')
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
                    # print(f'Merge - contains: {l[1]}-{l[2]} IN {start}-{end}')
                    merged = True
                if l[1] <= start <= l[2]:
                    # print(f'Merge - overlap - START: {start} to {l[1]}')
                    start = l[1]
                    merged = True
                if l[1] <= end <= l[2]:
                    # print(f'Merge - overlap - END: {end} to {l[2]}')
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
    test = [[1, 5, 10], [1, 7, 12], [1, 2, 6], [1, 20, 30], [1, 25, 29], [1, 1, 14], [1, 15, 16]]
    print(f'List before merge: {test}')
    test = merge_labels(test)
    print(f'List after  merge: {test}')


def main():
    test_merge()


if __name__ == '__main__':
    main()
