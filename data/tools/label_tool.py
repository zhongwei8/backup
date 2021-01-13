# Copyright 2020 Xiaomi
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tkinter.filedialog
import tkinter as tk
import numpy as np
import os, sys
from scipy import signal
from datetime import datetime
import csv
import platform
import matplotlib
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(src_dir)
from common.py.utils.data_utils import list_csv_files, read_csv_as_dict

label_dict = {
    "WalkOutSide": 1,
    "WalkInDoor": 2,
    "RuningOutSide": 3,
    "InCar": 4,
    "Lying": 5,
    "Biking": 6,
    "Sitting": 7,
    "Upstairs": 8,
    "Downstairs": 9,
    "Standing": 10,
    "Driving": 11,
    "RuningInDoor": 12,
    "RowingMachine": 13,
    "EllipticalMachine": 14,
    "Swimming": 15,
    "RopeJump": 16,
    "Nap": 17,
    "Others": 1000,
}

headers_rule = {
    'CurrentTimestamp': 0,
    'EventTimestamp': 1,
    'AccelX': 2,
    'AccelY': 3,
    'AccelZ': 4,
    'GyroX': 5,
    'GyroY': 6,
    'GyroZ': 7,
    'MagX': 8,
    'MagY': 9,
    'MagZ': 10,
    'Activity': 15
}


def get_activity_str(filename):
    items = filename.split('-')
    ret_val = ''
    for item in items:
        if item.startswith('scene_'):
            ret_val = item.replace('scene_', '')
        elif item in label_dict:
            ret_val = item
    return ret_val


class LowPassFilter(object):
    """ Class for butter-worth low pass filter, IIR2 """
    def __init__(self, order=8, cutoff=0.28):
        self._order = order
        self._cutoff = cutoff
        self._b, self._a = signal.butter(self._order, self._cutoff,
                                         'lowpass')  # N=8, 2*fn/fs
        self._buffer_len = min(len(self._b), len(self._a))
        self._buffer = np.zeros(self._buffer_len)

    def apply(self, val):
        tmp_sum = val
        n = self._buffer_len  # buffer is used to store previous values
        for j in range(1, n):
            tmp_sum -= self._a[j] * self._buffer[n - 1 - j]
        self._buffer[n - 1] = tmp_sum
        tmp_sum = 0
        for j in range(0, n):
            tmp_sum += self._b[j] * self._buffer[n - 1 - j]
        for j in range(0, n - 1):
            self._buffer[j] = self._buffer[j + 1]
        return tmp_sum


class MovingAvgFilter(object):
    """ Class for moving average filter """
    def __init__(self, win_len=25):
        self._len = win_len
        self._buffer = np.zeros(self._len)
        self._index = 0
        self._sum = 0

    def apply(self, val, index):
        self._index = index % self._len
        self._sum -= self._buffer[self._index]
        self._buffer[self._index] = val
        self._sum += val
        return self._sum / self._len  # it may be replaced with sum/cnt


class LabelTk(object):
    def __init__(self):
        self._file_name = ''
        self._header = ''
        self._label_indexes = []
        self._activity = 0
        self._data_len = 0
        self._data_dict = dict()

        self.top = tk.Tk()
        self.raw_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.out_path = self.output_path.get()

        self._changed = False

    def init_var(self):
        self._file_name = ''
        self._header = ''
        self._label_indexes = []
        self._activity = 0
        self._data_len = 0
        self._data_dict = dict()

        self._changed = False

    def open_file(self):
        self.init_var()
        path_ = tkinter.filedialog.askopenfilename()
        if path_ == '':
            return
        if platform.system() == 'Windows':
            path_ = path_.replace("/", "\\")
            self._file_name = path_.split("\\")[-1]
        else:
            self._file_name = path_.split("/")[-1]
        activity_str = get_activity_str(self._file_name)
        if activity_str not in self._file_name.split('-')[0]:
            self._file_name = activity_str + '-' + self._file_name
        self._activity = label_dict[activity_str]

        self.raw_path.set(path_)
        self.out_path = ''

        info, self._data_dict = read_csv_as_dict(path_)
        self._header = self._data_dict.keys()
        for i in self._header:
            if i not in headers_rule:
                headers_rule[i] = headers_rule['Activity']
                headers_rule['Activity'] += 1
        self._header = sorted(self._header, key=lambda x: headers_rule[x])
        self._data_len = len(self._data_dict['EventTimestamp'])

        self.plot(self._file_name, self._data_dict)

    def choos_dir(self):
        path_ = tkinter.filedialog.askdirectory()
        if path_ == '':
            return
        if platform.system() == 'Windows':
            path_ = path_.replace("/", "\\")

        self.output_path.set(path_)
        print(self.output_path.get())

    def plot(self, filename, data_dict):
        if 'Activity' not in self._header:
            data_dict['Activity'] = np.zeros(len(data_dict['EventTimestamp']))
            self._header.append('Activity')
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry('+0+0')
        plt.suptitle(filename, y=1)

        x = np.arange(0, len(data_dict['AccelX']))

        aspan_list = []

        def scroll(event):
            if event.inaxes:
                ax_tmp = event.inaxes
                x_min, x_max = ax_tmp.get_xlim()
                delta = (x_max - x_min) / 10
                if event.button == 'up':
                    ax_tmp.set(xlim=(x_min + delta, x_max - delta))
                elif event.button == 'down':
                    ax_tmp.set(xlim=(x_min - delta, x_max + delta))
                figure.canvas.draw_idle()

        def onselect(xmin, xmax):
            data_dict['Activity'] = np.zeros(len(data_dict['EventTimestamp']))
            indmin, indmax = np.searchsorted(x, (xmin, xmax))
            indmin = max(0, indmin)
            indmax = min(len(x) - 1, indmax)
            is_new = False
            if len(self._label_indexes) == 0:
                is_new = True
            else:
                for i in range(len(self._label_indexes)):
                    pre_min, pre_max = self._label_indexes[i]
                    if indmin in range(pre_min, pre_max) or indmax in range(
                            pre_min, pre_max) or pre_min in range(
                                indmin, indmax) or pre_max in range(
                                    indmin, indmax):
                        self._label_indexes[i] = [indmin, indmax]
                        aspans = aspan_list[i]
                        self._changed = True
                        for n in range(3):
                            aspans[n].remove()
                            aspans[n] = axes[n].axvspan(indmin,
                                                        indmax,
                                                        facecolor='0.5',
                                                        alpha=0.5)
                    else:
                        is_new = True
            if is_new:
                self._label_indexes.append([indmin, indmax])
                self._changed = True
                new_aspans = []
                for ax in axes:
                    aspan = ax.axvspan(indmin,
                                       indmax,
                                       facecolor='0.5',
                                       alpha=0.5)
                    new_aspans.append(aspan)
                aspan_list.append(new_aspans)
            figure.canvas.draw()

        def on_close(event):
            if self._changed is False:
                return

            for indexes in self._label_indexes:
                print("index:", indexes)
                for idx in range(indexes[0], indexes[1]):
                    data_dict['Activity'][idx] = self._activity
            path_ = self.output_path.get() + "/"
            print(self.output_path.get())
            if path_ == '':
                return
            if platform.system() == 'Windows':
                path_ = path_.replace("/", "\\")
            if filename in os.listdir(path_):
                label_file = os.path.join(path_, filename)
            else:
                now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                name = filename.split('.')[0]
                label_file = os.path.join(path_, name + '--' + now + '.csv')
            print(label_file)
            self.out_path = label_file
            with open(label_file, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(self._header)
                for i in range(self._data_len):
                    csv_writer.writerow(
                        [self._data_dict[j][i] for j in self._header])
            print(self.out_path)
            print('success')
            self._changed = False

        a = []
        b = []
        print(len(data_dict['Activity']))
        for i in data_dict['Activity']:
            if i != 0:
                a.append(10)
                b.append(5)
            else:
                a.append(0)
                b.append(0)

        figure.canvas.mpl_connect('scroll_event', scroll)
        # figure.canvas.mpl_connect('pick_event', on_pick)
        figure.canvas.mpl_connect('close_event', on_close)
        con = 0
        if ('AccelX' in self._header) and ('AccelY' in self._header) and (
                'AccelZ' in self._header):

            ax1 = axes[0]
            ax1.plot(data_dict['AccelX'], label='AccX')
            ax1.plot(data_dict['AccelY'], label='AccY')
            ax1.plot(data_dict['AccelZ'], label='AccZ')
            ax1.plot(a, label='Activity')
            ax1.grid(True)
            ax1.legend(loc=1)
            con += 1

            span1 = SpanSelector(ax1,
                                 onselect,
                                 'horizontal',
                                 useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'))
        #
        if ('GyroX' in self._header) and ('GyroY' in self._header) and (
                'GyroZ' in self._header):
            ax2 = axes[1]
            ax2.plot(data_dict['GyroX'], label='GyroX')
            ax2.plot(data_dict['GyroY'], label='GyroY')
            ax2.plot(data_dict['GyroZ'], label='GyroZ')
            ax2.plot(b, label='Activity')
            ax2.grid(True)
            ax2.legend(loc=1)
            con += 1

            span2 = SpanSelector(ax2,
                                 onselect,
                                 'horizontal',
                                 useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'))

        if ('MagX' in self._header) and ('MagY' in self._header) and (
                'MagZ' in self._header):
            ax3 = axes[2]
            ax3.plot(data_dict['MagX'], label='MagX')
            ax3.plot(data_dict['MagY'], label='MagY')
            ax3.plot(data_dict['MagZ'], label='MagZ')
            ax3.grid(True)
            ax3.legend(loc=1)
            con += 1

            span3 = SpanSelector(ax3,
                                 onselect,
                                 'horizontal',
                                 useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'))

        if con > 0:
            plt.tight_layout()
            plt.show()

    def callback(self):
        self.top.title('Label Window')
        self.top.geometry('1600x100')
        tk.Label(self.top, text='label_directory').grid(row=0,
                                                        column=0,
                                                        sticky='W')
        tk.Entry(self.top, textvariable=self.output_path,
                 width=160).grid(row=0, column=1)
        tk.Button(self.top, text='choose dir', command=self.choos_dir,
                  width=8).grid(row=0, column=2, sticky='W')
        tk.Label(self.top, text='raw_directory').grid(row=2,
                                                      column=0,
                                                      sticky='E')
        tk.Entry(self.top, textvariable=self.raw_path,
                 width=160).grid(row=2, column=1)
        tk.Button(self.top, text='open_file', command=self.open_file,
                  width=8).grid(row=2, column=2, sticky='W')

        self.top.mainloop()


if __name__ == '__main__':
    ann_tk = LabelTk()
    ann_tk.callback()
