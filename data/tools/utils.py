# Copyright 2020 Xiaomi

import csv
import math
import numpy as np
import os
import re
import struct
import zipfile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
import yaml


def unzip_directory(directory):
    """" This function unzips (and then deletes)
     all zip files in a directory """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if re.search(r'\.zip$', filename):
                folder_name = filename.strip('.zip')
                print("folder name:", folder_name)
                to_path = os.path.join(root, folder_name)
                zipped_file = os.path.join(root, filename)
                if not os.path.exists(to_path):
                    os.makedirs(to_path)
                    with zipfile.ZipFile(zipped_file, 'r') as zfile:
                        zfile.extractall(path=to_path)
                        print(to_path)
                    # deletes zip file
                    os.remove(zipped_file)


def read_csv_as_dict(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader)).astype(float).transpose()

    d = dict(zip(headers, [data[i] for i in range(data.shape[0])]))
    return headers, d


def exists_zip(directory):
    """ This function returns T/F whether any .zip file
     exists within the directory, recursively """
    is_zip = False
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if re.search(r'\.zip$', filename):
                is_zip = True
    return is_zip


def unzip_directory_recursively(directory, max_iter=1000):
    """ Calls unzip_directory until all contained zip files
     (and new ones from previous calls)
    are unzipped
    """
    print("Does the directory path exist? ", os.path.exists(directory))
    iterate = 0
    while exists_zip(directory) and iterate < max_iter:
        unzip_directory(directory)
        iterate += 1
    pre = "Did not " if iterate < max_iter else "Did"
    print(pre, "time out based on max_iter limit of", max_iter,
          ". Took iterations:", iterate)


def all_csv_files(directory):
    """ This function returns all .csv files in the directory, recursively """
    file_list = []
    if not os.path.isdir(directory):
        raise Exception("invalid input directory: {0}.".format(directory))
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if re.search(r'\.csv$', filename):
                file_list.append(os.path.join(root, filename))
    return file_list


def multi_plot(
        data_by_name={
            'x': np.zeros(10),
            'y': np.zeros(10),
            'anno': np.zeros(10),
            'step': np.zeros(10)
        },
        subplots=[{
            'line': ['x']
        }, {
            'dot': ['y'],
            'annotation': {
                'y': 'anno'
            }
        }],
        pickable=[],
        title='Default',
        label_file='label.log',
        change_csv=False):
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF'
    ]
    markers = [
        'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P',
        'X'
    ]

    fig, ax = plt.subplots(len(subplots),
                           sharex=data_has_same_length(data_by_name))
    fig.suptitle(title)

    def scroll(event):
        if event.inaxes:
            ax_tmp = event.inaxes
            x_min, x_max = ax_tmp.get_xlim()
            delta = (x_max - x_min) / 10
            if event.button == 'up':
                ax_tmp.set(xlim=(x_min + delta, x_max - delta))
            elif event.button == 'down':
                ax_tmp.set(xlim=(x_min - delta, x_max + delta))
            fig.canvas.draw_idle()

    positive = set()
    negative = set()

    def on_pick(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind

            if len(ind) > 2:
                print('WARN: more than one item picked, ',
                      'try zoom the figure and select 1 dot: ', xdata[ind],
                      ydata[ind])
                return

            if not event.mouseevent.dblclick:
                x, y = xdata[ind][0].item(), ydata[ind][0].item()
                if event.mouseevent.button == 1:  # left double click
                    print('+++: ', x, y)
                    positive.add(x)
                    axes = plt.gca()
                    axes.plot([x, x], [0, y], 'r', linewidth=2)
                    axes.figure.canvas.draw()
                    if x in negative:
                        negative.remove(x)
                        print('WARN: cleared previous negative label: ', x)
                elif event.mouseevent.button == 3:  # right double click
                    print('---: ', x, y)
                    negative.add(x)
                    axes = plt.gca()
                    axes.plot([x, x], [0, y], 'k--', linewidth=2)
                    axes.figure.canvas.draw()
                    if x in positive:
                        positive.remove(x)
                        print('WARN: cleared previous positive label: ', x)
                else:
                    print(
                        'WARN: wrong pick, use double click 1 dot to label it, ',
                        'left for positive, right for negative')

    def on_close(event):
        if len(positive) == 0 and len(negative) == 0:
            return
        now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        key = '%s %s' % (title, now)
        label_data = {
            key: {
                'positive': list(positive),
                'negative': list(negative)
            }
        }
        label_str = yaml.safe_dump(label_data)
        with open(label_file, "a") as f:
            f.write(label_str)

    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('close_event', on_close)

    for subplot_i in range(len(subplots)):
        ax_current = ax[subplot_i] if len(subplots) > 1 else ax
        subplot = subplots[subplot_i]
        # Draw lines
        color_i = 0
        if 'line' in subplot:
            for key in subplot['line']:
                if 'x' in subplot:
                    x = data_by_name[subplot['x']]
                else:
                    x = range(len(data_by_name[key]))
                if subplot_i != 2:
                    ax_current.plot(x,
                                    data_by_name[key],
                                    label=key,
                                    color=colors[color_i])
                    color_i = color_i + 1
                else:
                    ax_current.plot(x,
                                    data_by_name[key],
                                    label=key,
                                    color=colors[color_i],
                                    picker=3)
                    color_i = color_i + 1

        # Draw dots
        if 'dot' in subplot:
            marker_i = 0
            for key in subplot['dot']:
                tmp = []
                tmpp = []
                if 'x' in subplot:
                    x = data_by_name[subplot['x']]
                else:
                    x = range(len(data_by_name[key]))
                if subplot_i != 2:
                    ax_current.plot(x,
                                    data_by_name[key],
                                    markers[marker_i],
                                    label=key,
                                    color=colors[color_i],
                                    picker=5 if key in set(pickable) else None)
                else:
                    ax_current.stem(x,
                                    data_by_name[key],
                                    'g',
                                    use_line_collection=True,
                                    linefmt='none',
                                    basefmt='none',
                                    markerfmt='g^',
                                    label=key)
                    for i in range(len(data_by_name[key])):
                        if not np.isnan(data_by_name[key][i]):
                            tmp.append(x[i])
                            tmpp.append(data_by_name[key][i])
                            positive.add(x[i])
                            print('+a', ':', x[i], data_by_name[key][i])
                    ax_current.stem(tmp,
                                    tmpp,
                                    'r',
                                    use_line_collection=True,
                                    linefmt='r-',
                                    basefmt='none',
                                    markerfmt='none',
                                    label='auto_label')
                color_i = color_i + 1
                marker_i = marker_i + 1

        # Draw annotations
        if 'annotation' in subplot:
            for key, anno_key in subplot['annotation'].items():
                for anno_x in range(len(data_by_name[key])):
                    anno_y = data_by_name[key][anno_x]
                    anno_txt = data_by_name[anno_key][anno_x]
                    if not np.isnan(anno_y):
                        ax_current.annotate(anno_txt, (anno_x, anno_y))

        ax_current.legend()
        ax_current.grid(True, linestyle='dotted')

    plt.tight_layout()
    plt.show()
    for i in range(len(positive)):
        index = int(list(positive)[i])
        data_by_name['step'][index] = 1
    for i in range(len(negative)):
        index = int(list(negative)[i])
        data_by_name['step'][index] = -1
    if len(positive) == 0:
        print('There is not positive label!')
    if change_csv:
        f = open(title, 'r', encoding='utf-8')
        reader = csv.reader(f)
        file = []
        for line in reader:
            file.append(line)
        f.close()
        f = open(title, 'w', encoding='utf-8')
        for i in range(len(positive)):
            index = int(list(positive)[i])
            file[index + 1][-3] = 1
        for i in range(len(negative)):
            index = int(list(negative)[i])
            file[index + 1][-3] = -1
        writer = csv.writer(f, delimiter=",")
        for i in range(len(file)):
            writer.writerow(file[i])
        f.close()
    return data_by_name


def data_has_same_length(d):
    num_points = -1
    for key in d:
        if num_points == -1:
            num_points = len(d[key])
        elif num_points != len(d[key]):
            return False

    return True
