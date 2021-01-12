#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:46:26 2020

@author: mi
"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import argparse

# dir = "/home/mi/sensor-algorithms/dataset/activity-recognition/1/sensor-data/activity-recognition/collect/"
dir = ""
output = ""


def all_csv_files(directory):
    """ This function returns all .csv files in the directory, recursively """
    file_list = []
    # if not os.path.isdir(directory):
    #     raise Exception("invalid input directory: {0}.".format(directory))
    # for root, dirs, files in os.walk(directory):
    #     for filename in files:
    #         if re.search(r'\.csv$', filename):
    #             file_list.append(os.path.join(root, filename))
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            all_csv_files(file_path)
        elif os.path.splitext(file_path)[1] == '.csv':
            file_list.append(file_path)
    return file_list


def read_csv_as_dict(file_name, change=False):
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        if change:
            data = list(reader)
        else:
            data = np.array(list(reader)).transpose()
    if not change:
        d = dict(zip(headers, [data[i] for i in range(data.shape[0])]))
    else:
        d = data
    return headers, d


def find_nearest(array, value):
    array = np.asarray(array).astype(np.int64)
    value = np.asarray(value).astype(np.int64)
    idx = (np.abs(array - value)).argmin()
    return idx


def new_interp(dir, output):
    file_list = all_csv_files(dir)
    output_file = ""
    e = []
    for files in file_list:
        a = files.split('_')
        b = a[-1].split('.')
        c = b[0].split('-')
        e.append(c[1])
        # print(a)
        # print(b)
    a_path = a[0].split('/')
    print(a_path[-2])
    for i in range(len(a) - 1):
        if i == 0:
            output_file += "2020"
        else:
            output_file = output_file + '_' + a[i]
    output_file += '.csv'

    # print("\n")
    ff = output_file.split("/")
    # print(f)
    output_file = ff[0] + '.csv'
    # print(output_file)
    path = output + a_path[-2] + "/"
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    h_acc, d_acc = read_csv_as_dict(file_list[e.index('accel')])
    h_gyro, d_gyro = read_csv_as_dict(file_list[e.index('gyroscope')])
    h_bpm, d_bpm = read_csv_as_dict(file_list[e.index('heartrate')])
    h_mag, d_mag = read_csv_as_dict(file_list[e.index('magnet')])
    h_ppg, d_ppg = read_csv_as_dict(file_list[e.index('ppg')])
    h_user, d_user = read_csv_as_dict(file_list[e.index('user')], True)

    headers = []
    #acc
    idx = find_nearest(d_acc['CurrentTimeMillis'], d_user[0][1])
    idx2 = find_nearest(d_acc['CurrentTimeMillis'], d_user[1][1])
    old_file1 = []
    new_file = []
    for i in range(idx2 - idx):
        old_file_line = []
        for j in h_acc:
            old_file_line.append(d_acc[j][i + idx])
        old_file1.append(old_file_line)

    long_start = old_file1[0][1].astype(np.int64)
    long_stop = old_file1[-1][1].astype(np.int64)
    # long = long_start-long_stop

    new_star = 1000000 * (long_start // 1000000)
    new_stop = 1000000 * (long_stop // 1000000)
    new_x = range(new_star, new_stop, 19230770)
    old_file_np1 = np.array(list(old_file1)).transpose()
    # print(old_file_np1)
    # print(new_x)
    for i in range(old_file_np1.shape[0]):
        # new_file.append(np.interp(np.array(new_x).astype(np.float64),
        #                           np.array(old_file_np[1]).astype(np.float64),
        #                           np.array(old_file_np[i]).astype(np.float64)))
        o = np.interp(
            np.array(new_x).astype(np.float64),
            np.array(old_file_np1[1]).astype(np.float64),
            np.array(old_file_np1[i]).astype(np.float64))

        new_file.append(np.around(o, decimals=8))
        headers.append(h_acc[i])

#gyro
    idx = find_nearest(d_gyro['CurrentTimeMillis'], d_user[0][1])
    idx2 = find_nearest(d_gyro['CurrentTimeMillis'], d_user[1][1])
    old_file = []

    for i in range(idx2 - idx):
        old_file_line = []
        for j in h_gyro:
            old_file_line.append(d_gyro[j][i + idx])
        old_file.append(old_file_line)
    old_file_np = np.array(list(old_file)).transpose()

    for i in range(2, old_file_np.shape[0]):
        o = np.interp(
            np.array(new_x).astype(np.float64),
            np.array(old_file_np[1]).astype(np.float32),
            np.array(old_file_np[i]).astype(np.float32))
        new_file.append(np.around(o, decimals=8))
        headers.append(h_gyro[i])

    #bpm
    idx = find_nearest(d_bpm['CurrentTimeMillis'], d_user[0][1])
    idx2 = find_nearest(d_bpm['CurrentTimeMillis'], d_user[1][1])
    old_file = []

    for i in range(idx2 - idx):
        old_file_line = []
        for j in h_bpm:
            old_file_line.append(d_bpm[j][i + idx])
        old_file.append(old_file_line)
    old_file_np = np.array(list(old_file)).transpose()

    for i in range(2, old_file_np.shape[0]):
        new_file.append(
            np.interp(
                np.array(new_x).astype(np.float64),
                np.array(old_file_np[1]).astype(np.float64),
                np.array(old_file_np[i]).astype(np.float64)))
        headers.append(h_bpm[i])

    if old_file_np.shape[0] == 0:
        new_file.append(np.zeros(len(new_x)))
        headers.append(h_bpm[2])

    #mag
    idx = find_nearest(d_mag['CurrentTimeMillis'], d_user[0][1])
    idx2 = find_nearest(d_mag['CurrentTimeMillis'], d_user[1][1])
    old_file = []

    for i in range(idx2 - idx):
        old_file_line = []
        for j in h_mag:
            old_file_line.append(d_mag[j][i + idx])
        old_file.append(old_file_line)
    old_file_np = np.array(list(old_file)).transpose()

    for i in range(2, old_file_np.shape[0]):
        new_file.append(
            np.interp(
                np.array(new_x).astype(np.float64),
                np.array(old_file_np[1]).astype(np.float64),
                np.array(old_file_np[i]).astype(np.float64)))
        headers.append(h_mag[i])

    #ppg

    idx = find_nearest(d_ppg['CurrentTimeMillis'], d_user[0][1])
    idx2 = find_nearest(d_ppg['CurrentTimeMillis'], d_user[1][1])
    old_file = []

    for i in range(idx2 - idx):
        old_file_line = []
        for j in h_ppg:
            old_file_line.append(d_ppg[j][i + idx])
        old_file.append(old_file_line)
    old_file_np = np.array(list(old_file)).transpose()

    for i in range(2, old_file_np.shape[0]):
        new_file.append(
            np.interp(
                np.array(new_x).astype(np.float64),
                np.array(old_file_np[1]).astype(np.float64),
                np.array(old_file_np[i]).astype(np.float64)))
        headers.append(h_ppg[i])
    headers.append('style')
    new_file.append(np.zeros(len(new_file[0])))
    new_file = np.array(new_file).transpose()
    print(new_file.shape[0])

    pp = ff[0].split('_')
    p1 = pp[-2].split('-')
    p2 = p1[0]
    output_file = p2 + output_file
    out_dir = path + output_file
    with open(out_dir, 'w') as f:
        writer = csv.writer(f)

        writer.writerow(headers)
        for i in new_file:
            writer.writerow(i)

    with open(out_dir, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader)).astype(float).transpose()

    d = dict(zip(headers, [data[i] for i in range(data.shape[0])]))
    # print(d)
    data2 = np.array(list(old_file_np1[2])).astype(float).transpose()
    data3 = np.array(list(d_acc['accel_y'])).astype(float).transpose()
    data4 = np.array(list(d_acc['accel_z'])).astype(float).transpose()
    print(data2.shape)
    print(d['accel_x'].shape)
    # print(ff[0])
    print(p2)

    # figure, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
    # # fig = plt.figure()
    # # ax = fig.gca()
    # # ax.plot(d['accel_x'], label='acc_x')
    # # ax.plot(d['accel_y'], label='acc_y')
    # # ax.plot(d['accel_z'], label='acc_z')
    # # # ax.legend()
    # # # ax1 =fig.gca()

    # # # ax.plot(data2,label = 'acc_x_ture')
    # # # ax.plot(data3, label='acc_y')
    # # # ax.plot(data4, label='acc_z')
    # # ax.legend()
    # # plt.show()
    # ax1 = axes[0]
    # ax1.plot(d['accel_x'], label='accX')
    # ax1.plot(d['accel_y'], label='accY')
    # ax1.plot(d['accel_z'], label='accZ')
    # # ax1.plot(a,label = 'style')
    # ax1.grid(True)
    # ax1.legend(loc=1)
    # #
    # ax2 = axes[1]
    # ax2.plot(d['gyro_x'], label='gyroX')
    # ax2.plot(d['gyro_y'], label='gyroY')
    # ax2.plot(d['gyro_z'], label='gyroZ')
    # # ax2.plot(b,label = 'style')
    # ax2.grid(True)
    # ax2.legend(loc=1)
    # ax3 = axes[2]
    # ax3.plot(d['mag_x'], label='magX')
    # ax3.plot(d['mag_y'], label='magY')
    # ax3.plot(d['mag_z'], label='magZ')
    # # ax3.plot(,label = 'style')
    # ax3.grid(True)
    # ax3.legend(loc=1)

    # plt.tight_layout()
    # plt.show()

    # print(a)
    # print(len(new_file))
    # print(headers)
    # print(h_bpm)


# new_interp(dir,output)
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        dest='input',
                        help="Input data files directory")
    parser.add_argument("--output",
                        required=True,
                        dest='output',
                        help="Output data files directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.input[-1] != "/":
        dir = args.input + "/"
    else:
        dir = args.input
    dirs = os.listdir(dir)
    if args.output[-1] != "/":
        output = args.output + "/"
    else:
        output = args.output
    a = 0
    for i in dirs:
        dir_new = dir + i
        dirs_new = os.listdir(dir_new)
        for j in dirs_new:
            print("============================")
            a = a + 1
            print(a)
            dir_new_new = dir_new + "/" + j
            new_interp(dir_new_new, output)
            # print(dir_new_new)
    print("============================")
