# Copyright 2020 Xiaomi
import argparse
import itertools
from joblib import Memory
import os
from pathlib import Path
import random
import re
import time

import click
import csv
from keras.utils import to_categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import classification_report
from scipy import signal
from scipy import stats
import tabulate

mpl.use('Agg')

LabelDict = {
    'Undefined': 'Undefined',  # 0
    'Static': 'Others',  # 1
    'DailyActivity': 'Others',  # 2
    'OtherSports': 'Others',  # 3
    'BriskWalkInDoor': 'Walk',  # 4
    'BriskWalkOutSide': 'Walk',  # 5
    'RuningInDoor': 'Run',  # 6
    'RuningOutSide': 'Run',  # 7
    'BikingInDoor': 'Undefined',  # 8
    'BikingOutSide': 'Biking',  # 9
    'EBikingOutSide': 'Undefined',  # 10
    'MotoBikingOutSide': 'Undefined',  # 11
    'SwimmingInDoor': 'Others',  # 12
    'SwimmingOutSide': 'Others',  # 13
    'SubwayTaking': 'Others',  # 14
    'BusTaking': 'Others',  # 15
    'CarTaking': 'Others',  # 16
    'CoachTaking': 'Others',  # 17
    'Sitting': 'Others',  # 18
    'Standing': 'Others',  # 19
    'Lying': 'Others',  # 20
    'EllipticalMachine': 'Elliptical',  # 21
    'RowingMachine': 'Rowing',  # 22
    'Upstairs': 'Walk',  # 23
    'Downstairs': 'Walk',  # 24
    'Driving': 'Others',  # 25
    'RopeJump': 'Others',  # 26
    'SlowWalk': 'Walk'  # 27
}

Activities = [
    'Undefined',  # 0
    'Static',  # 1
    'DailyActivity',  # 2, Such as Working/Writing/Chatting
    'OtherSports',  # 3, Larger motions, such as Fitness/Playing ball/Washing
    'BriskWalkInDoor',  # 4
    'BriskWalkOutSide',  # 5
    'RuningInDoor',  # 6
    'RuningOutSide',  # 7
    'BikingInDoor',  # 8
    'BikingOutSide',  # 9
    'EBikingOutSide',  # 10
    'MotoBikingOutSide',  # 11
    'SwimmingInDoor',  # 12
    'SwimmingOutSide',  # 13
    'SubwayTaking',  # 14
    'BusTaking',  # 15
    'CarTaking',  # 16
    'CoachTaking',  # 17
    'Sitting',  # 18
    'Standing',  # 19
    'Lying',  # 20
    'EllipticalMachine',  # 21
    'RowingMachine',  # 22
    'Upstairs',  # 23
    'Downstairs',  # 24
    'Driving',  # 25
    'RopeJump',  # 26
    'SlowWalk'  # 27
]

CategoryNames = ["Others", "Walk", "Run", "Rowing", "Elliptical", "Biking"]

HEADER_NAMES = [
    'CurrentTimeMillis', 'EventTimestamp(ns)', 'AccelX', 'AccelY', 'AccelZ',
    'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Activity'
]

DATA_NAMES_TO_USE = ['AccelX', 'AccelY', 'AccelZ', 'Activity']


def label_to_category_idx(label):
    if label == 1000:
        label = len(Activities) - 1
    category = LabelDict[Activities[int(label)]]
    if category in CategoryNames:
        return CategoryNames.index(category)
    else:
        return -1


def label_to_category(label):
    if label == 1000:
        label = Activities[-1]
    return LabelDict[Activities[int(label)]]


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def read_csv_as_dict(file_name, has_header=True, custom_headers=None):
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        if has_header:
            headers = next(reader)
        if custom_headers is not None:
            headers = custom_headers
        data = np.array(list(reader)).astype(float).transpose()
    d = dict(zip(headers, [data[i] for i in range(data.shape[0])]))
    return headers, d


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


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


mem_1 = Memory(".cache")


@mem_1.cache()
def load_dataset(input_dir,
                 fs=26,
                 downsample=1,
                 duration=8,
                 shift=0.5,
                 use_amp=False,
                 filter_outlier=False,
                 lp_filter=False):
    data_x = []
    data_y = []
    for file_path in Path(input_dir).rglob('*.csv'):
        x, y = load_data_file(file_path, fs, downsample, duration, shift,
                              use_amp, filter_outlier, lp_filter)
        data_x.extend(x)
        data_y.extend(y)
    data_x = np.array(data_x, dtype=np.float32)
    data_y = np.array(data_y, dtype=np.float32)
    return data_x, data_y


def load_data_file(file_path,
                   fs=26,
                   downsample=1,
                   duration=8,
                   shift=0.5,
                   use_amp=True,
                   filter_outlier=False,
                   lp_filter=False):
    df = pd.read_csv(file_path, usecols=DATA_NAMES_TO_USE)
    data = df.values
    dest_fs = fs
    if downsample > 1:
        dest_fs = fs // downsample
        data = data[::downsample]

    data, activity = preprocess_data(data, use_amp, filter_outlier, lp_filter)
    x, y = slice_data(data, activity, duration, shift, dest_fs)

    return x, y


def preprocess_data(data, use_amp=True, filter_outlier=False, lp_filter=False):
    # Exception point handling, for AccX, AccY, AccZ
    if filter_outlier:
        exception_point_process(data[:, 0], inplace=True)
        exception_point_process(data[:, 1], inplace=True)
        exception_point_process(data[:, 2], inplace=True)

    if use_amp:
        activity = data[:, -1].copy()
        data[:, -1] = np.linalg.norm(data[:, :3], axis=1)
    else:
        activity = data[:, -1]
        data = data[:, :3]

    if lp_filter:
        data_lp = low_pass_filter(data[:, 0:3])
        data = np.hstack((data, data_lp))
    return data, activity


def exception_point_process(data, inplace=False, lower=-80, upper=80):
    res = data
    if not inplace:
        res = data.copy()
    max_idx = len(data) - 1
    for i, d in enumerate(data):
        if d > upper or d < lower:
            if i == 0:
                res[i] = 0
            elif i == max_idx:
                res[i] = data[i - 1]
            else:
                res[i] = (data[i - 1] + data[i + 1]) / 2
    return res


def low_pass_filter(data, N=5, Wn=0.05):
    b, a = signal.butter(N, Wn, 'lowpass', output='ba')
    data_lp = signal.filtfilt(b, a, data, axis=0)
    return data_lp


def ewma(data, inplace=False, lower=-80, upper=80):
    res = data
    if not inplace:
        res = data.copy()
    max_idx = len(data) - 1
    for i, d in enumerate(data):
        if d > upper or d < lower:
            if i == 0:
                res[i] = 0
            elif i == max_idx:
                res[i] = data[i - 1]
            else:
                res[i] = (data[i - 1] + data[i + 1]) / 2
    return res


def slice_data(in_data, activity, frame_len, shift, fs):
    win_len = int(frame_len * fs)
    stride = int(shift * fs)
    input_shape = np.shape(in_data)
    input_len = input_shape[0]
    n_frames = int((input_len - win_len) / stride) + 1
    if n_frames < 1:
        return None, None
    frames = []
    labels = []
    for i in range(0, in_data.shape[0] - win_len, stride):
        acc_raw = in_data[i:i + win_len]

        # Only accept labeled and pure data
        if activity[i] > 0 and np.all(activity[i:i + win_len] == activity[i]):
            # Convert activity to target category
            category = label_to_category_idx(activity[i])
            # Category <0 means this type data should not be use
            if category >= 0:
                frames.append(acc_raw)
                labels.append(category)
    return frames, labels


def reshape_data(data):
    data = np.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])
    return data


def plot_confusion_matrix(cm,
                          classes,
                          title='Confusion matrix',
                          cmap=plt.cm.binary):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm = cm.astype('float')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    tick_marks1 = CategoryNames
    plt.xticks(tick_marks, tick_marks1, rotation=45)
    plt.yticks(tick_marks, tick_marks1)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 '{:.3f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confuse(ypre, y_val, acc, title, plot=False):
    predictions = ypre
    y_val = np.array(y_val).astype(np.int64)
    print(y_val.shape)
    if len(y_val.shape) > 1:
        truelabel = y_val.argmax(axis=-1)
    else:
        truelabel = y_val
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    if plot:
        plt.figure(1)
        plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))
        # plt.title(loss)
        plt.title(title + ' Accuracy: ' + str(acc))
    return conf_mat


def get_pr(pos_prob, y_true):
    pos = y_true[y_true == 1]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    recall = []
    precision = []
    tp = 0
    fp = 0
    auc = 0
    for i in range(len(threshold)):
        if y[i] == 1:
            tp += 1
            recall.append(tp / len(pos))
            precision.append(tp / (tp + fp))
            auc += (recall[i] - recall[i - 1]) * precision[i]
        else:
            fp += 1
            recall.append(tp / len(pos))
            precision.append(tp / (tp + fp))
    return precision, recall, auc


def stats_confusion_matrix(y_true, y_pred, title='', plot=False):
    acc = accuracy_score(y_true, y_pred)
    conf_mat = plot_confuse(y_pred, y_true, acc, title, plot)
    return acc, conf_mat


def stats_precision_recall(y_true,
                           y_pred_probs,
                           num_classes,
                           title='',
                           plot=False):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_true_bin = to_categorical(y_true, num_classes=num_classes)
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_probs[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i],
                                                       y_pred_probs[:, i])
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_probs.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin,
                                                         y_pred_probs,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.3f}'.
          format(average_precision["micro"]))

    if plot:
        from itertools import cycle
        # setup plot details
        colors = cycle([
            'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red',
            'blue'
        ])

        plt.figure(2, figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=num_classes)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.3f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(num_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                          ''.format(CategoryNames[i], average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    return precision, recall, average_precision


def tabulate_conf_mat(conf_mat):
    conf_mat_score = conf_mat.astype('float') / conf_mat.sum(
        axis=1)[:, np.newaxis]

    cm_table_rows = []
    cm_score_table_rows = []
    for i in range(conf_mat.shape[0]):
        cm_row = [CategoryNames[i]]
        cm_score_row = [CategoryNames[i]]
        for j in range(conf_mat.shape[1]):
            cm_row.append(conf_mat[i][j])
            cm_score_row.append(conf_mat_score[i][j])
        cm_table_rows.append(cm_row)
        cm_score_table_rows.append(cm_score_row)

    cm_table_headers = ["Predict\nTrue"]
    cm_table_headers.extend(CategoryNames)
    print("\n              Confusion Matrix Count")
    print(
        tabulate.tabulate(cm_table_rows,
                          headers=cm_table_headers,
                          tablefmt='grid'))
    print("\n              Confusion Matrix Score")
    print(
        tabulate.tabulate(cm_score_table_rows,
                          headers=cm_table_headers,
                          tablefmt='grid',
                          floatfmt=".3f"))


def post_process(in_data, src_period, dest_period):
    stats_num = int(dest_period / src_period)
    data_len = in_data.shape[0]
    out_data = np.zeros(data_len)
    for i in range(data_len):
        start = i - stats_num
        end = i
        if start < 0:
            out_data[i] = in_data[i]
        else:
            out_data[i] = stats.mode(in_data[start:end])[0]
    return out_data


def stats_evaluation(y_test, y_pred_probs, num_classes, shift=0.5, show=False):
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc, conf_mat = stats_confusion_matrix(y_test, y_pred, title='', plot=show)
    precision, recall, average_precision = stats_precision_recall(
        y_test, y_pred_probs, num_classes=num_classes, title='', plot=show)

    print("Accuracy: ", acc)
    print(classification_report(y_test, y_pred, target_names=CategoryNames))
    tabulate_conf_mat(conf_mat)
    # confusion mat for 30s

    y_test_bin = to_categorical(y_test, num_classes=num_classes)
    y_test_30 = post_process(y_test, shift, 30)
    y_test_60 = post_process(y_test, shift, 60)
    y_pred_30 = post_process(y_pred, shift, 30)
    y_pred_60 = post_process(y_pred, shift, 60)
    acc_30, conf_mat_30 = stats_confusion_matrix(y_test_30,
                                                 y_pred_30,
                                                 title='30s',
                                                 plot=show)
    acc_60, conf_mat_60 = stats_confusion_matrix(y_test_60,
                                                 y_pred_60,
                                                 title='60s',
                                                 plot=show)

    print(
        "\n\n--------------------- 30s --------------------------------------------\n"
    )
    print("30s mode Accuracy: ", acc_30)
    print(
        classification_report(y_test_30, y_pred_30,
                              target_names=CategoryNames))
    tabulate_conf_mat(conf_mat_30)
    print(
        "\n\n--------------------- 60s ----------------------------------------------\n"
    )
    print("60s mode Accuracy: ", acc_60)
    print(
        classification_report(y_test_60, y_pred_60,
                              target_names=CategoryNames))
    tabulate_conf_mat(conf_mat_60)


def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)  #
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    return str_time


def save_model(model, model_name, out_path, type='.yaml'):
    if type == '.json':
        model_str = model.to_json()
    elif type == '.yaml':
        model_str = model.to_yaml()
    f_name = model_name + get_current_time()

    json_file = os.path.join(out_path, f_name + type)
    weisghts_file = os.path.join(out_path, f_name + '.weights')
    with open(json_file, 'w') as json_f:
        json_f.write(model_str)
    model.save_weights(weisghts_file)


def plot_har_prediction(filename, data_dict, predictions={}):
    print("plot data")
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry('+0+0')
    plt.suptitle(filename[:-4], y=1)

    def scroll(event):
        ax_tmp = event.inaxes
        x_min, x_max = ax_tmp.get_xlim()
        delta = (x_max - x_min) / 10
        if event.button == 'up':
            ax_tmp.set(xlim=(x_min + delta, x_max - delta))
        elif event.button == 'down':
            ax_tmp.set(xlim=(x_min - delta, x_max + delta))
        figure.canvas.draw_idle()

    # figure.canvas.mpl_connect('button_press_event', oneClick)
    figure.canvas.mpl_connect('scroll_event', scroll)
    """ plot data """
    ax1 = axes[0]
    ax1.plot(data_dict['accel_x'], label='accX')
    ax1.plot(data_dict['accel_y'], label='accY')
    ax1.plot(data_dict['accel_z'], label='accZ')
    if 'accel_amp' in data_dict:
        acc_amp = data_dict['accel_amp']
    else:
        acc_amp = np.sqrt(data_dict['accel_x']**2 + data_dict['accel_y']**2 +
                          data_dict['accel_z']**2)
    ax1.plot(acc_amp, '-', label='amp', zorder=0)
    ax1.grid(True)
    ax1.legend(loc=1)

    ax2 = axes[1]
    if len(predictions) > 0:
        idx = 0
        for key in predictions:
            offset = 10 * idx
            indicator = key + ' offset:' + str(offset)
            ax2.plot(predictions[key] + offset, '-', label=indicator, zorder=0)
            idx += 1
    ax2.grid(True)
    ax2.legend(loc=1)
    plt.tight_layout()
    plt.show()


def check_processed_data(data_file):
    raw = pd.read_csv(data_file, usecols=DATA_NAMES_TO_USE).values
    x = raw[:, 0].copy()
    y = raw[:, 1].copy()
    z = raw[:, 2].copy()
    data, _ = preprocess_data(raw, True, True, True)
    plt.figure('Processed data')
    plt.subplot(311)
    plt.plot(x, '-o', label='x', markersize=3)
    plt.plot(data[:, 0], '-o', label='x ep', markersize=3)
    plt.plot(data[:, 4], '-o', label='x lp', markersize=3)
    plt.legend()
    plt.subplot(312)
    plt.plot(y, '-o', label='y', markersize=3)
    plt.plot(data[:, 1], '-o', label='y ep', markersize=3)
    plt.plot(data[:, 5], '-o', label='y lp', markersize=3)
    plt.legend()
    plt.subplot(313)
    plt.plot(z, '-o', label='z', markersize=3)
    plt.plot(data[:, 2], '-o', label='z ep', markersize=3)
    plt.plot(data[:, 6], '-o', label='z lp', markersize=3)
    plt.legend()
    plt.show()


@click.command()
@click.argument('data-file')
def main(data_file):
    data_file = Path(data_file)
    x, y = load_data_file(data_file)
    x = np.asarray(x)
    y = np.asarray(y)
    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    x = x.reshape(-1, x.shape[-1])
    np.savetxt('./new.all.txt', x, fmt='%.8f')
    np.savetxt('./new.all.y.txt', y, fmt='%.0f')


if __name__ == "__main__":
    main()
