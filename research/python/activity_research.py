#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-15

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn import tree

from activity_data_labeler import LABEL_ITEMS
from activity_data_labeler import LABEL_ITEMS_INDEX_DICT
from activity_data_labeler import LABEL_DAILY
from activity_data_labeler import LABEL_OTHER_SPORTS
from activity_data_research import load_sensor_data_and_labels_by_name
from activity_data_research import load_and_split_data
from utils import plotting
from utils.plotting import pretty_plot_confusion_matrix


# Other type from type name to int label
LABEL_OTHER = LABEL_DAILY + LABEL_OTHER_SPORTS
LABEL_BRISKING = ['BriskWalkInDoor', 'BriskWalkOutSide']
LABEL_RUNNING = ['RuningInDoor', 'RuningOutSide']
LABEL_BIKING = ['BikingOutSide']
LABEL_ROWING = ['RowingMachine']
LABEL_ELLIPTICAL = ['EllipticalMachine']

LABEL_IN_USE = LABEL_OTHER + LABEL_BRISKING + LABEL_RUNNING + LABEL_BIKING + LABEL_ROWING + LABEL_ELLIPTICAL
LABEL_IN_USE_INDEX = [LABEL_ITEMS_INDEX_DICT.get(name) for name in LABEL_IN_USE]

# Map the origin index to activity type for model training
LABEL_ACTIVITY_NAME_MAP = [LABEL_OTHER, LABEL_BRISKING, LABEL_RUNNING, LABEL_BIKING, LABEL_ROWING, LABEL_ELLIPTICAL]
LABEL_ACTIVITY_TYPE_MAP = []
for label in LABEL_ACTIVITY_NAME_MAP:
    LABEL_ACTIVITY_TYPE_MAP.append([LABEL_ITEMS_INDEX_DICT.get(name) for name in label])

ACTIVITY_TYPE_NAME = ['Other', 'BriskWalking', 'Running', 'Biking', 'RowingMachine', 'EllipticalMachine']
ACTIVITY_TYPE = [i for i in range(len(ACTIVITY_TYPE_NAME))]
ACTIVITY_TYPE_WEIGHT = {
    0: 1,
    1: 5,
    2: 5,
    3: 5,
    4: 15,
    5: 5
}
ACTIVITY_TYPE_NAME_COLORS = {
    'Other': 'r',
    'BriskWalking': 'g',
    'Running': 'b',
    'Biking': 'r',
    'RowingMachine': 'g',
    'EllipticalMachine': 'b'
}

DATA_X_FILE = Path('../data/activity_data_x_raw.npy')
DATA_Y_FILE = Path('../data/activity_data_y_raw.npy')
FEAT_X_FILE = Path('../data/activity_data_x_feat.npy')
FEAT_Y_FILE = Path('../data/activity_data_y_feat.npy')

MODEL_PATH = Path('./models/model-alpha.dt.mdl')


def data_normalize(x, method='z-score'):
    """
    Data normalize along each channel
    """
    print(f'Norm type: {method}')
    data = x
    norm = []
    if method == 'z-score':
        # shape should be (batch num * seq_len * channel num)
        old_shape = data.shape
        # Reshape to (-1, channel num)
        data = data.reshape((-1, old_shape[-1]))
        # zero-score
        bias = np.mean(data, axis=0)
        scale = 1 / np.std(data, axis=0)
        data = (data - bias) * scale
        norm.append(bias)
        norm.append(scale)
        # reshape back
        data = data.reshape(old_shape)
    else:
        raise ValueError(f'Unsupported normalize method: {method}')
    norm = np.asarray(norm)
    print(f'Calculated norm parameter: \n{norm}')
    return data, norm


def normalize_data(data, norm, method='z-score'):
    print('Apply normalize')
    print(f'norm shape: {norm.shape}')
    print(f'norm: {norm}')
    if method == 'z-score':
        data = (data - norm[0]) * norm[1]

    return data


def evaluate(model, x, y, k_fold):
    results = model_selection.cross_val_score(model, x, y, cv=k_fold)
    return np.min(results), np.mean(results), np.max(results)


def lr(x, y, k_fold):
    model = LogisticRegression()
    result = evaluate(model, x, y, k_fold)
    print(f'LR  cv score: (min, mean, max) = {result}')


def lda(x, y, k_fold):
    model = LinearDiscriminantAnalysis()
    result = evaluate(model, x, y, k_fold)
    print(f'LDA cv score: (min, mean, max) = {result}')


def knn(x, y, k_fold):
    model = KNeighborsClassifier()
    result = evaluate(model, x, y, k_fold)
    print(f'KNN cv score: (min, mean, max) = {result}')


def naive_bayes(x, y, k_fold):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    result = evaluate(model, x, y, k_fold)
    print(f'NB  cv score: (min, mean, max) = {result}')


def decision_tree(x, y, k_fold):
    print('Evaluating by DT')
    model = DecisionTreeClassifier()
    result = evaluate(model, x, y, k_fold)
    print(f'DT  cv score: (min, mean, max) = {result}')


def svm(x, y, k_fold):
    model = SVC()
    result = evaluate(model, x, y, k_fold)
    print(f'SVM cv score: (min, mean, max) = {result}')


def load_data(feat_file, label_file):
    x = np.load(feat_file)
    y = np.load(label_file)
    y = np.repeat(y, x.shape[1], axis=0)
    x = x.reshape((-1, x.shape[-1]))
    return x, y


def try_the_best(train_x, train_y, test_x, test_y, record_dir=None):
    train_x, cv_x, train_y, cv_y = train_test_split(train_x,
                                                    train_y,
                                                    test_size=0.3,
                                                    random_state=42)
    # model = KNeighborsClassifier(n_neighbors=10)
    model = DecisionTreeClassifier(max_depth=8, max_features='auto', class_weight=ACTIVITY_TYPE_WEIGHT)
    # model = LogisticRegression()
    # model = SVC()
    for i in ACTIVITY_TYPE:
        print(f'activity type num for {i}: {np.sum(train_y == i)}')
    model.fit(train_x, train_y)
    with MODEL_PATH.open('wb') as fm:
        pickle.dump(model, fm)
    print(f'model results/train: {model.score(cv_x, cv_y)}')
    print(f'model results/test : {model.score(test_x, test_y)}')
    print(f'model parameters: {model.get_params()}')
    predicted = model.predict(test_x)
    print(f'Predicted[:50]: {predicted[:50]}')
    print(f'test_y[:50]: {test_y[:50]}')
    cm = confusion_matrix(y_true=test_y, y_pred=predicted)

    # plotting.plot_confusion_matrix(cm, SWIMMING_LABELS, True, title='SwimTCN')
    plotting.pretty_plot_confusion_matrix(cm, ACTIVITY_TYPE_NAME, title='Activity-DT')
    plt.show()

    # feat_names = []
    # for i in range(24):
    #     feat_names.append(f'feat-{i}')
    # # plt.figure(figsize=(600, 600))
    # tree.plot_tree(model, feature_names=feat_names, class_names=ACTIVITY_TYPE_NAME, node_ids=True, fontsize=6)
    # plt.show()

    if record_dir is not None:
        evaluate_record_with_model(model, record_dir)


def feature_select(train_x, train_y):
    # Use gyro feat
    train_x = train_x[:, :15]
    # repeats = train_x.shape[1]
    # print(f'repeats = {repeats}')
    # train_x = train_x.reshape((-1, train_x.shape[-1]))
    # train_y = np.repeat(train_y, repeats, axis=0)

    mask = train_y != 0
    train_x = train_x[mask]
    train_y = train_y[mask] - 1

    return train_x, train_y


def feature_extract(data_x):
    feats = []
    for x in data_x:
        feat = []
        mean = x.mean(axis=0)
        feat.append(x.min(axis=0))
        feat.append(x.max(axis=0))
        feat.append(mean)
        feat.append(x.std(axis=0))
        feat.append(x.sum(axis=0, where=x > 0))
        feat.append(-x.sum(axis=0, where=x < 0))
        ratio = np.asarray([mean[0] / mean[1], mean[0] / mean[2], mean[1] / mean[2]])
        feat.append(ratio)
        zc = np.where(np.diff(np.sign(x)))[0]
        sum1 = np.sum(x > 0)
        sum2 = -np.sum(x <= 0)
        addtion = np.asarray([zc.shape[0], sum1, sum2])
        feat.append(addtion)
        feats.append(np.asarray(feat).ravel())
    return np.asarray(feats)


def transform_labels(data_x, data_y):
    used_mask = np.isin(data_y, LABEL_IN_USE_INDEX)
    used_y = data_y[used_mask]
    print(f'Origin num: {data_y.shape[0]}, used num: {used_y.shape[0]}')
    for i, name in enumerate(LABEL_ITEMS):
        num_of_type = np.sum(used_y == i)
        if num_of_type > 0:
            print(f'win data num for type: {i:02d}:{name:<17} is {num_of_type:04d}')
    type_masks = []
    for i, map_type in enumerate(LABEL_ACTIVITY_TYPE_MAP):
        print(f'MAP label type: {map_type} as Activity type: {i}:{ACTIVITY_TYPE_NAME[i]}')
        type_masks.append(np.isin(used_y, map_type))
    for i, mask in enumerate(type_masks):
        used_y[mask] = i
    for i in range(len(type_masks)):
        print(f'activity type num for {i}: {np.sum(used_y == i)}')
    return data_x[used_mask], used_y


def custom_normalize(data_x):
    return normalize(data_x, axis=1)


def train(record_dir=None, force=False):
    if force or not FEAT_X_FILE.exists() or not FEAT_Y_FILE.exists():
        data_x = np.load(DATA_X_FILE)
        data_y = np.load(DATA_Y_FILE)
        print(f'raw data x shape: {data_x.shape}')
        print(f'raw data y shape: {data_y.shape}')
        data_x, data_y = transform_labels(data_x, data_y)

        # Feature extraction
        data_x = feature_extract(data_x)
        data_x = custom_normalize(data_x)
        print(f'data x shape: {data_x.shape}')
        print(f'data y shape: {data_y.shape}')
        np.save(FEAT_X_FILE, data_x)
        np.save(FEAT_Y_FILE, data_y)
    else:
        data_x = np.load(FEAT_X_FILE)
        data_y = np.load(FEAT_Y_FILE)
        data_x, data_y = shuffle(data_x, data_y)
        data_x = custom_normalize(data_x)
        print(f'data x shape: {data_x.shape}')
        print(f'data y shape: {data_y.shape}')
        print(f'data y[:10]: {data_y[:10]}')
        ploted = []
        plt.figure('feature 0')
        for i, label in enumerate(data_y):
            if label not in ploted:
                ploted.append(label)
                plt.plot(data_x[i], '-o', label=f'{label}')
            if len(ploted) >= 6:
                break
        plt.legend()
        plt.show()

    # Split test set
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=42)

    print(f'train x shape: {train_x.shape}')
    print(f'test  x shape: {test_x.shape}')

    try_the_best(train_x, train_y, test_x, test_y, record_dir)
    return

    x, y = train_x, train_y
    seed = 42
    k_fold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    # lr(x, y, k_fold)
    # lda(x, y, k_fold)
    # knn(x, y, k_fold)
    # naive_bayes(x, y, k_fold)
    decision_tree(x, y, k_fold)
    # svm(x, y, k_fold)


def predicted_result_smooth(predicted_results):
    smoothed = []
    cnt = 0
    py = 0
    is_stabled = False
    stable_y = 0
    win = []
    stat = np.zeros((len(ACTIVITY_TYPE),))
    for y in predicted_results:
        # Logic 1
        win.append(y)
        if (len(win) > 20):
            win.pop(0)
        stat.fill(0)
        for i in ACTIVITY_TYPE:
            for t in win:
                if t == i:
                    stat[i] += 1
        percent = stat / 20
        is_stabled = False
        for i, p in enumerate(percent):
            print(p)
            if p > 0.8:
                is_stabled = True
                stable_y = i
                break
        if not is_stabled:
            stable_y = 0

        # Logic 2
        # if py == y:
        #     if not is_stabled:
        #         cnt += 1
        #         # TODO: Optimize the stable condition
        #         if cnt >= 20:  # 10s
        #             is_stabled = True
        #             stable_y = y
        # else:
        #     is_stabled = False
        #     cnt = 1
        # py = y

        # Update
        smoothed.append(stable_y)
    return np.asarray(smoothed)


def evaluate_record_with_model(model, record_dir: Path, win_size=200, stride=25):
    ts, data, label = load_sensor_data_and_labels_by_name(record_dir)
    data_x = []
    data_x_index = []
    for i in range(0, len(data) - win_size, stride):
        data_x.append(data[i: i + win_size])
        data_x_index.append(i + win_size)
    data_x = np.asarray(data_x)

    # data_x, data_y = load_and_split_data(record_dir)
    print(f'test raw data x shape: {data_x.shape}')
    data_x = feature_extract(data_x)
    data_x = custom_normalize(data_x)
    print(f'test data x shape: {data_x.shape}')
    predicted = model.predict(data_x)
    predicted_labels = [ACTIVITY_TYPE_NAME[i] for i in predicted]
    smoothed = predicted_result_smooth(predicted)
    smoothed_labels = [ACTIVITY_TYPE_NAME[i] for i in smoothed]
    print(f'model parameters: {model.get_params()}')

    plt.figure('An real case')
    plt.subplot(211)
    ax1 = plt.gca()
    plt.title(record_dir.name)
    plt.plot(data)
    plt.subplot(212, sharex=ax1)
    plt.plot(data_x_index, predicted_labels, '-or', label='predicted', markersize=3)
    plt.plot(data_x_index, smoothed_labels, '-og', label='smoothed', markersize=3)
    # plt.plot(data_y, '-o', label='actual')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


def evaluate_record(record_dir: Path, win_size=200, stride=25):
    model = None
    with MODEL_PATH.open('rb') as fm:
        model = pickle.load(fm)
    evaluate_record_with_model(model, record_dir)


@click.command()
@click.argument('record-dir')
@click.option('-t', '--train-model', is_flag=True, help='Training model')
def main(record_dir, train_model):
    if train_model:
        train(Path(record_dir), force=True)
    else:
        evaluate_record(Path(record_dir))


if __name__ == '__main__':
    main()
