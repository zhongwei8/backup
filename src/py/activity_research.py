#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-15

import logging
from pathlib import Path
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from tabulate import tabulate
import torch
from torch import nn

from activity_data_labeler import (LABEL_DAILY, LABEL_ITEMS,
                                   LABEL_ITEMS_INDEX_DICT, LABEL_OTHER_SPORTS)
from activity_data_research import load_sensor_data_and_labels_by_name
from utils import log, plotting
from utils.common import ewma, load_data_file, load_dataset
from utils.model_utils import GeneralModelPredictor
from utils.plotting import pretty_plot_confusion_matrix
from utils.trainer import data_normalize, normalize_data, training

logger = log.create_logger('ActivityResearch', level=logging.DEBUG)

# Other type from type name to int label
LABEL_OTHER = LABEL_DAILY + LABEL_OTHER_SPORTS
LABEL_BRISKING = ['BriskWalkInDoor', 'BriskWalkOutSide', 'SlowWalk']
LABEL_RUNNING = ['RuningInDoor', 'RuningOutSide']
LABEL_BIKING = ['BikingOutSide']
LABEL_ROWING = ['RowingMachine']
LABEL_ELLIPTICAL = ['EllipticalMachine']

LABEL_IN_USE = LABEL_OTHER + LABEL_BRISKING + LABEL_RUNNING + LABEL_BIKING + LABEL_ROWING + LABEL_ELLIPTICAL
LABEL_IN_USE_INDEX = [
    LABEL_ITEMS_INDEX_DICT.get(name) for name in LABEL_IN_USE
]

# Map the origin index to activity type for model training
LABEL_ACTIVITY_NAME_MAP = [
    LABEL_OTHER, LABEL_BRISKING, LABEL_RUNNING, LABEL_BIKING, LABEL_ROWING,
    LABEL_ELLIPTICAL
]
LABEL_ACTIVITY_TYPE_MAP = []
for label in LABEL_ACTIVITY_NAME_MAP:
    LABEL_ACTIVITY_TYPE_MAP.append(
        [LABEL_ITEMS_INDEX_DICT.get(name) for name in label])

ACTIVITY_TYPE_NAME = [
    "Others", "Walk", "Run", "Rowing", "Elliptical", "Biking"
]
ACTIVITY_TYPE = [i for i in range(len(ACTIVITY_TYPE_NAME))]
ACTIVITY_TYPE_WEIGHT = {0: 1, 1: 1, 2: 1, 3: 1, 4: 3, 5: 1}
ACTIVITY_TYPE_NAME_COLORS = {
    'Other': 'r',
    'Walk': 'g',
    'Runn': 'b',
    'Rowing': 'g',
    'Elliptical': 'b',
    'Biking': 'r'
}

DATA_X_FILE = Path('../data/activity_data_x_raw.npy')
DATA_Y_FILE = Path('../data/activity_data_y_raw.npy')
FEAT_X_FILE = Path('../data/activity_data_x_feat.npy')
FEAT_Y_FILE = Path('../data/activity_data_y_feat.npy')

DATA_X_FILE_TRAIN = Path('../data/activity_data_x_raw_train.npy')
DATA_Y_FILE_TRAIN = Path('../data/activity_data_y_raw_train.npy')
FEAT_X_FILE_TRAIN = Path('../data/activity_data_x_feat_train.npy')
FEAT_Y_FILE_TRAIN = Path('../data/activity_data_y_feat_train.npy')

DATA_X_FILE_TEST = Path('../data/activity_data_x_raw_test.npy')
DATA_Y_FILE_TEST = Path('../data/activity_data_y_raw_test.npy')
FEAT_X_FILE_TEST = Path('../data/activity_data_x_feat_test.npy')
FEAT_Y_FILE_TEST = Path('../data/activity_data_y_feat_test.npy')

MODEL_PATH = Path('./models/model-alpha.lr.mdl')
CNN_MODEL_PATH = Path('./models/model-cnn-20200125.pth')
CNN_ONNX_MODEL_PATH = CNN_MODEL_PATH.with_suffix('.onnx')


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


def try_the_best(train_x, train_y, test_x, test_y, record_dir=None):
    train_x, cv_x, train_y, cv_y = train_test_split(train_x,
                                                    train_y,
                                                    test_size=0.3,
                                                    random_state=42)
    # model = KNeighborsClassifier(n_neighbors=10)
    model = DecisionTreeClassifier(max_depth=10,
                                   max_features='auto',
                                   class_weight=ACTIVITY_TYPE_WEIGHT)
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
    plotting.pretty_plot_confusion_matrix(cm,
                                          ACTIVITY_TYPE_NAME,
                                          title='Activity-DT')
    plt.show()

    # feat_names = []
    # for i in range(24):
    #     feat_names.append(f'feat-{i}')
    # # plt.figure(figsize=(600, 600))
    # tree.plot_tree(model, feature_names=feat_names, class_names=ACTIVITY_TYPE_NAME, node_ids=True, fontsize=6)
    # plt.show()


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
        lp = x[:, 3:]
        x = x[:, :3]
        mean = x.mean(axis=0)
        feat.append(x.min(axis=0))
        feat.append(x.max(axis=0))
        feat.append(mean)
        feat.append(lp.mean(axis=0))
        feat.append(x.std(axis=0))
        feat.append(x.sum(axis=0, where=x > 0))
        feat.append(-x.sum(axis=0, where=x < 0))
        ratio = np.asarray(
            [mean[0] / mean[1], mean[0] / mean[2], mean[1] / mean[2]])
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
            print(
                f'win data num for type: {i:02d}:{name:<17} is {num_of_type:04d}'
            )
    type_masks = []
    for i, map_type in enumerate(LABEL_ACTIVITY_TYPE_MAP):
        print(
            f'MAP label type: {map_type} as Activity type: {i}:{ACTIVITY_TYPE_NAME[i]}'
        )
        type_masks.append(np.isin(used_y, map_type))
    for i, mask in enumerate(type_masks):
        used_y[mask] = i
    for i in range(len(type_masks)):
        print(f'activity type num for {i}: {np.sum(used_y == i)}')
    return data_x[used_mask], used_y


def custom_normalize(data_x):
    data = normalize(data_x, axis=1)
    # data, norm = data_normalize(data_x)
    return data


def load_raw_data_and_extract_feature(x_file, y_file, x_feat_file,
                                      y_feat_file):
    print(f'\nProcessing x_file: {x_file}')
    print(f'Processing y_file: {y_file}')
    data_x = np.load(x_file)
    data_y = np.load(y_file)
    print(f'raw data x shape: {data_x.shape}')
    print(f'raw data y shape: {data_y.shape}')
    data_x, data_y = transform_labels(data_x, data_y)

    # Feature extraction
    data_x = feature_extract(data_x)
    data_x = custom_normalize(data_x)
    print(f'data x shape: {data_x.shape}')
    print(f'data y shape: {data_y.shape}')
    np.save(x_feat_file, data_x)
    np.save(y_feat_file, data_y)

    return data_x, data_y


def load_train_test_data(force=False):
    if not force and FEAT_X_FILE_TRAIN.exists():
        train_x = np.load(FEAT_X_FILE_TRAIN)
        train_y = np.load(FEAT_Y_FILE_TRAIN)
        test_x = np.load(FEAT_X_FILE_TEST)
        test_y = np.load(FEAT_Y_FILE_TEST)
    else:
        train_x, train_y = load_raw_data_and_extract_feature(
            DATA_X_FILE_TRAIN, DATA_Y_FILE_TRAIN, FEAT_X_FILE_TRAIN,
            FEAT_Y_FILE_TRAIN)
        test_x, test_y = load_raw_data_and_extract_feature(
            DATA_X_FILE_TEST, DATA_Y_FILE_TEST, FEAT_X_FILE_TEST,
            FEAT_Y_FILE_TEST)
    return train_x, test_x, train_y, test_y


def load_raw_train_test_data():
    train_x = np.load(DATA_X_FILE_TRAIN)
    train_y = np.load(DATA_Y_FILE_TRAIN)
    test_x = np.load(DATA_X_FILE_TEST)
    test_y = np.load(DATA_Y_FILE_TEST)
    train_x, train_y = transform_labels(train_x, train_y)
    test_x, test_y = transform_labels(test_x, test_y)
    train_y = train_y.reshape((-1, 1))
    test_y = test_y.reshape((-1, 1))

    return train_x, test_x, train_y, test_y


def train(train_dir, test_dir, force=False, USE_ML=False):
    fs = 26
    # duration = 4.04  # int(4.04 * 26) = 105
    duration = 8.08  # int(8.08 * 26) = 210
    shift = 2
    use_amp = False
    filter_outlier = True
    lp_filter = True
    seed = 13
    if USE_ML:
        train_x, test_x, train_y, test_y = load_train_test_data(force=force)
        print(f'train x shape: {train_x.shape}')
        print(f'test  x shape: {test_x.shape}')
        try_the_best(train_x, train_y, test_x, test_y, train_dir)
        return

        x, y = train_x, train_y
        seed = 42
        k_fold = model_selection.KFold(n_splits=10,
                                       shuffle=True,
                                       random_state=seed)
        lr(x, y, k_fold)
        lda(x, y, k_fold)
        knn(x, y, k_fold)
        naive_bayes(x, y, k_fold)
        decision_tree(x, y, k_fold)
        svm(x, y, k_fold)
    else:
        train_x, train_y = load_dataset(train_dir,
                                        fs=fs,
                                        duration=duration,
                                        shift=shift,
                                        use_amp=use_amp,
                                        filter_outlier=filter_outlier,
                                        lp_filter=lp_filter)
        test_x, test_y = load_dataset(test_dir,
                                      fs=fs,
                                      duration=duration,
                                      shift=shift,
                                      use_amp=use_amp,
                                      filter_outlier=filter_outlier,
                                      lp_filter=lp_filter)
        print(f'train x shape: {train_x.shape}')
        print(f'test  x shape: {test_x.shape}')

        model_path = CNN_MODEL_PATH
        start_training(train_x, train_y, test_x, test_y, model_path)
        seq_len = int(duration * fs)
        convert_model_to_onnx((1, 6, seq_len))


def predicted_result_smooth(predicted_results):
    smoothed = []
    cnt = 0
    py = 0
    is_stabled = False
    stable_y = 0
    win = []
    stat = np.zeros((len(ACTIVITY_TYPE), ))
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


def evaluate_single_file(file_path: Path, win_size=210, stride=52):
    predictor = ActivityPredictor()
    print(predictor.norm)
    predictor_onnx = GeneralModelPredictor(str(CNN_ONNX_MODEL_PATH), 'onnx')
    df = pd.read_csv(file_path)
    labels = df['Activity'].values
    data = df.values
    acc = data[:, 2:5]
    acc_lp = ewma(acc)
    acc = np.hstack((acc, acc_lp))

    data_x = []
    data_x_index = []
    for i in range(0, len(acc) - 210, stride):
        data_x_index.append(i + win_size // 2)
        data_x.append(acc[i:i + win_size])

    data_x = np.asarray(data_x).astype(np.float32)
    predicted_cnn = predictor.predict(data_x)
    norm_x = predictor.normalize_data(data_x)
    predicted_onnx = predictor_onnx.predict(norm_x)
    print(predicted_onnx.shape)
    predicted_onnx = np.argmax(np.squeeze(predicted_onnx), axis=1)
    predicted_labels_cnn = [ACTIVITY_TYPE_NAME[i] for i in predicted_cnn]
    predicted_labels_onnx = [ACTIVITY_TYPE_NAME[i] for i in predicted_onnx]

    labels_true = labels[data_x_index]
    plt.figure('An real case')
    plt.subplot(211)
    ax1 = plt.gca()
    plt.title(file_path.stem)
    plt.plot(acc[:, 0:3])
    plt.subplot(212, sharex=ax1)
    plt.plot(data_x_index,
             predicted_labels_cnn,
             '-ob',
             label='predicted cnn',
             markersize=3)
    plt.plot(data_x_index,
             predicted_labels_onnx,
             '-or',
             label='predicted onnx',
             markersize=3)
    plt.plot(data_x_index,
             labels_true,
             '-og',
             label='true label',
             markersize=3)
    # plt.plot(data_y, '-o', label='actual')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


class ActivityCNN(nn.Module):
    def __init__(self,
                 seq_len=90,
                 in_ch=6,
                 hidden_ch=20,
                 out_ch=6,
                 dropout_ratio=0.1):
        """
        :param seq_len:    输入序列长度
        :param in_ch:      输入channel
        :param hidden_ch:  隐藏层channel
        :param out_ch:     输出channel，即分类的类别
        """
        super(ActivityCNN, self).__init__()

        # Normalize or standardize parameter
        self.norm = None

        stride = seq_len // 15
        kernel_size = stride
        self.conv1 = nn.Conv1d(in_ch,
                               hidden_ch,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=0)
        logger.debug(f'conv1 weight shape: {self.conv1.weight.shape}')
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.out_len1 = (seq_len - kernel_size) // stride + 1
        # self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.out_len1 = self.out_len1 // 2
        logger.debug(f'conv1 output seq len: {self.out_len1}')

        kernel_size = 3
        self.conv2 = nn.Conv1d(hidden_ch,
                               hidden_ch,
                               kernel_size,
                               stride=2,
                               dilation=1)
        logger.debug(f'conv2 weight shape: {self.conv2.weight.shape}')
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.out_len2 = (self.out_len1 - kernel_size) // 2 + 1
        logger.debug(f'conv2 output seq len: {self.out_len2}')

        self.conv3 = nn.Conv1d(hidden_ch,
                               hidden_ch,
                               kernel_size,
                               stride=2,
                               dilation=1)
        logger.debug(f'conv3 weight shape: {self.conv3.weight.shape}')
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.out_len3 = (self.out_len2 - 1 * (kernel_size - 1) - 1) // 2 + 1
        logger.debug(f'conv3 output seq len: {self.out_len3}')

        k4 = 3
        self.conv4 = nn.Conv1d(hidden_ch,
                               out_ch,
                               kernel_size=k4,
                               stride=2,
                               dilation=1)
        logger.debug(f'conv4 weight shape: {self.conv4.weight.shape}')
        self.out_len = (self.out_len3 - 1 * (k4 - 1) - 1) // 2 + 1
        logger.debug(f'conv4 output seq len: {self.out_len}')

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2,
                                 self.conv3, self.relu3, self.dropout3,
                                 self.conv4)

    def forward(self, x):
        return self.net(x)


class ActivityPredictor:
    def __init__(self, seq_len=210):
        self.seq_len = seq_len
        self.model = ActivityCNN(seq_len=seq_len)
        state = torch.load(CNN_MODEL_PATH)
        self.norm = state['norm']
        logger.info(f'Using model: {CNN_MODEL_PATH.stem}')
        logger.info(f'Model norm param: {self.norm}')
        self.model.load_state_dict(state['state_dict'])
        self.model.eval()

    def normalize_data(self, x):
        test_x = normalize_data(x, self.norm)
        test_x = np.transpose(test_x, (0, 2, 1))
        return test_x

    def predict(self, x, return_prob=False):
        test_x = normalize_data(x, self.norm)
        test_x = np.transpose(test_x, (0, 2, 1))
        logger.debug(f'test_x shape after transpose: {test_x.shape}')

        test_x = torch.from_numpy(test_x.astype(np.float32))
        outputs = self.model(test_x)
        logger.debug(f'outputs shape: {outputs.shape}')
        y_score = torch.softmax(outputs, 1)
        y_score = y_score.detach().numpy()
        y_score = np.transpose(y_score, (0, 2, 1))
        prob = np.squeeze(outputs.detach().numpy())
        if return_prob:
            return np.argmax(prob, axis=1), prob
        else:
            return np.argmax(prob, axis=1)


def calculate_weights(y, type_num):
    w = np.zeros(type_num)
    for i in range(type_num):
        w[i] = np.sum(y == i)
        print(f'Type {i}: num: {w[i]}')
    if np.any(w <= 0):
        raise RuntimeError('Some type\'s data is missing')
    return w.min() / w


def start_training(train_x,
                   train_y,
                   test_x,
                   test_y,
                   model_path,
                   cv_size=0.33,
                   random_state=42):
    # Shuffle
    train_x, train_y = shuffle(train_x, train_y, random_state=random_state)
    test_x, test_y = shuffle(test_x, test_y, random_state=random_state)

    # Split cv set from train set
    train_x, cv_x, train_y, cv_y = train_test_split(train_x,
                                                    train_y,
                                                    test_size=cv_size,
                                                    random_state=random_state,
                                                    stratify=train_y)
    # Shift and balance training data set
    # train_x, train_y = load_training_data_post_process(train_x, train_y, enable_shift=False, do_balance=True)

    # Normalize
    train_x, norm = data_normalize(train_x)
    cv_x = normalize_data(cv_x, norm)
    test_x = normalize_data(test_x, norm)

    # Training
    train_x = np.transpose(train_x, (0, 2, 1))
    train_y = train_y.reshape((-1, 1))
    cv_x = np.transpose(cv_x, (0, 2, 1))
    cv_y = cv_y.reshape((-1, 1))
    test_x = np.transpose(test_x, (0, 2, 1))
    test_y = test_y.reshape((-1, 1))
    logger.info(f'Train x transposed shape: {train_x.shape}')
    logger.info(f'Train y shape: {train_y.shape}')
    logger.info(f'CV    x transposed shape: {cv_x.shape}')
    logger.info(f'Test  x transposed shape: {test_x.shape}')

    lr = 5e-4
    batch_size = 64
    epochs = 15

    # TODO: CHANGE THIS WHEN TRAINING DIFFERENT MODEL
    model = ActivityCNN(seq_len=train_x.shape[-1], in_ch=train_x.shape[-2])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights = calculate_weights(train_y, len(ACTIVITY_TYPE))
    type_weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f'Loss weight: {type_weights}')
    criterion = torch.nn.CrossEntropyLoss(weight=type_weights)

    # Remove other style
    # train_x, train_y = remove_other_type(train_x, train_y)
    # cv_x, cv_y = remove_other_type(cv_x, cv_y)
    # test_x, test_y = remove_other_type(test_x, test_y)
    cm_labels = ACTIVITY_TYPE
    cm_names = ACTIVITY_TYPE_NAME

    data = train_x, train_y, cv_x, cv_y, test_x, test_y, norm

    training(data,
             model,
             optimizer,
             criterion,
             batch_size,
             epochs,
             model_path,
             cm_labels=cm_labels,
             cm_names=cm_names)


def count_torch_model_parameters(model):
    table = [["Modules", "Parameters"]]
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.append([name, param])
        total_params += param
    print(tabulate(table))
    print(f"Total Trainable Params: {total_params}")
    return total_params


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def convert_model_to_onnx(input_shpae=(1, 6, 210)):
    print('\nConverting model to ONNX format:')
    print(f'From {CNN_MODEL_PATH} to {CNN_ONNX_MODEL_PATH}')
    model = ActivityCNN(seq_len=input_shpae[-1], in_ch=input_shpae[1])
    state = torch.load(CNN_MODEL_PATH)
    norm = state['norm']
    print(f'Norm param: \n{norm}')
    model.load_state_dict(state['state_dict'])
    model.eval()

    count_torch_model_parameters(model)

    input_name = 'input'
    output_name = 'output'
    x = torch.randn(1, 6, 210)
    torch.onnx.export(model,
                      x,
                      CNN_ONNX_MODEL_PATH,
                      verbose=True,
                      input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={
                          input_name: {
                              0: 'batch_size'
                          },
                          output_name: {
                              0: 'batch_size'
                          }
                      })
    # Test
    session = onnxruntime.InferenceSession(str(CNN_ONNX_MODEL_PATH))
    out_torch = model(x)
    out_onnx = session.run(None, {session.get_inputs()[0].name: to_numpy(x)})
    np.testing.assert_allclose(to_numpy(out_torch),
                               out_onnx[0],
                               rtol=1e-05,
                               atol=1e-06)


@click.command()
@click.option('-train', '--train-dir', help='Training data directory')
@click.option('-test', '--test-dir', help='Testing data directory')
@click.option('-file', '--test-file', help='Evaluate single file')
@click.option('-t', '--train-model', is_flag=True, help='Training model')
@click.option('-f', '--force', is_flag=True, help='force extract feature')
@click.option('-c',
              '--convert-model',
              is_flag=True,
              help='Convert Model to ONNX')
def main(train_dir, test_dir, test_file, train_model, force, convert_model):
    if train_model and train_dir is not None and test_dir is not None:
        train_dir = Path(train_dir)
        test_dir = Path(test_dir)
        train(train_dir, test_dir, force, USE_ML=False)
    elif convert_model:
        convert_model_to_onnx()
    else:
        evaluate_single_file(Path(test_file))


if __name__ == '__main__':
    main()
