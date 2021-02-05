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
import onnx
import onnxruntime
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
import torch
from torch import nn

from activity_data_labeler import (LABEL_DAILY, LABEL_ITEMS,
                                   LABEL_ITEMS_INDEX_DICT, LABEL_OTHER_SPORTS)
from utils import log, plotting
from utils.common import ewma, load_dataset
from utils.model_utils import GeneralModelPredictor
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

CNN_MODEL_PATH = Path('./models/model-cnn-20200201.pth')
CNN_ONNX_MODEL_PATH = CNN_MODEL_PATH.with_suffix('.onnx')


def prepare_training(train_dir, test_dir, train_cfg):
    fs = 26
    # duration = 4.04  # int(4.04 * 26) = 105
    # duration = 8.08  # int(8.08 * 26) = 210
    duration = 8.35  # int(8.35 * 26) = 217
    shift = 2
    use_amp = True
    filter_outlier = True
    lp_filter = False
    seed = 17
    cv_size = 0.33
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
    logger.info(f'Train x shape: {train_x.shape}')
    logger.info(f'Test  x shape: {test_x.shape}')

    # Normalize
    norm = None
    # train_x, norm = data_normalize(train_x)
    # test_x = normalize_data(test_x, norm)

    train_x = np.transpose(train_x, (0, 2, 1))
    test_x = np.transpose(test_x, (0, 2, 1))
    # train_y = train_y.reshape((-1, 1))
    # test_y = test_y.reshape((-1, 1))
    logger.info(f'Train x transposed shape: {train_x.shape}')
    logger.info(f'Train y shape: {train_y.shape}')
    logger.info(f'Test  x transposed shape: {test_x.shape}')

    # Change to 2d, (N,C,W) to (N,C,H,W), H=1
    old_shape = train_x.shape
    train_x = train_x.reshape(old_shape[0], old_shape[1], 1, old_shape[2])
    old_shape = test_x.shape
    test_x = test_x.reshape(old_shape[0], old_shape[1], 1, old_shape[2])

    # Training
    kfold = train_cfg['kfold']
    if kfold > 1:
        skf = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=seed)
        fold_idx = 0
        accuracy_scores = []
        for train_index, val_index in skf.split(train_x, train_y):
            print(f'KFold [{fold_idx + 1}/{kfold}]...')
            print("train", train_index, "val", val_index)
            x_train, x_cv = train_x[train_index], train_x[val_index]
            y_train, y_cv = train_y[train_index], train_y[val_index]
            model_path = CNN_MODEL_PATH.with_suffix(f'.cv.{fold_idx}.pth')
            accuracy = start_training(x_train, y_train, x_cv, y_cv, test_x,
                                      test_y, norm, model_path, train_cfg)
            accuracy_scores.append(accuracy)
            fold_idx += 1
        for i, accuracy in enumerate(accuracy_scores, 1):
            print(f'K-Fold Accuracy for fold {i}: {accuracy:0.4f}')
        print(f'KFold Accuracy mean: {np.mean(accuracy_scores):0.2f} '
              f'+/- {np.std(accuracy_scores):0.2f}')
    else:
        # Shuffle
        train_x, train_y = shuffle(train_x, train_y, random_state=seed)

        # Split cv set from train set
        train_x, cv_x, train_y, cv_y = train_test_split(train_x,
                                                        train_y,
                                                        test_size=cv_size,
                                                        random_state=seed,
                                                        stratify=train_y)
        model_path = CNN_MODEL_PATH
        start_training(train_x, train_y, cv_x, cv_y, test_x, test_y, norm,
                       model_path, train_cfg)
        # seq_len = int(duration * fs)
        # convert_model_to_onnx((1, 6, seq_len))


def predicted_result_smooth(predicted_results, win_size):
    smoothed = []
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


class ActivityCNN2D(nn.Module):
    def __init__(self,
                 seq_len=90,
                 in_ch=6,
                 hidden_ch=20,
                 out_ch=6,
                 dropout_p=0.1):
        """Using Conv2d for using MACE Micro

        Treat the sequence as image with shape (in_ch, 1, seq_len)
        Input size (N, in_ch, seq_len) should be reshape to (N, in_ch, 1, seq_len)

        Parameters
        ----------
        seq_len : int, optional
            Input sequence length, by default 90
        in_ch : int, optional
            Input channel, by default 6
        hidden_ch : int, optional
            Hidden channel, by default 20
        out_ch : int, optional
            Output channel size, by default 6
        dropout_p : float, optional
            Dropout ratio, by default 0.1
        """
        super(ActivityCNN2D, self).__init__()

        # Normalize or standardize parameter
        self.norm = None

        self.bn0 = nn.BatchNorm2d(in_ch)

        stride = seq_len // 31
        kernel_size = stride
        self.conv1 = nn.Conv2d(in_ch,
                               hidden_ch,
                               kernel_size=(1, kernel_size),
                               stride=(1, stride),
                               padding=0)
        logger.debug(f'conv1 weight shape: {self.conv1.weight.shape}')
        self.out_len1 = (seq_len - kernel_size) // stride + 1
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        # self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.out_len1 = self.out_len1 // 2
        logger.debug(f'conv1 output seq len: {self.out_len1}')

        kernel_size = 3
        self.conv2 = nn.Conv2d(hidden_ch,
                               hidden_ch,
                               kernel_size=(1, kernel_size),
                               stride=(1, 2),
                               dilation=1)
        logger.debug(f'conv2 weight shape: {self.conv2.weight.shape}')
        self.out_len2 = (self.out_len1 - kernel_size) // 2 + 1
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(hidden_ch)
        logger.debug(f'conv2 output seq len: {self.out_len2}')

        self.conv3 = nn.Conv2d(hidden_ch,
                               hidden_ch,
                               kernel_size=(1, kernel_size),
                               stride=(1, 2),
                               dilation=1)
        logger.debug(f'conv3 weight shape: {self.conv3.weight.shape}')
        self.out_len3 = (self.out_len2 - 1 * (kernel_size - 1) - 1) // 2 + 1
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(hidden_ch)
        logger.debug(f'conv3 output seq len: {self.out_len3}')

        k4 = 3
        self.conv4 = nn.Conv2d(hidden_ch,
                               hidden_ch,
                               kernel_size=(1, k4),
                               stride=(1, 2),
                               dilation=1)
        self.out_len = (self.out_len3 - 1 * (k4 - 1) - 1) // 2 + 1
        self.relu4 = nn.ReLU()
        self.flat = nn.Flatten()
        self.bn4 = nn.BatchNorm1d(num_features=self.out_len * hidden_ch)
        logger.debug(f'conv4 weight shape: {self.conv4.weight.shape}')
        logger.debug(f'conv4 output seq len: {self.out_len}')

        self.dense = nn.Linear(hidden_ch * self.out_len, out_ch)
        logger.debug(f'dense weight shape: {self.dense.weight.shape}')

        self.net = nn.Sequential(self.bn0, self.conv1, self.relu1, self.bn1,
                                 self.conv2, self.relu2, self.bn2, self.conv3,
                                 self.relu3, self.bn3, self.conv4, self.relu4,
                                 self.flat, self.bn4, self.dense)

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


def start_training(train_x, train_y, cv_x, cv_y, test_x, test_y, norm,
                   model_path, train_cfg):
    lr = train_cfg['lr']
    batch_size = train_cfg['batch-size']
    epochs = train_cfg['epochs']

    # Remove other style
    # train_x, train_y = remove_other_type(train_x, train_y)
    # cv_x, cv_y = remove_other_type(cv_x, cv_y)
    # test_x, test_y = remove_other_type(test_x, test_y)
    cm_labels = ACTIVITY_TYPE
    cm_names = ACTIVITY_TYPE_NAME

    # TODO: CHANGE THIS WHEN TRAINING DIFFERENT MODEL
    model = ActivityCNN2D(seq_len=train_x.shape[-1], in_ch=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weight = compute_class_weight('balanced', classes=cm_labels, y=train_y)
    weight = torch.tensor(weight, dtype=torch.float32)
    logger.info(f'Loss weight: {weight}')
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    data = train_x, train_y, cv_x, cv_y, test_x, test_y, norm

    return training(data,
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


def add_value_info_for_constants(model: onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)


def show_model_param(model_path, input_shape):
    print('\nConverting model to ONNX format:')
    state = torch.load(model_path)
    norm = state['norm']
    print(f'Norm param: \n{norm}')
    model = ActivityCNN2D(seq_len=input_shape[-1], in_ch=input_shape[1])
    model.load_state_dict(state['state_dict'])
    print(model.bn0.weight)
    print(1 / model.bn0.weight)
    print(model.bn0.bias)


def convert_model_to_onnx(torch_model_path,
                          onnx_model_path,
                          input_shape=(1, 6, 210)):
    print('\nConverting model to ONNX format:')
    print(f'From {torch_model_path} to {onnx_model_path}')
    model = ActivityCNN2D(seq_len=input_shape[-1], in_ch=input_shape[1])
    state = torch.load(torch_model_path)
    norm = state['norm']
    print(f'Norm param: \n{norm}')
    model.load_state_dict(state['state_dict'])
    model.eval()

    # torch.save(state,
    #            torch_model_path.with_suffix('.unzip.pth'),
    #            _use_new_zipfile_serialization=False)

    count_torch_model_parameters(model)

    input_name = 'input'
    output_name = 'output'
    x = torch.randn(input_shape)
    torch.onnx.export(model,
                      x,
                      onnx_model_path,
                      verbose=True,
                      input_names=[input_name],
                      output_names=[output_name],
                      keep_initializers_as_inputs=True)
    # Check
    print('Checking converted model...')
    onnx_model = onnx.load(str(onnx_model_path))
    onnx.helper.printable_graph(onnx_model.graph)
    onnx.checker.check_model(onnx_model)
    print('Checking done')

    # onnx_model = onnx.utils.polish_model(onnx_model)

    # Test
    print('Testing converted model...')
    session = onnxruntime.InferenceSession(str(onnx_model_path))
    out_torch = model(x)
    print(f'Output shape: {out_torch.shape}')
    out_onnx = session.run(None, {session.get_inputs()[0].name: to_numpy(x)})
    np.testing.assert_allclose(to_numpy(out_torch),
                               out_onnx[0],
                               rtol=1e-05,
                               atol=1e-05)


@click.group(invoke_without_command=False)
def main():
    pass


@main.command()
@click.option('-train', '--train-dir', help='Training data directory')
@click.option('-test', '--test-dir', help='Testing data directory')
@click.option('-lr', '--learning-rate', default=1e-3)
@click.option('-e', '--epochs', default=15)
@click.option('-k', '--kfold', default=0)
@click.option('-b', '--batch-size', default=128)
def train(train_dir, test_dir, learning_rate, epochs, kfold, batch_size):
    train_cfg = {
        'lr': learning_rate,
        'epochs': epochs,
        'kfold': kfold,
        'batch-size': batch_size
    }
    if train_dir is not None and test_dir is not None:
        train_dir = Path(train_dir)
        test_dir = Path(test_dir)
        prepare_training(train_dir, test_dir, train_cfg)


@main.command()
@click.argument('torch-model-path', default=str(CNN_MODEL_PATH))
@click.argument('onnx-model-path', default=str(CNN_ONNX_MODEL_PATH))
def convert(torch_model_path, onnx_model_path):
    convert_model_to_onnx(Path(torch_model_path),
                          Path(onnx_model_path),
                          input_shape=(1, 4, 1, 217))


@main.command()
@click.argument('file-path')
def eval(file_path):
    evaluate_single_file(Path(file_path))


@main.command()
@click.argument('model-path', default=str(CNN_MODEL_PATH))
def show(model_path):
    show_model_param(model_path, input_shape=(1, 4, 1, 217))


if __name__ == '__main__':
    main()
