# Copyright 2020 Xiaomi
# Usage: python3 train.py --dir=dataset/activity-recognition/sensor-data/ --type=cnn --epochs 30 --batch 16 --save_best=True
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from keras.layers import LSTM
from keras.layers import Dense, Embedding
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
import itertools
import math
import cmath
import random
import logging
import numpy as np
import argparse
import os
from enum import Enum
import math
import csv
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import load_model
from common import *
from models import cnn_model, lstm_model, ann_model
from models import cnn_model_base, focal_loss, cnn_features_model
import tabulate
from models import GHM_Loss

ghm = GHM_Loss(bins=30)


class ModelType(Enum):
    CNN = 0
    ANN = 1
    LSTM = 2
    NA = 3
    CNNF = 4
    CNNM = 5


model_type = {
    'cnn': ModelType.CNN,
    'lstm': ModelType.LSTM,
    'ann': ModelType.ANN,
    'cnn_feature': ModelType.CNNF,
    'cnn_m': ModelType.CNNM
}

CKPT_DIR = 'tmp/checkpoints'


def str2ints(v, delimiter=','):
    str_list = v.split(delimiter)
    return [int(item) for item in str_list]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training DNN Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--kfold',
                        type=int,
                        default=5,
                        help='k fold',
                        required=False)
    parser.add_argument('--model-file',
                        type=str,
                        default='',
                        dest='model_file',
                        help='load model file',
                        required=False)
    parser.add_argument('--shift',
                        type=float,
                        default=1.0,
                        help='frame shift length',
                        required=False)
    parser.add_argument('--win-len',
                        type=float,
                        default=8.0,
                        dest='win_len',
                        help='frame shift length',
                        required=False)
    parser.add_argument('--epochs',
                        type=int,
                        default=256,
                        help='epochs',
                        required=False)
    parser.add_argument('--batch',
                        type=int,
                        default=64,
                        help='batch size',
                        required=False)
    parser.add_argument('--hidden_size',
                        type=int,
                        default=64,
                        help='lstm hidden size',
                        required=False)
    parser.add_argument('--dense_size',
                        type=int,
                        default=32,
                        help='dense lalyer size',
                        required=False)
    parser.add_argument('--save_best',
                        type=str2bool,
                        default=True,
                        help='save best weights only or not.',
                        required=False)
    parser.add_argument('--type',
                        type=str,
                        default='cnn',
                        help='use cnn or lstm model, default is cnn.')
    parser.add_argument("--dir",
                        required=True,
                        dest='dir',
                        help="Trainset and Valset directory")
    parser.add_argument("--test",
                        required=True,
                        dest='test',
                        help="Testset directory")
    parser.add_argument("--ckpt-dir",
                        required=False,
                        default=CKPT_DIR,
                        dest='ckpt_dir',
                        help="Checkpoints directory")
    parser.add_argument('--show',
                        type=str2bool,
                        help="plot waveforms and detected results.",
                        dest='show',
                        default=False)
    parser.add_argument('--use-amp',
                        type=str2bool,
                        help="use amp channel or not.",
                        dest='use_amp',
                        default=True)
    parser.add_argument('--shuffle',
                        type=str2bool,
                        help="use shuffle or not when splitting dataset.",
                        dest='shuffle',
                        default=False)
    parser.add_argument('--hidden_layers',
                        type=str2ints,
                        dest='hidden_layers',
                        default='8,16,32,16')
    parser.add_argument('--kernels',
                        dest='kernels',
                        type=str2ints,
                        default='5,5,5,5')
    parser.add_argument('--pool_size', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--l2', type=float, default=0.05)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--num_classes', type=int, default=len(CategoryNames))
    args = parser.parse_args()
    return args


def train(args, x_train, y_train, x_val, y_val, x_test, y_test, fold_idx):
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(y_train),
    #                                                   y_train)
    y_train_bin = to_categorical(y_train, num_classes=args.num_classes)
    y_val_bin = to_categorical(y_val, num_classes=args.num_classes)
    y_test_bin = to_categorical(y_test, num_classes=args.num_classes)
    print('train shape:', x_train.shape, y_train.shape)
    print('val shape:', x_val.shape, y_val.shape)
    print('test shape:', x_test.shape, y_test.shape)

    print('Build model...')
    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
    # n_features = x_train.shape[1]
    n_outputs = y_train_bin.shape[1]
    if os.path.isfile(args.model_file):
        model = load_model(args.model_file,
                           custom_objects={
                               'focal_loss_fixed': focal_loss(),
                               'ghm_class_loss': ghm.ghm_class_loss
                           })
    else:
        model = None

    type = model_type[args.type]

    if type == ModelType.ANN:
        x_train = reshape_data(x_train)
        n_features = x_train.shape[1]

    elif type == ModelType.LSTM:
        folder_name = '/lstm'
        if model is None:
            model = lstm_model(n_timesteps, n_features, n_outputs,
                               args.hidden_size, args.dense_size)
    elif type == ModelType.CNN:
        folder_name = '/cnn'
        if model is None:
            model = cnn_model_base(n_timesteps,
                                   n_features,
                                   n_outputs,
                                   hidden_layers=args.hidden_layers,
                                   kernels=args.kernels,
                                   dropout_rate=args.dropout_rate,
                                   activation=args.activation)
            # model = cnn_model(n_timesteps, n_features, n_outputs)

    elif type == ModelType.CNNF:
        folder_name = '/cnn_feature'
        if model is None:
            model = cnn_features_model(n_timesteps, n_features, n_outputs,
                                       nb_features)
    elif type == ModelType.CNNM:
        folder_name = '/cnn_m'
        if model is None:
            model = cnn_model(n_timesteps, n_features, n_outputs)

    elif type == ModelType.ANN:
        folder_name = '/ann'
        if model is None:
            model = ann_model(n_features, n_outputs)
    else:
        raise argparse.ArgumentTypeError(
            'Unsupported model type value encountered.')
    print(model.summary())
    ckpt_dir = args.ckpt_dir + folder_name
    log_dir = args.ckpt_dir + '/logs' + folder_name
    out_path = args.ckpt_dir + folder_name
    # checkpoint
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    if args.save_best:
        filepath = ckpt_dir + "/weights.best-" + str(fold_idx) + ".hdf5"
        fold_idx += 1
    else:
        filepath= ckpt_dir + "/weights-improvement-" \
                            "{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_categorical_accuracy',
                                 verbose=1,
                                 save_best_only=args.save_best,
                                 mode='max')
    lr_reduce = ReduceLROnPlateau(monitor='val_categorical_accuracy',
                                  factor=0.5,
                                  patience=20,
                                  verbose=0)
    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=30,
                                  verbose=1,
                                  mode='min',
                                  restore_best_weights=args.save_best)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    csv_log = CSVLogger(log_dir + '/trainning.log',
                        separator=',',
                        append=False)
    tensorboard = TensorBoard(log_dir=log_dir)

    callbacks_list = [checkpoint, lr_reduce, csv_log, tensorboard]
    print('Train...')
    if type == ModelType.CNNF:
        model.fit(
            [x_train, features_train],
            y_train_bin,
            batch_size=args.batch,
            epochs=args.epochs,
            validation_data=([x_val, features_val], y_val_bin),
            # class_weight=class_weights,
            callbacks=callbacks_list)
    else:
        model.fit(
            x_train,
            y_train_bin,
            batch_size=args.batch,
            epochs=args.epochs,
            validation_data=(x_val, y_val_bin),
            # class_weight=class_weights,
            callbacks=callbacks_list)
    save_model(model, 'har_cnn' + str(fold_idx), out_path)

    model = load_model(filepath,
                       custom_objects={
                           'focal_loss_fixed': focal_loss(),
                           'ghm_class_loss': ghm.ghm_class_loss
                       })
    if type == ModelType.CNNF:
        loss, acc = model.evaluate([x_test, features_test],
                                   y_test_bin,
                                   batch_size=args.batch,
                                   verbose=1)
    else:
        loss, acc = model.evaluate(x_test,
                                   y_test_bin,
                                   batch_size=args.batch,
                                   verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    print("model's inputs:", model.inputs)
    print("model's outputs:", model.outputs)
    print(f'0 num: {np.sum(y_test == 0)}')
    print(f'1 num: {np.sum(y_test == 1)}')
    print(f'2 num: {np.sum(y_test == 2)}')
    print(f'3 num: {np.sum(y_test == 3)}')
    print(f'4 num: {np.sum(y_test == 4)}')
    print(f'5 num: {np.sum(y_test == 5)}')
    if type == ModelType.CNNF:
        y_pred_probs = model.predict([x_test, features_test])
    else:
        y_pred_probs = model.predict(x_test)
    stats_evaluation(y_test,
                     y_pred_probs,
                     num_classes=len(CategoryNames),
                     shift=args.shift,
                     show=False)
    return acc


def main():
    args = parse_args()
    seed = 6
    fold_idx = 0
    if args.kfold <= 0:
        print('Loading data...')
        x_data, y_data = load_dataset(args.dir,
                                      duration=args.win_len,
                                      shift=args.shift,
                                      use_amp=args.use_amp)
        x_test, y_test = load_dataset(args.test,
                                      duration=args.win_len,
                                      shift=args.shift,
                                      use_amp=args.use_amp)
        x_train, x_val, y_train, y_val = train_test_split(x_data,
                                                          y_data,
                                                          test_size=0.33,
                                                          random_state=seed)
        acc = train(args, x_train, y_train, x_val, y_val, x_test, y_test,
                    fold_idx)
        logging.info("accuracy score is:%s" % acc)
    else:
        print('Loading data...')
        x_data, y_data = load_dataset(args.dir,
                                      duration=args.win_len,
                                      shift=args.shift,
                                      use_amp=args.use_amp)
        x_test, y_test = load_dataset(args.test,
                                      duration=args.win_len,
                                      shift=args.shift,
                                      use_amp=args.use_amp)

        skf = StratifiedKFold(n_splits=args.kfold,
                              shuffle=args.shuffle,
                              random_state=seed)
        cv_scores = []
        for train_index, val_index in skf.split(x_data, y_data):
            print("train", train_index, "val", val_index)
            x_train, x_val = x_data[train_index], x_data[val_index]
            y_train, y_val = y_data[train_index], y_data[val_index]
            acc = train(args, x_train, y_train, x_val, y_val, x_test, y_test,
                        fold_idx)
            cv_scores.append(acc)
            fold_idx += 1
        logging.info("%.2f%% (+/- %.2f%%)" %
                     (np.mean(cv_scores), np.std(cv_scores)))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO)

    main()
