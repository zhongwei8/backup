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
from keras.callbacks import TensorBoard, CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
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
import tabulate

CHECKPOINT_DIR = '/fds/checkpoints'
DATASET = '/fds/dataset/labeled-0525'
LOGDIR = '/fds/logs'


def cnn_model(n_timesteps,
              n_features,
              n_outputs,
              kernel_size=5,
              channels=16,
              mid_layers=1,
              mid_channels=32,
              mid_kernel_size=7,
              pool_size=2,
              dropout_rate=0.5,
              learning_rate=0.0003):
    model = Sequential()
    model.add(
        Conv1D(channels,
               kernel_size,
               input_shape=(n_timesteps, n_features),
               padding='same',
               activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding='same', strides=2))
    model.add(BatchNormalization())
    for i in range(mid_layers):
        model.add(
            Conv1D(mid_channels,
                   mid_kernel_size,
                   padding='same',
                   activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size, padding='same', strides=2))
    model.add(Conv1D(channels, kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, strides=2, padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(n_outputs,
              activation='softmax',
              kernel_regularizer=regularizers.l2(0.05)))
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training DNN Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--kfold',
                        type=int,
                        default=5,
                        help='k fold',
                        required=False)
    parser.add_argument('--epochs',
                        type=int,
                        default=30,
                        help='epochs',
                        required=False)
    parser.add_argument('--batch',
                        type=int,
                        default=128,
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
                        required=False,
                        default=DATASET,
                        dest='dir',
                        help="Dataset directory")
    parser.add_argument('--show',
                        type=str2bool,
                        help="plot waveforms and detected results.",
                        dest='show',
                        default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Loading data...')
    x_data, y_data = load_dataset(args.dir,
                                  downsample=2,
                                  duration=8,
                                  shift=1.0,
                                  use_amp=True)
    n_timesteps, n_features = x_data.shape[1], x_data.shape[2]
    n_outputs = 6
    print('Train...')

    model = KerasClassifier(build_fn=cnn_model, verbose=1)
    param_grid = dict(n_timesteps=np.array([n_timesteps]),
                      n_features=np.array([n_features]),
                      n_outputs=np.array([n_outputs]),
                      kernel_size=np.array([3, 5, 7]),
                      learning_rate=np.array([0.0003]),
                      dropout_rate=np.array([0.3, 0.5, 0.8]),
                      channels=np.array([8]),
                      mid_layers=np.array([1, 2, 3]),
                      mid_channels=np.array([8, 16, 32]),
                      mid_kernel_size=np.array([3, 5, 7]),
                      pool_size=np.array([2]),
                      nb_epoch=np.array([32, 64, 128]),
                      batch_size=np.array([8, 32, 64]))
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=-1,
                        cv=5)
    grid_result = grid.fit(x_data, y_data)

    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO)

    main()
