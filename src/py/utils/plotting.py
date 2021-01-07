#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: farmer
# @Date: 2020-04-14

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import OneHotEncoder

from utils.confusion_matrix_pretty_print import pretty_plot_confusion_matrix as pp_cm


def plot_x(x, title="", x_label='x', y_label='y'):
    plt.figure(title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, '-o')
    plt.legend()
    plt.show()


def plot_x_with_idx2(x, idx, idx2, title="", x_label='x', y_label='y'):
    plt.figure(title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, '--or')
    cnt = 0
    if len(idx) > 0:
        for i in range(len(idx)):
            if idx2[i] == 1:
                plt.plot(idx[i], x[idx[i]], '-og')
            else:
                plt.plot(idx[i], x[idx[i]], '-ob')
            cnt += 1
            plt.annotate(cnt, xy=(idx[i], x[idx[i]]), xycoords='data',
                         xytext=(+0, +30), textcoords='offset points', fontsize=10,
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.legend()
    plt.show()


def plot_xy_in_one(x, y, title="", x_label='x', y_label='y', x_legend='x', y_legend='y'):
    plt.figure(title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, '-or', label=x_legend)
    plt.plot(y, '-og', label=y_legend)
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()
    return fig


def pretty_plot_confusion_matrix(cm, classes, title='Confusion Matrix', annot=True, cmap='Oranges', fmt='.2f', fz=11,
                                 lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='y'):
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    return pp_cm(df_cm, title, annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


def f_measure(beta, precision, recall):
    beta_2 = beta ** 2
    v = (beta_2 + 1) * precision * recall
    return v / (beta_2 * precision + recall)


def f1_measure(precision, recall):
    precision[precision <= 0] = 1e-4
    recall[recall <= 0] = 1e-4
    v = 2 * precision * recall
    return v / (precision + recall)


def plot_pr_curve(y_test: np.ndarray, y_score: np.ndarray):
    # For each class
    precision = dict()
    recall = dict()
    threshold = {}
    average_precision = dict()
    n_classes = len(y_score[0])
    enc = OneHotEncoder()
    y_test = enc.fit_transform(y_test).toarray()
    print(f'y_test: {y_test.shape}')
    print(f'y_score: {y_score.shape}')
    if not y_test.shape == y_score.shape:
        raise ValueError(f'Shape not match: y_test shape: {y_test.shape}, y_score_shape: {y_score.shape}')
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    plt.figure('PR-Curve', figsize=(7, 8))
    f_scores = np.linspace(0.3, 0.9, num=7)
    lines = []
    labels = []
    line = None
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        line, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    if line is not None:
        lines.append(line)
        labels.append('iso-f1 curves')
    line, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(line)
    labels.append('micro-average PR (area = {0:0.2f})'.format(average_precision["micro"]))

    for i in range(n_classes):
        f1 = f1_measure(precision[i], recall[i])
        j = np.argmax(f1)
        line, = plt.plot(recall[i], precision[i], lw=2)
        info = (recall[i][j], precision[i][j], threshold[i][j], f1[j])
        plt.scatter(recall[i][j], precision[i][j], s=100)
        plt.annotate(f'(%.2f,%.2f,%.3f,%.3f)' % info, xy=info[:2], xytext=(-20, 10), textcoords='offset points')
        lines.append(line)
        labels.append('PR for class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for multi-class')
    plt.legend(lines, labels, loc=(0, 0), prop=dict(size=12))
    # plt.show()
