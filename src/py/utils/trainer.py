#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Farmer Li
# @Date: 2020-12-21

import logging
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils import tensorboard as tb
import torch.utils.data as torch_data

from utils import plotting

log_level = logging.INFO
logger = logging.getLogger('Trainer')
logger.setLevel(log_level)
ch = logging.StreamHandler()
ch.setLevel(log_level)
formatter = logging.Formatter(
    '%(asctime)s/%(name)s/%(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def save_check_point(model_path, model, optimizer, norm):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'norm': norm
    }
    torch.save(state, model_path)
    logger.info(f'Save model to: {model_path}')


def load_check_point(model_path, model):
    state = torch.load(model_path)
    norm = state['norm']
    logger.info(f'Loading model: {model_path.stem}')
    logger.info(f'Model norm param: {norm}')
    model.load_state_dict(state['state_dict'])

    return model, norm


def training(data,
             model,
             optimizer,
             criterion,
             batch_size,
             epochs,
             model_save_path,
             enable_log=True,
             log_dir: Path = None,
             log_tag=None,
             cm_labels=(1, 2, 3, 4),
             cm_names=None,
             desc=None):
    train_x, train_y, cv_x, cv_y, test_x, test_y, norm = data
    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.long))
    cv_x = torch.from_numpy(cv_x.astype(np.float32))
    train_dataset = torch_data.TensorDataset(train_x, train_y)
    train_data = torch_data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8)

    sw = None
    if enable_log:
        tag = 'train'
        if log_tag is not None:
            tag = log_tag
        logger.info(f'Using training log tag: {tag}')
        if log_dir is None:
            log_dir = f'./runs/{model_save_path.stem}/{tag}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
        else:
            log_dir = log_dir / f'{tag}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
        sw = tb.SummaryWriter(log_dir)
        if desc is not None:
            sw.add_text('Discription', desc)
        sw.add_graph(model, cv_x)
    backup_model_path = Path(log_dir) / model_save_path.name

    total_epochs = 0
    cv_num = len(cv_x)
    cv_loss_arr = []
    cv_accuracy_arr = []
    best_macro_f1 = 0
    cv_labels = None
    predictions = None
    predicted = None
    if not enable_log:
        plt.figure('Training CV data accuracy and loss')
    for i, epoch in enumerate(range(1, epochs + 1)):
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_data):
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, 1)[1]
            num_corrected = torch.eq(predicted, batch_y).sum().item()
            if epoch == 1 and step == 0:
                logger.info(f'batch_y shape: {batch_y.shape}')
                logger.info(f'loss shape: {loss.shape}')
                logger.info(f'outputs shape: {outputs.shape}')
                logger.info(f'Corrected num: {num_corrected}')
                logger.info(f'batch loss: {loss.item()}')
                print(f'outputs: {outputs}')
                print(f'predicted: {predicted}')
                print(f'actual: {batch_y}')
                # exit(0)
        # Cross validation
        model.eval()
        outputs = model(cv_x)
        out_seq_len = outputs.shape[2]
        if cv_labels is None:
            logger.info(f'Repeat cv_y in axis 1 with {out_seq_len} times')
            cv_y = np.repeat(cv_y, out_seq_len, axis=1)
            cv_y = torch.from_numpy(cv_y.astype(np.long))
            cv_labels = cv_y.flatten()
        # logger.info(f'Output shape: {outputs.shape}')
        cv_loss = criterion(outputs, cv_y)
        predicted = torch.max(outputs, 1)[1]
        num_corrected = (predicted == cv_y).sum().item()
        num_total = cv_num * out_seq_len
        cv_accuracy = num_corrected / num_total
        macro_f1 = f1_score(cv_y, predicted, average='macro')
        logger.info(
            f'Total num: {num_total}, corrected: {num_corrected}, accuracy: {cv_accuracy}'
        )

        # For summary writer
        predictions = torch.softmax(outputs, dim=1)
        if enable_log:
            for t in range(predictions.shape[1]):
                sw.add_pr_curve(f'PR - {t}', cv_labels == t,
                                predictions[:, t, :].flatten(), epoch)
            sw.add_scalar('Loss', cv_loss.item(), epoch)
            sw.add_scalar('Number Correct', num_corrected, epoch)
            sw.add_scalar('Accuracy', cv_accuracy, epoch)
            sw.add_scalar('Macro F1 Score', macro_f1, epoch)
            sw.add_scalar('Best Macro F1 Score', best_macro_f1, epoch)
            sw.add_histogram('conv1.bias', model.conv1.bias, epoch)
            sw.add_histogram('conv1.weight', model.conv1.weight, epoch)
            sw.add_histogram('conv2.bias', model.conv2.bias, epoch)
            sw.add_histogram('conv2.weight', model.conv2.weight, epoch)
        logger.info(
            f'Epoch: {total_epochs + epoch}, CV - Loss: {cv_loss}, Accuracy: {cv_accuracy}'
        )
        report = classification_report(cv_y,
                                       predicted,
                                       labels=cm_labels,
                                       target_names=cm_names)
        logger.info(f'Classification report:\n{report}')

        cv_accuracy_arr.append(cv_accuracy)
        cv_loss_arr.append(cv_loss.detach().item())
        if not enable_log:
            plt.cla()
            plt.subplot(211)
            plt.plot(cv_accuracy_arr, "-og", linewidth=2.0, label="accuracy")
            plt.subplot(212)
            plt.plot(cv_loss_arr, "-og", linewidth=2.0, label="cv loss")
            plt.pause(0.1)

        save_check_point(backup_model_path.with_suffix(f'.{i}.pth'), model,
                         optimizer, norm)
        if macro_f1 > best_macro_f1:
            logger.info(
                f'Marco F1: {macro_f1} better than previous: {best_macro_f1}')
            best_macro_f1 = macro_f1
            save_check_point(model_save_path, model, optimizer, norm)
    if not enable_log:
        plt.ioff()
        plt.show()

    cv_labels = cv_labels.detach().numpy().flatten().reshape((-1, 1))
    pred = predicted.detach().numpy().flatten().reshape((-1, 1))
    logger.info(f'cv_labels shape: {cv_labels.shape}')
    cv_y_score = np.transpose(predictions.detach().numpy(), (0, 2, 1))
    cv_y_score = cv_y_score.reshape((-1, cv_y_score.shape[-1]))
    logger.info(f'cv_y_score shape: {cv_y_score.shape}')
    cv_cm = confusion_matrix(y_true=cv_labels, y_pred=pred, labels=cm_labels)
    if enable_log:
        cv_fig = plotting.pretty_plot_confusion_matrix(
            cv_cm, cm_names, title=f'{model_save_path.stem} CM-CV')
        sw.add_figure('ConfusionMatrix-CV', cv_fig)

    # Plot test set confusion matrix and save
    test_x = torch.from_numpy(test_x.astype(np.float32))
    test_y = test_y.reshape((-1, 1))
    model.eval()
    outputs = model(test_x)
    test_y = np.repeat(test_y, outputs.shape[-1], axis=0)
    labels = test_y.flatten()
    predicted = torch.max(outputs, 1)[1]
    logger.info(f'predicted shape: {predicted.shape}')
    predicted = predicted.detach().numpy()
    predicted = predicted.flatten()
    test_cm = confusion_matrix(y_true=labels,
                               y_pred=predicted,
                               labels=cm_labels)
    if enable_log:
        test_fig = plotting.pretty_plot_confusion_matrix(
            test_cm, cm_names, title=f'{model_save_path.stem} CM-Test')
        sw.add_figure('ConfusionMatrix-Test', test_fig)
        sw.close()
    else:
        plotting.pretty_plot_confusion_matrix(
            cv_cm, cm_names, title=f'{model_save_path.stem} CM-CV')
        plotting.pretty_plot_confusion_matrix(
            test_cm, cm_names, title=f'{model_save_path.stem} CM-Test')
        # PR curve
        plotting.plot_pr_curve(cv_labels, cv_y_score)
        plt.show()


def data_normalize(x, method='z-score'):
    """Data normalize along each channel

    Args:
        x (np.ndarray): Data to normalize, shape should be (batch num * seq_len * channel num)
        method (str, optional): Normalize method. Defaults to 'z-score'.

    Returns:
        tuple: tuple of normalize data and norm parameter
    """
    logger.info(f'Norm type: {method}')
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
    norm = np.asarray(norm)
    logger.info(f'Calculated norm parameter: {norm}')
    return data, norm


def normalize_data(data, norm):
    data = (data - norm[0]) * norm[1]
    return data
