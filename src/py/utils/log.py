#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: farmer
# @Date: 2020-04-14

import logging

DEFAULT_FORMAT = '%(asctime)s/%(name)s/%(levelname)s: %(message)s'


def create_logger(name=None, level=logging.INFO, format_str=DEFAULT_FORMAT):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class Log:
    FAIL = '\033[91m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    OK_BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def d(log):
        print(f'{Log.OK_BLUE}Debug: {log}{Log.ENDC}')

    @staticmethod
    def w(log):
        print(f'{Log.WARNING}Warn: {log}{Log.ENDC}')

    @staticmethod
    def i(log):
        print(f'{Log.OK_GREEN}Info: {log}{Log.ENDC}')

    @staticmethod
    def e(log):
        print(f'{Log.FAIL}Error: {log}{Log.ENDC}')
